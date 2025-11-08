import asyncio
import cv2
import numpy as np
import json
import time
import torch
from av import VideoFrame
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer, MediaRecorder, MediaRelay
from aiortc.contrib.signaling import TcpSocketSignaling, add_signaling_arguments
import aiohttp
from aiohttp import web
import logging
from collections import deque

import sys
import os

depth_path = os.path.join(os.path.dirname(__file__), "../depth-anything-v2")
sys.path.insert(0, depth_path)
print(f"[DEBUG] Added depth path: {depth_path}")
# Import the backend module so we can access and update globals reliably
import main as backend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Live preview / throttling knobs
SHOW_PREVIEW = True     # Turn off for maximum performance. True opens an OpenCV window so you can watch the feed.
PREVIEW_EVERY = 5       # When preview is enabled, show every Nth processed frame.
PROCESS_EVERY = 20      # Run inference on every Nth incoming frame (reduces CPU/GPU load).
INFER_INPUT_SIZE = 476  # Square size fed into YOLO + Depth models (matches values used in Combined.py tuning).

# Connected WebSocket clients to push analysis results
WS_CLIENTS: set[web.WebSocketResponse] = set()

async def broadcast_result(payload: dict):
    if not WS_CLIENTS:
        return
    dead = []
    msg = json.dumps(payload)
    for ws in list(WS_CLIENTS):
        try:
            await ws.send_str(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        WS_CLIENTS.discard(ws)

async def health(request):
    return web.json_response({"status": "ok"})

# --- Obstacle detection utility adapted from Combined.py ---

def analyze_obstacles(depth_np: np.ndarray, yolo_results, h_orig: int, w_orig: int, h_input: int, w_input: int):
    """
    Inspect the lower half of the depth map to decide whether the walking lane is clear.

    Returns:
        event: 'obstacle in path' or 'path clear'
        mask: boolean mask highlighting blocked pixels (used for debugging / preview)
    """
    # Remove areas covered by detected objects so we only look at the free-ground depth.
    yolo_mask = np.zeros(depth_np.shape, dtype=np.uint8)
    for result in yolo_results or []:
        for box in result.boxes:
            x1_in, y1_in, x2_in, y2_in = box.xyxy[0]
            x1, y1 = int(x1_in * w_orig / w_input), int(y1_in * h_orig / h_input)
            x2, y2 = int(x2_in * w_orig / w_input), int(y2_in * h_orig / h_input)
            cv2.rectangle(yolo_mask, (x1, y1), (x2, y2), 255, -1)

    masked_depth_np = depth_np.copy()
    masked_depth_np[yolo_mask == 255] = np.nan  # ignore pixels already claimed by detections

    # Treat the lower half of the frame as the walkable ground
    roi_start_row = int(h_orig * 0.5)
    roi_depth = masked_depth_np[roi_start_row:, :]

    # Look for strong negative vertical gradients (ground suddenly rising = obstacle)
    v_offset = 5
    padded_roi = np.pad(roi_depth, ((v_offset, 0), (0, 0)), mode='edge')
    diff = padded_roi[v_offset:, :] - padded_roi[:-v_offset, :]
    OBSTACLE_THRESHOLD = -0.4
    obstacle_mask = (diff < OBSTACLE_THRESHOLD) & np.isfinite(roi_depth)

    # Inflate the mask so scattered pixels merge into a single blob
    kernel_size = 5
    iterations = 8
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    obstacle_mask = cv2.dilate(obstacle_mask.astype(np.uint8), kernel, iterations=iterations).astype(bool)

    # Isolate the central corridor that approximates the user's walking lane
    pathway_width_ratio = 0.3
    path_start_col = int(w_orig * (0.5 - pathway_width_ratio / 2))
    path_end_col = int(w_orig * (0.5 + pathway_width_ratio / 2))
    pathway_mask = np.zeros_like(obstacle_mask, dtype=bool)
    pathway_mask[:, path_start_col:path_end_col] = True

    pathway_obstacles_mask = obstacle_mask & pathway_mask

    pathway_area = np.sum(pathway_mask)
    blocked_area = np.sum(pathway_obstacles_mask)
    blocked_ratio = blocked_area / (pathway_area + 1e-6)

    # Same 5% threshold used in Combined.py to decide if the path is blocked
    event = "obstacle in path" if blocked_ratio > 0.05 else "path clear"
    return event, pathway_obstacles_mask

def detect_wall_by_uniformity(raw_depth_map: np.ndarray, uniformity_threshold=0.3, proximity_threshold=1.0) -> tuple[str, tuple]:
    h, w = raw_depth_map.shape
    roi_y_start, roi_y_end = int(h * 0.2), int(h * 0.8)
    roi_x_start, roi_x_end = int(w * 0.2), int(w * 0.8)
    roi_coords = (roi_x_start, roi_y_start, roi_x_end, roi_y_end)
    central_roi = raw_depth_map[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    
    if central_roi.size == 0: return None, None
    valid_points = central_roi[np.isfinite(central_roi)]
    if valid_points.size < (central_roi.size * 0.8): return None, None

    stable_points = valid_points.astype(np.float32)

    std_dev = np.std(stable_points)
    mean_depth = np.mean(stable_points)
    
    if std_dev < uniformity_threshold and mean_depth > proximity_threshold:
        return "A wall is right in front of you.", roi_coords
        
    return None, None
class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from one peer to another.
    """
    kind = "video"

    def __init__(self, track, pc, websocket_queue):
        super().__init__()
        self.track = track
        self.pc = pc
        self.websocket_queue = websocket_queue
        self.frame_count = 0
        self.last_analysis_time = 0
        self.processing_frame = False
        self._depth_toggle = 0
        self._last_depth_np = None
        
        # Temporal smoothing buffers
        self._depth_map_history = deque(maxlen=3)
        self._depth_event_history = deque(maxlen=3)
        
        # --- Trackers for obstacle alert anti-spam logic ---
        self._last_announced_obstacle_event = None
        self._last_obstacle_announcement_time = 0

        # --- NEW: Add a state tracker for detected objects to prevent spam ---
        self._announced_objects_state = {} # Key: label, Value: {time, proximity, direction}

    async def recv(self):
        frame = await self.track.recv()
        
        try:
            img = frame.to_ndarray(format="bgr24")
            if SHOW_PREVIEW and (self.frame_count % PREVIEW_EVERY == 0):
                cv2.imshow("WebRTC Preview", img)
                cv2.waitKey(1)
        except Exception as e:
            logger.error(f"Error converting frame: {e}")
            img = None
        
        if (self.frame_count % PROCESS_EVERY == 0) and (not self.processing_frame) and (img is not None):
            self.processing_frame = True
            asyncio.create_task(self.process_frame(img))
        
        self.frame_count += 1
        return frame

    async def process_frame(self, frame_bgr):
        """Process frame with YOLO and depth models"""
        try:
            if backend.yolo_model is None or backend.depth_model is None:
                logger.info("[INIT] Initializing models for processing...")
                backend.initialize_models()

            DEVICE = backend.DEVICE
            USE_FP16 = backend.USE_FP16

            input_frame = cv2.resize(frame_bgr, (INFER_INPUT_SIZE, INFER_INPUT_SIZE))
            
            results = backend.yolo_model.predict(input_frame, show=False, conf=0.5, verbose=False)
            
            need_depth = (self._depth_toggle % 2 == 0) or (self._last_depth_np is None)
            self._depth_toggle += 1
            if need_depth:
                img_rgb = input_frame[:, :, ::-1].copy()
                img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
                if USE_FP16 and DEVICE != 'cpu':
                    img_tensor = img_tensor.half()
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=USE_FP16 and DEVICE != 'cpu'):
                        depth = backend.depth_model(img_tensor)
                    depth = torch.nn.functional.interpolate(
                        depth[:, None], frame_bgr.shape[:2], mode="bilinear", align_corners=True
                    )[0, 0]
                    self._last_depth_np = depth.cpu().numpy()
            depth_np = self._last_depth_np if self._last_depth_np is not None else np.zeros(frame_bgr.shape[:2], dtype=np.float32)

            self._depth_map_history.append(depth_np)
            if len(self._depth_map_history) < self._depth_map_history.maxlen:
                return
            smoothed_depth_np = np.mean(np.array(list(self._depth_map_history)), axis=0)

            h_orig, w_orig = frame_bgr.shape[:2]
            h_input, w_input = (INFER_INPUT_SIZE, INFER_INPUT_SIZE)
            obstacle_event, pathway_mask = analyze_obstacles(smoothed_depth_np, results, h_orig, w_orig, h_input, w_input)
            wall_description, wall_roi = detect_wall_by_uniformity(smoothed_depth_np, uniformity_threshold=0.2, proximity_threshold=0.8)
            self._depth_event_history.append(obstacle_event)
            if len(self._depth_event_history) == self._depth_event_history.maxlen:
                obstacle_event = max(set(self._depth_event_history), key=list(self._depth_event_history).count)


            processed_objects = []
            overlay = frame_bgr if SHOW_PREVIEW else None

            for result in results:
                for box in result.boxes:
                    x1_input, y1_input, x2_input, y2_input = box.xyxy[0]
                    x1, y1 = int(x1_input * w_orig / w_input), int(y1_input * h_orig / h_input)
                    x2, y2 = int(x2_input * w_orig / w_input), int(y2_input * h_orig / h_input)
                    cls = int(box.cls[0])
                    label = backend.yolo_model.names[cls]
                    if label == '8':
                        label = 'door'
                    conf = float(box.conf[0]) if hasattr(box, 'conf') else 0.0

                    if SHOW_PREVIEW and overlay is not None:
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(overlay, f"{label} {conf:.2f}", (x1, max(0, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    cx, cy = min(max(cx, 0), w_orig-1), min(max(cy, 0), h_orig-1)

                    distance = smoothed_depth_np[cy, cx]
                    proximity = backend.get_proximity_label(distance)
                    direction = backend.get_clock_direction(cx, w_orig)
                    priority = backend.compute_priority(distance, direction, label)

                    processed_objects.append({
                        "label": label, "direction": direction,
                        "proximity": proximity, "priority": priority
                    })
            # Only add wall detection if no door is detected by YOLO
            has_door_detection = any(obj['label'] == 'door' for obj in processed_objects)
            if wall_description is not None:
                print(f"[DEBUG] Wall detected, has_door_detection: {has_door_detection}")
            if wall_description is not None and not has_door_detection:
                wall_obj = {
                    "label": "wall",
                    "direction": "12 o'clock",   # always straight ahead
                    "proximity": "very close",   # critical hazard
                    "priority": 100,    # High priority but not infinite
                    "description": wall_description
                }
                processed_objects.append(wall_obj)
            if SHOW_PREVIEW and overlay is not None:
                roi_start_row = int(h_orig * 0.5)
                if pathway_mask is not None and pathway_mask.any():
                    obstacle_vis = np.zeros_like(overlay)
                    roi_slice = obstacle_vis[roi_start_row:, :]
                    roi_slice[pathway_mask] = (0, 0, 255)
                    overlay[:] = cv2.addWeighted(overlay, 1.0, obstacle_vis, 0.5, 0)
                cv2.imshow("Detections", overlay)
                cv2.waitKey(1)

            summary_lines = []
            spatial_audio_objects = []  # Initialize spatial audio objects list
            OBSTACLE_COOLDOWN_SECONDS = 4

            has_state_changed = (obstacle_event != self._last_announced_obstacle_event)
            cooldown_passed = (time.time() - self._last_obstacle_announcement_time > OBSTACLE_COOLDOWN_SECONDS)

            if has_state_changed and cooldown_passed:
                if obstacle_event == "obstacle in path":
                    summary_lines.append("Warning: obstacle in your path.")
                    spatial_audio_objects.append({
                        "name": "obstacle",
                        "direction": "12 o'clock",
                        "proximity": "in path"
                    })
                elif obstacle_event == "path clear":
                    if self._last_announced_obstacle_event == "obstacle in path":
                        summary_lines.append("Path clear ahead.")
                        # Don't add "path clear" to spatial audio as it's not an object
                
                self._last_announced_obstacle_event = obstacle_event
                self._last_obstacle_announcement_time = time.time()
            
            # --- MODIFIED: Implement anti-spam logic for YOLO objects ---
            processed_objects.sort(key=lambda x: -x["priority"])
            
            # Debug: Print detected objects
            if processed_objects:
                print(f"[DEBUG] Detected objects: {[(obj['label'], obj['proximity'], obj['direction']) for obj in processed_objects]}")
            
            top_k = 3 # Consider the top 3 most important objects
            OBJECT_COOLDOWN_SECONDS = 5.0 # Cooldown for announcing the same object type

            current_time = time.time()
            announced_labels_in_frame = set() # Prevent announcing the same label twice in one frame

            for obj in processed_objects[:top_k]:
                label = obj['label']
                if label in announced_labels_in_frame:
                    continue

                last_state = self._announced_objects_state.get(label)
                
                # Conditions to announce an object:
                # 1. It's a new object we haven't seen.
                # 2. Or, enough time has passed AND its state (proximity/direction) has changed.
                should_announce = False
                if not last_state:
                    should_announce = True
                else:
                    cooldown_ok = (current_time - last_state['time']) > OBJECT_COOLDOWN_SECONDS
                    state_changed = (obj['proximity'] != last_state['proximity'] or obj['direction'] != last_state['direction'])
                    if cooldown_ok and state_changed:
                        should_announce = True

                if should_announce:
                    # Construct the sentence for the object
                    sentence = ""
                    if label == "wall":
                        sentence = obj.get("description", "A wall is right in front of you.")
                    elif label in ['green', 'red', 'yellow']:
                        sentence = f"A traffic light is {obj['proximity']}, around {obj['direction']}. The light is {label}."
                    elif label in ['road', 'sidewalk']:
                        if obj['direction'] == "12 o'clock":
                            sentence = "Caution, you are on a road." if label == 'road' else "The path ahead appears to be a sidewalk."
                    elif label == 'person' and obj['direction'] == "12 o'clock":
                        sentence = "There is a person in your path."
                    else: # Generic announcement for all other objects
                        sentence = f"A {label} is {obj['proximity']}, around {obj['direction']}."

                    if sentence:
                        summary_lines.append(sentence)
                        
                        # Collect structured data for spatial audio (just object name, no direction)
                        object_name = label
                        if label == "wall":
                            object_name = "wall"
                        elif label in ['green', 'red', 'yellow']:
                            object_name = "traffic light"
                        elif label == 'person':
                            object_name = "person"
                        
                        spatial_audio_objects.append({
                            "name": object_name,
                            "direction": obj['direction'],
                            "proximity": obj['proximity']
                        })
                        
                        # Update the state for this object label
                        self._announced_objects_state[label] = {
                            "time": current_time,
                            "proximity": obj['proximity'],
                            "direction": obj['direction']
                        }
                        announced_labels_in_frame.add(label)

            # --- END OF MODIFICATION ---
            
            # Clean up old entries from the state tracker to prevent it from growing indefinitely
            stale_time = current_time - (OBJECT_COOLDOWN_SECONDS * 2)
            stale_keys = [k for k, v in self._announced_objects_state.items() if v['time'] < stale_time]
            for k in stale_keys:
                del self._announced_objects_state[k]

            if not summary_lines:
                return # Nothing new or important to say

            analysis_result = "\n".join(summary_lines)
            
            # Print analysis result to terminal
            print(f"[ANALYSIS] {analysis_result}")
            
            # We now send any time we have a valid summary line, as the logic above handles spam
            await broadcast_result({
                "analysis_result": analysis_result, 
                "timestamp": time.time(),
                "frame_number": self.frame_count,
                "detections": sum(len(r.boxes) for r in results) if results else 0,
                "obstacle_event": obstacle_event,
                "spatial_audio": spatial_audio_objects,  # Structured data for 8D audio
            })

        except Exception as e:
            logger.error(f"Error in frame processing: {e}")
        finally:
            self.processing_frame = False

# WebRTC peer connections
pcs = set()
websocket_queue = None

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    if backend.yolo_model is None or backend.depth_model is None:
        logger.info("[INIT] Initializing models for WebRTC server...")
        backend.initialize_models()

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        logger.info("Track %s received", track.kind)
        if track.kind == "video":
            pc.addTrack(VideoTransformTrack(track, pc, websocket_queue))
        
        @track.on("ended")
        async def on_ended():
            logger.info("Track %s ended", track.kind)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})
    )

async def ws_handler(request):
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    WS_CLIENTS.add(ws)
    logger.info("[WS] Client connected (%d total)", len(WS_CLIENTS))

    try:
        async for _ in ws:
            pass
    finally:
        WS_CLIENTS.discard(ws)
        logger.info("[WS] Client disconnected (%d total)", len(WS_CLIENTS))

    return ws

async def on_shutdown(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    for ws in list(WS_CLIENTS):
        await ws.close()
    WS_CLIENTS.clear()

def create_app():
    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_post("/offer", offer)
    app.router.add_get("/ws", ws_handler)
    app.on_shutdown.append(on_shutdown)
    return app

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="WebRTC video streaming server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=9000, type=int)
    args = parser.parse_args()

    if backend.yolo_model is None or backend.depth_model is None:
        logger.info("[INIT] Initializing models on startup...")
        backend.initialize_models()

    app = create_app()
    web.run_app(app, host=args.host, port=args.port)
