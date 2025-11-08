import cv2
import torch
import numpy as np
import base64
import io
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
from datetime import datetime
from ultralytics import YOLO
import torch.nn.functional as F
import time
import asyncio
from fastapi import WebSocket, WebSocketDisconnect
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(".."))

# Import custom modules
try:
    sys.path.insert(0, os.path.abspath("../depth-anything-v2"))
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    sys.exit(1)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ImageRequest(BaseModel):
    image: str  # base64 encoded image

class AnalysisResponse(BaseModel):
    objects: list
    depth_info: str
    scene_description: str
    timestamp: str

class StreamRequest(BaseModel):
    image: str  # base64 encoded image
    frame_number: int

class StreamResponse(BaseModel):
    analysis_result: str
    timestamp: str

# Global variables for state tracking
last_known_objects = {}
DANGEROUS_OBJECTS = ['person', 'car', 'truck', 'bicycle', 'motorbike', 'bus', 'door']
proximity_map = {"far": 0, "nearby": 1, "right in front of you": 2}
last_analysis_time = 0
COOLDOWN_SECONDS = 7

# Model configuration
ENCODER = "vits"
USE_FP16 = True
INPUT_SIZE = 896  # Must be divisible by 14 for DepthAnythingV2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model paths
depth_ckpt = os.path.join(os.path.dirname(__file__), "../depth-anything-v2/checkpoints/depth_anything_v2_vits.pth")
yolo_model_path = os.path.join(os.path.dirname(__file__), "YOLO.pt")

# Global model variables
depth_model = None
yolo_model = None

# WebSocket helpers
_processing_lock = asyncio.Lock()
_last_ws_result = ""


def initialize_models():
    """Initialize YOLO and DepthAnythingV2 models"""
    global depth_model, yolo_model
    
    print("[INFO] Loading models...")
    print(f"[DEBUG] Current working directory: {os.getcwd()}")
    print(f"[DEBUG] Depth checkpoint path: {depth_ckpt}")
    print(f"[DEBUG] YOLO model path: {yolo_model_path}")
    print(f"[DEBUG] Depth checkpoint exists: {os.path.exists(depth_ckpt)}")
    print(f"[DEBUG] YOLO model exists: {os.path.exists(yolo_model_path)}")
    
    # Load DepthAnythingV2
    depth_cfg = {"vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}}
    depth_model = DepthAnythingV2(**depth_cfg[ENCODER])
    depth_model.load_state_dict(torch.load(depth_ckpt, map_location="cpu"))
    depth_model = depth_model.to(DEVICE).eval()
    if USE_FP16 and DEVICE != 'cpu':
        depth_model = depth_model.half()
    
    # Load YOLO
    yolo_model = YOLO(yolo_model_path)
    
    print(f"[OK] Models loaded. Device: {DEVICE}, FP16: {USE_FP16}")

def get_clock_direction(cx, frame_width):
    """Given center-x and frame width, return a clock-face direction (10–2 o'clock)."""
    x_ratio = cx / frame_width
    if x_ratio < 0.20:
        return "10 o'clock"
    elif x_ratio < 0.40:
        return "11 o'clock"
    elif x_ratio < 0.60:
        return "12 o'clock"
    elif x_ratio < 0.80:
        return "1 o'clock"
    else:
        return "2 o'clock"
    

def get_proximity_label(norm_depth):
    """Classify normalized depth into proximity categories."""
    if norm_depth > 0.75:
        return "right in front of you"
    elif norm_depth > 0.4:
        return "nearby"
    else:
        return "far"

def compute_priority(depth, direction, cls_name):
    """Compute priority score for objects based on various factors."""
    score = 0
    if cls_name in ['person', 'car', 'motorbike', 'bicycle', 'truck']:
        score += 2
    if direction == "12 o'clock":
        score += 2
    elif direction in ["11 o'clock", "1 o'clock"]:
        score += 1
    if depth > 0.75:  # very close
        score += 3
    elif depth > 0.4:  # medium range
        score += 2
    else:
        score += 1
    return score

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    initialize_models()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": depth_model is not None and yolo_model is not None}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(request: ImageRequest):
    """Legacy endpoint for single image analysis"""
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Resize for model input
        input_frame = cv2.resize(opencv_image, (INPUT_SIZE, INPUT_SIZE))
        
        # YOLO detection
        yolo_results = yolo_model.predict(input_frame, show=False, conf=0.5, verbose=False)
        
        # Depth estimation
        img_rgb = input_frame[:, :, ::-1].copy()
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        if USE_FP16 and DEVICE != 'cpu':
            img_tensor = img_tensor.half()
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=USE_FP16 and DEVICE != 'cpu'):
                depth = depth_model(img_tensor)
            
            depth = F.interpolate(depth[:, None], opencv_image.shape[:2], mode="bilinear", align_corners=True)[0, 0]
            depth_np = depth.cpu().numpy()
        
        # Process results
        objects = []
        h_orig, w_orig = opencv_image.shape[:2]
        h_input, w_input = (INPUT_SIZE, INPUT_SIZE)
        
        for result in yolo_results:
            for box in result.boxes:
                x1_input, y1_input, x2_input, y2_input = box.xyxy[0]
                cx = int((x1_input + x2_input) / 2 * w_orig / w_input)
                cy = int((y1_input + y2_input) / 2 * h_orig / h_input)
                cls = int(box.cls[0])
                label = yolo_model.names[cls]
                conf = float(box.conf[0])
                
                if 0 <= cy < depth_np.shape[0] and 0 <= cx < depth_np.shape[1]:
                    distance = depth_np[cy, cx]
                    proximity = get_proximity_label(distance)
                    direction = get_clock_direction(cx, w_orig)
                    priority = compute_priority(distance, direction, label)
                    
                    objects.append({
                        "label": label,
                        "confidence": conf,
                        "direction": direction,
                        "proximity": proximity,
                        "priority": priority,
                        "bbox": [int(x1_input * w_orig / w_input), int(y1_input * h_orig / h_input),
                                int(x2_input * w_orig / w_input), int(y2_input * h_orig / h_input)]
                    })
        
        # Sort by priority
        objects.sort(key=lambda x: -x["priority"])
        
        # Generate scene description
        if objects:
            scene_description = "\n".join([f"A {obj['label']} is {obj['proximity']}, around {obj['direction']}." for obj in objects[:3]])
        else:
            scene_description = "No objects detected in the scene."
        
        return AnalysisResponse(
            objects=objects,
            depth_info=f"Depth range: {depth_np.min():.3f} - {depth_np.max():.3f}",
            scene_description=scene_description,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")

@app.post("/analyze_stream", response_model=StreamResponse)
async def analyze_stream_frame(request: StreamRequest):
    """Analyze streaming frame with event-driven logic"""
    global last_known_objects, last_analysis_time
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to OpenCV format and rotate 90 degrees clockwise
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        opencv_image = cv2.rotate(opencv_image, cv2.ROTATE_90_CLOCKWISE)
        
        # Save test image for debugging
        test_image_path = f"test_image_{request.frame_number}.jpg"
        cv2.imwrite(test_image_path, opencv_image)
        print(f"[DEBUG] Saved test image: {test_image_path}, size: {opencv_image.shape}")
        
        # Resize for model input
        input_frame = cv2.resize(opencv_image, (INPUT_SIZE, INPUT_SIZE))
        
        # YOLO detection
        yolo_results = yolo_model.predict(input_frame, show=False, conf=0.5, verbose=False)
        
        # Depth estimation
        img_rgb = input_frame[:, :, ::-1].copy()
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        if USE_FP16 and DEVICE != 'cpu':
            img_tensor = img_tensor.half()
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=USE_FP16 and DEVICE != 'cpu'):
                depth = depth_model(img_tensor)
            
            depth = F.interpolate(depth[:, None], opencv_image.shape[:2], mode="bilinear", align_corners=True)[0, 0]
            depth_np = depth.cpu().numpy()
        
        # Event-driven analysis logic
        trigger_analysis = False
        current_objects = {}
        processed_objects = []
        h_orig, w_orig = opencv_image.shape[:2]
        h_input, w_input = (INPUT_SIZE, INPUT_SIZE)
        
        for result in yolo_results:
            for box in result.boxes:
                x1_input, y1_input, x2_input, y2_input = box.xyxy[0]
                cx = int((x1_input + x2_input) / 2 * w_orig / w_input)
                cy = int((y1_input + y2_input) / 2 * h_orig / h_input)
                cls = int(box.cls[0])
                label = yolo_model.names[cls]
                
                if 0 <= cy < depth_np.shape[0] and 0 <= cx < depth_np.shape[1]:
                    distance = depth_np[cy, cx]
                    proximity = get_proximity_label(distance)
                    direction = get_clock_direction(cx, w_orig)
                    priority = compute_priority(distance, direction, label)
                    
                    processed_objects.append({
                        "label": label, "direction": direction, 
                        "proximity": proximity, "priority": priority
                    })
                    
                    current_objects[label] = proximity
                    
                    # Check for analysis triggers
                    if not trigger_analysis:
                        if label in DANGEROUS_OBJECTS and label not in last_known_objects and proximity == "right in front of you":
                            print(f"[EVENT] URGENT: New {label} appeared right in front.")
                            trigger_analysis = True
                        elif label in last_known_objects:
                            old_score = proximity_map.get(last_known_objects.get(label), -1)
                            new_score = proximity_map.get(proximity, -1)
                            if new_score > old_score:
                                print(f"[EVENT] A {label} is getting closer.")
                                trigger_analysis = True
                        elif label in DANGEROUS_OBJECTS and label not in last_known_objects:
                            print(f"[EVENT] New important object: {label} has entered the scene.")
                            trigger_analysis = True
                        if not trigger_analysis and (time.time() - last_analysis_time > COOLDOWN_SECONDS):
                            if label not in last_known_objects:
                                print(f"[EVENT] New regular object: {label} has entered the scene.")
                                trigger_analysis = True
        
        analysis_result = ""
        if trigger_analysis:
            print(f"[INFO] Analysis triggered by scene change.")
            processed_objects.sort(key=lambda x: -x["priority"])
            top_k = 3
            
            object_descriptions = []
            for obj in processed_objects[:top_k]:
                object_descriptions.append(
                    f"A {obj['label']} is {obj['proximity']}, around {obj['direction']}."
                )
            
            if object_descriptions:
                analysis_result = "\n".join(object_descriptions)
                print(analysis_result)
                print("[ANALYSIS RESULT]")
                print(analysis_result)
                print("-" * 20)
            
            last_known_objects = current_objects.copy()
            last_analysis_time = time.time()
        
        # Print to terminal
        print("\n" + "="*60)
        print(f"Frame {request.frame_number} - Objects detected: {len(processed_objects)}")
        if processed_objects:
            for obj in processed_objects[:3]:  # Show top 3
                print(f"  {obj['label']}: {obj['proximity']}, {obj['direction']}, priority: {obj['priority']}")
        print("="*60)
        
        return StreamResponse(
            analysis_result=analysis_result,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing frame: {str(e)}")

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global _last_ws_result, last_known_objects, last_analysis_time
    await ws.accept()
    print("[WS] Client connected")
    try:
        while True:
            msg = await ws.receive_text()
            # Expect JSON: {"image": base64, "frame_number": int}
            try:
                data = json.loads(msg)
                b64 = data.get("image")
                frame_number = int(data.get("frame_number", 0))
            except Exception as e:
                await ws.send_text(json.dumps({"error": f"invalid message: {e}"}))
                continue

            if not b64:
                await ws.send_text(json.dumps({"error": "no image"}))
                continue

            # Backpressure: if busy, immediately return last result
            if _processing_lock.locked():
                await ws.send_text(json.dumps({
                    "analysis_result": _last_ws_result,
                    "timestamp": datetime.now().isoformat(),
                    "skipped": True
                }))
                continue

            async with _processing_lock:
                try:
                    image_data = base64.b64decode(b64)
                    image = Image.open(io.BytesIO(image_data))
                    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    opencv_image = cv2.rotate(opencv_image, cv2.ROTATE_90_CLOCKWISE)

                    input_frame = cv2.resize(opencv_image, (INPUT_SIZE, INPUT_SIZE))

                    # YOLO detection
                    yolo_results = yolo_model.predict(input_frame, show=False, conf=0.5, verbose=False)

                    # Depth estimation
                    img_rgb = input_frame[:, :, ::-1].copy()
                    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
                    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
                    if USE_FP16 and DEVICE != 'cpu':
                        img_tensor = img_tensor.half()

                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=USE_FP16 and DEVICE != 'cpu'):
                            depth = depth_model(img_tensor)
                        depth = F.interpolate(depth[:, None], opencv_image.shape[:2], mode="bilinear", align_corners=True)[0, 0]
                        depth_np = depth.cpu().numpy()

                    # Event-driven analysis logic (same as HTTP)
                    trigger_analysis = False
                    current_objects = {}
                    processed_objects = []
                    h_orig, w_orig = opencv_image.shape[:2]
                    h_input, w_input = (INPUT_SIZE, INPUT_SIZE)

                    for result in yolo_results:
                        for box in result.boxes:
                            x1_input, y1_input, x2_input, y2_input = box.xyxy[0]
                            cx = int((x1_input + x2_input) / 2 * w_orig / w_input)
                            cy = int((y1_input + y2_input) / 2 * h_orig / h_input)
                            cls = int(box.cls[0])
                            label = yolo_model.names[cls]

                            if 0 <= cy < depth_np.shape[0] and 0 <= cx < depth_np.shape[1]:
                                distance = depth_np[cy, cx]
                                proximity = get_proximity_label(distance)
                                direction = get_clock_direction(cx, w_orig)
                                priority = compute_priority(distance, direction, label)

                                processed_objects.append({
                                    "label": label, "direction": direction, 
                                    "proximity": proximity, "priority": priority
                                })

                                current_objects[label] = proximity

                                if not trigger_analysis:
                                    if label in DANGEROUS_OBJECTS and label not in last_known_objects and proximity == "right in front of you":
                                        trigger_analysis = True
                                    elif label in last_known_objects:
                                        old_score = proximity_map.get(last_known_objects.get(label), -1)
                                        new_score = proximity_map.get(proximity, -1)
                                        if new_score > old_score:
                                            trigger_analysis = True
                                    elif label in DANGEROUS_OBJECTS and label not in last_known_objects:
                                        trigger_analysis = True
                                    if not trigger_analysis and (time.time() - last_analysis_time > COOLDOWN_SECONDS):
                                        if label not in last_known_objects:
                                            trigger_analysis = True

                    analysis_result = _last_ws_result
                    if trigger_analysis:
                        processed_objects.sort(key=lambda x: -x["priority"])
                        top_k = 3
                        object_descriptions = []
                        for obj in processed_objects[:top_k]:
                            object_descriptions.append(
                                f"A {obj['label']} is {obj['proximity']}, around {obj['direction']}."
                            )
                        if object_descriptions:
                            analysis_result = "\n".join(object_descriptions)
                            last_known_objects = current_objects.copy()
                            last_analysis_time = time.time()

                    _last_ws_result = analysis_result

                    await ws.send_text(json.dumps({
                        "analysis_result": analysis_result,
                        "timestamp": datetime.now().isoformat(),
                        "frame_number": frame_number,
                    }))
                except Exception as e:
                    print(f"[WS] Error processing frame: {e}")
                    await ws.send_text(json.dumps({"error": str(e)}))
    except WebSocketDisconnect:
        print("[WS] Client disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 