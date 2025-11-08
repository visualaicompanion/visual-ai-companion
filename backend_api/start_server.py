#!/usr/bin/env python3
"""
Startup script for the Vision Assistant API
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(".."))

try:
    # Add depth-anything-v2 to path
    depth_path = os.path.join(os.path.dirname(__file__), "../depth-anything-v2")
    sys.path.insert(0, depth_path)
    print(f"[DEBUG] Added depth path: {depth_path}")
    
    from main import app
    import uvicorn
    
    if __name__ == "__main__":
        print("Starting Vision Assistant API server...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
        
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    sys.exit(1) 