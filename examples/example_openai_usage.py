#!/usr/bin/env python3
"""
Example usage of OpenAI Lip-Sync Application

This script demonstrates how to use the OpenAI lip-sync integration
to process a video, detect words, generate speech, and play audio with hotkeys.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from run.openai_lipsync import LipSyncOpenAI
from infra.cnn import CNN
from constants.constants import CLASSES

def main():
    # Configuration
    VIDEO_PATH = "path/to/your/video.mp4"  # Replace with actual video path
    MODEL_PATH = "path/to/your/model.pth"  # Replace with actual model path
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # Set this environment variable
    
    # Validate inputs
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Usage: export OPENAI_API_KEY='your-api-key' && python example_openai_usage.py")
        sys.exit(1)
    
    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: Video file not found: {VIDEO_PATH}")
        print("Please update VIDEO_PATH in this script")
        sys.exit(1)
    
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found: {MODEL_PATH}")
        print("Please update MODEL_PATH in this script")
        sys.exit(1)
    
    # Load model
    print(f"Loading model from: {MODEL_PATH}")
    model = CNN(num_classes=len(CLASSES), num_rows=60, num_cols=4)
    import torch
    model.load_state_dict(torch.load(MODEL_PATH))
    
    # Create LipSyncOpenAI instance
    print("Initializing OpenAI Lip-Sync Application...")
    app = LipSyncOpenAI(model, openai_api_key=OPENAI_API_KEY)
    
    # Process video and generate speech
    print(f"\nProcessing video: {VIDEO_PATH}")
    audio_file, txt_file = app.process_video_and_generate_speech(VIDEO_PATH)
    
    if audio_file and txt_file:
        print(f"\n✓ Success!")
        print(f"  Audio: {audio_file}")
        print(f"  Text: {txt_file}")
    else:
        print("\n✗ Failed to process video")

if __name__ == "__main__":
    main()
