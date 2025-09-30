"""
OpenAI Lip-Sync Application
Captures lip movements, predicts words, sends to OpenAI TTS, and plays audio with hotkeys.
No visual overlay - headless operation.
"""
import os
import sys
from datetime import datetime
from time import time
import cv2
import mediapipe as mp
import pandas as pd
import torch
from openai import OpenAI
import keyboard
from pathlib import Path

from constants.constants import CLASSES
from utils.extract_features import extract_features
from utils.numpy_utils import pad_sequence
from utils.pandas_utils import normalize_dataframe

# Configuration
OUTPUT_DIR = Path("openai_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


class LipSyncOpenAI:
    def __init__(self, model, openai_api_key=None):
        self.model = model
        self.model.eval()
        
        # Initialize OpenAI client
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass it to constructor.")
        
        self.client = OpenAI(api_key=self.openai_api_key)
        
        # Audio playback state
        self.current_audio_file = None
        self.audio_playing = False
        
        # Setup hotkeys
        self.setup_hotkeys()
        
    def setup_hotkeys(self):
        """Setup keyboard hotkeys for audio playback"""
        keyboard.add_hotkey('space', self.play_audio)
        keyboard.add_hotkey('s', self.stop_audio)
        print("Hotkeys registered:")
        print("  SPACE - Play audio")
        print("  S - Stop audio")
        
    def play_audio(self):
        """Play the current audio file"""
        if self.current_audio_file and os.path.exists(self.current_audio_file):
            print(f"Playing audio: {self.current_audio_file}")
            # Use OS-specific audio player
            if sys.platform == "darwin":  # macOS
                os.system(f"afplay {self.current_audio_file} &")
            elif sys.platform == "linux":  # Linux
                os.system(f"aplay {self.current_audio_file} &")
            elif sys.platform == "win32":  # Windows
                os.system(f"start {self.current_audio_file}")
            self.audio_playing = True
        else:
            print("No audio file available to play")
    
    def stop_audio(self):
        """Stop audio playback"""
        print("Stopping audio...")
        if sys.platform == "darwin":  # macOS
            os.system("killall afplay")
        elif sys.platform == "linux":  # Linux
            os.system("killall aplay")
        self.audio_playing = False
    
    def predict_word(self, video_path):
        """Process video and predict words from lip movements"""
        buffer = []
        tmp_buffer = []
        predictions = []
        
        print(f"Processing video: {video_path}")
        mp_face_mesh = mp.solutions.face_mesh
        
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_mesh:
            
            cap = cv2.VideoCapture(video_path)
            success = 1
            silence = 0
            rise = False
            last_features = [0, 0, 0, 0]
            time1 = time()
            
            while success:
                success, image = cap.read()
                if not success:
                    break
                
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image)
                
                if results.multi_face_landmarks:
                    features, area, _ = extract_features(results, image)
                    if not rise:
                        if area > 600:
                            rise = True
                            silence = 0
                            tmp_buffer.append(last_features)
                    
                    if rise:
                        tmp_buffer.append(features)
                        if area < 500:
                            silence += 1
                            if silence == 1:
                                if len(tmp_buffer) >= 6:
                                    buffer.extend(tmp_buffer)
                                    tmp_buffer.clear()
                                else:
                                    tmp_buffer.clear()
                                    silence = 0
                                    rise = False
                            elif silence > 4:
                                rise = False
                                silence = 0
                                normalized = normalize_dataframe(buffer)
                                if 60 > len(buffer) > 15:
                                    padded = pad_sequence(pd.DataFrame(normalized), 60)
                                    tensor = torch.tensor(padded).float()
                                    tensor = tensor.unsqueeze(0)
                                    
                                    if not (torch.all(torch.eq(tensor, 0))):
                                        with torch.no_grad():
                                            output = self.model(tensor)
                                        
                                        pred = torch.argmax(output).item()
                                        prob = torch.softmax(output, dim=1)[0][pred].item()
                                        
                                        time2 = time() - time1
                                        time1 = time()
                                        
                                        word = CLASSES[pred]
                                        print(f"\nDetected: {word} (confidence: {prob:.2f})")
                                        print(f"Frames: {len(buffer)}, Time: {time2:.2f}s")
                                        
                                        predictions.append({
                                            'word': word,
                                            'confidence': prob,
                                            'frames': len(buffer),
                                            'time': time2
                                        })
                                
                                buffer.clear()
                                tmp_buffer.clear()
                        else:
                            buffer.append([0, 0, 0, 0])
                            tmp_buffer.clear()
                    else:
                        silence = 0
                    last_features = features
            
            cap.release()
        
        return predictions
    
    def send_to_openai_tts(self, text, output_file=None):
        """Send text to OpenAI TTS API and save audio file"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = OUTPUT_DIR / f"speech_{timestamp}.mp3"
        
        print(f"Sending to OpenAI TTS: {text}")
        
        try:
            response = self.client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text
            )
            
            # Save audio file
            response.stream_to_file(str(output_file))
            print(f"Audio saved to: {output_file}")
            
            # Save text to file
            txt_file = output_file.with_suffix('.txt')
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Text saved to: {txt_file}")
            
            self.current_audio_file = str(output_file)
            return str(output_file), str(txt_file)
            
        except Exception as e:
            print(f"Error calling OpenAI TTS: {e}")
            return None, None
    
    def process_video_and_generate_speech(self, video_path, language="Hebrew"):
        """
        Main workflow:
        1. Process video to detect words
        2. Send detected words to OpenAI TTS
        3. Save audio and text files
        4. Audio ready to play with hotkeys
        """
        print("=" * 50)
        print("Starting OpenAI Lip-Sync Processing")
        print("=" * 50)
        
        # Step 1: Predict words from video
        predictions = self.predict_word(video_path)
        
        if not predictions:
            print("No words detected in video")
            return None, None
        
        # Step 2: Get the best prediction
        best_prediction = max(predictions, key=lambda x: x['confidence'])
        detected_word = best_prediction['word']
        confidence = best_prediction['confidence']
        
        print(f"\nBest prediction: {detected_word} (confidence: {confidence:.2f})")
        
        # Step 3: Send to OpenAI TTS
        # Create a sentence for TTS (you can customize this)
        tts_text = f"The detected word is: {detected_word}"
        
        audio_file, txt_file = self.send_to_openai_tts(tts_text)
        
        if audio_file and txt_file:
            print("\n" + "=" * 50)
            print("Processing Complete!")
            print("=" * 50)
            print(f"Audio file: {audio_file}")
            print(f"Text file: {txt_file}")
            print("\nPress SPACE to play audio")
            print("Press S to stop audio")
            print("Press Q to quit")
            print("=" * 50)
            
            # Wait for hotkey input
            keyboard.wait('q')
            
        return audio_file, txt_file


def main():
    """Main entry point"""
    import argparse
    from infra.cnn import CNN
    
    parser = argparse.ArgumentParser(description='OpenAI Lip-Sync Application')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.pth file)')
    parser.add_argument('--api-key', type=str, help='OpenAI API key (or set OPENAI_API_KEY env var)')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from: {args.model}")
    model = CNN(num_classes=len(CLASSES), num_rows=60, num_cols=4)
    model.load_state_dict(torch.load(args.model))
    
    # Create LipSyncOpenAI instance
    app = LipSyncOpenAI(model, openai_api_key=args.api_key)
    
    # Process video
    app.process_video_and_generate_speech(args.video)
    
    print("\nApplication terminated.")


if __name__ == "__main__":
    main()
