"""
Real-time lip-reading testing module.

This module provides functionality for testing the trained CNN model
on video files in real-time, extracting features and making predictions.
"""

from time import time
from typing import List, Tuple, Optional, Union
import cv2
import mediapipe as mp
import pandas as pd
import torch
import numpy as np

from constants.constants import CLASSES
from utils.extract_features import extract_features
from utils.numpy_utils import pad_sequence
from utils.pandas_utils import normalize_dataframe


def test_word(model: torch.nn.Module, word: str, path: str, 
              count: int = 0, miss: int = 0) -> Tuple[int, int]:
    """
    Test a specific word prediction on a video file.
    
    This function processes a video file to extract lip features and predict
    the spoken word using the trained CNN model.
    
    Args:
        model (torch.nn.Module): Trained CNN model for prediction
        word (str): Expected word being spoken in the video
        path (str): Path to the video file
        count (int): Number of correct predictions so far
        miss (int): Number of incorrect predictions so far
        
    Returns:
        Tuple[int, int]: Updated (count, miss) statistics
        
    Raises:
        FileNotFoundError: If video file doesn't exist
        RuntimeError: If video cannot be opened
    """
    # Validate inputs
    if not isinstance(word, str) or not word.strip():
        raise ValueError("Word must be a non-empty string")
    
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Path must be a non-empty string")
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize processing variables
    buffer: List[List[float]] = []
    tmp_buffer: List[List[float]] = []
    
    print(f"🎬 Starting testing for word: {word.upper()}")
    print(f"📁 Video path: {path}")
    
    # Initialize MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    
    try:
        with mp_face_mesh.FaceMesh(
                static_image_mode=False,  # Changed to False for video processing
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
        ) as face_mesh:
            
            # Open video capture
            cap = cv2.VideoCapture(path)
            
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video file: {path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            print(f"📊 Video info: {frame_count} frames, {fps:.1f} FPS, {duration:.1f}s duration")
            
            # Processing state variables
            success = True
            silence = 0
            rise = False
            last_features = [0.0, 0.0, 0.0, 0.0]
            start_time = time()
            frame_idx = 0
            
            while success:
                success, image = cap.read()
                if not success:
                    print("📋 Reached end of video or failed to read frame")
                    break
                
                frame_idx += 1
                
                # Check for user interrupt
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("⏹️  Processing interrupted by user")
                    break

                # Prepare image for processing
                image.flags.writeable = False  # Improve performance
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Process frame with MediaPipe
                results = face_mesh.process(image_rgb)
                
                if results.multi_face_landmarks:
                    # Extract lip features
                    feature_result = extract_features(results, image_rgb)
                    
                    if feature_result is not None:
                        features, area, ratio = feature_result
                        
                        # State machine for speech detection
                        if not rise:
                            # Waiting for speech to start
                            if area > 600:
                                rise = True
                                silence = 0
                                tmp_buffer.append(last_features)
                                print(f"🗣️  Speech detected at frame {frame_idx} (area: {area:.1f})")
                        
                        if rise:
                            # Currently in speech
                            tmp_buffer.append(features)
                            
                            if area < 500:
                                # Potential end of speech
                                silence += 1
                                
                                if silence == 1:
                                    # First silence frame - check if we have enough data
                                    if len(tmp_buffer) >= 6:
                                        buffer.extend(tmp_buffer)
                                        tmp_buffer.clear()
                                        print(f"📝 Speech segment captured: {len(buffer)} frames")
                                    else:
                                        # Not enough data, reset
                                        tmp_buffer.clear()
                                        silence = 0
                                        rise = False
                                        print("⚠️  Insufficient speech data, resetting")
                                        
                                elif silence > 4:
                                    # End of speech detected
                                    rise = False
                                    silence = 0
                                    
                                    print(f"🎯 Processing speech segment: {len(buffer)} frames")
                                    
                                    # Process the captured speech
                                    if 60 > len(buffer) > 15:
                                        try:
                                            count, miss = _process_speech_segment(
                                                model, buffer, word, count, miss
                                            )
                                        except Exception as e:
                                            print(f"❌ Error processing speech segment: {e}")
                                    else:
                                        print(f"⚠️  Invalid segment length: {len(buffer)} frames")
                                    
                                    # Reset buffers
                                    buffer.clear()
                                    tmp_buffer.clear()
                                    
                            else:
                                # Still in speech, reset silence counter
                                silence = 0
                        
                        # Update last features
                        last_features = features
                    
                else:
                    # No face detected in frame
                    if frame_idx % 30 == 0:  # Print every 30 frames
                        print(f"👤 No face detected at frame {frame_idx}")

            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            processing_time = time() - start_time
            print(f"⏱️  Processing completed in {processing_time:.2f} seconds")
            print(f"📈 Results: {count} correct, {miss} incorrect")
            
            return count, miss
            
    except Exception as e:
        print(f"❌ Error during video processing: {e}")
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        raise


def _process_speech_segment(model: torch.nn.Module, buffer: List[List[float]], 
                          expected_word: str, count: int, miss: int) -> Tuple[int, int]:
    """
    Process a captured speech segment and make prediction.
    
    Args:
        model (torch.nn.Module): Trained model
        buffer (List[List[float]]): Captured features
        expected_word (str): Expected word
        count (int): Current correct count
        miss (int): Current miss count
        
    Returns:
        Tuple[int, int]: Updated (count, miss)
    """
    try:
        # Normalize features
        normalized = normalize_dataframe(buffer)
        
        if normalized is None or len(normalized) == 0:
            print("⚠️  Failed to normalize features")
            return count, miss
        
        # Pad sequence to expected length
        padded = pad_sequence(pd.DataFrame(normalized), 60)
        
        if padded is None:
            print("⚠️  Failed to pad sequence")
            return count, miss
        
        # Convert to tensor
        tensor = torch.tensor(padded, dtype=torch.float32)
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        
        # Check for valid input
        if torch.all(torch.eq(tensor, 0)):
            print("⚠️  All-zero input tensor, skipping prediction")
            return count, miss
        
        # Make prediction
        with torch.no_grad():
            output = model(tensor)
            probabilities = torch.softmax(output, dim=1)
            
            # Get prediction
            pred_idx = torch.argmax(output).item()
            confidence = probabilities[0][pred_idx].item()
            predicted_word = CLASSES[pred_idx]
            
            # Get top 3 predictions for analysis
            top_probs, top_indices = torch.topk(probabilities, k=min(3, len(CLASSES)))
            
            print(f"\n🤖 Prediction Results:")
            print(f"   Predicted: {predicted_word}")
            print(f"   Expected:  {expected_word}")
            print(f"   Confidence: {confidence:.2%}")
            
            print(f"   Top predictions:")
            for i in range(top_probs.size(1)):
                idx = top_indices[0][i].item()
                prob = top_probs[0][i].item()
                print(f"     {i+1}. {CLASSES[idx]}: {prob:.2%}")
            
            # Update statistics
            if predicted_word == expected_word:
                count += 1
                print(f"✅ Correct prediction!")
            else:
                miss += 1
                print(f"❌ Incorrect prediction")
            
            return count, miss
            
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        return count, miss


def test_video_batch(model: torch.nn.Module, video_paths: List[str], 
                    expected_words: List[str]) -> dict:
    """
    Test multiple videos in batch mode.
    
    Args:
        model (torch.nn.Module): Trained model
        video_paths (List[str]): List of video file paths
        expected_words (List[str]): List of expected words
        
    Returns:
        dict: Batch testing results
    """
    if len(video_paths) != len(expected_words):
        raise ValueError("Number of videos must match number of expected words")
    
    print(f"🎬 Starting batch testing of {len(video_paths)} videos")
    
    total_correct = 0
    total_miss = 0
    results = []
    
    start_time = time()
    
    for i, (video_path, expected_word) in enumerate(zip(video_paths, expected_words)):
        print(f"\n{'='*50}")
        print(f"Testing video {i+1}/{len(video_paths)}: {video_path}")
        
        try:
            count, miss = test_word(model, expected_word, video_path, 0, 0)
            total_correct += count
            total_miss += miss
            
            results.append({
                'video_path': video_path,
                'expected_word': expected_word,
                'correct': count,
                'miss': miss
            })
            
        except Exception as e:
            print(f"❌ Failed to process {video_path}: {e}")
            results.append({
                'video_path': video_path,
                'expected_word': expected_word,
                'correct': 0,
                'miss': 1,
                'error': str(e)
            })
            total_miss += 1
    
    total_time = time() - start_time
    total_tests = total_correct + total_miss
    accuracy = (total_correct / total_tests * 100) if total_tests > 0 else 0
    
    batch_results = {
        'total_videos': len(video_paths),
        'total_correct': total_correct,
        'total_miss': total_miss,
        'accuracy': accuracy,
        'processing_time': total_time,
        'individual_results': results
    }
    
    print(f"\n{'='*50}")
    print(f"🎯 Batch Testing Complete!")
    print(f"   Total videos: {len(video_paths)}")
    print(f"   Correct: {total_correct}")
    print(f"   Incorrect: {total_miss}")
    print(f"   Accuracy: {accuracy:.2f}%")
    print(f"   Total time: {total_time:.2f} seconds")
    print(f"   Average time per video: {total_time/len(video_paths):.2f} seconds")
    
    return batch_results


def main() -> None:
    """Main function for interactive testing."""
    try:
        # This would need to be implemented based on the specific model loading logic
        print("🤖 Real-time lip-reading testing")
        print("This module provides functions for testing trained models.")
        print("Use test_word() or test_video_batch() functions to test videos.")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
    model.eval()
    buffer = []
    tmp_buffer = []
    print(f"Starting testing {word.upper()}..")
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
        cap = cv2.VideoCapture(path)
        # checks whether frames were extracted
        success = 1
        silence = 0
        rise = False
        last_features = [0, 0, 0, 0]
        time1 = time()

        while success:
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera image.")
                continue

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
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
                                        output = model(tensor)

                                    # Get the class with the highest probability
                                    pred = torch.argmax(output).item()
                                    prob = torch.softmax(output, dim=1)[0][pred].item()

                                    # Print the prediction and the confidence
                                    time2 = time() - time1
                                    time1 = time()
                                    print("\nGathered:", str(len(buffer)), "frames in", str(time2), "seconds")
                                    print("Prediction:", CLASSES[pred], "with confidence:", prob, "\n")
                                    if CLASSES[pred] == word:
                                        count += 1
                                    else:
                                        miss += 1
                            buffer.clear()
                            tmp_buffer.clear()
                        else:
                            buffer.append([0, 0, 0, 0])
                            tmp_buffer.clear()
                    else:
                        silence = 0
                last_features = features

                # Exit if the 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    # Release the video capture and close the windows
    cap.release()
    cv2.destroyAllWindows()
    print(f"I was correct {count} times out of {count + miss} times..")