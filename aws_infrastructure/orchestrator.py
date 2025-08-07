"""
Main processing orchestrator for the AWS lip-sync pipeline.

This module coordinates the entire processing flow:
user -> telegram -> alfred -> queue -> processor -> hermes -> s3 -> results
"""

import os
import json
import logging
import asyncio
import tempfile
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import uuid

import torch
import cv2
import numpy as np

from aws_infrastructure.alfred_telegram import AlfredTelegramBot
from aws_infrastructure.hermes_dynamodb import HermesDynamoDB
from aws_infrastructure.s3_storage import S3StorageManager
from aws_infrastructure.sqs_queue import SQSMessageQueue
from infra.cnn import CNN
from constants.constants import CLASSES
from utils.extract_features import extract_features
from utils.numpy_utils import pad_sequence
from utils.pandas_utils import normalize_dataframe
import mediapipe as mp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LipSyncProcessor:
    """
    Main processor for lip-sync video analysis.
    
    Handles the complete processing pipeline from video input to results.
    """
    
    def __init__(self) -> None:
        """Initialize the processor with AWS components and ML model."""
        # AWS components
        self.hermes = HermesDynamoDB()
        self.s3_manager = S3StorageManager(
            bucket_name=os.getenv('S3_BUCKET_NAME'),
            region=os.getenv('AWS_REGION', 'us-east-1')
        )
        self.queue = SQSMessageQueue(
            queue_url=os.getenv('SQS_QUEUE_URL'),
            dead_letter_queue_url=os.getenv('SQS_DEAD_LETTER_QUEUE_URL')
        )
        
        # ML model setup
        self.model = None
        self.model_loaded = False
        self.mp_face_mesh = mp.solutions.face_mesh
        
        logger.info("LipSync Processor initialized")
    
    async def load_model(self, model_name: str = "cnn_model", version: str = "latest") -> bool:
        """
        Load the CNN model from S3 or local storage.
        
        Args:
            model_name (str): Model identifier
            version (str): Model version
            
        Returns:
            bool: True if model loaded successfully
        """
        try:
            # Try to load from S3 first
            model_path = await self.s3_manager.download_model(model_name, version)
            
            if model_path is None:
                # Fallback to local model
                local_model_path = os.path.join(
                    os.path.dirname(__file__), '..', 'cnn_model', 'cnn_model.pth'
                )
                if os.path.exists(local_model_path):
                    model_path = local_model_path
                else:
                    logger.error("No model found in S3 or locally")
                    return False
            
            # Initialize and load model
            self.model = CNN(num_classes=len(CLASSES), num_rows=60, num_cols=4)
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.model.eval()
            self.model_loaded = True
            
            logger.info(f"Model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    async def process_video(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a video for lip-sync analysis.
        
        Args:
            job_data (Dict[str, Any]): Job information from queue
            
        Returns:
            Dict[str, Any]: Processing results
        """
        job_id = job_data.get('job_id', str(uuid.uuid4()))
        user_id = job_data.get('user_id')
        video_key = job_data.get('video_key')
        chat_id = job_data.get('chat_id')
        
        logger.info(f"Processing job {job_id} for user {user_id}")
        
        # Update job status to processing
        await self.hermes.update_job_status(job_id, 'processing')
        
        try:
            # Ensure model is loaded
            if not self.model_loaded:
                model_loaded = await self.load_model()
                if not model_loaded:
                    raise Exception("Failed to load model")
            
            # Download video from S3
            logger.info(f"Downloading video: {video_key}")
            video_path = await self.s3_manager.download_video(video_key)
            if video_path is None:
                raise Exception("Failed to download video")
            
            # Extract features from video
            logger.info("Extracting features from video")
            features = await self._extract_video_features(video_path)
            
            if features is None or len(features) == 0:
                raise Exception("No features extracted from video")
            
            # Process features and make predictions
            logger.info("Making predictions")
            predictions, confidence_scores = await self._predict_from_features(features)
            
            # Calculate processing time and create results
            processing_time = 2.5  # Placeholder - would calculate actual time
            
            result_data = {
                'success': True,
                'predictions': predictions,
                'confidence_scores': confidence_scores,
                'processing_time': processing_time,
                'feature_count': len(features),
                'model_version': '1.0'
            }
            
            # Save results to S3 and DynamoDB
            await self.s3_manager.save_processing_result(job_id, result_data)
            await self.hermes.save_result(
                job_id, predictions, confidence_scores, processing_time
            )
            
            # Update job status to completed
            await self.hermes.update_job_status(job_id, 'completed')
            
            # Update user stats
            await self.hermes.update_user_stats(user_id, 1)
            
            # Send results back to user via Telegram
            if chat_id:
                await self._send_results_to_user(chat_id, result_data)
            
            # Cleanup temporary files
            if os.path.exists(video_path):
                os.unlink(video_path)
            
            logger.info(f"Successfully processed job {job_id}")
            return result_data
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing job {job_id}: {error_msg}")
            
            # Update job status to failed
            await self.hermes.update_job_status(job_id, 'failed', error_msg)
            
            # Send error notification to user
            if chat_id:
                error_result = {
                    'success': False,
                    'error': error_msg,
                    'job_id': job_id
                }
                await self._send_results_to_user(chat_id, error_result)
            
            return {'success': False, 'error': error_msg}
    
    async def _extract_video_features(self, video_path: str) -> Optional[List[List[float]]]:
        """
        Extract lip features from video file.
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            Optional[List[List[float]]]: Extracted features or None
        """
        try:
            features_buffer = []
            
            with self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            ) as face_mesh:
                
                cap = cv2.VideoCapture(video_path)
                
                while True:
                    success, frame = cap.read()
                    if not success:
                        break
                    
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(frame_rgb)
                    
                    if results.multi_face_landmarks:
                        features, area, _ = extract_features(results, frame_rgb)
                        
                        # Only include frames with significant mouth area
                        if area > 300:
                            features_buffer.append(features)
                
                cap.release()
            
            if len(features_buffer) < 10:
                logger.warning(f"Only extracted {len(features_buffer)} features from video")
                return None
            
            # Normalize features
            normalized_features = normalize_dataframe(features_buffer)
            
            logger.info(f"Extracted {len(normalized_features)} features from video")
            return normalized_features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    async def _predict_from_features(self, features: List[List[float]]) -> tuple[List[str], List[float]]:
        """
        Make predictions from extracted features.
        
        Args:
            features (List[List[float]]): Normalized features
            
        Returns:
            tuple[List[str], List[float]]: Predictions and confidence scores
        """
        try:
            # Pad sequence to expected length
            padded_features = pad_sequence(np.array(features), 60)
            
            if padded_features is None:
                raise Exception("Failed to pad features")
            
            # Convert to tensor
            tensor = torch.tensor(padded_features, dtype=torch.float32)
            tensor = tensor.unsqueeze(0)  # Add batch dimension
            
            # Make prediction
            with torch.no_grad():
                output = self.model(tensor)
                probabilities = torch.softmax(output, dim=1)
                
                # Get top predictions
                top_probs, top_indices = torch.topk(probabilities, k=min(3, len(CLASSES)))
                
                predictions = []
                confidence_scores = []
                
                for i in range(top_probs.size(1)):
                    class_idx = top_indices[0][i].item()
                    confidence = top_probs[0][i].item()
                    
                    predictions.append(CLASSES[class_idx])
                    confidence_scores.append(confidence)
            
            logger.info(f"Predictions: {predictions} with confidence: {confidence_scores}")
            return predictions, confidence_scores
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return [], []
    
    async def _send_results_to_user(self, chat_id: int, result_data: Dict[str, Any]) -> None:
        """
        Send processing results to user via Telegram.
        
        Args:
            chat_id (int): Telegram chat ID
            result_data (Dict[str, Any]): Processing results
        """
        try:
            # Create bot instance (in production, this would be a singleton)
            bot = AlfredTelegramBot(
                token=os.getenv('TELEGRAM_BOT_TOKEN'),
                sqs_queue_url=os.getenv('SQS_QUEUE_URL'),
                s3_bucket=os.getenv('S3_BUCKET_NAME')
            )
            
            await bot.send_result_to_user(chat_id, result_data)
            
        except Exception as e:
            logger.error(f"Error sending results to user: {e}")


class LipSyncOrchestrator:
    """
    Main orchestrator for the complete lip-sync system.
    
    Coordinates Telegram bot, message processing, and result delivery.
    """
    
    def __init__(self) -> None:
        """Initialize orchestrator components."""
        self.processor = LipSyncProcessor()
        self.bot = None
        self.queue = None
        
    async def start_system(self) -> None:
        """Start the complete lip-sync system."""
        logger.info("Starting Lip Sync Core system...")
        
        # Initialize components
        await self._initialize_components()
        
        # Start processing tasks
        tasks = [
            asyncio.create_task(self._start_message_processor()),
            asyncio.create_task(self._start_telegram_bot()),
            asyncio.create_task(self._monitor_system())
        ]
        
        # Run all tasks concurrently
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _initialize_components(self) -> None:
        """Initialize all system components."""
        try:
            # Create bot instance
            self.bot = AlfredTelegramBot(
                token=os.getenv('TELEGRAM_BOT_TOKEN'),
                sqs_queue_url=os.getenv('SQS_QUEUE_URL'),
                s3_bucket=os.getenv('S3_BUCKET_NAME')
            )
            
            # Create queue instance
            self.queue = SQSMessageQueue(
                queue_url=os.getenv('SQS_QUEUE_URL'),
                dead_letter_queue_url=os.getenv('SQS_DEAD_LETTER_QUEUE_URL')
            )
            
            # Load ML model
            await self.processor.load_model()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    async def _start_message_processor(self) -> None:
        """Start the message processing worker."""
        logger.info("Starting message processor...")
        
        await self.queue.start_worker(
            processor_function=self.processor.process_video,
            max_concurrent=3
        )
    
    async def _start_telegram_bot(self) -> None:
        """Start the Telegram bot."""
        logger.info("Starting Telegram bot...")
        
        # Run bot in thread to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.bot.run)
    
    async def _monitor_system(self) -> None:
        """Monitor system health and performance."""
        logger.info("Starting system monitor...")
        
        while True:
            try:
                # Get queue statistics
                queue_attrs = await self.queue.get_queue_attributes()
                pending_messages = queue_attrs.get('ApproximateNumberOfMessages', 0)
                
                # Get system statistics
                hermes = HermesDynamoDB()
                system_stats = await hermes.get_system_stats()
                
                logger.info(f"System Status - Pending messages: {pending_messages}, "
                           f"Total users: {system_stats.get('total_users', 0)}")
                
                # Sleep for monitoring interval
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Error in system monitor: {e}")
                await asyncio.sleep(30)


def main():
    """Main entry point for the application."""
    # Verify environment variables
    required_env_vars = [
        'TELEGRAM_BOT_TOKEN',
        'SQS_QUEUE_URL', 
        'S3_BUCKET_NAME',
        'DYNAMODB_USERS_TABLE',
        'DYNAMODB_JOBS_TABLE',
        'DYNAMODB_RESULTS_TABLE'
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        return
    
    # Start the orchestrator
    orchestrator = LipSyncOrchestrator()
    
    try:
        asyncio.run(orchestrator.start_system())
    except KeyboardInterrupt:
        logger.info("System shutdown requested")
    except Exception as e:
        logger.error(f"System error: {e}")


if __name__ == "__main__":
    main()