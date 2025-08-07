"""
S3 storage management for sentences and media files.

This module handles all file storage operations for the lip-sync application,
including video uploads, model artifacts, and processed results.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone
import boto3
from botocore.exceptions import ClientError
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class S3StorageManager:
    """
    S3 storage management for lip-sync application.
    
    Handles storage and retrieval of videos, models, results,
    and other application assets.
    """
    
    def __init__(self, bucket_name: str, region: str = 'us-east-1') -> None:
        """
        Initialize S3 client and bucket configuration.
        
        Args:
            bucket_name (str): S3 bucket name
            region (str): AWS region
        """
        self.bucket_name = bucket_name
        self.region = region
        self.s3_client = boto3.client('s3', region_name=region)
        
        # Storage prefixes for organization
        self.prefixes = {
            'videos': 'videos/',
            'models': 'models/',
            'results': 'results/',
            'sentences': 'sentences/',
            'temp': 'temp/',
            'processed': 'processed/'
        }
        
        logger.info(f"S3 Storage Manager initialized for bucket: {bucket_name}")
    
    # Video Management
    
    async def upload_video(self, file_path: str, user_id: int, 
                          file_name: str) -> Optional[str]:
        """
        Upload video file to S3.
        
        Args:
            file_path (str): Local path to video file
            user_id (int): User ID for organization
            file_name (str): Original file name
            
        Returns:
            Optional[str]: S3 key if successful, None otherwise
        """
        try:
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            s3_key = f"{self.prefixes['videos']}{user_id}/{timestamp}_{file_name}"
            
            # Upload with metadata
            extra_args = {
                'ContentType': 'video/mp4',
                'Metadata': {
                    'user_id': str(user_id),
                    'original_name': file_name,
                    'upload_time': datetime.now(timezone.utc).isoformat()
                }
            }
            
            self.s3_client.upload_file(file_path, self.bucket_name, s3_key, ExtraArgs=extra_args)
            
            logger.info(f"Uploaded video: {s3_key}")
            return s3_key
            
        except ClientError as e:
            logger.error(f"Error uploading video: {e}")
            return None
    
    async def download_video(self, s3_key: str, local_path: str = None) -> Optional[str]:
        """
        Download video from S3.
        
        Args:
            s3_key (str): S3 object key
            local_path (str, optional): Local download path
            
        Returns:
            Optional[str]: Local file path if successful
        """
        try:
            if local_path is None:
                # Create temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                local_path = temp_file.name
                temp_file.close()
            
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            
            logger.info(f"Downloaded video from {s3_key} to {local_path}")
            return local_path
            
        except ClientError as e:
            logger.error(f"Error downloading video {s3_key}: {e}")
            return None
    
    async def get_video_metadata(self, s3_key: str) -> Optional[Dict[str, Any]]:
        """
        Get video metadata from S3.
        
        Args:
            s3_key (str): S3 object key
            
        Returns:
            Optional[Dict[str, Any]]: Metadata dictionary
        """
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return {
                'size': response.get('ContentLength'),
                'last_modified': response.get('LastModified'),
                'content_type': response.get('ContentType'),
                'metadata': response.get('Metadata', {})
            }
            
        except ClientError as e:
            logger.error(f"Error getting metadata for {s3_key}: {e}")
            return None
    
    # Model Management
    
    async def upload_model(self, model_path: str, model_name: str, 
                          version: str = "latest") -> Optional[str]:
        """
        Upload trained model to S3.
        
        Args:
            model_path (str): Local path to model file
            model_name (str): Model identifier
            version (str): Model version
            
        Returns:
            Optional[str]: S3 key if successful
        """
        try:
            s3_key = f"{self.prefixes['models']}{model_name}/{version}/model.pth"
            
            extra_args = {
                'Metadata': {
                    'model_name': model_name,
                    'version': version,
                    'upload_time': datetime.now(timezone.utc).isoformat()
                }
            }
            
            self.s3_client.upload_file(model_path, self.bucket_name, s3_key, ExtraArgs=extra_args)
            
            logger.info(f"Uploaded model: {s3_key}")
            return s3_key
            
        except ClientError as e:
            logger.error(f"Error uploading model: {e}")
            return None
    
    async def download_model(self, model_name: str, version: str = "latest", 
                           local_path: str = None) -> Optional[str]:
        """
        Download model from S3.
        
        Args:
            model_name (str): Model identifier
            version (str): Model version
            local_path (str, optional): Local download path
            
        Returns:
            Optional[str]: Local file path if successful
        """
        try:
            s3_key = f"{self.prefixes['models']}{model_name}/{version}/model.pth"
            
            if local_path is None:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pth')
                local_path = temp_file.name
                temp_file.close()
            
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            
            logger.info(f"Downloaded model from {s3_key} to {local_path}")
            return local_path
            
        except ClientError as e:
            logger.error(f"Error downloading model {model_name}: {e}")
            return None
    
    # Results Management
    
    async def save_processing_result(self, job_id: str, result_data: Dict[str, Any]) -> Optional[str]:
        """
        Save processing results to S3.
        
        Args:
            job_id (str): Job identifier
            result_data (Dict[str, Any]): Processing results
            
        Returns:
            Optional[str]: S3 key if successful
        """
        try:
            s3_key = f"{self.prefixes['results']}{job_id}/result.json"
            
            # Add timestamp to result data
            result_data['saved_at'] = datetime.now(timezone.utc).isoformat()
            
            # Upload JSON result
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json.dumps(result_data, indent=2),
                ContentType='application/json',
                Metadata={
                    'job_id': job_id,
                    'result_type': 'lip_sync_processing'
                }
            )
            
            logger.info(f"Saved result: {s3_key}")
            return s3_key
            
        except ClientError as e:
            logger.error(f"Error saving result for job {job_id}: {e}")
            return None
    
    async def get_processing_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve processing results from S3.
        
        Args:
            job_id (str): Job identifier
            
        Returns:
            Optional[Dict[str, Any]]: Processing results
        """
        try:
            s3_key = f"{self.prefixes['results']}{job_id}/result.json"
            
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            result_data = json.loads(response['Body'].read().decode('utf-8'))
            
            return result_data
            
        except ClientError as e:
            logger.error(f"Error retrieving result for job {job_id}: {e}")
            return None
    
    # Sentence Database Management
    
    async def upload_sentence_database(self, sentences: List[Dict[str, Any]]) -> Optional[str]:
        """
        Upload sentence database to S3.
        
        Args:
            sentences (List[Dict[str, Any]]): List of sentence data
            
        Returns:
            Optional[str]: S3 key if successful
        """
        try:
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            s3_key = f"{self.prefixes['sentences']}database_{timestamp}.json"
            
            database = {
                'sentences': sentences,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'version': timestamp,
                'count': len(sentences)
            }
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json.dumps(database, indent=2, ensure_ascii=False),
                ContentType='application/json',
                Metadata={
                    'type': 'sentence_database',
                    'sentence_count': str(len(sentences))
                }
            )
            
            logger.info(f"Uploaded sentence database: {s3_key}")
            return s3_key
            
        except ClientError as e:
            logger.error(f"Error uploading sentence database: {e}")
            return None
    
    async def get_sentence_database(self, version: str = "latest") -> Optional[Dict[str, Any]]:
        """
        Retrieve sentence database from S3.
        
        Args:
            version (str): Database version or "latest"
            
        Returns:
            Optional[Dict[str, Any]]: Sentence database
        """
        try:
            if version == "latest":
                # Find the latest database file
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=self.prefixes['sentences']
                )
                
                if 'Contents' not in response:
                    return None
                
                # Sort by last modified and get the latest
                latest_obj = max(response['Contents'], key=lambda x: x['LastModified'])
                s3_key = latest_obj['Key']
            else:
                s3_key = f"{self.prefixes['sentences']}database_{version}.json"
            
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            database = json.loads(response['Body'].read().decode('utf-8'))
            
            return database
            
        except ClientError as e:
            logger.error(f"Error retrieving sentence database: {e}")
            return None
    
    # Utility Methods
    
    async def list_files(self, prefix: str, max_keys: int = 100) -> List[Dict[str, Any]]:
        """
        List files in S3 with given prefix.
        
        Args:
            prefix (str): S3 prefix to search
            max_keys (int): Maximum number of files to return
            
        Returns:
            List[Dict[str, Any]]: List of file information
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            files = []
            for obj in response.get('Contents', []):
                files.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'],
                    'etag': obj['ETag']
                })
            
            return files
            
        except ClientError as e:
            logger.error(f"Error listing files with prefix {prefix}: {e}")
            return []
    
    async def delete_file(self, s3_key: str) -> bool:
        """
        Delete file from S3.
        
        Args:
            s3_key (str): S3 object key
            
        Returns:
            bool: True if successful
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"Deleted file: {s3_key}")
            return True
            
        except ClientError as e:
            logger.error(f"Error deleting file {s3_key}: {e}")
            return False
    
    async def copy_file(self, source_key: str, dest_key: str) -> bool:
        """
        Copy file within S3 bucket.
        
        Args:
            source_key (str): Source S3 key
            dest_key (str): Destination S3 key
            
        Returns:
            bool: True if successful
        """
        try:
            copy_source = {'Bucket': self.bucket_name, 'Key': source_key}
            self.s3_client.copy_object(
                CopySource=copy_source,
                Bucket=self.bucket_name,
                Key=dest_key
            )
            
            logger.info(f"Copied file from {source_key} to {dest_key}")
            return True
            
        except ClientError as e:
            logger.error(f"Error copying file: {e}")
            return False
    
    async def get_bucket_size(self) -> Dict[str, Any]:
        """
        Get bucket size statistics.
        
        Returns:
            Dict[str, Any]: Bucket statistics
        """
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
            
            total_size = 0
            file_count = 0
            prefix_stats = {prefix: {'size': 0, 'count': 0} for prefix in self.prefixes.values()}
            
            for obj in response.get('Contents', []):
                size = obj['Size']
                key = obj['Key']
                
                total_size += size
                file_count += 1
                
                # Update prefix statistics
                for prefix_name, prefix_path in self.prefixes.items():
                    if key.startswith(prefix_path):
                        prefix_stats[prefix_path]['size'] += size
                        prefix_stats[prefix_path]['count'] += 1
                        break
            
            return {
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'total_files': file_count,
                'prefix_statistics': prefix_stats,
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
            
        except ClientError as e:
            logger.error(f"Error getting bucket statistics: {e}")
            return {}


def create_s3_manager() -> S3StorageManager:
    """
    Factory function to create S3 storage manager.
    
    Returns:
        S3StorageManager: Configured S3 manager instance
    """
    bucket_name = os.getenv('S3_BUCKET_NAME')
    region = os.getenv('AWS_REGION', 'us-east-1')
    
    if not bucket_name:
        raise ValueError("S3_BUCKET_NAME environment variable is required")
    
    return S3StorageManager(bucket_name, region)


if __name__ == "__main__":
    # Example usage
    s3_manager = create_s3_manager()
    logger.info("S3 Storage Manager initialized")