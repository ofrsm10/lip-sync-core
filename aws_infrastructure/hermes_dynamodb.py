"""
DynamoDB data storage module (Hermes).

This module manages all data persistence for the lip-sync application
using AWS DynamoDB. Hermes handles user data, processing jobs,
results, and system metrics.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from decimal import Decimal

import boto3
from boto3.dynamodb.conditions import Key, Attr
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HermesDynamoDB:
    """
    DynamoDB data management for lip-sync application.
    
    Hermes provides a high-level interface for storing and retrieving
    user data, processing jobs, results, and system metrics.
    """
    
    def __init__(self, region: str = 'us-east-1') -> None:
        """
        Initialize DynamoDB client and tables.
        
        Args:
            region (str): AWS region for DynamoDB
        """
        self.region = region
        self.dynamodb = boto3.resource('dynamodb', region_name=region)
        
        # Table references
        self.users_table_name = os.getenv('DYNAMODB_USERS_TABLE', 'lip-sync-users')
        self.jobs_table_name = os.getenv('DYNAMODB_JOBS_TABLE', 'lip-sync-jobs')
        self.results_table_name = os.getenv('DYNAMODB_RESULTS_TABLE', 'lip-sync-results')
        
        self.users_table = self.dynamodb.Table(self.users_table_name)
        self.jobs_table = self.dynamodb.Table(self.jobs_table_name)
        self.results_table = self.dynamodb.Table(self.results_table_name)
        
        logger.info("Hermes DynamoDB initialized")
    
    # User Management
    
    async def create_user(self, user_id: int, username: str = None, 
                         first_name: str = None, last_name: str = None) -> bool:
        """
        Create or update user record.
        
        Args:
            user_id (int): Telegram user ID
            username (str, optional): Telegram username
            first_name (str, optional): User's first name
            last_name (str, optional): User's last name
            
        Returns:
            bool: True if successful
        """
        try:
            user_data = {
                'user_id': user_id,
                'username': username,
                'first_name': first_name,
                'last_name': last_name,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'updated_at': datetime.now(timezone.utc).isoformat(),
                'total_videos_processed': 0,
                'is_active': True
            }
            
            self.users_table.put_item(Item=user_data)
            logger.info(f"Created/updated user: {user_id}")
            return True
            
        except ClientError as e:
            logger.error(f"Error creating user {user_id}: {e}")
            return False
    
    async def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve user data.
        
        Args:
            user_id (int): Telegram user ID
            
        Returns:
            Optional[Dict[str, Any]]: User data or None if not found
        """
        try:
            response = self.users_table.get_item(Key={'user_id': user_id})
            return response.get('Item')
            
        except ClientError as e:
            logger.error(f"Error retrieving user {user_id}: {e}")
            return None
    
    async def update_user_stats(self, user_id: int, increment_videos: int = 1) -> bool:
        """
        Update user statistics.
        
        Args:
            user_id (int): Telegram user ID
            increment_videos (int): Number to increment video count
            
        Returns:
            bool: True if successful
        """
        try:
            self.users_table.update_item(
                Key={'user_id': user_id},
                UpdateExpression='ADD total_videos_processed :inc SET updated_at = :time',
                ExpressionAttributeValues={
                    ':inc': increment_videos,
                    ':time': datetime.now(timezone.utc).isoformat()
                }
            )
            return True
            
        except ClientError as e:
            logger.error(f"Error updating user stats {user_id}: {e}")
            return False
    
    # Job Management
    
    async def create_job(self, job_id: str, user_id: int, video_key: str, 
                        file_name: str, chat_id: int) -> bool:
        """
        Create a new processing job.
        
        Args:
            job_id (str): Unique job identifier
            user_id (int): User who submitted the job
            video_key (str): S3 key for the video file
            file_name (str): Original file name
            chat_id (int): Telegram chat ID
            
        Returns:
            bool: True if successful
        """
        try:
            job_data = {
                'job_id': job_id,
                'user_id': user_id,
                'video_key': video_key,
                'file_name': file_name,
                'chat_id': chat_id,
                'status': 'pending',
                'created_at': datetime.now(timezone.utc).isoformat(),
                'updated_at': datetime.now(timezone.utc).isoformat(),
                'processing_start_time': None,
                'processing_end_time': None,
                'error_message': None
            }
            
            self.jobs_table.put_item(Item=job_data)
            logger.info(f"Created job: {job_id}")
            return True
            
        except ClientError as e:
            logger.error(f"Error creating job {job_id}: {e}")
            return False
    
    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve job data.
        
        Args:
            job_id (str): Job identifier
            
        Returns:
            Optional[Dict[str, Any]]: Job data or None if not found
        """
        try:
            response = self.jobs_table.get_item(Key={'job_id': job_id})
            return response.get('Item')
            
        except ClientError as e:
            logger.error(f"Error retrieving job {job_id}: {e}")
            return None
    
    async def update_job_status(self, job_id: str, status: str, 
                               error_message: str = None) -> bool:
        """
        Update job status.
        
        Args:
            job_id (str): Job identifier
            status (str): New status (pending, processing, completed, failed)
            error_message (str, optional): Error message if failed
            
        Returns:
            bool: True if successful
        """
        try:
            update_expression = 'SET #status = :status, updated_at = :time'
            expression_values = {
                ':status': status,
                ':time': datetime.now(timezone.utc).isoformat()
            }
            expression_names = {'#status': 'status'}
            
            if status == 'processing':
                update_expression += ', processing_start_time = :start_time'
                expression_values[':start_time'] = datetime.now(timezone.utc).isoformat()
            elif status in ['completed', 'failed']:
                update_expression += ', processing_end_time = :end_time'
                expression_values[':end_time'] = datetime.now(timezone.utc).isoformat()
            
            if error_message:
                update_expression += ', error_message = :error'
                expression_values[':error'] = error_message
            
            self.jobs_table.update_item(
                Key={'job_id': job_id},
                UpdateExpression=update_expression,
                ExpressionAttributeValues=expression_values,
                ExpressionAttributeNames=expression_names
            )
            
            logger.info(f"Updated job {job_id} status to {status}")
            return True
            
        except ClientError as e:
            logger.error(f"Error updating job status {job_id}: {e}")
            return False
    
    async def get_pending_jobs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get pending jobs for processing.
        
        Args:
            limit (int): Maximum number of jobs to retrieve
            
        Returns:
            List[Dict[str, Any]]: List of pending jobs
        """
        try:
            response = self.jobs_table.scan(
                FilterExpression=Attr('status').eq('pending'),
                Limit=limit
            )
            return response.get('Items', [])
            
        except ClientError as e:
            logger.error(f"Error retrieving pending jobs: {e}")
            return []
    
    # Results Management
    
    async def save_result(self, job_id: str, predictions: List[str], 
                         confidence_scores: List[float], processing_time: float,
                         model_version: str = "1.0") -> bool:
        """
        Save processing results.
        
        Args:
            job_id (str): Job identifier
            predictions (List[str]): Predicted words
            confidence_scores (List[float]): Confidence scores for predictions
            processing_time (float): Processing time in seconds
            model_version (str): Model version used
            
        Returns:
            bool: True if successful
        """
        try:
            result_data = {
                'job_id': job_id,
                'predictions': predictions,
                'confidence_scores': [Decimal(str(score)) for score in confidence_scores],
                'processing_time': Decimal(str(processing_time)),
                'model_version': model_version,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'success': True
            }
            
            self.results_table.put_item(Item=result_data)
            logger.info(f"Saved result for job: {job_id}")
            return True
            
        except ClientError as e:
            logger.error(f"Error saving result for job {job_id}: {e}")
            return False
    
    async def get_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve processing results.
        
        Args:
            job_id (str): Job identifier
            
        Returns:
            Optional[Dict[str, Any]]: Results or None if not found
        """
        try:
            response = self.results_table.get_item(Key={'job_id': job_id})
            return response.get('Item')
            
        except ClientError as e:
            logger.error(f"Error retrieving result for job {job_id}: {e}")
            return None
    
    async def get_user_results(self, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent results for a user.
        
        Args:
            user_id (int): User ID
            limit (int): Maximum number of results
            
        Returns:
            List[Dict[str, Any]]: List of user results
        """
        try:
            # First get user's jobs
            jobs_response = self.jobs_table.scan(
                FilterExpression=Attr('user_id').eq(user_id) & Attr('status').eq('completed'),
                Limit=limit
            )
            
            results = []
            for job in jobs_response.get('Items', []):
                result = await self.get_result(job['job_id'])
                if result:
                    result['job_info'] = job
                    results.append(result)
            
            return results
            
        except ClientError as e:
            logger.error(f"Error retrieving user results {user_id}: {e}")
            return []
    
    # Analytics and Metrics
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system-wide statistics.
        
        Returns:
            Dict[str, Any]: System statistics
        """
        try:
            # Get user count
            users_response = self.users_table.scan(Select='COUNT')
            user_count = users_response['Count']
            
            # Get job counts by status
            jobs_response = self.jobs_table.scan()
            jobs = jobs_response.get('Items', [])
            
            job_stats = {
                'total': len(jobs),
                'pending': len([j for j in jobs if j.get('status') == 'pending']),
                'processing': len([j for j in jobs if j.get('status') == 'processing']),
                'completed': len([j for j in jobs if j.get('status') == 'completed']),
                'failed': len([j for j in jobs if j.get('status') == 'failed'])
            }
            
            return {
                'total_users': user_count,
                'job_statistics': job_stats,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except ClientError as e:
            logger.error(f"Error retrieving system stats: {e}")
            return {}


def create_hermes() -> HermesDynamoDB:
    """
    Factory function to create Hermes DynamoDB instance.
    
    Returns:
        HermesDynamoDB: Configured Hermes instance
    """
    region = os.getenv('AWS_REGION', 'us-east-1')
    return HermesDynamoDB(region)


if __name__ == "__main__":
    # Example usage
    hermes = create_hermes()
    logger.info("Hermes DynamoDB initialized")