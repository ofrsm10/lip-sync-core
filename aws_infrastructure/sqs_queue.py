"""
SQS Message Queue System for asynchronous processing.

This module handles all message queue operations for the lip-sync application,
providing reliable asynchronous processing of video analysis jobs.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Callable
import asyncio
from datetime import datetime, timezone
import boto3
from botocore.exceptions import ClientError
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SQSMessageQueue:
    """
    SQS message queue for asynchronous job processing.
    
    Handles queuing, processing, and monitoring of lip-sync analysis jobs.
    """
    
    def __init__(self, queue_url: str, dead_letter_queue_url: str = None, 
                 region: str = 'us-east-1') -> None:
        """
        Initialize SQS client and queue configuration.
        
        Args:
            queue_url (str): Main processing queue URL
            dead_letter_queue_url (str, optional): Dead letter queue URL for failed messages
            region (str): AWS region
        """
        self.queue_url = queue_url
        self.dead_letter_queue_url = dead_letter_queue_url
        self.region = region
        self.sqs_client = boto3.client('sqs', region_name=region)
        
        # Queue attributes
        self.visibility_timeout = 300  # 5 minutes
        self.message_retention_period = 1209600  # 14 days
        self.max_receive_count = 3
        
        logger.info(f"SQS Message Queue initialized: {queue_url}")
    
    # Message Sending
    
    async def send_job_message(self, job_data: Dict[str, Any], 
                              delay_seconds: int = 0) -> Optional[str]:
        """
        Send a job message to the queue.
        
        Args:
            job_data (Dict[str, Any]): Job data to process
            delay_seconds (int): Delay before message becomes visible
            
        Returns:
            Optional[str]: Message ID if successful
        """
        try:
            # Add message metadata
            message_body = {
                'job_id': job_data.get('job_id', str(uuid.uuid4())),
                'job_type': 'lip_sync_processing',
                'data': job_data,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'retry_count': 0
            }
            
            # Message attributes for filtering and routing
            message_attributes = {
                'JobType': {
                    'StringValue': 'lip_sync_processing',
                    'DataType': 'String'
                },
                'Priority': {
                    'StringValue': job_data.get('priority', 'normal'),
                    'DataType': 'String'
                },
                'UserId': {
                    'StringValue': str(job_data.get('user_id', 0)),
                    'DataType': 'String'
                }
            }
            
            response = self.sqs_client.send_message(
                QueueUrl=self.queue_url,
                MessageBody=json.dumps(message_body),
                MessageAttributes=message_attributes,
                DelaySeconds=delay_seconds
            )
            
            message_id = response['MessageId']
            logger.info(f"Sent message to queue: {message_id}")
            return message_id
            
        except ClientError as e:
            logger.error(f"Error sending message to queue: {e}")
            return None
    
    async def send_batch_messages(self, jobs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Send multiple messages in a batch (up to 10).
        
        Args:
            jobs (List[Dict[str, Any]]): List of job data
            
        Returns:
            Dict[str, Any]: Batch operation results
        """
        try:
            if len(jobs) > 10:
                logger.warning("Batch size exceeds SQS limit of 10, taking first 10")
                jobs = jobs[:10]
            
            entries = []
            for i, job_data in enumerate(jobs):
                message_body = {
                    'job_id': job_data.get('job_id', str(uuid.uuid4())),
                    'job_type': 'lip_sync_processing',
                    'data': job_data,
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'retry_count': 0
                }
                
                entries.append({
                    'Id': str(i),
                    'MessageBody': json.dumps(message_body),
                    'MessageAttributes': {
                        'JobType': {
                            'StringValue': 'lip_sync_processing',
                            'DataType': 'String'
                        }
                    }
                })
            
            response = self.sqs_client.send_message_batch(
                QueueUrl=self.queue_url,
                Entries=entries
            )
            
            successful = len(response.get('Successful', []))
            failed = len(response.get('Failed', []))
            
            logger.info(f"Batch send: {successful} successful, {failed} failed")
            return {
                'successful': successful,
                'failed': failed,
                'successful_messages': response.get('Successful', []),
                'failed_messages': response.get('Failed', [])
            }
            
        except ClientError as e:
            logger.error(f"Error sending batch messages: {e}")
            return {'successful': 0, 'failed': len(jobs)}
    
    # Message Receiving and Processing
    
    async def receive_messages(self, max_messages: int = 1, 
                              wait_time_seconds: int = 20) -> List[Dict[str, Any]]:
        """
        Receive messages from the queue.
        
        Args:
            max_messages (int): Maximum number of messages to receive (1-10)
            wait_time_seconds (int): Long polling wait time (0-20 seconds)
            
        Returns:
            List[Dict[str, Any]]: List of received messages
        """
        try:
            response = self.sqs_client.receive_message(
                QueueUrl=self.queue_url,
                MaxNumberOfMessages=min(max_messages, 10),
                WaitTimeSeconds=wait_time_seconds,
                MessageAttributeNames=['All'],
                AttributeNames=['All']
            )
            
            messages = response.get('Messages', [])
            logger.info(f"Received {len(messages)} messages from queue")
            
            # Parse message bodies
            parsed_messages = []
            for message in messages:
                try:
                    body = json.loads(message['Body'])
                    parsed_message = {
                        'sqs_message': message,
                        'job_data': body,
                        'receipt_handle': message['ReceiptHandle'],
                        'message_id': message['MessageId']
                    }
                    parsed_messages.append(parsed_message)
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing message body: {e}")
            
            return parsed_messages
            
        except ClientError as e:
            logger.error(f"Error receiving messages: {e}")
            return []
    
    async def delete_message(self, receipt_handle: str) -> bool:
        """
        Delete a processed message from the queue.
        
        Args:
            receipt_handle (str): Receipt handle from received message
            
        Returns:
            bool: True if successful
        """
        try:
            self.sqs_client.delete_message(
                QueueUrl=self.queue_url,
                ReceiptHandle=receipt_handle
            )
            
            logger.info("Message deleted successfully")
            return True
            
        except ClientError as e:
            logger.error(f"Error deleting message: {e}")
            return False
    
    async def change_message_visibility(self, receipt_handle: str, 
                                      visibility_timeout: int) -> bool:
        """
        Change message visibility timeout.
        
        Args:
            receipt_handle (str): Receipt handle from received message
            visibility_timeout (int): New visibility timeout in seconds
            
        Returns:
            bool: True if successful
        """
        try:
            self.sqs_client.change_message_visibility(
                QueueUrl=self.queue_url,
                ReceiptHandle=receipt_handle,
                VisibilityTimeout=visibility_timeout
            )
            
            logger.info(f"Changed message visibility to {visibility_timeout} seconds")
            return True
            
        except ClientError as e:
            logger.error(f"Error changing message visibility: {e}")
            return False
    
    # Queue Management
    
    async def get_queue_attributes(self) -> Dict[str, Any]:
        """
        Get queue attributes and statistics.
        
        Returns:
            Dict[str, Any]: Queue attributes
        """
        try:
            response = self.sqs_client.get_queue_attributes(
                QueueUrl=self.queue_url,
                AttributeNames=['All']
            )
            
            attributes = response.get('Attributes', {})
            
            # Convert numeric attributes
            numeric_attrs = [
                'ApproximateNumberOfMessages',
                'ApproximateNumberOfMessagesNotVisible',
                'ApproximateNumberOfMessagesDelayed'
            ]
            
            for attr in numeric_attrs:
                if attr in attributes:
                    attributes[attr] = int(attributes[attr])
            
            return attributes
            
        except ClientError as e:
            logger.error(f"Error getting queue attributes: {e}")
            return {}
    
    async def purge_queue(self) -> bool:
        """
        Purge all messages from the queue.
        
        Returns:
            bool: True if successful
        """
        try:
            self.sqs_client.purge_queue(QueueUrl=self.queue_url)
            logger.info("Queue purged successfully")
            return True
            
        except ClientError as e:
            logger.error(f"Error purging queue: {e}")
            return False
    
    # Dead Letter Queue Management
    
    async def get_dead_letter_messages(self, max_messages: int = 10) -> List[Dict[str, Any]]:
        """
        Get messages from dead letter queue for analysis.
        
        Args:
            max_messages (int): Maximum number of messages to retrieve
            
        Returns:
            List[Dict[str, Any]]: Dead letter messages
        """
        if not self.dead_letter_queue_url:
            logger.warning("No dead letter queue configured")
            return []
        
        try:
            response = self.sqs_client.receive_message(
                QueueUrl=self.dead_letter_queue_url,
                MaxNumberOfMessages=min(max_messages, 10),
                MessageAttributeNames=['All']
            )
            
            messages = response.get('Messages', [])
            logger.info(f"Retrieved {len(messages)} messages from dead letter queue")
            
            return messages
            
        except ClientError as e:
            logger.error(f"Error retrieving dead letter messages: {e}")
            return []
    
    async def requeue_dead_letter_message(self, message_body: str, 
                                        message_attributes: Dict[str, Any] = None) -> bool:
        """
        Requeue a message from dead letter queue back to main queue.
        
        Args:
            message_body (str): Message body to requeue
            message_attributes (Dict[str, Any], optional): Message attributes
            
        Returns:
            bool: True if successful
        """
        try:
            # Parse the original message and increment retry count
            try:
                body_data = json.loads(message_body)
                body_data['retry_count'] = body_data.get('retry_count', 0) + 1
                body_data['requeued_at'] = datetime.now(timezone.utc).isoformat()
                message_body = json.dumps(body_data)
            except json.JSONDecodeError:
                logger.warning("Could not parse message body for retry count update")
            
            self.sqs_client.send_message(
                QueueUrl=self.queue_url,
                MessageBody=message_body,
                MessageAttributes=message_attributes or {}
            )
            
            logger.info("Message requeued from dead letter queue")
            return True
            
        except ClientError as e:
            logger.error(f"Error requeuing dead letter message: {e}")
            return False
    
    # Worker Process Management
    
    async def start_worker(self, processor_function: Callable, 
                          max_concurrent: int = 5) -> None:
        """
        Start a worker process to handle messages.
        
        Args:
            processor_function (Callable): Function to process messages
            max_concurrent (int): Maximum concurrent message processing
        """
        logger.info(f"Starting worker with max concurrent: {max_concurrent}")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        while True:
            try:
                messages = await self.receive_messages(max_messages=max_concurrent)
                
                if not messages:
                    await asyncio.sleep(1)
                    continue
                
                # Process messages concurrently
                tasks = []
                for message in messages:
                    task = asyncio.create_task(
                        self._process_message_with_semaphore(
                            semaphore, message, processor_function
                        )
                    )
                    tasks.append(task)
                
                await asyncio.gather(*tasks, return_exceptions=True)
                
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                await asyncio.sleep(5)
    
    async def _process_message_with_semaphore(self, semaphore: asyncio.Semaphore,
                                            message: Dict[str, Any],
                                            processor_function: Callable) -> None:
        """
        Process a single message with semaphore control.
        
        Args:
            semaphore: Asyncio semaphore for concurrency control
            message: Message to process
            processor_function: Function to process the message
        """
        async with semaphore:
            try:
                # Process the message
                result = await processor_function(message['job_data'])
                
                if result.get('success', False):
                    # Delete message on successful processing
                    await self.delete_message(message['receipt_handle'])
                else:
                    # Extend visibility timeout for retry
                    await self.change_message_visibility(
                        message['receipt_handle'], 
                        self.visibility_timeout
                    )
                
            except Exception as e:
                logger.error(f"Error processing message {message['message_id']}: {e}")
                # Let message return to queue for retry


def create_sqs_queue() -> SQSMessageQueue:
    """
    Factory function to create SQS message queue.
    
    Returns:
        SQSMessageQueue: Configured SQS queue instance
    """
    queue_url = os.getenv('SQS_QUEUE_URL')
    dead_letter_queue_url = os.getenv('SQS_DEAD_LETTER_QUEUE_URL')
    region = os.getenv('AWS_REGION', 'us-east-1')
    
    if not queue_url:
        raise ValueError("SQS_QUEUE_URL environment variable is required")
    
    return SQSMessageQueue(queue_url, dead_letter_queue_url, region)


if __name__ == "__main__":
    # Example usage
    queue = create_sqs_queue()
    logger.info("SQS Message Queue initialized")