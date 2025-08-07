"""
Telegram Bot integration module (Alfred).

This module handles all Telegram API interactions for the lip-sync application.
Alfred serves as the main interface between users and the lip-sync processing pipeline.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
import asyncio
from datetime import datetime

import boto3
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlfredTelegramBot:
    """
    Telegram bot interface for lip-sync processing.
    
    Alfred handles user interactions, video uploads, and coordinates
    with the processing pipeline through AWS services.
    """
    
    def __init__(self, token: str, sqs_queue_url: str, s3_bucket: str) -> None:
        """
        Initialize the Telegram bot.
        
        Args:
            token (str): Telegram bot token
            sqs_queue_url (str): AWS SQS queue URL for processing jobs
            s3_bucket (str): AWS S3 bucket for file storage
        """
        self.token = token
        self.sqs_queue_url = sqs_queue_url
        self.s3_bucket = s3_bucket
        
        # AWS clients
        self.sqs_client = boto3.client('sqs')
        self.s3_client = boto3.client('s3')
        
        # Bot application
        self.application = Application.builder().token(token).build()
        self._setup_handlers()
        
        logger.info("Alfred Telegram Bot initialized")
    
    def _setup_handlers(self) -> None:
        """Set up message and command handlers."""
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("status", self.status_command))
        self.application.add_handler(MessageHandler(filters.VIDEO, self.handle_video))
        self.application.add_handler(MessageHandler(filters.TEXT, self.handle_text))
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        welcome_message = (
            "🎬 Welcome to Lip Sync Core!\n\n"
            "I can analyze lip movements in videos and predict spoken words.\n\n"
            "Send me a video file to get started!\n\n"
            "Commands:\n"
            "/help - Show this help message\n"
            "/status - Check processing status"
        )
        await update.message.reply_text(welcome_message)
        logger.info(f"Start command from user {update.effective_user.id}")
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        help_message = (
            "🤖 Lip Sync Core Bot Help\n\n"
            "How to use:\n"
            "1. Send me a video file (MP4, MOV, AVI)\n"
            "2. I'll process the lip movements\n"
            "3. You'll receive the predicted words\n\n"
            "Supported formats: MP4, MOV, AVI\n"
            "Max file size: 50MB\n\n"
            "Commands:\n"
            "/start - Start the bot\n"
            "/status - Check processing status"
        )
        await update.message.reply_text(help_message)
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status command."""
        # TODO: Implement status checking from DynamoDB
        await update.message.reply_text("🟢 Bot is running and ready to process videos!")
    
    async def handle_video(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle video uploads from users.
        
        Args:
            update: Telegram update containing the video
            context: Bot context
        """
        try:
            video = update.message.video
            user_id = update.effective_user.id
            
            logger.info(f"Received video from user {user_id}: {video.file_name}")
            
            # Check file size
            if video.file_size > 50 * 1024 * 1024:  # 50MB limit
                await update.message.reply_text("❌ File too large. Maximum size is 50MB.")
                return
            
            await update.message.reply_text("📥 Downloading video...")
            
            # Download video file
            file = await context.bot.get_file(video.file_id)
            file_key = f"videos/{user_id}/{datetime.now().isoformat()}_{video.file_name}"
            
            # Upload to S3
            await self._upload_to_s3(file, file_key)
            
            # Send processing job to queue
            job_data = {
                'user_id': user_id,
                'video_key': file_key,
                'file_name': video.file_name,
                'timestamp': datetime.now().isoformat(),
                'chat_id': update.effective_chat.id
            }
            
            await self._send_to_queue(job_data)
            
            await update.message.reply_text(
                "✅ Video received and queued for processing!\n"
                "I'll send you the results when ready. ⏳"
            )
            
        except Exception as e:
            logger.error(f"Error handling video: {e}")
            await update.message.reply_text("❌ Error processing video. Please try again.")
    
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle text messages."""
        await update.message.reply_text(
            "📝 I understand text, but I specialize in video analysis!\n"
            "Send me a video file to analyze lip movements."
        )
    
    async def _upload_to_s3(self, file, file_key: str) -> None:
        """Upload file to S3 bucket."""
        try:
            # Download file to memory
            file_bytes = await file.download_as_bytearray()
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=file_key,
                Body=file_bytes,
                ContentType='video/mp4'
            )
            
            logger.info(f"Uploaded file to S3: {file_key}")
            
        except Exception as e:
            logger.error(f"Error uploading to S3: {e}")
            raise
    
    async def _send_to_queue(self, job_data: Dict[str, Any]) -> None:
        """Send processing job to SQS queue."""
        try:
            response = self.sqs_client.send_message(
                QueueUrl=self.sqs_queue_url,
                MessageBody=json.dumps(job_data),
                MessageAttributes={
                    'JobType': {
                        'StringValue': 'lip_sync_processing',
                        'DataType': 'String'
                    }
                }
            )
            
            logger.info(f"Sent job to queue: {response['MessageId']}")
            
        except Exception as e:
            logger.error(f"Error sending to queue: {e}")
            raise
    
    async def send_result_to_user(self, chat_id: int, result: Dict[str, Any]) -> None:
        """
        Send processing results back to user.
        
        Args:
            chat_id (int): Telegram chat ID
            result (Dict[str, Any]): Processing results
        """
        try:
            bot = Bot(token=self.token)
            
            if result.get('success'):
                message = (
                    f"✅ Analysis complete!\n\n"
                    f"🗣️ Predicted words: {', '.join(result.get('predictions', []))}\n"
                    f"📊 Confidence: {result.get('confidence', 0):.2%}\n"
                    f"⏱️ Processing time: {result.get('processing_time', 0):.2f}s"
                )
            else:
                message = f"❌ Processing failed: {result.get('error', 'Unknown error')}"
            
            await bot.send_message(chat_id=chat_id, text=message)
            logger.info(f"Sent result to chat {chat_id}")
            
        except Exception as e:
            logger.error(f"Error sending result to user: {e}")
    
    def run(self) -> None:
        """Start the bot."""
        logger.info("Starting Alfred Telegram Bot...")
        self.application.run_polling()


def create_bot() -> AlfredTelegramBot:
    """
    Factory function to create and configure the Telegram bot.
    
    Returns:
        AlfredTelegramBot: Configured bot instance
    """
    # Get configuration from environment variables
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    sqs_queue_url = os.getenv('SQS_QUEUE_URL')
    s3_bucket = os.getenv('S3_BUCKET_NAME')
    
    if not all([token, sqs_queue_url, s3_bucket]):
        raise ValueError("Missing required environment variables")
    
    return AlfredTelegramBot(token, sqs_queue_url, s3_bucket)


if __name__ == "__main__":
    bot = create_bot()
    bot.run()