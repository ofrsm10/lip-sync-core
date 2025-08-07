#!/bin/bash

# Lip Sync Core - AWS Deployment Script
# This script deploys the complete AWS infrastructure for the lip-sync application

set -e

# Configuration
PROJECT_NAME="lip-sync-core"
AWS_REGION=${AWS_REGION:-"us-east-1"}
ENVIRONMENT=${ENVIRONMENT:-"prod"}
STACK_NAME="$PROJECT_NAME-$ENVIRONMENT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

echo_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

echo_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check required environment variables
check_requirements() {
    echo_info "Checking deployment requirements..."
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        echo_error "AWS CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo_error "Docker is not installed. Please install it first."
        exit 1
    fi
    
    # Check required environment variables
    if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
        echo_error "TELEGRAM_BOT_TOKEN environment variable is required."
        echo_info "Get a bot token from @BotFather on Telegram."
        exit 1
    fi
    
    # Verify AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        echo_error "AWS credentials not configured. Please run 'aws configure'."
        exit 1
    fi
    
    echo_success "All requirements met."
}

# Create ECR repository and push Docker image
build_and_push_image() {
    echo_info "Building and pushing Docker image..."
    
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    ECR_REGISTRY="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
    ECR_REPOSITORY="$PROJECT_NAME-$ENVIRONMENT"
    IMAGE_TAG="latest"
    ECR_IMAGE_URI="$ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG"
    
    # Create ECR repository if it doesn't exist
    if ! aws ecr describe-repositories --repository-names "$ECR_REPOSITORY" --region "$AWS_REGION" &> /dev/null; then
        echo_info "Creating ECR repository: $ECR_REPOSITORY"
        aws ecr create-repository \
            --repository-name "$ECR_REPOSITORY" \
            --region "$AWS_REGION" \
            --image-scanning-configuration scanOnPush=true
    fi
    
    # Get Docker login token
    echo_info "Logging into ECR..."
    aws ecr get-login-password --region "$AWS_REGION" | \
        docker login --username AWS --password-stdin "$ECR_REGISTRY"
    
    # Build Docker image
    echo_info "Building Docker image..."
    docker build -t "$ECR_REPOSITORY:$IMAGE_TAG" .
    
    # Tag for ECR
    docker tag "$ECR_REPOSITORY:$IMAGE_TAG" "$ECR_IMAGE_URI"
    
    # Push to ECR
    echo_info "Pushing Docker image to ECR..."
    docker push "$ECR_IMAGE_URI"
    
    echo_success "Docker image pushed successfully: $ECR_IMAGE_URI"
    echo "$ECR_IMAGE_URI" > .ecr_image_uri
}

# Deploy CloudFormation stack
deploy_infrastructure() {
    echo_info "Deploying AWS infrastructure..."
    
    ECR_IMAGE_URI=$(cat .ecr_image_uri)
    
    # Prepare CloudFormation parameters
    PARAMETERS="ParameterKey=ProjectName,ParameterValue=$PROJECT_NAME"
    PARAMETERS="$PARAMETERS ParameterKey=Environment,ParameterValue=$ENVIRONMENT"
    PARAMETERS="$PARAMETERS ParameterKey=TelegramBotToken,ParameterValue=$TELEGRAM_BOT_TOKEN"
    PARAMETERS="$PARAMETERS ParameterKey=ECRImageURI,ParameterValue=$ECR_IMAGE_URI"
    
    # Deploy stack
    if aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$AWS_REGION" &> /dev/null; then
        echo_info "Updating existing CloudFormation stack: $STACK_NAME"
        aws cloudformation update-stack \
            --stack-name "$STACK_NAME" \
            --template-body file://deployment/cloudformation.yaml \
            --parameters $PARAMETERS \
            --capabilities CAPABILITY_NAMED_IAM \
            --region "$AWS_REGION"
    else
        echo_info "Creating new CloudFormation stack: $STACK_NAME"
        aws cloudformation create-stack \
            --stack-name "$STACK_NAME" \
            --template-body file://deployment/cloudformation.yaml \
            --parameters $PARAMETERS \
            --capabilities CAPABILITY_NAMED_IAM \
            --region "$AWS_REGION"
    fi
    
    # Wait for stack to complete
    echo_info "Waiting for CloudFormation stack to complete..."
    aws cloudformation wait stack-create-complete \
        --stack-name "$STACK_NAME" \
        --region "$AWS_REGION" 2>/dev/null || \
    aws cloudformation wait stack-update-complete \
        --stack-name "$STACK_NAME" \
        --region "$AWS_REGION"
    
    echo_success "CloudFormation stack deployed successfully!"
}

# Get stack outputs
get_stack_outputs() {
    echo_info "Retrieving stack outputs..."
    
    aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$AWS_REGION" \
        --query 'Stacks[0].Outputs[*].[OutputKey,OutputValue]' \
        --output table
}

# Upload model to S3
upload_model() {
    echo_info "Uploading model files to S3..."
    
    S3_BUCKET=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$AWS_REGION" \
        --query 'Stacks[0].Outputs[?OutputKey==`S3BucketName`].OutputValue' \
        --output text)
    
    # Check if model exists locally
    if [ -f "cnn_model/cnn_model.pth" ]; then
        echo_info "Uploading model to S3: s3://$S3_BUCKET/models/cnn_model/latest/model.pth"
        aws s3 cp cnn_model/cnn_model.pth "s3://$S3_BUCKET/models/cnn_model/latest/model.pth"
        echo_success "Model uploaded successfully"
    else
        echo_warning "Model file not found locally. The application will need to be trained first."
    fi
}

# Setup monitoring and logs
setup_monitoring() {
    echo_info "Setting up monitoring and logging..."
    
    # Create CloudWatch dashboard
    cat > dashboard.json << EOF
{
    "widgets": [
        {
            "type": "metric",
            "properties": {
                "metrics": [
                    [ "AWS/SQS", "ApproximateNumberOfVisibleMessages", "QueueName", "${PROJECT_NAME}-${ENVIRONMENT}-processing" ],
                    [ ".", "NumberOfMessagesSent", ".", "." ],
                    [ ".", "NumberOfMessagesReceived", ".", "." ]
                ],
                "period": 300,
                "stat": "Average",
                "region": "${AWS_REGION}",
                "title": "SQS Metrics"
            }
        },
        {
            "type": "metric",
            "properties": {
                "metrics": [
                    [ "AWS/DynamoDB", "ConsumedReadCapacityUnits", "TableName", "${PROJECT_NAME}-${ENVIRONMENT}-users" ],
                    [ ".", "ConsumedWriteCapacityUnits", ".", "." ]
                ],
                "period": 300,
                "stat": "Sum",
                "region": "${AWS_REGION}",
                "title": "DynamoDB Metrics"
            }
        }
    ]
}
EOF
    
    aws cloudwatch put-dashboard \
        --dashboard-name "${PROJECT_NAME}-${ENVIRONMENT}" \
        --dashboard-body file://dashboard.json \
        --region "$AWS_REGION"
    
    rm dashboard.json
    echo_success "Monitoring dashboard created"
}

# Test deployment
test_deployment() {
    echo_info "Testing deployment..."
    
    # Get queue URL
    QUEUE_URL=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$AWS_REGION" \
        --query 'Stacks[0].Outputs[?OutputKey==`ProcessingQueueURL`].OutputValue' \
        --output text)
    
    # Send test message
    aws sqs send-message \
        --queue-url "$QUEUE_URL" \
        --message-body '{"test": true, "message": "Deployment test"}' \
        --region "$AWS_REGION"
    
    echo_success "Test message sent to queue"
}

# Print deployment summary
print_summary() {
    echo ""
    echo_success "🎉 Deployment completed successfully!"
    echo ""
    echo_info "Deployment Summary:"
    echo "  Project: $PROJECT_NAME"
    echo "  Environment: $ENVIRONMENT"
    echo "  Region: $AWS_REGION"
    echo "  Stack: $STACK_NAME"
    echo ""
    echo_info "Next Steps:"
    echo "1. Set up your Telegram bot webhook (if using webhooks)"
    echo "2. Train your model if not already done"
    echo "3. Test the bot by sending a video to your Telegram bot"
    echo "4. Monitor the CloudWatch dashboard for system health"
    echo ""
    echo_info "Useful Commands:"
    echo "  View logs: aws logs tail /ecs/${PROJECT_NAME}-${ENVIRONMENT}-telegram-bot --follow"
    echo "  Check queue: aws sqs get-queue-attributes --queue-url \$(aws cloudformation describe-stacks --stack-name $STACK_NAME --query 'Stacks[0].Outputs[?OutputKey==\`ProcessingQueueURL\`].OutputValue' --output text)"
    echo "  Update stack: ./deploy.sh"
}

# Cleanup function
cleanup() {
    echo_warning "Cleaning up temporary files..."
    rm -f .ecr_image_uri dashboard.json
}

# Main deployment flow
main() {
    echo_info "Starting deployment of Lip Sync Core to AWS..."
    echo_info "Project: $PROJECT_NAME, Environment: $ENVIRONMENT, Region: $AWS_REGION"
    echo ""
    
    # Set cleanup trap
    trap cleanup EXIT
    
    # Run deployment steps
    check_requirements
    build_and_push_image
    deploy_infrastructure
    get_stack_outputs
    upload_model
    setup_monitoring
    test_deployment
    print_summary
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "destroy")
        echo_warning "Destroying CloudFormation stack: $STACK_NAME"
        aws cloudformation delete-stack --stack-name "$STACK_NAME" --region "$AWS_REGION"
        aws cloudformation wait stack-delete-complete --stack-name "$STACK_NAME" --region "$AWS_REGION"
        echo_success "Stack destroyed successfully"
        ;;
    "status")
        aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$AWS_REGION" \
            --query 'Stacks[0].{StackName:StackName,Status:StackStatus,LastUpdated:LastUpdatedTime}'
        ;;
    "logs")
        aws logs tail "/ecs/${PROJECT_NAME}-${ENVIRONMENT}-telegram-bot" --follow --region "$AWS_REGION"
        ;;
    *)
        echo "Usage: $0 [deploy|destroy|status|logs]"
        echo "  deploy  - Deploy the infrastructure (default)"
        echo "  destroy - Destroy the infrastructure"
        echo "  status  - Check stack status"
        echo "  logs    - Follow application logs"
        exit 1
        ;;
esac