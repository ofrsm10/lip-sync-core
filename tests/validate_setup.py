#!/usr/bin/env python3
"""
Basic functionality tests for the Lip Sync Core project.

This script runs basic validation tests to ensure the core components
are working correctly.
"""

import sys
import os
import numpy as np
import torch
from typing import List, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports() -> bool:
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        # Core modules
        from constants.constants import CLASSES, MODEL_PATH
        from infra.cnn import CNN
        from infra.dataset import CustomDataset
        
        # AWS modules
        from aws_infrastructure.hermes_dynamodb import HermesDynamoDB
        from aws_infrastructure.s3_storage import S3StorageManager
        from aws_infrastructure.sqs_queue import SQSMessageQueue
        
        # Utility modules
        from utils.numpy_utils import pad_sequence, interpolate_matrix
        from utils.extract_features import extract_features
        
        print("✅ All imports successful")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_constants() -> bool:
    """Test constants module."""
    print("🔍 Testing constants...")
    
    try:
        from constants.constants import CLASSES, MODEL_PATH, FACEMESH_LIPS
        
        # Check classes
        assert isinstance(CLASSES, list), "CLASSES should be a list"
        assert len(CLASSES) > 0, "CLASSES should not be empty"
        
        # Check paths
        assert isinstance(MODEL_PATH, str), "MODEL_PATH should be a string"
        
        # Check facemesh landmarks
        assert isinstance(FACEMESH_LIPS, frozenset), "FACEMESH_LIPS should be a frozenset"
        assert len(FACEMESH_LIPS) > 0, "FACEMESH_LIPS should not be empty"
        
        print(f"✅ Constants valid - {len(CLASSES)} classes, {len(FACEMESH_LIPS)} landmarks")
        return True
        
    except Exception as e:
        print(f"❌ Constants test failed: {e}")
        return False


def test_cnn_model() -> bool:
    """Test CNN model creation and forward pass."""
    print("🔍 Testing CNN model...")
    
    try:
        from infra.cnn import CNN
        from constants.constants import CLASSES
        
        # Create model
        model = CNN(num_classes=len(CLASSES), num_rows=60, num_cols=4)
        
        # Test forward pass
        batch_size = 2
        test_input = torch.randn(batch_size, 1, 60, 4)
        
        with torch.no_grad():
            output = model(test_input)
        
        # Check output shape
        expected_shape = (batch_size, len(CLASSES))
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Check output is finite
        assert torch.isfinite(output).all(), "Output contains non-finite values"
        
        print(f"✅ CNN model working - input: {test_input.shape}, output: {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ CNN test failed: {e}")
        return False


def test_numpy_utils() -> bool:
    """Test numpy utility functions."""
    print("🔍 Testing numpy utilities...")
    
    try:
        from utils.numpy_utils import pad_sequence, interpolate_matrix, normalize_sequence
        
        # Test pad_sequence
        test_seq = np.random.randn(30, 4)
        padded = pad_sequence(test_seq, 60)
        
        assert padded is not None, "pad_sequence returned None"
        assert padded.shape == (60, 4), f"Expected (60, 4), got {padded.shape}"
        assert np.array_equal(padded[:30], test_seq), "Original data not preserved"
        
        # Test interpolate_matrix
        test_matrix = np.random.randn(20, 4)
        interpolated = interpolate_matrix(test_matrix, 60)
        
        assert interpolated.shape == (60, 4), f"Expected (60, 4), got {interpolated.shape}"
        
        # Test normalize_sequence
        normalized = normalize_sequence(test_matrix, method='minmax')
        
        assert normalized.shape == test_matrix.shape, "Shape changed during normalization"
        
        print("✅ Numpy utilities working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Numpy utils test failed: {e}")
        return False


def test_aws_components() -> bool:
    """Test AWS component initialization (without actual AWS calls)."""
    print("🔍 Testing AWS components...")
    
    try:
        # Test DynamoDB (without actual connection)
        from aws_infrastructure.hermes_dynamodb import HermesDynamoDB
        hermes = HermesDynamoDB()
        assert hasattr(hermes, 'users_table_name'), "Hermes missing table configuration"
        
        # Test S3 Storage (without actual connection)
        from aws_infrastructure.s3_storage import S3StorageManager
        s3_manager = S3StorageManager("test-bucket")
        assert hasattr(s3_manager, 'prefixes'), "S3Manager missing prefix configuration"
        
        # Test SQS Queue (without actual connection)
        from aws_infrastructure.sqs_queue import SQSMessageQueue
        sqs_queue = SQSMessageQueue("test-queue-url")
        assert hasattr(sqs_queue, 'queue_url'), "SQSQueue missing URL configuration"
        
        print("✅ AWS components initialized correctly")
        return True
        
    except Exception as e:
        print(f"❌ AWS components test failed: {e}")
        return False


def test_feature_extraction() -> bool:
    """Test feature extraction functions."""
    print("🔍 Testing feature extraction...")
    
    try:
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test extract_features function signature (can't test without MediaPipe results)
        from utils.extract_features import extract_features
        
        # Check function exists and is callable
        assert callable(extract_features), "extract_features is not callable"
        
        print("✅ Feature extraction functions available")
        return True
        
    except Exception as e:
        print(f"❌ Feature extraction test failed: {e}")
        return False


def run_all_tests() -> Tuple[int, int]:
    """Run all tests and return (passed, total) counts."""
    tests = [
        ("Imports", test_imports),
        ("Constants", test_constants),
        ("CNN Model", test_cnn_model),
        ("Numpy Utils", test_numpy_utils),
        ("AWS Components", test_aws_components),
        ("Feature Extraction", test_feature_extraction),
    ]
    
    passed = 0
    total = len(tests)
    
    print("🧪 Running Lip Sync Core validation tests...\n")
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            print()  # Add spacing between tests
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}\n")
    
    return passed, total


def main():
    """Main test execution."""
    print("=" * 60)
    print("🚀 Lip Sync Core - Basic Functionality Tests")
    print("=" * 60)
    
    passed, total = run_all_tests()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        sys.exit(0)
    else:
        print(f"⚠️  {total - passed} tests failed. Please check the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()