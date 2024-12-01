import pytest
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.alexnet import AlexNetWithSkip

def test_model_initialization():
    model = AlexNetWithSkip()
    assert model is not None
    
def test_forward_pass():
    model = AlexNetWithSkip()
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)
    output = model(x)
    
    # 출력 크기 확인
    assert output.shape == (batch_size, 10)  # CIFAR10은 10개 클래스
    
def test_skip_connection():
    model = AlexNetWithSkip()
    
    # Skip connection 추가
    model.add_skip_connection('conv2', 'conv4')
    
    # Forward pass
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    
    # 중간 특징맵 확인
    feature_maps = model.get_feature_maps()
    assert 'conv2' in feature_maps
    assert 'conv4' in feature_maps
    
def test_linear_classifier_replacement():
    model = AlexNetWithSkip()
    
    # Classifier 교체
    model.replace_classifier_with_linear()
    
    # Forward pass 확인
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    assert output.shape == (1, 10)
    
    # 모델 정보 확인
    stats = model.get_model_stats()
    assert stats['classifier'] == 'Linear Regressor'
    
def test_model_stats():
    model = AlexNetWithSkip()
    stats = model.get_model_stats()
    
    # 기본 정보 확인
    assert stats['name'] == 'AlexNet with Skip Connections'
    assert stats['input_size'] == (3, 32, 32)
    assert stats['num_classes'] == 10
    
    # Skip connection 추가 후 정보 확인
    model.add_skip_connection('conv2', 'conv4')
    stats = model.get_model_stats()
    assert len(stats['skip_connections']) == 1