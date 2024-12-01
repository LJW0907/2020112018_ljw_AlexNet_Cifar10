import pytest
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import CIFAR10DataLoader

def test_data_loader_initialization():
    data_loader = CIFAR10DataLoader()
    assert data_loader is not None
    
def test_data_shapes_and_types():
    data_loader = CIFAR10DataLoader()
    train_loader, test_loader = data_loader.get_dataloaders()
    
    # 첫 번째 배치 가져오기
    images, labels = next(iter(train_loader))
    
    # 데이터 형태 검사
    assert isinstance(images, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert len(images.shape) == 4  # [batch_size, channels, height, width]
    assert images.shape[1:] == (3, 32, 32)
    
def test_data_statistics():
    data_loader = CIFAR10DataLoader()
    train_loader, test_loader = data_loader.get_dataloaders()
    stats = data_loader.get_data_stats()
    
    # 기본 통계 검사
    assert stats['train']['num_samples'] == 50000
    assert stats['test']['num_samples'] == 10000
    assert stats['train']['num_classes'] == 10
    assert len(stats['train']['class_names']) == 10
    
    # 클래스 분포 검사
    assert len(stats['train']['class_distribution']) == 10
    assert sum(stats['train']['class_distribution']) == 50000
    
def test_data_augmentation():
    data_loader = CIFAR10DataLoader()
    stats = data_loader.get_data_stats()
    
    # Augmentation 설정 검사
    if data_loader.use_augmentation:
        assert 'augmentations' in stats['train']
        assert len(stats['train']['augmentations']) > 0