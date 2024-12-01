import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import yaml
import os

class CIFAR10DataLoader:
    def __init__(self, config_path='configs/config.yaml'):
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 데이터 관련 설정값들
        self.batch_size = self.config['training']['batch_size']
        self.num_workers = self.config['data']['num_workers']
        self.use_augmentation = self.config['augmentation']['use_augmentation']
        
        # 보고서용 데이터 특성 저장 변수
        self.train_data_stats = {}
        self.test_data_stats = {}
        
        # Transform 설정
        self.train_transform = self._get_train_transform()
        self.test_transform = self._get_test_transform()
        
    def _get_train_transform(self):
        """학습용 데이터 전처리 정의 (보고서에 사용될 augmentation 정보 포함)"""
        transforms_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ]
        
        if self.use_augmentation:
            aug_transforms = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
            transforms_list = aug_transforms + transforms_list
            
            # 보고서용 augmentation 정보 저장
            self.train_data_stats['augmentations'] = [
                'RandomCrop(32, padding=4)',
                'RandomHorizontalFlip()',
                'Normalization'
            ]
        
        return transforms.Compose(transforms_list)
    
    def _get_test_transform(self):
        """테스트용 데이터 전처리 정의"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])
    
    def get_dataloaders(self):
        """데이터 로더 생성 및 데이터 특성 저장"""
        # 학습 데이터
        train_dataset = datasets.CIFAR10(
            root='data/cifar10',
            train=True,
            download=True,
            transform=self.train_transform
        )
        
        # 테스트 데이터
        test_dataset = datasets.CIFAR10(
            root='data/cifar10',
            train=False,
            download=True,
            transform=self.test_transform
        )
        
        # 보고서용 데이터 특성 저장
        self.train_data_stats.update({
            'num_samples': len(train_dataset),
            'num_classes': 10,
            'image_size': (32, 32, 3),
            'class_names': train_dataset.classes,
            'class_distribution': self._get_class_distribution(train_dataset)
        })
        
        self.test_data_stats.update({
            'num_samples': len(test_dataset),
            'class_distribution': self._get_class_distribution(test_dataset)
        })
        
        # DataLoader 생성
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, test_loader
    
    def _get_class_distribution(self, dataset):
        """클래스별 샘플 수 계산 (보고서용)"""
        counts = [0] * 10
        for _, label in dataset:
            counts[label] += 1
        return counts
    
    def get_data_stats(self):
        """보고서 작성에 필요한 데이터 특성 반환"""
        return {
            'train': self.train_data_stats,
            'test': self.test_data_stats
        }