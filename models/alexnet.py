import torch
import torch.nn as nn
from typing import List, Optional, Dict
import yaml

class AlexNetWithSkip(nn.Module):
    def __init__(self, config_path: str = 'configs/config.yaml'):
        super(AlexNetWithSkip, self).__init__()
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        self.model_stats = {
            'name': 'AlexNet with Skip Connections',
            'input_size': (3, 32, 32),
            'num_classes': self.config['model']['num_classes'],
            'skip_connections': []
        }
        
        # Conv layers with their channel sizes
        self.features = nn.Sequential(
            # conv1: 3 -> 64
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # conv2: 64 -> 64 (수정됨: 192->64)
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # conv3: 64 -> 64 (수정됨: 384->64)
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # conv4: 64 -> 64 (수정됨: 256->64)
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # conv5: 64 -> 64 (수정됨: 256->64)
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 4 * 4, 1024),  # 수정됨: 더 작은 크기의 은닉층
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.config['model']['num_classes'])
        )
        
        self.intermediate_features = {}
        self.skip_connections = []
        
    def add_skip_connection(self, from_layer: str, to_layer: str):
        """Skip connection을 추가합니다."""
        self.skip_connections.append((from_layer, to_layer))
        self.model_stats['skip_connections'].append({
            'from': from_layer,
            'to': to_layer
        })
        
    def get_feature_maps(self) -> Dict[str, torch.Tensor]:
        return self.intermediate_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.intermediate_features = {}
        
        for name, layer in self.features.named_children():
            x = layer(x)
            self.intermediate_features[f'conv{name}'] = x.clone()
            
            for from_layer, to_layer in self.skip_connections:
                if f'conv{name}' == to_layer and from_layer in self.intermediate_features:
                    x = x + self.intermediate_features[from_layer]
        
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
    
    def replace_classifier_with_linear(self):
        """Classifier를 Linear Regressor로 교체합니다."""
        in_features = 64 * 4 * 4  # flattened feature size
        self.classifier = nn.Sequential(
            nn.Linear(in_features, self.config['model']['num_classes'])
        )
        self.model_stats['classifier'] = 'Linear Regressor'
        
    def get_model_stats(self) -> dict:
        return self.model_stats