import yaml
import torch
from utils.data_loader import CIFAR10DataLoader
from models.alexnet import AlexNetWithSkip
from utils.trainer import AlexNetTrainer
import os

def load_pretrained_model(model, model_path):
    """사전 학습된 모델을 로드하고 feature extractor를 고정합니다."""
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Feature extractor를 고정 (gradient 계산 비활성화)
    for param in model.features.parameters():
        param.requires_grad = False
    
    return model

def main():
    # 설정 파일 로드
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 데이터 로더 생성
    data_loader = CIFAR10DataLoader(config_path='configs/config.yaml')
    train_loader, test_loader = data_loader.get_dataloaders()
    
    # 두 가지 모델에 대해 실험 진행
    models = {
        'base_alexnet': 'results/logs/base_alexnet/best_model.pth',
        'alexnet_with_skip': 'results/logs/alexnet_with_skip/best_model.pth'
    }
    
    for model_name, model_path in models.items():
        print(f"\nTraining Linear Regressor on {model_name}'s features...")
        
        # 모델 생성 및 사전 학습된 가중치 로드
        model = AlexNetWithSkip(config_path='configs/config.yaml')
        model = load_pretrained_model(model, model_path)
        
        # Classifier를 Linear Regressor로 교체
        model.replace_classifier_with_linear()
        
        # Linear Regressor 학습
        trainer = AlexNetTrainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            config=config,
            experiment_name=f'{model_name}_linear'
        )
        
        # Learning rate를 조정하여 Linear Regressor 학습
        trainer.optimizer = torch.optim.SGD(
            model.classifier.parameters(),  # classifier의 파라미터만 학습
            lr=0.001,  # 더 작은 learning rate 사용
            momentum=0.9
        )
        
        trainer.train(epochs=config['training']['epochs'])
        
    print("\nLinear Regressor training completed! Check the results directory for detailed statistics.")

if __name__ == "__main__":
    main()