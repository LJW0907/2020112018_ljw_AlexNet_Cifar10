import yaml
from utils.data_loader import CIFAR10DataLoader
from models.alexnet import AlexNetWithSkip
from utils.trainer import AlexNetTrainer
import torch

def main():
    # 설정 파일을 UTF-8로 명시적으로 로드
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 데이터 로더 생성
    data_loader = CIFAR10DataLoader(config_path='configs/config.yaml')
    train_loader, test_loader = data_loader.get_dataloaders()
    
    # 데이터 통계 출력
    data_stats = data_loader.get_data_stats()
    print("\nData Statistics:")
    print(f"Training samples: {data_stats['train']['num_samples']}")
    print(f"Test samples: {data_stats['test']['num_samples']}")
    print(f"Number of classes: {data_stats['train']['num_classes']}")
    
    # 기본 AlexNet 모델 학습
    print("\nTraining base AlexNet...")
    model = AlexNetWithSkip(config_path='configs/config.yaml')
    trainer = AlexNetTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        experiment_name='base_alexnet'
    )
    trainer.train(epochs=config['training']['epochs'])
    
    # Skip Connection이 있는 AlexNet 학습
    print("\nTraining AlexNet with Skip Connections...")
    model_skip = AlexNetWithSkip(config_path='configs/config.yaml')
    model_skip.add_skip_connection('conv2', 'conv4')
    trainer_skip = AlexNetTrainer(
        model=model_skip,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        experiment_name='alexnet_with_skip'
    )
    trainer_skip.train(epochs=config['training']['epochs'])
    
    print("\nTraining completed! Check the results directory for detailed statistics and visualizations.")

if __name__ == "__main__":
    main()