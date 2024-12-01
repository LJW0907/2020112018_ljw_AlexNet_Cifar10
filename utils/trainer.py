import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime

class AlexNetTrainer:
    """AlexNet 모델의 학습을 담당하는 클래스입니다."""
    
    def __init__(self, model, train_loader, test_loader, config, experiment_name="default"):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        
        # 학습 설정
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=config['training']['momentum'],
            weight_decay=config['training']['weight_decay']
        )
        
        # 결과 저장을 위한 설정
        self.experiment_name = experiment_name
        self.result_dir = os.path.join(config['logging']['log_dir'], experiment_name)
        os.makedirs(self.result_dir, exist_ok=True)
        
        # 학습 기록
        self.train_losses = []
        self.test_losses = []
        self.train_accs = []
        self.test_accs = []
        self.best_acc = 0.0
        
        # 보고서용 정보 저장
        self.training_stats = {
            'start_time': None,
            'end_time': None,
            'best_accuracy': 0.0,
            'best_epoch': 0,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'learning_rate': config['training']['learning_rate'],
            'batch_size': config['training']['batch_size'],
            'optimizer': 'SGD',
            'momentum': config['training']['momentum'],
            'weight_decay': config['training']['weight_decay']
        }

    def train_epoch(self, epoch):
        """한 에포크 동안의 학습을 수행합니다."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # tqdm으로 진행률 표시
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for inputs, targets in pbar:
            # 순전파
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # 역전파
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 통계 업데이트
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 진행률 표시 업데이트
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    def test_epoch(self):
        """테스트 데이터로 평가를 수행합니다."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        epoch_loss = running_loss / len(self.test_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    def train(self, epochs):
        """전체 학습을 수행합니다."""
        self.training_stats['start_time'] = datetime.now()
        
        for epoch in range(epochs):
            # 학습
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # 평가
            test_loss, test_acc = self.test_epoch()
            self.test_losses.append(test_loss)
            self.test_accs.append(test_acc)
            
            # 최고 성능 모델 저장
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                self.training_stats['best_accuracy'] = test_acc
                self.training_stats['best_epoch'] = epoch
                self.save_model(f'best_model.pth')
            
            # 주기적인 모델 저장
            if (epoch + 1) % self.config['logging']['save_freq'] == 0:
                self.save_model(f'model_epoch_{epoch+1}.pth')
            
            # 학습 곡선 업데이트
            self.plot_learning_curves()
        
        self.training_stats['end_time'] = datetime.now()
        self.save_training_stats()

    def save_model(self, filename):
        """모델을 저장합니다."""
        save_path = os.path.join(self.result_dir, filename)
        torch.save({
            'epoch': len(self.train_losses),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'train_accs': self.train_accs,
            'test_accs': self.test_accs,
            'best_acc': self.best_acc,
            'model_stats': self.model.get_model_stats()
        }, save_path)

    def plot_learning_curves(self):
        """학습 곡선을 그립니다."""
        plt.figure(figsize=(12, 5))
        
        # Loss 그래프
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Testing Loss')
        
        # Accuracy 그래프
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='Train Acc')
        plt.plot(self.test_accs, label='Test Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Training and Testing Accuracy')
        
        # 저장
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, 'learning_curves.png'))
        plt.close()

    def save_training_stats(self):
        """학습 통계를 저장합니다."""
        stats_path = os.path.join(self.result_dir, 'training_stats.txt')
        with open(stats_path, 'w') as f:
            for key, value in self.training_stats.items():
                f.write(f"{key}: {value}\n")
            
            f.write("\nTraining History:\n")
            for epoch in range(len(self.train_losses)):
                f.write(f"Epoch {epoch + 1}:\n")
                f.write(f"  Train Loss: {self.train_losses[epoch]:.4f}\n")
                f.write(f"  Train Acc:  {self.train_accs[epoch]:.2f}%\n")
                f.write(f"  Test Loss:  {self.test_losses[epoch]:.4f}\n")
                f.write(f"  Test Acc:   {self.test_accs[epoch]:.2f}%\n")