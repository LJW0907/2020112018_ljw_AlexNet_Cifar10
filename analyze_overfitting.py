import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
from pathlib import Path
import seaborn as sns

class OverfittingAnalyzer:
    """학습 과정에서의 overfitting을 분석하는 클래스입니다.
    
    이 클래스는 학습 과정에서 저장된 손실값과 정확도를 분석하여
    모델의 일반화 능력과 overfitting 여부를 평가합니다.
    """
    
    def __init__(self, model_dir, result_dir):
        self.model_dir = Path(model_dir)
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
    def load_training_history(self):
        """체크포인트에서 학습 기록을 로드합니다."""
        checkpoint = torch.load(self.model_dir / 'best_model.pth')
        history = {
            'train_losses': checkpoint['train_losses'],
            'test_losses': checkpoint['test_losses'],
            'train_accs': checkpoint['train_accs'],
            'test_accs': checkpoint['test_accs']
        }
        return history
        
    def analyze_loss_curves(self, history):
        """손실 곡선을 분석하고 시각화합니다."""
        plt.figure(figsize=(12, 5))
        
        # 손실 곡선 그래프
        plt.subplot(1, 2, 1)
        plt.plot(history['train_losses'], label='Training Loss', marker='o')
        plt.plot(history['test_losses'], label='Testing Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curves Analysis')
        plt.legend()
        plt.grid(True)
        
        # 정확도 곡선 그래프
        plt.subplot(1, 2, 2)
        plt.plot(history['train_accs'], label='Training Accuracy', marker='o')
        plt.plot(history['test_accs'], label='Testing Accuracy', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy Curves Analysis')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.result_dir / 'learning_curves_analysis.png')
        plt.close()
        
    def compute_overfitting_metrics(self, history):
        """Overfitting 관련 지표들을 계산합니다."""
        metrics = {}
        
        # 훈련/테스트 간의 최종 성능 차이
        final_epoch = len(history['train_losses']) - 1
        metrics['final_train_loss'] = history['train_losses'][final_epoch]
        metrics['final_test_loss'] = history['test_losses'][final_epoch]
        metrics['final_train_acc'] = history['train_accs'][final_epoch]
        metrics['final_test_acc'] = history['test_accs'][final_epoch]
        
        # 성능 격차 (train-test gap)
        metrics['loss_gap'] = metrics['final_train_loss'] - metrics['final_test_loss']
        metrics['acc_gap'] = metrics['final_train_acc'] - metrics['final_test_acc']
        
        # 조기 종료가 필요했을지 판단 (validation loss가 증가하기 시작한 시점)
        test_losses = history['test_losses']
        min_loss_epoch = np.argmin(test_losses)
        metrics['best_epoch'] = min_loss_epoch
        metrics['early_stopping_needed'] = min_loss_epoch < final_epoch
        
        return metrics
        
    def analyze_and_save_results(self):
        """전체 분석을 수행하고 결과를 저장합니다."""
        print(f"\nAnalyzing model in {self.model_dir}...")
        
        # 학습 기록 로드
        history = self.load_training_history()
        
        # 학습 곡선 분석
        self.analyze_loss_curves(history)
        
        # Overfitting 지표 계산
        metrics = self.compute_overfitting_metrics(history)
        
        # 결과 저장
        results_path = self.result_dir / 'overfitting_analysis.txt'
        with open(results_path, 'w') as f:
            f.write("Overfitting Analysis Results\n")
            f.write("===========================\n\n")
            
            f.write("Final Performance Metrics:\n")
            f.write(f"Training Loss: {metrics['final_train_loss']:.4f}\n")
            f.write(f"Testing Loss: {metrics['final_test_loss']:.4f}\n")
            f.write(f"Training Accuracy: {metrics['final_train_acc']:.2f}%\n")
            f.write(f"Testing Accuracy: {metrics['final_test_acc']:.2f}%\n\n")
            
            f.write("Generalization Gap:\n")
            f.write(f"Loss Gap (Train-Test): {metrics['loss_gap']:.4f}\n")
            f.write(f"Accuracy Gap (Train-Test): {metrics['acc_gap']:.2f}%\n\n")
            
            f.write("Early Stopping Analysis:\n")
            f.write(f"Best Epoch: {metrics['best_epoch']}\n")
            f.write(f"Early Stopping Needed: {metrics['early_stopping_needed']}\n")
        
        return metrics

def main():
    # 설정 파일 로드
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 각 모델에 대해 분석 수행
    models = {
        'base_alexnet': 'results/logs/base_alexnet',
        'alexnet_with_skip': 'results/logs/alexnet_with_skip'
    }
    
    all_results = {}
    for model_name, model_dir in models.items():
        analyzer = OverfittingAnalyzer(
            model_dir=model_dir,
            result_dir=f'results/analysis/{model_name}/overfitting'
        )
        all_results[model_name] = analyzer.analyze_and_save_results()
    
    # 모델 간 비교 결과 저장
    comparison_path = 'results/analysis/model_comparison.txt'
    with open(comparison_path, 'w') as f:
        f.write("Model Performance Comparison\n")
        f.write("=========================\n\n")
        
        for model_name, metrics in all_results.items():
            f.write(f"\n{model_name}:\n")
            f.write(f"Final Test Accuracy: {metrics['final_test_acc']:.2f}%\n")
            f.write(f"Accuracy Gap: {metrics['acc_gap']:.2f}%\n")
            f.write(f"Loss Gap: {metrics['loss_gap']:.4f}\n")
            f.write(f"Early Stopping Needed: {metrics['early_stopping_needed']}\n")
    
    print("\nOverfitting analysis completed! Check the results directory for detailed analysis.")

if __name__ == "__main__":
    main()