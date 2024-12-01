import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models.alexnet import AlexNetWithSkip
from utils.data_loader import CIFAR10DataLoader
import yaml
import os
from tqdm import tqdm

class PerformanceAnalyzer:
    """모델의 세부적인 성능을 분석하는 클래스입니다.
    
    이 클래스는 Confusion Matrix와 Top-k 정확도를 계산하고 시각화합니다.
    각 클래스별 성능과 모델의 전반적인 예측 패턴을 이해하는 데 도움을 줍니다.
    """
    
    def __init__(self, model, data_loader, config, result_dir):
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.result_dir = result_dir
        os.makedirs(result_dir, exist_ok=True)
        
        # CIFAR-10 클래스 이름
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        
    def compute_predictions(self):
        """테스트 데이터에 대한 예측을 수행합니다."""
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        self.model.eval()
        with torch.no_grad():
            for inputs, targets in tqdm(self.data_loader, desc="Computing predictions"):
                # 모델의 예측 계산
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)
                
                # 결과 저장
                all_predictions.extend(predictions.numpy())
                all_targets.extend(targets.numpy())
                all_probabilities.extend(probabilities.numpy())
        
        return (np.array(all_predictions), 
                np.array(all_targets),
                np.array(all_probabilities))
    
    def create_confusion_matrix(self, predictions, targets):
        """Confusion Matrix를 계산하고 시각화합니다."""
        # Confusion Matrix 계산
        conf_matrix = np.zeros((len(self.classes), len(self.classes)))
        for t, p in zip(targets, predictions):
            conf_matrix[t, p] += 1
            
        # 퍼센트로 정규화
        conf_matrix_percent = (conf_matrix / conf_matrix.sum(axis=1)[:, None] * 100)
        
        # 시각화
        plt.figure(figsize=(12, 10))
        sns.heatmap(conf_matrix_percent, annot=True, fmt='.1f', 
                    xticklabels=self.classes, yticklabels=self.classes)
        plt.title('Confusion Matrix (%)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, 'confusion_matrix.png'))
        plt.close()
        
        return conf_matrix_percent
    
    def compute_per_class_metrics(self, conf_matrix):
        """클래스별 성능 지표를 계산합니다."""
        metrics = {}
        for i, class_name in enumerate(self.classes):
            # True Positives, False Positives, False Negatives 계산
            tp = conf_matrix[i, i]
            fp = conf_matrix[:, i].sum() - tp
            fn = conf_matrix[i, :].sum() - tp
            
            # 정밀도(Precision)와 재현율(Recall) 계산
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'accuracy': conf_matrix[i, i]  # 클래스별 정확도
            }
        
        return metrics
    
    def compute_topk_accuracy(self, probabilities, targets, k=(1, 3)):
        """Top-k 정확도를 계산합니다."""
        results = {}
        
        for k_val in k:
            # 상위 k개의 예측 클래스 선택
            topk_indices = np.argsort(probabilities, axis=1)[:, -k_val:]
            
            # 각 샘플에 대해 정답이 상위 k개 안에 있는지 확인
            correct = 0
            for i, target in enumerate(targets):
                if target in topk_indices[i]:
                    correct += 1
            
            accuracy = (correct / len(targets)) * 100
            results[f'top{k_val}'] = accuracy
            
        return results
    
    def analyze(self):
        """전체 성능 분석을 수행합니다."""
        print("\nStarting performance analysis...")
        
        # 예측 수행
        predictions, targets, probabilities = self.compute_predictions()
        
        # Confusion Matrix 생성
        print("Creating confusion matrix...")
        conf_matrix = self.create_confusion_matrix(predictions, targets)
        
        # 클래스별 메트릭 계산
        print("Computing per-class metrics...")
        class_metrics = self.compute_per_class_metrics(conf_matrix)
        
        # Top-k 정확도 계산
        print("Computing Top-k accuracies...")
        topk_accuracies = self.compute_topk_accuracy(probabilities, targets)
        
        # 결과 저장
        results = {
            'class_metrics': class_metrics,
            'topk_accuracies': topk_accuracies
        }
        
        # 결과를 텍스트 파일로 저장
        with open(os.path.join(self.result_dir, 'performance_metrics.txt'), 'w') as f:
            f.write("Per-class Metrics:\n")
            for class_name, metrics in class_metrics.items():
                f.write(f"\n{class_name}:\n")
                for metric_name, value in metrics.items():
                    f.write(f"  {metric_name}: {value:.2f}%\n")
            
            f.write("\nTop-k Accuracies:\n")
            for k, acc in topk_accuracies.items():
                f.write(f"{k}: {acc:.2f}%\n")
        
        return results

def main():
    # 설정 로드
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 데이터 로더 생성
    data_loader = CIFAR10DataLoader(config_path='configs/config.yaml')
    _, test_loader = data_loader.get_dataloaders()
    
    # 두 모델에 대해 성능 분석 수행
    models = {
        'base_alexnet': 'results/logs/base_alexnet/best_model.pth',
        'alexnet_with_skip': 'results/logs/alexnet_with_skip/best_model.pth'
    }
    
    all_results = {}
    for model_name, model_path in models.items():
        print(f"\nAnalyzing performance of {model_name}...")
        
        # 모델 로드
        model = AlexNetWithSkip(config_path='configs/config.yaml')
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 성능 분석
        analyzer = PerformanceAnalyzer(
            model=model,
            data_loader=test_loader,
            config=config,
            result_dir=f'results/analysis/{model_name}'
        )
        
        results = analyzer.analyze()
        all_results[model_name] = results
        
    print("\nPerformance analysis completed!")
    print("Check the results directory for detailed metrics and visualizations.")

if __name__ == "__main__":
    main()