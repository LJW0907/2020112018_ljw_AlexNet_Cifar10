import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.pca import PCA
from models.alexnet import AlexNetWithSkip
from utils.data_loader import CIFAR10DataLoader
import yaml
import os

class FeatureAnalyzer:
    """모델의 특징 공간을 분석하는 클래스입니다."""
    
    def __init__(self, model, data_loader, config, result_dir):
        """
        Args:
            model: 분석할 모델
            data_loader: 데이터 로더
            config: 설정 정보
            result_dir: 결과를 저장할 디렉토리
        """
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.result_dir = result_dir
        os.makedirs(result_dir, exist_ok=True)
        
    def extract_features(self):
        """데이터셋에서 특징과 레이블을 추출합니다."""
        features = []
        labels = []
        
        # 모델을 평가 모드로 설정
        self.model.eval()
        
        with torch.no_grad():
            for inputs, targets in self.data_loader:
                # 특징 추출 (classifier 직전 층의 출력)
                x = inputs
                for layer in self.model.features:
                    x = layer(x)
                x = torch.flatten(x, 1)
                
                # CPU로 이동하고 numpy 배열로 변환
                features.append(x.numpy())
                labels.append(targets.numpy())
        
        # 리스트를 하나의 배열로 결합
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        
        return features, labels
    
    def analyze_with_pca(self, features, labels):
        """PCA를 사용하여 특징을 분석합니다."""
        # PCA 수행
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
        
        # 결과 시각화
        plt.figure(figsize=(10, 8))
        
        # 클래스별로 다른 색상으로 산점도 그리기
        classes = self.data_loader.dataset.classes
        colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))
        
        for i, cls in enumerate(classes):
            mask = labels == i
            plt.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                c=[colors[i]],
                label=cls,
                alpha=0.6
            )
        
        # 그래프 꾸미기
        plt.title('Feature Space Visualization (PCA)')
        plt.xlabel(f'First PC (var ratio: {pca.explained_variance_ratio[0]:.3f})')
        plt.ylabel(f'Second PC (var ratio: {pca.explained_variance_ratio[1]:.3f})')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # 저장
        plt.savefig(os.path.join(self.result_dir, 'pca_visualization.png'))
        plt.close()
        
        return pca

def main():
    # 설정 로드
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 데이터 로더 생성
    data_loader = CIFAR10DataLoader(config_path='configs/config.yaml')
    _, test_loader = data_loader.get_dataloaders()
    
    # 두 모델에 대해 분석 수행
    models = {
        'base_alexnet': 'results/logs/base_alexnet/best_model.pth',
        'alexnet_with_skip': 'results/logs/alexnet_with_skip/best_model.pth'
    }
    
    for model_name, model_path in models.items():
        print(f"\nAnalyzing features of {model_name}...")
        
        # 모델 로드
        model = AlexNetWithSkip(config_path='configs/config.yaml')
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 특징 분석
        analyzer = FeatureAnalyzer(
            model=model,
            data_loader=test_loader,
            config=config,
            result_dir=f'results/analysis/{model_name}'
        )
        
        # 특징 추출 및 PCA 분석
        features, labels = analyzer.extract_features()
        pca = analyzer.analyze_with_pca(features, labels)
        
        # 분산 설명 비율 저장
        var_ratio, cum_ratio = pca.get_explained_variance()
        with open(os.path.join(analyzer.result_dir, 'variance_explained.txt'), 'w') as f:
            f.write(f"Individual explained variance ratios: {var_ratio}\n")
            f.write(f"Cumulative explained variance ratio: {cum_ratio}\n")
        
        print(f"Analysis completed for {model_name}")

if __name__ == "__main__":
    main()