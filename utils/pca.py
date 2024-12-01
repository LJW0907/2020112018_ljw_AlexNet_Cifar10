import numpy as np

class PCA:
    """PCA(Principal Component Analysis) 구현
    
    이 클래스는 고차원 데이터를 저차원으로 축소하기 위한 PCA를 구현합니다.
    sklearn을 사용하지 않고 numpy만으로 구현하여, PCA의 작동 원리를 명확히 보여줍니다.
    """
    
    def __init__(self, n_components=2):
        """PCA 초기화
        
        Args:
            n_components (int): 축소할 차원 수. 기본값은 2(시각화를 위해)
        """
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ratio = None
        
    def fit(self, X):
        """데이터에 맞추어 PCA 변환을 학습합니다.
        
        Args:
            X (np.ndarray): 형태가 (n_samples, n_features)인 입력 데이터
        """
        # 데이터 중심화 (평균을 0으로)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # 공분산 행렬 계산
        # 데이터가 매우 크면 SVD를 사용하는 것이 좋지만, 
        # 여기서는 이해를 위해 공분산 행렬을 직접 계산합니다
        cov_matrix = np.cov(X_centered.T)
        
        # 고유값과 고유벡터 계산
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 고유값이 큰 순서대로 정렬
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 주성분 저장 (최상위 n_components개)
        self.components = eigenvectors[:, :self.n_components]
        
        # 설명된 분산 비율 계산
        self.explained_variance_ratio = (
            eigenvalues[:self.n_components] / np.sum(eigenvalues)
        )
        
        return self
        
    def transform(self, X):
        """학습된 PCA를 사용하여 데이터를 변환합니다.
        
        Args:
            X (np.ndarray): 변환할 입력 데이터
            
        Returns:
            np.ndarray: 변환된 데이터
        """
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
    
    def fit_transform(self, X):
        """데이터에 맞추고 바로 변환을 수행합니다."""
        self.fit(X)
        return self.transform(X)
    
    def get_feature_importance(self):
        """각 원본 특징의 중요도를 계산합니다.
        
        Returns:
            np.ndarray: 각 특징의 중요도 점수
        """
        # 각 특징이 주성분에 기여하는 정도를 계산
        importance = np.abs(self.components).sum(axis=1)
        # 중요도 정규화
        return importance / np.sum(importance)
    
    def get_explained_variance(self):
        """설명된 분산 비율과 누적 비율을 반환합니다.
        
        Returns:
            tuple: (분산 비율, 누적 분산 비율)
        """
        cumulative = np.cumsum(self.explained_variance_ratio)
        return self.explained_variance_ratio, cumulative