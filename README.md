# CIFAR10 데이터셋을 활용한 AlexNet 구조 분석 및 성능 향상 연구

## 프로젝트 개요

본 프로젝트는 CIFAR10 데이터셋을 사용하여 AlexNet의 구조적 변화가 모델의 성능과 특징 학습에 미치는 영향을 체계적으로 분석합니다. 특히 Skip Connection의 도입과 Linear Regressor의 활용이 모델의 학습 동태와 성능에 어떤 변화를 가져오는지 심층적으로 연구했습니다.

## 주요 연구 내용

### 1. 모델 구현 및 기본 성능 평가

기본 AlexNet과 Skip Connection이 추가된 변형 모델을 구현했습니다. Skip Connection 모델은 기본 모델(48.92%)보다 높은 55.27%의 테스트 정확도를 달성했으며, 특히 학습 과정에서 더 안정적인 성능 향상을 보였습니다.

### 2. Skip Connection의 영향 분석

Skip Connection의 도입은 모델의 특징 학습 방식을 근본적으로 변화시켰습니다. 특히:

- 특정 클래스(bird: 72.2%, ship: 91.8%)에서 현저히 높은 성능을 보임
- 더 복잡한 특징 공간 형성 (PCA 분산 설명률 33.9% vs 43.9%)
- 학습 과정에서 더 낮은 일반화 격차 (Loss Gap: 0.1031 vs 0.1517)

### 3. Linear Regressor 실험

기존 분류기를 Linear Regressor로 교체하여 특징 추출기의 성능을 평가했습니다. 이 실험은 각 모델이 학습한 특징의 선형 분리 가능성을 보여주었으며, Skip Connection 모델이 더 복잡하지만 풍부한 특징 표현을 학습했음을 시사합니다.

### 4. 특징 공간 분석

자체 구현한 PCA를 통해 특징 공간을 2차원으로 시각화했습니다. 이 분석은 각 모델이 형성하는 특징 공간의 구조적 차이를 명확히 보여주었으며, 특히 Skip Connection이 특징들의 비선형적 관계를 더 잘 포착함을 확인했습니다.

## 실험 환경

- 데이터셋: CIFAR10 (50,000 학습 이미지, 10,000 테스트 이미지)
- 학습 파라미터:
  - Batch Size: 256
  - Learning Rate: 0.01
  - Optimizer: SGD (momentum: 0.9)
  - Epochs: 5
- 평가 지표: Top-1/Top-3 정확도, Confusion Matrix, PCA 분석

## 주요 발견

1. Skip Connection의 효과:

   - 전반적인 성능 향상
   - 특정 클래스에 대한 특화된 성능
   - 더 안정적인 학습 과정

2. 특징 학습의 차이:

   - 기본 모델: 더 선형적이고 균일한 특징 분포
   - Skip Connection 모델: 더 복잡하고 특화된 특징 학습

3. 일반화 성능:
   - 두 모델 모두 과적합 없이 안정적인 학습
   - Skip Connection 모델이 더 낮은 일반화 격차 보임

## 디렉토리 구조

```
cifar10_alexnet_project/
├── data/                    # CIFAR10 데이터셋
├── models/                  # 모델 구현 파일
├── utils/                   # 유틸리티 함수
├── results/                 # 실험 결과
│   ├── analysis/           # 분석 결과
│   └── logs/               # 학습 로그
└── configs/                # 설정 파일
```

## 실행 방법

```bash
# 환경 설정
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 학습 실행
python train.py

# 성능 분석
python analyze_performance.py
python analyze_features.py
```

## 결론 및 시사점

본 연구는 Skip Connection이 딥러닝 모델의 특징 학습에 미치는 영향을 실증적으로 분석했습니다. 특히 특정 클래스에 대한 성능 향상과 특징 공간의 구조적 변화를 통해, 네트워크 구조의 수정이 모델의 학습 동태에 미치는 영향을 보다 깊이 이해할 수 있었습니다. 이러한 발견은 향후 태스크 특화적인 네트워크 구조 설계에 중요한 통찰을 제공할 것으로 기대됩니다.
