# CIFAR10 Classification with AlexNet

이 프로젝트는 CIFAR10 데이터셋에 대한 AlexNet 기반의 분류 실험을 포함합니다. Skip Connection의 효과와 Linear Regressor의 성능을 분석합니다.

## 프로젝트 구조

```
cifar10_alexnet_project/
├── data/                    # 데이터셋 저장 폴더
├── models/                  # 모델 관련 파일들
├── utils/                   # 유틸리티 함수들
├── configs/                 # 설정 파일들
├── results/                 # 결과 저장 폴더
├── tests/                   # 테스트 코드
└── notebooks/              # 분석용 노트북
```

## 설치 방법

1. 가상환경 생성 및 활성화:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

2. 필요한 패키지 설치:

```bash
pip install -r requirements.txt
```

## 실행 방법

1. 기본 학습:

```bash
python train.py
```

2. Linear Regressor 실험:

```bash
python train_linear.py
```

## 주요 기능

1. AlexNet 구현

   - CIFAR10에 맞게 수정된 구조
   - Skip Connection 지원
   - Feature Extractor와 Classifier 분리

2. 실험 기능
   - Skip Connection 성능 비교
   - Linear Regressor 성능 평가
   - PCA를 통한 특징 분석

## 결과 분석

실험 결과는 results 디렉토리에 저장됩니다:

- 학습 곡선 (learning_curves.png)
- 모델 가중치 (.pth 파일)
- 학습 통계 (training_stats.txt)

## 참고사항

- CIFAR10 데이터셋은 자동으로 다운로드됩니다
- GPU가 없어도 실행 가능하도록 설계되었습니다
- 모든 실험 설정은 configs/config.yaml에서 관리됩니다
