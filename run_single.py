"""
개별 실험 실행 스크립트

사용 방법:
    python run_single.py --experiment [experiment_name]

실험 옵션:
    - train: 기본 모델 훈련
    - linear: Linear Regressor 실험
    - analysis: 모든 분석 수행
"""

import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description='CIFAR10 AlexNet 실험 실행기')
    parser.add_argument('--experiment', type=str, required=True,
                      choices=['train', 'linear', 'analysis'],
                      help='실행할 실험 선택')

    args = parser.parse_args()

    if args.experiment == 'train':
        subprocess.run("python train.py", shell=True)
    elif args.experiment == 'linear':
        subprocess.run("python train_linear.py", shell=True)
    elif args.experiment == 'analysis':
        subprocess.run("python analyze_performance.py", shell=True)
        subprocess.run("python analyze_features.py", shell=True)
        subprocess.run("python analyze_overfitting.py", shell=True)

if __name__ == "__main__":
    main()