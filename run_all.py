"""
CIFAR10 AlexNet Experiments Runner
이 스크립트는 모든 실험을 순차적으로 실행합니다.

실행 방법:
    python run_all.py

주의사항:
    - 실행 전 requirements.txt의 모든 패키지가 설치되어 있어야 합니다.
    - 충분한 저장 공간이 필요합니다.
"""

import os
import subprocess
import time

def run_command(command, description):
    print(f"\n=== {description} ===")
    print(f"실행 명령어: {command}")
    start_time = time.time()
    subprocess.run(command, shell=True)
    duration = time.time() - start_time
    print(f"소요 시간: {duration:.2f}초")

def main():
    # 필요한 디렉토리 생성
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/analysis", exist_ok=True)

    # 1. 기본 모델 훈련
    run_command("python train.py", "기본 AlexNet과 Skip Connection 모델 훈련")

    # 2. Linear Regressor 실험
    run_command("python train_linear.py", "Linear Regressor 실험")

    # 3. 성능 분석
    run_command("python analyze_performance.py", "성능 분석 (Confusion Matrix, Top-k)")
    
    # 4. 특징 공간 분석
    run_command("python analyze_features.py", "PCA 분석")

    # 5. Overfitting 분석
    run_command("python analyze_overfitting.py", "과적합 분석")

if __name__ == "__main__":
    main()