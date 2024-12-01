import sys
import torch
import torchvision
import numpy as np
import yaml
import os

def check_cuda():
    """CUDA 가용성과 버전을 확인합니다."""
    print("\n=== CUDA 환경 검사 ===")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 버전: {torch.version.cuda}")
        print(f"현재 CUDA 디바이스: {torch.cuda.current_device()}")
        print(f"CUDA 디바이스 이름: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 디바이스 개수: {torch.cuda.device_count()}")

def check_versions():
    """주요 패키지들의 버전을 확인합니다."""
    print("\n=== 패키지 버전 검사 ===")
    print(f"Python 버전: {sys.version}")
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"Torchvision 버전: {torchvision.__version__}")
    print(f"Numpy 버전: {np.__version__}")

def check_directories():
    """필요한 디렉토리들이 존재하는지 확인합니다."""
    print("\n=== 디렉토리 구조 검사 ===")
    required_dirs = ['data', 'models', 'utils', 'configs', 'results', 'tests', 'notebooks']
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"{dir_name} 디렉토리 존재: ✓")
        else:
            print(f"{dir_name} 디렉토리 없음: ✗")
            try:
                os.makedirs(dir_name)
                print(f"- {dir_name} 디렉토리 생성됨")
            except Exception as e:
                print(f"- {dir_name} 디렉토리 생성 실패: {e}")

def check_config():
    """config.yaml 파일이 올바른지 확인합니다."""
    print("\n=== 설정 파일 검사 ===")
    config_path = 'configs/config.yaml'
    
    if not os.path.exists(config_path):
        print("config.yaml 파일이 없습니다!")
        return False
        
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            required_keys = ['training', 'data', 'model', 'augmentation', 'logging']
            
            for key in required_keys:
                if key in config:
                    print(f"{key} 설정 존재: ✓")
                else:
                    print(f"{key} 설정 없음: ✗")
    except Exception as e:
        print(f"설정 파일 로드 실패: {e}")
        return False
    
    return True

def test_torch_operations():
    """간단한 PyTorch 연산을 테스트합니다."""
    print("\n=== PyTorch 기본 연산 테스트 ===")
    try:
        # CPU 테스트
        x = torch.randn(2, 3)
        y = torch.randn(3, 2)
        z = torch.mm(x, y)
        print("CPU 행렬 곱셈 테스트: ✓")
        
        # CUDA 테스트
        if torch.cuda.is_available():
            x_cuda = x.cuda()
            y_cuda = y.cuda()
            z_cuda = torch.mm(x_cuda, y_cuda)
            print("CUDA 행렬 곱셈 테스트: ✓")
            
        # 간단한 신경망 연산 테스트
        conv = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
        test_input = torch.randn(1, 3, 32, 32)
        output = conv(test_input)
        print("신경망 연산 테스트: ✓")
        
    except Exception as e:
        print(f"PyTorch 연산 테스트 실패: {e}")

def main():
    """모든 검사를 실행합니다."""
    print("=== 환경 검사 시작 ===")
    check_versions()
    check_cuda()
    check_directories()
    check_config()
    test_torch_operations()
    print("\n=== 환경 검사 완료 ===")

if __name__ == "__main__":
    main()