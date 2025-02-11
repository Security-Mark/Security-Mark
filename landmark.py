import cv2
import numpy as np
import face_alignment
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as transforms

def apply_noise_and_transparency(image_path, output_path, noise_strength=0.05, transparency=80):
    """
    얼굴의 중요 부위(눈, 코, 입)에 노이즈를 추가하고 불투명도를 적용하는 함수
    
    Parameters:
    - image_path: 입력 이미지 경로
    - output_path: 출력 이미지 경로
    - noise_strength: 노이즈 강도 (0-1 사이 값)
    - transparency: 불투명도 (0-255 사이 값, 0:투명, 255:불투명)
    """
    
    # 1. 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print("이미지를 불러올 수 없습니다.")
        return
    
    # BGR을 RGB로 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 2. 얼굴 랜드마크 검출
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cpu')
    landmarks = fa.get_landmarks(image_rgb)
    
    if not landmarks:
        print("얼굴을 찾을 수 없습니다.")
        return
    
    # 3. 중요 부위 좌표 정의
    features = {
        'right_eye': list(range(36, 42)),
        'left_eye': list(range(42, 48)),
        'nose': list(range(27, 36)),
        'mouth': list(range(48, 68))
    }
    
    # 4. 마스크 생성
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # 각 특징에 대해 마스크 생성
    for landmark in landmarks:
        for feature in features.values():
            points = landmark[feature].astype(np.int32)
            cv2.fillPoly(mask, [points], 255)
    
    # 5. 노이즈 생성 및 적용
    noise = np.random.normal(0, noise_strength, image.shape).astype(np.float32)
    noised_image = image.astype(np.float32) + noise * mask[:, :, np.newaxis]
    noised_image = np.clip(noised_image, 0, 255).astype(np.uint8)
    
    # 6. 불투명도 적용
    alpha = np.zeros((height, width), dtype=np.uint8)
    alpha[mask > 0] = transparency
    
    # 7. 최종 이미지 생성
    result = cv2.cvtColor(noised_image, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = alpha
    
    # 8. 이미지 저장
    cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_BGRA2BGR))
    print(f"처리된 이미지가 {output_path}에 저장되었습니다.")

# 실행 코드
if __name__ == "__main__":
    input_path = "jeong.jpg"  # 입력 이미지
    output_path = "landmark3.jpg"  # 출력 이미지
    
    # 함수 실행
    apply_noise_and_transparency(
        image_path=input_path,
        output_path=output_path,
        noise_strength=0.00,  # 노이즈 강도 (0-1 사이 값)
        transparency=150    # 불투명도 (0-255 사이 값)
    )