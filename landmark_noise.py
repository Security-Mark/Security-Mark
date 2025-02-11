import numpy as np
import dlib
import cv2

# 68개 랜드마크 인덱스 정의
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
NOSE = list(range(27, 36))
MOUTH = list(range(48, 68))

# 데이터 파일과 이미지 파일 경로
predictor_file = 'C:/Users/jun/deepfake/shape_predictor_68_face_landmarks.dat'  # 자신의 개발 환경에 맞게 변경할 것
image_file = 'yayaya.jpg'  # 자신의 개발 환경에 맞게 변경할 것

# 얼굴 검출기와 랜드마크 예측기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_file)

# 이미지 로드
image = cv2.imread(image_file)
if image is None:
    print("이미지를 불러올 수 없습니다.")
    exit()

# 원본 이미지를 복사 (랜드마크 표시를 피하기 위해)
original_image = image.copy()

# 그레이스케일 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 얼굴 검출
rects = detector(gray, 1)
print("Number of faces detected: {}".format(len(rects)))

# 마스크 생성
height, width = image.shape[:2]
mask = np.zeros((height, width), dtype=np.uint8)

for (i, rect) in enumerate(rects):
    # 랜드마크 추출
    shape = predictor(gray, rect)
    points = np.array([[p.x, p.y] for p in shape.parts()])

    # 랜드마크 표시 부분을 제거하였습니다.

    # 눈, 코, 입 영역을 마스크에 채우기
    features = [RIGHT_EYE, LEFT_EYE, NOSE, MOUTH]
    for feature in features:
        pts = points[feature]
        pts = pts.astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)

# 노이즈 생성 및 적용
noise_strength = 0.05  # 노이즈 강도 (0 - 1 사이 값, 값이 클수록 노이즈가 강해집니다)
noise = np.random.normal(0, noise_strength * 255, original_image.shape).astype(np.float32)

# 마스크를 3차원으로 확장
mask_3d = mask[:, :, np.newaxis]
mask_3d = np.repeat(mask_3d, 3, axis=2)

# 마스크를 0과 1로 정규화
mask_norm = mask_3d / 255.0

# 이미지에 노이즈 적용 (마스크된 영역에만)
noised_image = original_image.astype(np.float32)
noised_image += noise * mask_norm
noised_image = np.clip(noised_image, 0, 255).astype(np.uint8)

# 저장할 이미지 파일 경로
output_file = 'noise.jpg'  # 저장할 파일 이름

# 처리된 이미지를 파일로 저장
cv2.imwrite(output_file, noised_image)

print(f"Image with noise saved as {output_file}")