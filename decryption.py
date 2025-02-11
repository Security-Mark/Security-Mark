import cv2
import numpy as np
from scipy.fftpack import dct, idct

def apply_dct(img):
    return dct(dct(img.T, norm='ortho').T, norm='ortho')

def apply_idct(img):
    return idct(idct(img.T, norm='ortho').T, norm='ortho')

def extract_watermark(image_path, h, w, watermark_length):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # DCT 변환
    dct_img = np.zeros_like(img, dtype=np.float32)
    for i in range(3):
        dct_img[:, :, i] = apply_dct(img[:, :, i].astype(np.float32))

    # 워터마크 추출
    extracted_bits = []
    for i in range(watermark_length):
        x, y = divmod(i, w)
        if x + 50 < h and y + 50 < w:
            value = dct_img[x+50, y+50, 0]
            extracted_bits.append(int(value > 75))  # 임계값을 75로 설정

    # 추출된 비트열을 텍스트로 변환
    extracted_text = ''.join(chr(int(''.join(map(str, extracted_bits[i:i+8])), 2)) for i in range(0, len(extracted_bits), 8))

    # 디버깅용 중간 출력
    for i in range(0, len(extracted_bits), 8):
        byte_str = ''.join(map(str, extracted_bits[i:i+8]))
        print(f"Byte {i//8}: {byte_str} -> {int(byte_str, 2)} -> {chr(int(byte_str, 2))}")
    
    return extracted_text


# 테스트 코드
image_path = '/Users/sonjeongmin/watermark/watermarked_image.png'
h, w = 256, 256  # 이미지 크기
watermark_length = min(h * w, len('SecretMessage') * 8)  # 워터마크 길이를 최소값으로 설정

# 워터마크 추출 및 출력
restored_watermark = extract_watermark(image_path, h, w, watermark_length)
print(f"Restored Watermark: {restored_watermark}")
