import cv2
import numpy as np
from scipy.fftpack import dct, idct

def apply_dct(img):
    return dct(dct(img.T, norm='ortho').T, norm='ortho')

def apply_idct(img):
    return idct(idct(img.T, norm='ortho').T, norm='ortho')

def embed_watermark(image_path, watermark_text, output_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    h, w, c = img.shape

    # DCT 변환
    dct_img = np.zeros_like(img, dtype=np.float32)
    for i in range(c):
        dct_img[:, :, i] = apply_dct(img[:, :, i].astype(np.float32))

    # 워터마크 삽입을 위한 텍스트 비트 변환
    watermark = np.zeros((h, w), dtype=np.float32)
    text_bits = [int(b) for b in ''.join(f'{ord(c):08b}' for c in watermark_text)]
    
    # 워터마크를 중간 주파수 영역에 삽입
    for i, bit in enumerate(text_bits):
        x, y = divmod(i, w)
        if x + 30 < h and y + 30 < w:  # 워터마크를 이미지의 중앙 부근에 삽입
            watermark[x+30, y+30] = bit * 0  # 워터마크 강도를 조정

    # 워터마크 삽입
    watermarked_dct = dct_img + np.repeat(watermark[:, :, np.newaxis], c, axis=2)

    # IDCT 변환
    watermarked_img = np.zeros_like(img, dtype=np.float32)
    for i in range(c):
        watermarked_img[:, :, i] = apply_idct(watermarked_dct[:, :, i])

    # 이미지 저장 (무손실 PNG 포맷)
    watermarked_img = np.clip(watermarked_img, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path.replace('.jpeg', '.png'), watermarked_img)
    print(f"Watermarked image saved at {output_path.replace('.jpeg', '.png')}")


# 테스트
embed_watermark('/Users/sonjeongmin/watermark/input_image.jpeg', 'SecretMessage', '/Users/sonjeongmin/watermark/watermarked_image.png')
