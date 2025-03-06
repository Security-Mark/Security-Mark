import numpy as np
import dlib
import cv2
import hashlib
import os
from scipy.fftpack import dct, idct
from PIL import Image
import imagehash
from flask import Flask, request, jsonify

app = Flask(__name__)

# 랜드마크 인덱스 정의
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
NOSE = list(range(27, 36))
MOUTH = list(range(48, 68))

def apply_dct(img):
    return dct(dct(img.T, norm='ortho').T, norm='ortho')

def apply_idct(img):
    return idct(idct(img.T, norm='ortho').T, norm='ortho')

def generate_image_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def add_fake_landmark_signals(image, mask, fake_strength=50):
    fake_signal = np.zeros_like(image, dtype=np.float32)
    height, width = image.shape[:2]
    
    for _ in range(35):  # 가짜 랜드마크 35개 추가 (기존보다 15개 증가)
        fake_x = np.random.randint(0, width)
        fake_y = np.random.randint(0, height)
        
        if mask[fake_y, fake_x] == 0:  # 얼굴 영역(mask가 255인 곳)과 겹치지 않도록 확인
            cv2.circle(fake_signal, (fake_x, fake_y), 1, (fake_strength, fake_strength, fake_strength), -1)  # 점 크기 축소
    
    return fake_signal

def process_image(image_path, predictor_file, output_path, noise_strength=0.08, watermark_strength=150, offset=50, dct_strength=5, noise_intensity=2):
    if not os.path.exists(image_path):
        raise FileNotFoundError("❌ 이미지 파일이 존재하지 않습니다.")
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_file)
    
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    for rect in rects:
        shape = predictor(gray, rect)
        points = np.array([[p.x, p.y] for p in shape.parts()])
        for feature in [RIGHT_EYE, LEFT_EYE, NOSE, MOUTH]:
            cv2.fillPoly(mask, [points[feature].astype(np.int32)], 255)
    
    noise = np.random.normal(0, noise_strength * 255, image.shape).astype(np.float32)
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2) / 255.0
    processed_image = image.astype(np.float32) + noise * mask_3d
    
    fake_signal = add_fake_landmark_signals(processed_image, mask)
    processed_image += fake_signal  # 가짜 랜드마크는 얼굴 영역 밖에만 적용
    
    processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)
    
    image_hash = generate_image_hash(image_path)[:16]
    dct_img = np.zeros_like(processed_image, dtype=np.float32)
    for i in range(3):
        dct_img[:, :, i] = apply_dct(processed_image[:, :, i].astype(np.float32))
    
    watermark = np.zeros(processed_image.shape[:2], dtype=np.float32)
    text_bits = [int(b) for b in ''.join(f'{ord(c):08b}' for c in image_hash)]
    for i, bit in enumerate(text_bits):
        x, y = divmod(i, processed_image.shape[1])
        if x + offset < processed_image.shape[0] and y + offset < processed_image.shape[1]:
            watermark[x + offset, y + offset] = bit * watermark_strength
    
    watermarked_dct = dct_img + np.repeat(watermark[:, :, np.newaxis], 3, axis=2)
    watermarked_img = np.zeros_like(processed_image, dtype=np.float32)
    for i in range(3):
        watermarked_img[:, :, i] = apply_idct(watermarked_dct[:, :, i])
    
    dct_pattern = np.random.normal(0, dct_strength, processed_image.shape[:2])
    for i in range(3):
        watermarked_img[:, :, i] += dct_pattern
    
    noise = np.random.normal(0, noise_intensity, processed_image.shape).astype(np.float32)
    final_img = np.clip(watermarked_img + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, final_img)
    print(f"✅ 최종 이미지 저장 완료! 저장 파일: {output_path}")
    return output_path

@app.route("/")
def read_root():
    return {"message": "Flask 서버 실행 중"}

@app.route('/process', methods=['POST'])
def process_request():
    file = request.files['image']
    predictor_file = "shape_predictor_68_face_landmarks.dat"
    input_path = "1.jpg"
    output_path = "real1.jpg"
    
    file.save(input_path)
    try:
        processed_path = process_image(input_path, predictor_file, output_path)
        return jsonify({"status": "success", "processed_image": processed_path})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
