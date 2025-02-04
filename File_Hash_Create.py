import hashlib

def generate_image_hash(file_path, hash_algorithm='sha256'):
    """이미지 파일의 해시를 생성"""
    hash_func = getattr(hashlib, hash_algorithm, None)
    if hash_func is None:
        raise ValueError(f"지원되지 않는 해시 알고리즘: {hash_algorithm}")
    
    hasher = hash_func()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)

    return hasher.hexdigest()

# 예제 사용법
image_path = "D:\File hash\Hash Create.png" # 해시를 생성할 이미지 파일 경로
hash_value = generate_image_hash(image_path)
print(f"SHA-256 Hash: {hash_value}")