import numpy as np
import dlib
import cv2


# 194개 랜드마크 인덱스 (예시로 설정된 값)
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
MOUTH = list(range(48, 68))
NOSE = list(range(27, 36))
EYEBROWS = list(range(17, 27))
JAWLINE = list(range(1, 17))
ALL = list(range(0, 194))  # 194개 랜드마크 전체 포인트


#-- 데이터 파일과 이미지 파일 경로
predictor_file = './shape_predictor_194_face_landmarks.dat' #-- 자신의 개발 환경에 맞게 변경할 것
image_file = './selfi.jpg' #-- 자신의 개발 환경에 맞게 변경할 것

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_file)

image = cv2.imread(image_file)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = detector(gray, 1)
print("Number of faces detected: {}".format(len(rects)))


for (i, rect) in enumerate(rects):
    points = np.matrix([[p.x, p.y] for p in predictor(gray, rect).parts()])
    show_parts = points[ALL]
    for (i, point) in enumerate(show_parts):
        x = point[0,0]
        y = point[0,1]
        cv2.circle(image, (x, y), 1, (0, 255, 255), -1)
        cv2.putText(image, "{}".format(i + 1), (x, y - 2),
		cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)


# 저장할 이미지 파일 경로
output_file = './output_image.jpg' # 저장할 파일 이름


# 처리된 이미지를 파일로 저장
cv2.imwrite(output_file, image)


print(f"Image with landmarks savaed as {output_file}")
