from ultralytics import RTDETR
from utils import split_image, merge_image, apply_nms, draw_boxes
import cv2

# 모델 로드
model = RTDETR("rtdetr-l.pt")

# 테스트 이미지 경로
image_path = "test_data/test.png"
image = cv2.imread(image_path)

# 1. 원본 이미지 추론 (Raw Result)
raw_result = model(image)
raw_detections = raw_result[0].boxes

# 2. 분할 후 병합된 추론 (Processed Result)
# 이미지 4분할
split_images = split_image(image)

# 각 분할된 이미지에 대해 추론 수행
detections = []
offsets = []
for split_img, offset in split_images:
    result = model(split_img)
    detections.append(result[0].boxes)
    offsets.append(offset)

# 분할된 추론 결과를 원래 이미지 좌표계로 변환 및 NMS 적용
merged_detections = merge_image(image, detections, offsets)
final_detections = apply_nms(merged_detections, iou_threshold=0.5)


# 3. 결과 그리기
raw_image = draw_boxes(image.copy(), [{"xyxy": box.xyxy.clone(), "conf": box.conf.clone(), "cls": box.cls.clone()} for box in raw_detections], model.names)
processed_image = draw_boxes(image.copy(), final_detections, model.names)

# 4. 결과 저장 및 비교
# 결과 저장
raw_result_path = "./test_data/output/raw_result.png"
processed_result_path = "./test_data/output/split_result.png"
cv2.imwrite(f"{raw_result_path}", raw_image)
cv2.imwrite(f"{processed_result_path}", processed_image)

print(f"Raw 결과 이미지는 {raw_result_path}에 저장되었습니다.")
print(f"Processed 결과 이미지는 {processed_result_path}에 저장되었습니다.")