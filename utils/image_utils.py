import cv2
import numpy as np

def split_image(image):
    """
    이미지를 4분할로 나눕니다.
    """
    h, w = image.shape[:2]
    h_half, w_half = h // 2, w // 2

    return [
        (image[0:h_half, 0:w_half], (0, 0)),  # 왼쪽 위
        (image[0:h_half, w_half:w], (w_half, 0)),  # 오른쪽 위
        (image[h_half:h, 0:w_half], (0, h_half)),  # 왼쪽 아래
        (image[h_half:h, w_half:w], (w_half, h_half))  # 오른쪽 아래
    ]

def merge_image(image, detections, offsets):
    """
    분할된 추론 결과를 원래 이미지 좌표계로 변환합니다.

    Returns:
        list: 변환된 바운딩 박스 정보 리스트 (dict 구조).
    """
    merged = []
    for det, (x_offset, y_offset) in zip(detections, offsets):
        for box in det:
            xyxy = box.xyxy.clone()
            # 바운딩 박스 좌표 이동
            xyxy[0][0] += x_offset  # x1
            xyxy[0][1] += y_offset  # y1
            xyxy[0][2] += x_offset  # x2
            xyxy[0][3] += y_offset  # y2
            
            # dict 형태로 저장
            merged.append({
                "xyxy": xyxy,
                "conf": box.conf.clone(),
                "cls": box.cls.clone()
            })
    return merged


def draw_boxes(image, detections, class_names, conf_threshold=0.5):
    """
    이미지에 바운딩 박스를 그리고 클래스 이름과 신뢰도를 표시합니다.

    Args:
        image (np.ndarray): 입력 이미지.
        detections (list): NMS가 적용된 바운딩 박스 정보 리스트 (dict 구조).
        class_names (list): 클래스 이름 리스트.
        conf_threshold (float): 신뢰도 임계값.

    Returns:
        np.ndarray: 바운딩 박스가 그려진 이미지.
    """
    for det in detections:
        if det["conf"][0] >= conf_threshold:  # 신뢰도 확인
            x1, y1, x2, y2 = map(int, det["xyxy"][0])  # 바운딩 박스 좌표 추출
            class_id = int(det["cls"][0])  # 클래스 ID 추출
            class_name = class_names[class_id]  # 클래스 이름 매핑
            confidence = det["conf"][0]  # 신뢰도 추출

            # 바운딩 박스 그리기
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{class_name} {confidence:.2f}"
            cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image