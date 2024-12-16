from ultralytics import RTDETR
from utils import split_image, merge_image, apply_nms, draw_boxes
import cv2
import argparse
import os
import time
import json

def process_image(image_path, output_path, model, json_path):
    """
    이미지를 처리하고 Bounding Box 결과를 JSON에 저장합니다.

    Args:
        image_path (str): 입력 이미지 경로.
        output_path (str): 처리된 결과를 저장할 경로.
        model (RTDETR): RTDETR 모델 객체.
        json_path (str): JSON 결과를 저장할 경로.

    Returns:
        None
    """
    # 이미지 읽기
    image = cv2.imread(image_path)

    # 1. 원본 이미지 추론
    raw_result = model(image)
    raw_detections = raw_result[0].boxes

    # 2. 이미지 분할 및 각 분할에 대한 추론
    split_images = split_image(image)

    detections = []
    offsets = []
    for split_img, offset in split_images:
        result = model(split_img)
        detections.append(result[0].boxes)
        offsets.append(offset)

    # 3. 병합 및 NMS 적용
    merged_detections = merge_image(image, detections, offsets)
    final_detections = apply_nms(merged_detections, iou_threshold=0.5)

    # 4. JSON에 Bounding Box 결과 저장
    result_data = []
    for det in final_detections:
        bbox = det["xyxy"].tolist()  # Bounding Box 좌표 (Tensor → List)
        confidence = float(det["conf"])  # 신뢰도 (Tensor → Float)
        class_id = int(det["cls"])  # 클래스 ID (Tensor → Int)
        class_name = model.names[class_id]  # 클래스 이름

        # 결과 저장
        result_data.append({
            "bbox": bbox,
            "confidence": confidence,
            "class_name": class_name
        })

    # JSON 저장
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump({"frame_0": result_data}, f, indent=4)

    # 5. 결과 그리기
    processed_image = draw_boxes(image.copy(), final_detections, model.names)

    # 결과 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed_result_path = os.path.splitext(output_path)[0] + "_processed.png"
    cv2.imwrite(processed_result_path, processed_image)

    print(f"Processed 결과 이미지는 {processed_result_path}에 저장되었습니다.")
    print(f"Bounding Box 결과는 {json_path}에 저장되었습니다.")


def process_video(video_path, output_path, model, json_path):
    """
    비디오를 처리하고 Bounding Box 결과를 JSON에 저장합니다.

    Args:
        video_path (str): 입력 비디오 경로.
        output_path (str): 결과 비디오 저장 경로.
        model (RTDETR): RTDETR 모델 객체.
        json_path (str): JSON 결과를 저장할 경로.

    Returns:
        None
    """
    # 비디오 읽기
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 비디오 저장 설정
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    results = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        print(f"Processing frame {frame_count}")

        # 1. 원본 프레임 추론
        raw_result = model(frame)
        raw_detections = raw_result[0].boxes

        # 2. 프레임 분할 및 추론
        split_images = split_image(frame)
        detections = []
        offsets = []
        for split_img, offset in split_images:
            result = model(split_img)
            detections.append(result[0].boxes)
            offsets.append(offset)

        # 3. 병합 및 NMS 적용
        merged_detections = merge_image(frame, detections, offsets)
        final_detections = apply_nms(merged_detections, iou_threshold=0.5)

        # 4. JSON에 Bounding Box 결과 저장
        frame_data = []
        for det in final_detections:
            bbox = det["xyxy"].tolist()  # Bounding Box 좌표 (Tensor → List)
            confidence = float(det["conf"])  # 신뢰도 (Tensor → Float)
            class_id = int(det["cls"])  # 클래스 ID (Tensor → Int)
            class_name = model.names[class_id]  # 클래스 이름

            frame_data.append({
                "bbox": bbox,
                "confidence": confidence,
                "class_name": class_name
            })

        results[f"frame_{frame_count}"] = frame_data

        # 5. 결과 그리기
        processed_frame = draw_boxes(frame.copy(), final_detections, model.names)

        # 비디오에 프레임 저장
        out.write(processed_frame)

    cap.release()
    out.release()

    # JSON 저장
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Processed video saved to {output_path}")
    print(f"Bounding Box 결과는 {json_path}에 저장되었습니다.")


def main():
    # ArgumentParser를 이용해 입력 파일(이미지/비디오)과 출력 파일 경로를 받아 처리
    parser = argparse.ArgumentParser(description="RT-DETR Object Detection for Image/Video")
    parser.add_argument("input", type=str, help="Input file name (image or video)")
    args = parser.parse_args()

    input_path = f'./test_data/{args.input}'
    output_path = f'./test_data/output/{args.input}'
    json_path = f'./test_data/output/{os.path.splitext(args.input)[0]}_results.json'

    # RTDETR 모델 초기화
    model = RTDETR("rtdetr-l.pt")

    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} does not exist.")
        return

    # 입력 파일 유형에 따라 이미지 또는 비디오 처리
    if input_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        process_image(input_path, output_path, model, json_path)
    elif input_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        process_video(input_path, output_path, model, json_path)
    else:
        print("Error: Unsupported file type. Please provide an image or video file.")


if __name__ == "__main__":
    main()