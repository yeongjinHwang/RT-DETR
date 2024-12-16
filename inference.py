from ultralytics import RTDETR
from utils import split_image, merge_image, apply_nms, draw_boxes
import cv2
import argparse
import os

def process_image(image_path, output_path, model):
    """
    이미지를 처리하여 원본 결과와 분할 병합 결과를 생성합니다.

    Args:
        image_path (str): 입력 이미지 경로.
        output_path (str): 처리된 결과를 저장할 경로.
        model (RTDETR): RTDETR 모델 객체.

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

    # 4. 결과 그리기
    raw_image = draw_boxes(
        image.copy(),
        [{"xyxy": box.xyxy.clone(), "conf": box.conf.clone(), "cls": box.cls.clone()} for box in raw_detections],
        model.names,
    )
    processed_image = draw_boxes(image.copy(), final_detections, model.names)

    # 결과 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    raw_result_path = os.path.splitext(output_path)[0] + "_raw.png"
    processed_result_path = os.path.splitext(output_path)[0] + "_processed.png"
    cv2.imwrite(raw_result_path, raw_image)
    cv2.imwrite(processed_result_path, processed_image)

    print(f"Raw 결과 이미지는 {raw_result_path}에 저장되었습니다.")
    print(f"Processed 결과 이미지는 {processed_result_path}에 저장되었습니다.")

def process_video(video_path, output_path, model):
    """
    비디오를 처리하여 각 프레임에 대한 분할 병합 결과를 생성합니다.

    Args:
        video_path (str): 입력 비디오 경로.
        output_path (str): 처리된 결과 비디오를 저장할 경로.
        model (RTDETR): RTDETR 모델 객체.

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

        # 4. 결과 그리기
        processed_frame = draw_boxes(frame.copy(), final_detections, model.names)

        # 비디오에 프레임 저장
        out.write(processed_frame)

    cap.release()
    out.release()
    print(f"Processed video saved to {output_path}")


def main():
    # ArgumentParser를 이용해 입력 파일(이미지/비디오)과 출력 파일 경로를 받아 처리
    parser = argparse.ArgumentParser(description="RT-DETR Object Detection for Image/Video")
    parser.add_argument("input", type=str, help="Input file name (image or video)")
    args = parser.parse_args()

    input_path = f'./test_data/{args.input}'
    output_path = f'./test_data/output/{args.input}'

    # RTDETR 모델 초기화
    model = RTDETR("rtdetr-l.pt")

    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} does not exist.")
        return

    # 입력 파일 유형에 따라 이미지 또는 비디오 처리
    if input_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        process_image(input_path, output_path, model)
    elif input_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        process_video(input_path, output_path, model)
    else:
        print("Error: Unsupported file type. Please provide an image or video file.")


if __name__ == "__main__":
    main()