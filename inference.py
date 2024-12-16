from ultralytics import RTDETR
from utils import split_image, merge_image, apply_nms, draw_boxes
import cv2
import argparse
import os
import torch


def process_image(image_path, output_path, model, device):
    """
    이미지를 처리하여 원본 결과와 분할 병합 결과를 생성합니다.

    Args:
        image_path (str): 입력 이미지 경로.
        output_path (str): 결과를 저장할 경로.
        model (RTDETR): RTDETR 모델 객체.
        device (torch.device): GPU 또는 CPU 장치.
    """
    # 이미지 읽기 및 RGB 변환
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]  # 원본 이미지 크기 (H, W)

    # 모델 입력 크기(640x640)로 리사이징
    resized_image = cv2.resize(image_rgb, (640, 640))
    image_tensor = torch.from_numpy(resized_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    image_tensor = image_tensor.to(device)

    # 1. 원본 이미지 추론
    raw_result = model(image_tensor)
    raw_detections = raw_result[0].boxes

    # 2. 결과를 원본 이미지 크기에 맞게 리사이징
    resized_image_with_boxes = draw_boxes(resized_image.copy(), raw_detections, model.names)
    processed_image = cv2.resize(resized_image_with_boxes, (original_size[1], original_size[0]))

    # 3. 결과 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path.replace(".png", "_processed.png"), cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))

    print(f"Processed image saved to {output_path}")


def process_video(video_path, output_path, model, device):
    """
    비디오를 처리하여 각 프레임에 대한 결과를 생성합니다.

    Args:
        video_path (str): 입력 비디오 경로.
        output_path (str): 결과를 저장할 경로.
        model (RTDETR): RTDETR 모델 객체.
        device (torch.device): GPU 또는 CPU 장치.
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"Processing frame {frame_count}")

        # 프레임을 RGB로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 모델 입력 크기(640x640)로 리사이징
        resized_frame = cv2.resize(frame_rgb, (640, 640))
        frame_tensor = torch.from_numpy(resized_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        frame_tensor = frame_tensor.to(device)

        # 1. 모델 추론
        raw_result = model(frame_tensor)
        raw_detections = raw_result[0].boxes

        # 2. 결과를 그려 리사이징된 프레임에 표시
        resized_frame_with_boxes = draw_boxes(resized_frame.copy(), raw_detections, model.names)

        # 결과 프레임을 원본 해상도로 리사이징
        processed_frame = cv2.resize(resized_frame_with_boxes, (width, height))
        out.write(cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))

    cap.release()
    out.release()
    print(f"Processed video saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="RT-DETR Object Detection for Image/Video")
    parser.add_argument("input", type=str, help="Input file name (image or video)")
    args = parser.parse_args()

    input_path = f'./test_data/{args.input}'
    output_path = f'./test_data/output/{args.input}'

    # GPU 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # RTDETR 모델 초기화 및 GPU로 이동
    model = RTDETR("rtdetr-l.pt").to(device)

    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} does not exist.")
        return

    if input_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        process_image(input_path, output_path, model, device)
    elif input_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        process_video(input_path, output_path, model, device)
    else:
        print("Error: Unsupported file type. Supported formats are: jpg, jpeg, png, bmp, mp4, avi, mov, mkv.")


if __name__ == "__main__":
    main()
