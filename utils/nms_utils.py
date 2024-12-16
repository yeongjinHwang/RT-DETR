import torch

def apply_nms(detections, iou_threshold=0.5, device=None):
    """
    NMS(Non-Maximum Suppression)를 적용하여 겹치는 바운딩 박스를 제거합니다.

    Args:
        detections (list): 추론된 바운딩 박스 정보 리스트 (dict 구조 포함).
        iou_threshold (float): IoU 임계값.
        device (torch.device): 연산을 수행할 디바이스 (CPU 또는 CUDA).

    Returns:
        list: NMS가 적용된 바운딩 박스 정보 리스트.
    """
    if len(detections) == 0:
        return []

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 바운딩 박스 좌표 및 신뢰도 추출
    boxes = torch.vstack([det["xyxy"].to(device) for det in detections])  # 2D 텐서 (N, 4)
    scores = torch.tensor([det["conf"] for det in detections], device=device)  # (N,)

    # NMS 적용
    keep_indices = torch.ops.torchvision.nms(boxes, scores, iou_threshold)

    # NMS 결과 필터링
    return [detections[i] for i in keep_indices]
