"""
YOLO Segmentation 레이블을 Label Studio JSON 형식으로 변환
이미지를 base64로 인코딩하여 포함, 50MB 단위로 배치 분할
"""

import json
import base64
from pathlib import Path
from PIL import Image
import uuid


def image_to_base64(image_path: Path) -> str:
    """이미지를 base64 문자열로 변환"""
    with open(image_path, "rb") as f:
        img_data = f.read()

    # 확장자에 따른 MIME 타입
    ext = image_path.suffix.lower()
    mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"

    b64_str = base64.b64encode(img_data).decode("utf-8")
    return f"data:{mime_type};base64,{b64_str}"


def create_task(img_path: Path, label_path: Path, task_id: int, use_base64: bool = True) -> dict:
    """단일 태스크 생성"""
    # 이미지 크기 가져오기
    with Image.open(img_path) as img:
        img_width, img_height = img.size

    # 레이블 읽기
    annotations = []
    if label_path.exists():
        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9:  # class_id + 4 points (8 values)
                continue

            points = [float(p) for p in parts[1:]]

            # YOLO 정규화 좌표 (0~1) -> Label Studio 퍼센트 좌표 (0~100)
            polygon_points = []
            for i in range(0, len(points), 2):
                x_percent = points[i] * 100
                y_percent = points[i + 1] * 100
                polygon_points.append([x_percent, y_percent])

            annotation = {
                "id": str(uuid.uuid4())[:8],
                "type": "polygonlabels",
                "value": {
                    "points": polygon_points,
                    "polygonlabels": ["id_card"],
                },
                "from_name": "label",
                "to_name": "image",
                "original_width": img_width,
                "original_height": img_height,
            }
            annotations.append(annotation)

    # 이미지 데이터
    if use_base64:
        image_data = image_to_base64(img_path)
    else:
        image_data = f"/data/local-files/?d={img_path.absolute()}"

    task = {
        "id": task_id,
        "data": {
            "image": image_data,
        },
        "annotations": [
            {
                "id": task_id,
                "result": annotations,
            }
        ] if annotations else [],
    }
    return task


def convert_yolo_to_labelstudio_batched(
    image_dir: str,
    label_dir: str,
    output_dir: str,
    max_batch_size_mb: float = 50.0,
    use_base64: bool = True,
):
    """
    YOLO Segmentation 레이블을 Label Studio JSON으로 변환 (배치 분할)

    Args:
        image_dir: 이미지 폴더 경로
        label_dir: 레이블 폴더 경로
        output_dir: 출력 폴더 경로
        max_batch_size_mb: 배치당 최대 크기 (MB)
        use_base64: True면 이미지를 base64로 인코딩
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_batch_size_bytes = max_batch_size_mb * 1024 * 1024

    # 이미지 파일 순회
    image_files = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))

    batch_num = 1
    current_batch = []
    current_batch_size = 0
    task_id = 1
    total_tasks = 0

    for img_path in image_files:
        label_path = label_dir / f"{img_path.stem}.txt"

        # 태스크 생성
        task = create_task(img_path, label_path, task_id, use_base64)
        task_json = json.dumps(task, ensure_ascii=False)
        task_size = len(task_json.encode("utf-8"))

        # 배치 크기 초과 시 저장하고 새 배치 시작
        if current_batch and (current_batch_size + task_size > max_batch_size_bytes):
            batch_path = output_dir / f"batch_{batch_num:03d}.json"
            with open(batch_path, "w", encoding="utf-8") as f:
                json.dump(current_batch, f, ensure_ascii=False)

            print(f"  배치 {batch_num}: {len(current_batch)}개 태스크, {current_batch_size / 1024 / 1024:.2f}MB")

            batch_num += 1
            current_batch = []
            current_batch_size = 0

        current_batch.append(task)
        current_batch_size += task_size
        task_id += 1
        total_tasks += 1

    # 마지막 배치 저장
    if current_batch:
        batch_path = output_dir / f"batch_{batch_num:03d}.json"
        with open(batch_path, "w", encoding="utf-8") as f:
            json.dump(current_batch, f, ensure_ascii=False)

        print(f"  배치 {batch_num}: {len(current_batch)}개 태스크, {current_batch_size / 1024 / 1024:.2f}MB")

    print(f"\n✅ 총 {total_tasks}개 태스크를 {batch_num}개 배치로 변환 완료")
    print(f"   출력 경로: {output_dir}")

    return batch_num


def main():
    base_dir = Path(__file__).parent

    print("Label Studio JSON 변환 (base64 포함, 50MB 배치 분할)\n")

    convert_yolo_to_labelstudio_batched(
        image_dir=base_dir / "output" / "bg_gen",
        label_dir=base_dir / "output" / "bg_gen",
        output_dir=base_dir / "output" / "bg_gen" / "batch",
        max_batch_size_mb=50.0,
        use_base64=True,
    )


if __name__ == "__main__":
    main()
