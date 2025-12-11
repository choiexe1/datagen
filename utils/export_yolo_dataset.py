"""
YOLO Segmentation 훈련용 데이터셋 생성
- images/train, images/val
- labels/train, labels/val
- data.yaml
"""

import shutil
import random
from pathlib import Path


def export_yolo_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    class_names: list = None,
):
    """
    YOLO Segmentation 훈련용 데이터셋 구조로 내보내기

    Args:
        source_dir: 원본 이미지/레이블 폴더
        output_dir: 출력 폴더
        train_ratio: 훈련 데이터 비율 (0.8 = 80% train, 20% val)
        class_names: 클래스 이름 리스트
    """
    if class_names is None:
        class_names = ["id_card"]

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    # 출력 디렉토리 구조 생성
    (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # 이미지 파일 목록
    image_files = sorted(list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png")))

    # 셔플 후 train/val 분할
    random.seed(42)
    random.shuffle(image_files)

    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    print(f"전체: {len(image_files)}장")
    print(f"Train: {len(train_files)}장")
    print(f"Val: {len(val_files)}장")

    # 파일 복사
    def copy_files(files, split_name):
        for img_path in files:
            label_path = source_dir / f"{img_path.stem}.txt"

            # 이미지 복사
            dst_img = output_dir / "images" / split_name / img_path.name
            shutil.copy2(img_path, dst_img)

            # 레이블 복사
            if label_path.exists():
                dst_label = output_dir / "labels" / split_name / label_path.name
                shutil.copy2(label_path, dst_label)

    print("\nTrain 데이터 복사 중...")
    copy_files(train_files, "train")

    print("Val 데이터 복사 중...")
    copy_files(val_files, "val")

    # data.yaml 생성
    yaml_content = f"""# YOLO Segmentation Dataset
# 주민등록증 탐지용 데이터셋

path: {output_dir.absolute()}
train: images/train
val: images/val

# Classes
names:
"""
    for i, name in enumerate(class_names):
        yaml_content += f"  {i}: {name}\n"

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"\n✅ YOLO 데이터셋 생성 완료!")
    print(f"   경로: {output_dir}")
    print(f"   data.yaml: {yaml_path}")

    return output_dir


def main():
    base_dir = Path(__file__).parent

    export_yolo_dataset(
        source_dir=base_dir / "output" / "bg_gen",
        output_dir=base_dir / "output" / "yolo_dataset",
        train_ratio=0.8,
        class_names=["id_card"],
    )


if __name__ == "__main__":
    main()
