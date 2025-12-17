#!/usr/bin/env python
"""
EKYC 증강 파이프라인
- output/id_cards의 합성 이미지에 EKYC 증강 적용
- output/augmented_id_cards에 출력
"""

import argparse
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from PIL import Image


def get_ekyc_transform(intensity: str = "light") -> A.Compose:
    """EKYC 증강 파이프라인 생성 (E-KYC 환경은 대부분 화질이 양호함)

    Args:
        intensity: "light", "medium", "heavy"
    """
    if intensity == "light":
        # E-KYC 기본: 가벼운 증강
        jpeg_p, blur_p, brightness_p, noise_p, color_p = 0.25, 0.15, 0.25, 0.15, 0.2
    elif intensity == "heavy":
        jpeg_p, blur_p, brightness_p, noise_p, color_p = 0.5, 0.35, 0.5, 0.35, 0.4
    else:  # medium
        jpeg_p, blur_p, brightness_p, noise_p, color_p = 0.35, 0.25, 0.35, 0.25, 0.3

    return A.Compose(
        [
            # 1. JPEG 압축 (화질 높음)
            A.ImageCompression(
                quality_range=(75, 98), compression_type="jpeg", p=jpeg_p
            ),
            # 2. 블러 (약한 블러만)
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=(3, 5)),
                    A.GaussianBlur(blur_limit=(3, 5)),
                    A.Defocus(radius=(2, 3), alias_blur=(0.1, 0.3)),
                ],
                p=blur_p,
            ),
            # 3. 밝기/대비 (미세 조정)
            A.RandomBrightnessContrast(
                brightness_limit=(-0.12, 0.12),
                contrast_limit=(-0.08, 0.08),
                p=brightness_p,
            ),
            # 4. 노이즈 (약한 노이즈)
            A.OneOf(
                [
                    A.ISONoise(intensity=(0.05, 0.15)),
                    A.GaussNoise(std_range=(0.01, 0.05)),  # pyright: ignore[reportCallIssue]
                ],
                p=noise_p,
            ),
            # 5. 색상 변화 (미세한 색조 변화)
            A.OneOf(
                [
                    A.HueSaturationValue(
                        hue_shift_limit=5, sat_shift_limit=8, val_shift_limit=6
                    ),
                    A.RGBShift(r_shift_limit=6, g_shift_limit=6, b_shift_limit=6),
                ],
                p=color_p,
            ),
        ]
    )


def add_glare(img: np.ndarray, p: float = 0.15) -> np.ndarray:
    """커스텀 글레어/빛반사 효과 (신분증 표면 반사)"""
    if random.random() > p:
        return img

    h, w = img.shape[:2]
    result = img.astype(np.float32)

    # 글레어 마스크 생성
    glare_mask = np.zeros((h, w), dtype=np.float32)

    # 랜덤 위치
    cx = random.randint(int(w * 0.15), int(w * 0.85))
    cy = random.randint(int(h * 0.15), int(h * 0.85))

    # 길고 자연스러운 타원형 (가로로 긴 빛 반사)
    long_axis = random.randint(int(w * 0.15), int(w * 0.35))  # 적당한 가로 길이
    short_axis = random.randint(int(h * 0.05), int(h * 0.12))  # 적당한 세로 길이
    axes = (long_axis, short_axis)
    angle = random.randint(-25, 25)  # 약간의 기울기
    intensity = random.uniform(0.3, 0.55)  # 자연스러운 강도

    # 메인 글레어
    cv2.ellipse(glare_mask, (cx, cy), axes, angle, 0, 360, intensity, -1)

    # 중심부 더 밝게 (핫스팟)
    hotspot_axes = (long_axis // 3, short_axis // 2)
    cv2.ellipse(glare_mask, (cx, cy), hotspot_axes, angle, 0, 360, min(intensity + 0.15, 0.7), -1)

    # 부드럽게 블러
    blur_size = max(31, int(min(h, w) * 0.15) | 1)  # 홀수로 만들기
    glare_mask = cv2.GaussianBlur(glare_mask, (blur_size, blur_size), 0)

    # 밝기 증가 적용
    for c in range(3):
        result[:, :, c] = result[:, :, c] + (255 - result[:, :, c]) * glare_mask

    return np.clip(result, 0, 255).astype(np.uint8)


def add_shadow(img: np.ndarray, p: float = 0.15) -> np.ndarray:
    """커스텀 그림자 효과 (손/물체에 의한 그림자)"""
    if random.random() > p:
        return img

    h, w = img.shape[:2]
    shadow_mask = np.ones((h, w), dtype=np.float32)

    shadow_type = random.choice(["corner", "edge", "diagonal"])

    if shadow_type == "corner":
        # 모서리 그림자
        corner = random.choice(["tl", "tr", "bl", "br"])
        radius = random.randint(int(min(h, w) * 0.2), int(min(h, w) * 0.4))
        shadow_intensity = random.uniform(0.5, 0.7)

        if corner == "tl":
            cv2.circle(shadow_mask, (0, 0), radius, shadow_intensity, -1)
        elif corner == "tr":
            cv2.circle(shadow_mask, (w, 0), radius, shadow_intensity, -1)
        elif corner == "bl":
            cv2.circle(shadow_mask, (0, h), radius, shadow_intensity, -1)
        else:
            cv2.circle(shadow_mask, (w, h), radius, shadow_intensity, -1)

    elif shadow_type == "edge":
        # 가장자리 그림자
        edge = random.choice(["top", "bottom", "left", "right"])
        shadow_width = random.randint(int(min(h, w) * 0.1), int(min(h, w) * 0.2))
        shadow_intensity = random.uniform(0.6, 0.8)

        if edge == "top":
            shadow_mask[:shadow_width, :] = shadow_intensity
        elif edge == "bottom":
            shadow_mask[-shadow_width:, :] = shadow_intensity
        elif edge == "left":
            shadow_mask[:, :shadow_width] = shadow_intensity
        else:
            shadow_mask[:, -shadow_width:] = shadow_intensity

    else:  # diagonal
        # 대각선 그림자
        pts = np.array(
            [
                [0, 0],
                [int(w * random.uniform(0.3, 0.5)), 0],
                [0, int(h * random.uniform(0.3, 0.5))],
            ],
            dtype=np.int32,
        )

        if random.random() > 0.5:
            # 우측 하단으로 이동
            pts[:, 0] = w - pts[:, 0]
            pts[:, 1] = h - pts[:, 1]

        shadow_intensity = random.uniform(0.5, 0.7)
        cv2.fillPoly(shadow_mask, [pts], shadow_intensity)

    # 블러로 부드럽게
    shadow_mask = cv2.GaussianBlur(shadow_mask, (31, 31), 0)

    # 그림자 적용
    result = img.astype(np.float32)
    for c in range(3):
        result[:, :, c] = result[:, :, c] * shadow_mask

    return np.clip(result, 0, 255).astype(np.uint8)


def augment_image(
    img: np.ndarray, transform: A.Compose, glare_p: float = 0.15, shadow_p: float = 0.15
) -> np.ndarray:
    """이미지에 EKYC 증강 적용"""
    # Albumentations 증강
    augmented = transform(image=img)
    result = augmented["image"]

    # 커스텀 글레어
    result = add_glare(result, p=glare_p)

    # 커스텀 그림자
    result = add_shadow(result, p=shadow_p)

    return result


def process_images(
    input_dir: Path,
    output_dir: Path,
    num_augments: int = 1,
    intensity: str = "light",
    glare_p: float = 0.15,
    shadow_p: float = 0.15,
):
    """디렉토리 내 모든 이미지 처리"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 이미지 파일 목록
    image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))

    if not image_files:
        print(f"이미지를 찾을 수 없음: {input_dir}")
        return

    print(f"발견된 이미지: {len(image_files)}개")
    print(f"증강 배수: {num_augments}x")
    print(f"강도: {intensity}")
    print(f"예상 출력: {len(image_files) * num_augments}개")
    print()

    # 증강 파이프라인 생성
    transform = get_ekyc_transform(intensity)

    total_generated = 0

    for img_path in image_files:
        # 이미지 로드 (알파 채널 유지)
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"로드 실패: {img_path}")
            continue

        # RGBA → RGB 변환 (알파 채널이 있는 경우)
        if img.shape[-1] == 4:
            # 알파 채널 저장
            alpha = img[:, :, 3]
            img_rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
            has_alpha = True
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            has_alpha = False
            alpha = None

        # 증강 적용
        for aug_idx in range(num_augments):
            augmented = augment_image(
                img_rgb, transform, glare_p=glare_p, shadow_p=shadow_p
            )

            # 출력 파일명
            stem = img_path.stem
            if num_augments > 1:
                output_name = f"{stem}_aug{aug_idx:02d}.png"
            else:
                output_name = f"{stem}_aug.png"

            output_path = output_dir / output_name

            # RGB → BGR 변환 후 저장
            if has_alpha:
                # 알파 채널 복원
                augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                augmented_rgba = np.dstack([augmented_bgr, alpha])
                cv2.imwrite(str(output_path), augmented_rgba)
            else:
                augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), augmented_bgr)

            total_generated += 1

            if total_generated % 10 == 0:
                print(f"생성: {total_generated}개...")

    print(f"\n완료: {total_generated}개 생성됨")
    print(f"출력 경로: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="EKYC 이미지 증강")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="output/id_cards",
        help="입력 디렉토리 (기본값: output/id_cards)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output/augmented_id_cards",
        help="출력 디렉토리 (기본값: output/augmented_id_cards)",
    )
    parser.add_argument(
        "-n",
        "--num-augments",
        type=int,
        default=1,
        help="이미지당 증강 횟수 (기본값: 1)",
    )
    parser.add_argument(
        "--intensity",
        type=str,
        choices=["light", "medium", "heavy"],
        default="light",
        help="증강 강도 (기본값: light, E-KYC 환경에 적합)",
    )
    parser.add_argument(
        "--glare",
        type=float,
        default=0.15,
        help="글레어 확률 (기본값: 0.15)",
    )
    parser.add_argument(
        "--shadow",
        type=float,
        default=0.15,
        help="그림자 확률 (기본값: 0.15)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"입력 디렉토리가 존재하지 않음: {input_dir}")
        return

    process_images(
        input_dir=input_dir,
        output_dir=output_dir,
        num_augments=args.num_augments,
        intensity=args.intensity,
        glare_p=args.glare,
        shadow_p=args.shadow,
    )


if __name__ == "__main__":
    main()
