"""
주민등록증 합성 데이터 생성기
Kbank 블로그 방식 참고: TextRecognitionDataGenerator 활용

핵심 방식:
1. 템플릿 이미지에 필드별로 텍스트를 누적 합성
2. 각 필드마다 정확한 좌표(x, y)와 폰트 크기 지정
3. bbox 좌표와 텍스트를 레이블 파일로 저장
"""

import os
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from trdg.generators import GeneratorFromStrings
import json


class IDCardField:
    """주민등록증 필드 정의"""
    def __init__(
        self,
        field_id: str,
        x: int,
        y: int,
        font_size: int,
        font_type: str = "gothic",
        font_color: tuple = (0, 0, 0),
        alignment: str = "left",  # left, center
        max_width: int | None = None,
    ):
        self.field_id = field_id
        self.x = x
        self.y = y
        self.font_size = font_size
        self.font_type = font_type
        self.font_color = font_color
        self.alignment = alignment
        self.max_width = max_width


class IDCardGenerator:
    """주민등록증 합성 이미지 생성기"""

    def __init__(
        self,
        config_path: str,
        output_dir: str,
    ):
        self.config = self._load_config(config_path)
        self.base_dir = Path(config_path).parent.parent

        self.template_path = self.base_dir / self.config["template"]["path"]

        # 폰트 경로들 로드
        self.font_paths = {}
        for font_type, font_path in self.config["fonts"].items():
            self.font_paths[font_type] = self.base_dir / font_path

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # config에서 필드 정의 로드
        self.fields = {}
        for field_id, field_config in self.config["fields"].items():
            self.fields[field_id] = IDCardField(
                field_id=field_id,
                x=field_config["x"],
                y=field_config["y"],
                font_size=field_config["font_size"],
                font_type=field_config.get("font_type", "gothic"),
                font_color=tuple(field_config.get("font_color", [0, 0, 0])),
                alignment=field_config.get("alignment", "left"),
                max_width=field_config.get("max_width"),
            )

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def generate_fake_data(self) -> dict:
        """가상 데이터 생성"""
        # 이름 목록 (실제로는 txt 파일에서 로드)
        names = ["홍길동", "김철수", "이영희", "박민수", "최지연", "정대현", "강서윤", "조현우"]
        hanja_names = ["洪吉童", "金哲洙", "李英姬", "朴民秀", "崔智妍", "鄭大賢", "姜瑞潤", "趙賢宇"]

        # 주소 목록
        addresses = [
            "서울특별시 강남구 테헤란로 123",
            "서울특별시 서초구 반포대로 45",
            "경기도 성남시 분당구 판교로 256",
            "부산광역시 해운대구 센텀중앙로 78",
            "인천광역시 연수구 송도과학로 32",
        ]

        # 발급기관
        issuers = [
            "서울특별시 강남구청장",
            "서울특별시 서초구청장",
            "경기도 성남시 분당구청장",
            "부산광역시 해운대구청장",
            "인천광역시 연수구청장",
        ]

        idx = random.randint(0, len(names) - 1)
        name = names[idx]
        hanja = hanja_names[idx]

        # 주민번호 생성 (YYMMDD-GNNNNNN)
        year = random.randint(50, 99)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        gender = random.choice([1, 2, 3, 4])  # 1, 3: 남, 2, 4: 여
        seq = random.randint(0, 999999)
        jumin = f"{year:02d}{month:02d}{day:02d}-{gender}{seq:06d}"

        # 발급일자
        issue_year = random.randint(2000, 2024)
        issue_month = random.randint(1, 12)
        issue_day = random.randint(1, 28)
        issue_date = f"{issue_year}. {issue_month:02d}. {issue_day:02d}."

        return {
            "title": "주 민 등 록 증",
            "name": f"{name} ({hanja})",
            "jumin": jumin,
            "address": random.choice(addresses),
            "issue_date": issue_date,
            "issuer": random.choice(issuers),
        }

    def draw_text_on_image(
        self,
        image: Image.Image,
        text: str,
        field: IDCardField,
        font: ImageFont.FreeTypeFont,
    ) -> tuple[int, int, int, int]:
        """
        이미지에 텍스트를 그리고 bbox 반환

        Returns:
            (x1, y1, x2, y2) bbox 좌표
        """
        draw = ImageDraw.Draw(image)

        # 텍스트 크기 계산
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # 정렬에 따른 x 좌표 조정
        x = field.x
        if field.alignment == "center" and field.max_width:
            x = field.x + (field.max_width - text_width) // 2

        y = field.y

        # 텍스트 그리기
        draw.text((x, y), text, font=font, fill=field.font_color)

        # bbox 반환 (x1, y1, x2, y2)
        return (x, y, x + text_width, y + text_height)

    def generate_single(self, index: int, use_fixed_text: bool = False) -> tuple[str, str]:
        """
        단일 합성 이미지 생성

        Args:
            index: 이미지 인덱스
            use_fixed_text: True면 config의 고정 텍스트 사용, False면 랜덤 생성

        Returns:
            (image_path, label_path)
        """
        # 템플릿 로드
        template = Image.open(self.template_path).convert("RGB")

        # 가상 데이터 생성 또는 고정 텍스트 사용
        if use_fixed_text:
            data = {}
            for field_id, field_config in self.config["fields"].items():
                if "text" in field_config:
                    data[field_id] = field_config["text"]
        else:
            data = self.generate_fake_data()

        # 레이블 저장용 리스트
        labels = []

        # 각 필드별로 텍스트 합성 (누적)
        for field_id, field in self.fields.items():
            if field_id not in data:
                continue

            text = data[field_id]

            # 필드별 폰트 로드
            font_path = self.font_paths.get(field.font_type, self.font_paths["gothic"])
            font = ImageFont.truetype(str(font_path), field.font_size)

            # 텍스트 그리기
            x1, y1, x2, y2 = self.draw_text_on_image(template, text, field, font)

            # 레이블 추가 (Kbank 형식: x1,y1,x2,y1,x2,y2,x1,y2,text)
            # 4점 bbox 형식
            label_line = f"{x1},{y1},{x2},{y1},{x2},{y2},{x1},{y2},{text}"
            labels.append(label_line)

        # 이미지 저장
        image_filename = f"id_card_{index:06d}.jpg"
        image_path = self.output_dir / image_filename
        template.save(image_path, "JPEG", quality=95)

        # 레이블 저장
        label_filename = f"id_card_{index:06d}.txt"
        label_path = self.output_dir / label_filename
        with open(label_path, "w", encoding="utf-8") as f:
            for label in labels:
                f.write(label + "\n")

        return str(image_path), str(label_path)

    def generate_batch(self, count: int) -> list[tuple[str, str]]:
        """배치 생성"""
        results = []
        for i in range(count):
            result = self.generate_single(i)
            results.append(result)
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{count}")
        return results


def main():
    """테스트 실행"""
    base_dir = Path(__file__).parent.parent

    generator = IDCardGenerator(
        config_path=base_dir / "config" / "id_card_config.json",
        output_dir=base_dir / "output" / "trdg_test",
    )

    # Kbank 고정 텍스트로 1장 생성 (테스트용)
    print("주민등록증 합성 데이터 생성 시작 (Kbank 고정 텍스트)...")
    img_path, label_path = generator.generate_single(0, use_fixed_text=True)

    print(f"\n✅ 생성 완료:")
    print(f"  - {img_path}")


if __name__ == "__main__":
    main()
