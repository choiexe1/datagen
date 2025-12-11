"""
주민등록증 이미지 생성 파이프라인
- 얼굴 합성 + 텍스트 합성 통합
- 템플릿별 개별 설정 파일 지원
"""

import json
import random
from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np

from trdg import computer_text_generator
from src.fake_data import generate_single_record


class IDCardGenerator:
    """주민등록증 생성 파이프라인"""

    def __init__(self, config_dir: str = None):
        self.base_dir = Path(__file__).parent.parent  # src/ -> datagen/

        if config_dir is None:
            config_dir = self.base_dir / "config"
        self.config_dir = Path(config_dir)

        # assets 디렉토리
        self.assets_dir = self.base_dir / "assets"

        # 폰트 설정 로드
        self.fonts_config = self._load_json(self.config_dir / "fonts.json")
        self.font_paths = {}
        for font_type, font_path in self.fonts_config["fonts"].items():
            self.font_paths[font_type] = str(self.assets_dir / font_path)

        # 얼굴 설정 로드
        self.faces_config = self._load_json(self.config_dir / "faces.json")
        faces_dir = self.assets_dir / self.faces_config["dir"]
        self.male_faces = sorted(faces_dir.glob(f"{self.faces_config['male_prefix']}*"))
        self.female_faces = sorted(faces_dir.glob(f"{self.faces_config['female_prefix']}*"))

        # 템플릿 설정 로드 (개별 파일)
        self.templates = {}
        templates_config_dir = self.config_dir / "templates"
        for config_file in sorted(templates_config_dir.glob("template_*.json")):
            config = self._load_json(config_file)
            template_path = config["template_path"]
            self.templates[template_path] = config

        self.template_paths = sorted(self.templates.keys())

        # 출력 디렉토리
        self.output_dir = self.base_dir / "output" / "id_cards"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"템플릿: {len(self.templates)}개")
        print(f"폰트: {len(self.font_paths)}개")
        print(f"남성 얼굴: {len(self.male_faces)}개")
        print(f"여성 얼굴: {len(self.female_faces)}개")

    def _load_json(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_face_by_gender(self, is_male: bool) -> Path:
        """성별에 맞는 랜덤 얼굴 선택"""
        faces = self.male_faces if is_male else self.female_faces
        if not faces:
            faces = self.male_faces + self.female_faces
        return random.choice(faces)

    # ========== 얼굴 합성 ==========

    def _remove_white_background(self, img: Image.Image, threshold: int = 240) -> Image.Image:
        """흰색 배경을 투명하게 변환"""
        datas = img.getdata()
        new_data = []
        for item in datas:
            if len(item) == 4:
                r, g, b, a = item
            else:
                r, g, b = item
                a = 255
            if r > threshold and g > threshold and b > threshold:
                new_data.append((r, g, b, 0))
            else:
                new_data.append((r, g, b, a))
        img.putdata(new_data)
        return img

    def _apply_face_effects(
        self,
        img: Image.Image,
        effects: dict,
    ) -> Image.Image:
        """
        얼굴 이미지에 효과 적용

        Args:
            effects: {
                "blur": 0.9,      # 블러 강도
                "noise": 9,       # 노이즈 강도
                "aged": true,     # 오래된 사진 효과
                "contrast": 0.88, # 대비 (1.0이면 변화 없음)
                "saturation": 0.82  # 채도 (1.0이면 변화 없음)
            }
        """
        blur = effects.get("blur", 0)
        noise = effects.get("noise", 0)
        aged = effects.get("aged", False)
        contrast = effects.get("contrast", 1.0)
        saturation = effects.get("saturation", 1.0)

        # 블러 적용
        if blur > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=blur))

        img_array = np.array(img).astype(np.float32)

        # 노이즈 추가
        if noise > 0:
            noise_arr = np.random.randint(-noise, noise + 1, img_array.shape, dtype=np.int16)
            img_array = np.clip(img_array + noise_arr, 0, 255)

        # 대비 조정
        if contrast != 1.0:
            img_array = img_array * contrast + 128 * (1 - contrast)

        # 오래된 사진 효과 (황변)
        if aged:
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.05, 0, 255)  # R
            img_array[:, :, 1] = np.clip(img_array[:, :, 1] * 1.06, 0, 255)  # G
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 0.89, 0, 255)  # B

        # 채도 조정
        if saturation != 1.0:
            gray = np.mean(img_array[:, :, :3], axis=2, keepdims=True)
            img_array[:, :, :3] = img_array[:, :, :3] * saturation + gray * (1 - saturation)

        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    def composite_face(
        self,
        template_path: str,
        face_path: Path,
        template_config: dict,
    ) -> Image.Image:
        """템플릿에 얼굴 합성"""
        face_region = template_config["face_region"]
        face_effects = face_region.get("effects", {})

        # 템플릿 로드
        template = Image.open(self.assets_dir / template_path).convert("RGBA")

        # 얼굴 이미지 로드
        face = Image.open(face_path).convert("RGBA")

        # 얼굴 영역
        target_w = face_region["width"]
        target_h = face_region["height"]

        # 비율 유지하면서 리사이즈
        face_w, face_h = face.size
        ratio = min(target_w / face_w, target_h / face_h)
        new_w = int(face_w * ratio)
        new_h = int(face_h * ratio)

        face_resized = face.resize((new_w, new_h), Image.LANCZOS)
        face_resized = self._remove_white_background(face_resized)

        # 얼굴 효과 적용
        if face_effects:
            face_resized = self._apply_face_effects(face_resized, face_effects)

        # 회전
        rotation_angle = face_region.get("rotation", 0)
        if rotation_angle != 0:
            face_resized = face_resized.rotate(rotation_angle, expand=True, resample=Image.BICUBIC)
            new_w, new_h = face_resized.size

        # 위치 (중앙 정렬)
        face_x = face_region["x"] + (target_w - new_w) // 2
        face_y = face_region["y"] + (target_h - new_h) // 2

        # 합성
        template.paste(face_resized, (face_x, face_y), face_resized)
        return template

    # ========== 텍스트 합성 ==========

    def generate_text_image(
        self,
        text: str,
        font_path: str,
        font_size: int,
        text_color: str = "#000000",
        space_width: float = 1.0,
        stroke_width: int = 0,
        stroke_fill: str = "#000000",
    ) -> Image.Image:
        """trdg로 텍스트 이미지 생성"""
        result = computer_text_generator.generate(
            text=text,
            font=font_path,
            text_color=text_color,
            font_size=font_size,
            orientation=0,
            space_width=space_width,
            character_spacing=0,
            fit=True,
            word_split=False,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )
        img = result[0] if isinstance(result, tuple) else result
        img = img.convert("RGBA")

        white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        white_bg.paste(img, mask=img)
        return white_bg

    def _apply_text_effects(
        self,
        img: Image.Image,
        blur: float = 0,
        noise: int = 0,
        fade: float = 0,
    ) -> Image.Image:
        """텍스트 이미지에 효과 적용"""
        if blur > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=blur))

        img_array = np.array(img).astype(np.float32)

        if noise > 0:
            noise_arr = np.random.randint(-noise, noise + 1, img_array.shape, dtype=np.int16)
            img_array = np.clip(img_array + noise_arr, 0, 255)

        if fade > 0:
            fade_value = fade * 255
            img_array[:, :, :3] = img_array[:, :, :3] * (1 - fade) + fade_value * fade

        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    def _remove_text_white_background(self, img: Image.Image, threshold: int = 120) -> Image.Image:
        """흰색 배경을 투명하게"""
        datas = img.getdata()
        new_data = []
        for item in datas:
            if item[0] > threshold and item[1] > threshold and item[2] > threshold:
                new_data.append((255, 255, 255, 0))
            else:
                new_data.append(item)
        img.putdata(new_data)
        return img

    def _wrap_text(self, text: str, font_path: str, font_size: int, max_width: int) -> list[str]:
        """텍스트를 max_width에 맞게 줄바꿈"""
        from PIL import ImageFont

        font = ImageFont.truetype(font_path, font_size)
        lines = []
        current_line = ""

        for char in text:
            test_line = current_line + char
            bbox = font.getbbox(test_line)
            width = bbox[2] - bbox[0]

            if width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = char

        if current_line:
            lines.append(current_line)

        return lines

    def draw_text_on_image(
        self,
        image: Image.Image,
        text: str,
        field: dict,
        rotation: float = 0,
    ) -> tuple[int, int, int, int]:
        """이미지에 텍스트 합성"""
        font_type = field.get("font_type", "gothic")
        font_path = self.font_paths.get(font_type, list(self.font_paths.values())[0])
        font_size = field["font_size"]
        font_color = field.get("font_color", [0, 0, 0])
        space_width = field.get("space_width", 1.0)
        effects = field.get("effects", {})
        max_width = field.get("max_width", None)
        line_height = field.get("line_height", font_size + 5)
        stroke_width = field.get("stroke_width", 0)
        stroke_fill = field.get("stroke_fill", "#000000")

        r, g, b = font_color
        text_color = f"#{r:02x}{g:02x}{b:02x}"

        x = field["x"]
        y = field["y"]

        if max_width:
            lines = self._wrap_text(text, font_path, font_size, max_width)
        else:
            lines = [text]

        all_bboxes = []

        for i, line in enumerate(lines):
            line_y = y + (i * line_height)

            text_img = self.generate_text_image(
                text=line,
                font_path=font_path,
                font_size=font_size,
                text_color=text_color,
                space_width=space_width,
                stroke_width=stroke_width,
                stroke_fill=stroke_fill,
            )

            text_img = self._remove_text_white_background(text_img)

            if effects:
                text_img = self._apply_text_effects(
                    text_img,
                    blur=effects.get("blur", 0),
                    noise=effects.get("noise", 0),
                    fade=effects.get("fade", 0),
                )

            if rotation != 0:
                text_img = text_img.rotate(rotation, expand=True, resample=Image.BICUBIC)

            text_width, text_height = text_img.size
            image.paste(text_img, (x, line_y), text_img)
            all_bboxes.append((x, line_y, x + text_width, line_y + text_height))

        if all_bboxes:
            min_x = min(b[0] for b in all_bboxes)
            min_y = min(b[1] for b in all_bboxes)
            max_x = max(b[2] for b in all_bboxes)
            max_y = max(b[3] for b in all_bboxes)
            return (min_x, min_y, max_x, max_y)

        return (x, y, x, y)

    def composite_text(
        self,
        image: Image.Image,
        data: dict,
        template_config: dict,
    ) -> list[tuple]:
        """이미지에 텍스트 합성"""
        text_config = template_config["text"]
        fields = text_config["fields"]
        rotation = text_config.get("rotation", 0)

        labels = []

        for field_id, field in fields.items():
            if field_id not in data:
                continue

            text = data[field_id]
            x1, y1, x2, y2 = self.draw_text_on_image(image, text, field, rotation)
            labels.append((x1, y1, x2, y2, text))

        return labels

    # ========== 통합 생성 ==========

    def generate_single(
        self,
        index: int,
        template_path: str = None,
        is_male: bool = None,
        data: dict = None,
    ) -> tuple[str, dict]:
        """단일 주민등록증 생성"""
        # 템플릿 선택
        if template_path is None:
            template_path = random.choice(self.template_paths)

        template_config = self.templates[template_path]

        # 성별 결정
        if is_male is None:
            is_male = random.choice([True, False])

        # 얼굴 선택
        face_path = self.get_face_by_gender(is_male)

        # 데이터 생성
        if data is None:
            data = generate_single_record()

        # 출력 경로
        output_path = self.output_dir / f"id_card_{index:06d}.jpg"
        label_path = self.output_dir / f"id_card_{index:06d}.txt"

        # 1. 얼굴 합성
        image = self.composite_face(template_path, face_path, template_config)

        # 2. 텍스트 합성
        labels = self.composite_text(image, data, template_config)

        # 저장
        image_rgb = image.convert("RGB")
        image_rgb.save(output_path, "JPEG", quality=95)

        # 레이블 저장
        with open(label_path, "w", encoding="utf-8") as f:
            for x1, y1, x2, y2, text in labels:
                f.write(f"{x1},{y1},{x2},{y1},{x2},{y2},{x1},{y2},{text}\n")

        # 메타데이터
        metadata = {
            "index": index,
            "template": template_path,
            "face": face_path.name,
            "is_male": is_male,
            "data": data,
            "labels": labels,
        }

        return str(output_path), metadata

    def generate_batch(
        self,
        count: int,
        template_path: str = None,
    ) -> list[tuple[str, dict]]:
        """배치 생성"""
        results = []

        if template_path:
            for i in range(count):
                result = self.generate_single(i, template_path=template_path)
                results.append(result)
                if (i + 1) % 10 == 0:
                    print(f"생성: {i + 1}/{count}")
        else:
            total = len(self.template_paths) * count
            idx = 0
            for template in self.template_paths:
                for i in range(count):
                    result = self.generate_single(idx, template_path=template)
                    results.append(result)
                    idx += 1
                    if idx % 10 == 0:
                        print(f"생성: {idx}/{total}")

        print(f"✅ {len(results)}개 주민등록증 생성 완료")
        return results

    def generate_all_templates(self, count_per_template: int = 1) -> list[tuple[str, dict]]:
        """모든 템플릿에서 각각 count_per_template개씩 생성"""
        return self.generate_batch(count_per_template, template_path=None)
