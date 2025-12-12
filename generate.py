#!/usr/bin/env python
"""주민등록증 생성 CLI"""

import argparse
from pathlib import Path
from src.generator import IDCardGenerator


def main():
    parser = argparse.ArgumentParser(description="주민등록증 이미지 생성")
    parser.add_argument(
        "-t", "--template",
        type=str,
        help="템플릿 번호 (예: 01, 02, 12) 또는 전체 경로. 미지정시 모든 템플릿 사용",
    )
    parser.add_argument(
        "-n", "--count",
        type=int,
        default=1,
        help="생성할 이미지 수 (기본값: 1)",
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="사용 가능한 템플릿 목록 출력",
    )
    parser.add_argument(
        "-r", "--random",
        action="store_true",
        help="템플릿 랜덤 분배 (총 n개 생성, 템플릿 랜덤 선택)",
    )

    args = parser.parse_args()

    generator = IDCardGenerator()

    # 템플릿 목록 출력
    if args.list:
        print("\n사용 가능한 템플릿:")
        for i, tp in enumerate(generator.template_paths, 1):
            print(f"  {i:2d}. {tp}")
        return

    # 템플릿 경로 결정
    template_path = None
    if args.template:
        if args.template.isdigit() or (len(args.template) == 2 and args.template.isdigit()):
            num = args.template.zfill(2)
            template_path = f"templates/id_template_empty_{num}.png"
        else:
            template_path = args.template

        if template_path not in generator.templates:
            print(f"❌ 템플릿을 찾을 수 없음: {template_path}")
            print("\n사용 가능한 템플릿:")
            for tp in generator.template_paths:
                print(f"  - {tp}")
            return

    # 생성
    print("\n주민등록증 생성 시작...")

    if template_path:
        results = generator.generate_batch(args.count, template_path=template_path)
    elif args.random:
        # 랜덤 분배: 총 n개, 템플릿 랜덤 선택
        results = []
        for i in range(args.count):
            result = generator.generate_single(i)  # template_path=None이면 랜덤
            results.append(result)
            if (i + 1) % 10 == 0:
                print(f"생성: {i + 1}/{args.count}")
        print(f"✅ {len(results)}개 주민등록증 생성 완료")
    else:
        results = generator.generate_all_templates(count_per_template=args.count)

    print(f"\n생성된 파일:")
    for img_path, meta in results:
        print(f"  - {Path(img_path).name}")
        print(f"    템플릿: {meta['template']}")
        print(f"    얼굴: {meta['face']}")
        print(f"    이름: {meta['data']['name']}")


if __name__ == "__main__":
    main()
