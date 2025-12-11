"""
합성 신분증 데이터 생성기
이름, 한자이름, 주소, 주민등록번호, 발급일, 발급청 랜덤 생성
"""

import random
import json
from pathlib import Path
from PIL import ImageFont


# 이름 데이터 (성 + 이름 조합)
LAST_NAMES = [
    ("김", "金"), ("이", "李"), ("박", "朴"), ("최", "崔"), ("정", "鄭"),
    ("강", "姜"), ("조", "趙"), ("윤", "尹"), ("장", "張"), ("임", "林"),
    ("한", "韓"), ("오", "吳"), ("서", "徐"), ("신", "申"), ("권", "權"),
    ("황", "黃"), ("안", "安"), ("송", "宋"), ("류", "柳"), ("홍", "洪"),
    ("전", "全"), ("고", "高"), ("문", "文"), ("양", "梁"), ("손", "孫"),
    ("배", "裴"), ("백", "白"), ("허", "許"), ("유", "劉"), ("남", "南"),
    ("심", "沈"), ("노", "盧"), ("하", "河"), ("곽", "郭"), ("성", "成"),
    ("차", "車"), ("주", "朱"), ("우", "禹"), ("구", "具"), ("민", "閔"),
    ("진", "陳"), ("나", "羅"), ("지", "池"), ("엄", "嚴"), ("채", "蔡"),
    ("원", "元"), ("천", "千"), ("방", "方"), ("공", "孔"), ("현", "玄"),
    ("함", "咸"), ("변", "卞"), ("염", "廉"), ("여", "呂"), ("추", "秋"),
    ("도", "都"), ("석", "石"), ("선", "宣"), ("설", "薛"), ("마", "馬"),
    ("길", "吉"), ("연", "延"), ("위", "魏"), ("표", "表"), ("명", "明"),
    ("기", "奇"), ("반", "潘"), ("왕", "王"), ("금", "琴"), ("옥", "玉"),
    ("육", "陸"), ("인", "印"), ("맹", "孟"), ("제", "諸"), ("모", "毛"),
]

FIRST_NAMES = [
    # 남자 이름
    ("민준", "民俊"), ("서준", "瑞俊"), ("도윤", "道允"), ("예준", "禮俊"), ("시우", "時宇"),
    ("하준", "夏俊"), ("주원", "周元"), ("지호", "志浩"), ("지후", "志厚"), ("준서", "俊瑞"),
    ("현우", "賢宇"), ("준혁", "俊赫"), ("도현", "道賢"), ("건우", "建宇"), ("우진", "宇鎭"),
    ("성민", "成民"), ("정훈", "正勳"), ("철수", "哲洙"), ("영수", "永洙"), ("동현", "東賢"),
    ("승현", "承賢"), ("태현", "泰賢"), ("정우", "正宇"), ("승민", "承民"), ("유준", "裕俊"),
    ("민성", "民成"), ("지훈", "志勳"), ("승호", "承浩"), ("진우", "鎭宇"), ("민재", "民宰"),
    ("현준", "賢俊"), ("재원", "宰元"), ("한결", "韓結"), ("윤호", "允浩"), ("시현", "時賢"),
    ("민혁", "民赫"), ("정현", "正賢"), ("준영", "俊英"), ("상현", "尙賢"), ("기현", "基賢"),
    ("성준", "成俊"), ("형준", "亨俊"), ("상우", "尙宇"), ("재민", "宰民"), ("동욱", "東旭"),
    ("광수", "光洙"), ("영호", "永浩"), ("상철", "尙哲"), ("병철", "炳哲"), ("경수", "景洙"),
    # 여자 이름
    ("서연", "瑞妍"), ("서윤", "瑞潤"), ("지우", "智宇"), ("서현", "瑞賢"), ("민서", "民瑞"),
    ("하은", "夏恩"), ("하윤", "夏潤"), ("윤서", "允瑞"), ("지민", "智敏"), ("지유", "智裕"),
    ("수빈", "秀彬"), ("예은", "禮恩"), ("수아", "秀雅"), ("다은", "多恩"), ("채원", "彩媛"),
    ("영희", "英姬"), ("미영", "美英"), ("지영", "智英"), ("수진", "秀珍"), ("미정", "美貞"),
    ("혜진", "惠珍"), ("지현", "智賢"), ("유진", "裕珍"), ("소연", "昭妍"), ("하영", "河英"),
    ("민지", "旼智"), ("수연", "秀娟"), ("지원", "志媛"), ("은지", "恩智"), ("서영", "瑞英"),
    ("예진", "藝珍"), ("소희", "昭熙"), ("유나", "侑娜"), ("민아", "旼雅"), ("정은", "貞恩"),
    ("수현", "秀賢"), ("예지", "藝智"), ("소영", "昭英"), ("미선", "美善"), ("경희", "敬姬"),
    ("순자", "順子"), ("옥순", "玉順"), ("영자", "英子"), ("정숙", "貞淑"), ("명숙", "明淑"),
    ("춘자", "春子"), ("말순", "末順"), ("복순", "福順"), ("금자", "今子"), ("점순", "點順"),
]

# 주소 데이터 (시/도, 구/군, 상세주소)
CITIES = [
    "서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시",
    "대전광역시", "울산광역시", "세종특별자치시", "경기도", "강원도",
]

DISTRICTS = {
    "서울특별시": ["강남구", "강동구", "강북구", "강서구", "관악구", "광진구", "구로구", "금천구", "노원구", "도봉구", "동대문구", "동작구", "마포구", "서대문구", "서초구", "성동구", "성북구", "송파구", "양천구", "영등포구", "용산구", "은평구", "종로구", "중구", "중랑구"],
    "부산광역시": ["강서구", "금정구", "남구", "동구", "동래구", "부산진구", "북구", "사상구", "사하구", "서구", "수영구", "연제구", "영도구", "중구", "해운대구"],
    "대구광역시": ["남구", "달서구", "동구", "북구", "서구", "수성구", "중구", "달성군"],
    "인천광역시": ["계양구", "남동구", "동구", "미추홀구", "부평구", "서구", "연수구", "중구", "강화군", "옹진군"],
    "광주광역시": ["광산구", "남구", "동구", "북구", "서구"],
    "대전광역시": ["대덕구", "동구", "서구", "유성구", "중구"],
    "울산광역시": ["남구", "동구", "북구", "중구", "울주군"],
    "세종특별자치시": ["세종시"],
    "경기도": ["수원시", "성남시", "고양시", "용인시", "부천시", "안산시", "안양시", "남양주시", "화성시", "평택시", "의정부시", "시흥시", "파주시", "김포시", "광명시", "광주시", "군포시", "이천시", "오산시", "하남시"],
    "강원도": ["춘천시", "원주시", "강릉시", "동해시", "태백시", "속초시", "삼척시"],
}

ROAD_NAMES = [
    "테헤란로", "강남대로", "역삼로", "삼성로", "봉은사로", "선릉로", "논현로", "학동로",
    "중앙로", "대학로", "문화로", "평화로", "자유로", "통일로", "민주로", "번영로",
    "해운대로", "광안리로", "센텀로", "마린시티로", "수영로", "남천로",
    "동성로", "중앙대로", "달구벌대로", "신천대로", "앞산로",
    "부평대로", "경인로", "연수로", "청라로", "송도대로",
]

DONG_NAMES = [
    "역삼동", "삼성동", "대치동", "논현동", "청담동", "압구정동", "신사동",
    "해운대동", "우동", "중동", "좌동", "송정동", "재송동",
    "수성동", "범어동", "만촌동", "황금동", "두산동",
    "부평동", "산곡동", "청천동", "갈산동", "삼산동",
    "문화동", "목동", "신정동", "등촌동", "화곡동",
]

APT_NAMES = [
    "래미안", "자이", "푸르지오", "힐스테이트", "롯데캐슬", "아이파크", "더샵",
    "e편한세상", "SK뷰", "포스코더샵", "현대아이파크", "대림e편한세상",
]

# 주소 영역 설정 (픽셀 기준)
ADDRESS_MAX_WIDTH = 560  # 최대 너비
ADDRESS_FONT_SIZE = 48
BASE_DIR = Path(__file__).parent.parent
ADDRESS_FONT_PATH = str(BASE_DIR / "fonts" / "Pretendard-Regular.ttf")

# 폰트 로드 (전역)
try:
    ADDRESS_FONT = ImageFont.truetype(ADDRESS_FONT_PATH, ADDRESS_FONT_SIZE)
except:
    ADDRESS_FONT = None


def measure_text_width(text: str) -> int:
    """텍스트 너비 측정 (픽셀)"""
    if ADDRESS_FONT is None:
        return len(text) * 25  # 폰트 로드 실패시 추정값
    bbox = ADDRESS_FONT.getbbox(text)
    return bbox[2] - bbox[0]


def generate_name() -> tuple[str, str]:
    """이름과 한자이름 생성"""
    last = random.choice(LAST_NAMES)
    first = random.choice(FIRST_NAMES)
    name = last[0] + first[0]
    hanja = last[1] + first[1]
    return name, f"({hanja})"


def wrap_address_lines(parts: list[str], max_width: int = ADDRESS_MAX_WIDTH) -> list[str]:
    """주소 파트들을 max_width에 맞게 줄바꿈 (최대 3줄)"""
    lines = []
    current_line = ""

    for part in parts:
        if not current_line:
            test_line = part
        else:
            test_line = f"{current_line} {part}"

        if measure_text_width(test_line) <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = part

    if current_line:
        lines.append(current_line)

    # 최대 3줄까지만 반환
    return lines[:3]


def generate_address() -> tuple[str, str, str]:
    """주소 생성 (전체 주소 문자열, 시, 구 반환) - 2~3줄 분량"""
    city = random.choice(CITIES)
    district = random.choice(DISTRICTS[city])

    # 2줄 또는 3줄 랜덤 선택
    if random.random() < 0.5:
        # 짧은 버전 (2줄) - 도로명 짧게
        road_num = random.randint(1, 50)
        gil_num = random.randint(1, 30)
        unit = random.randint(101, 999)
        full_address = f"{city} {district} {road_num}번길 {gil_num}, {unit}호"
    else:
        # 긴 버전 (3줄) - 도로명 + 아파트명
        road = random.choice(ROAD_NAMES)
        road_num = random.randint(1, 200)
        dong = random.choice(DONG_NAMES)
        apt = random.choice(APT_NAMES)
        bldg = random.randint(101, 115)
        unit = random.randint(101, 1500)
        gil_num = random.randint(1, 99)
        full_address = f"{city} {district} {road}{road_num}번길 {gil_num}, {bldg}동 {unit}호 ({dong}, {apt})"

    return full_address, city, district


def generate_jumin() -> str:
    """주민등록번호 생성 (공백 포함)"""
    year = random.randint(50, 99)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    gender = random.choice([1, 2, 3, 4])
    seq = random.randint(0, 999999)

    # 숫자 사이에 공백 추가 (space_width 적용용)
    front = f"{year:02d}{month:02d}{day:02d}"
    back = f"{gender}{seq:06d}"

    # 글자 사이 공백 추가
    front_spaced = " ".join(front)
    back_spaced = " ".join(back)

    return f"{front_spaced} - {back_spaced}"


def generate_issue_date() -> str:
    """발급일자 생성"""
    year = random.randint(2000, 2024)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return f"{year}.  {month}.  {day}."


def generate_issuer(city: str, district: str) -> str:
    """발급청 생성"""
    if city == "세종특별자치시":
        return "세종특별자치시장"
    elif "특별시" in city or "광역시" in city:
        return f"{city} {district}청장"
    else:
        return f"{city} {district}장"


def generate_single_record() -> dict:
    """단일 레코드 생성"""
    name, hanja = generate_name()
    address, city, district = generate_address()

    record = {
        "name": name,
        "name_hanja": hanja,
        "address": address,
        "jumin": generate_jumin(),
        "issue_date": generate_issue_date(),
        "issuer": generate_issuer(city, district),
    }

    return record


def generate_dataset(count: int, output_path: str):
    """데이터셋 생성"""
    records = []
    for _ in range(count):
        records.append(generate_single_record())

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"✅ {count}개 레코드 생성 완료: {output_path}")
    return records


def generate_txt_dataset(count: int, output_dir: str):
    """개별 txt 파일로 데이터셋 생성"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for i in range(count):
        record = generate_single_record()

        txt_path = output_path / f"record_{i:06d}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"name={record['name']}\n")
            f.write(f"name_hanja={record['name_hanja']}\n")
            f.write(f"address={record['address']}\n")
            f.write(f"jumin={record['jumin']}\n")
            f.write(f"issue_date={record['issue_date']}\n")
            f.write(f"issuer={record['issuer']}\n")

    print(f"✅ {count}개 txt 파일 생성 완료: {output_dir}")


if __name__ == "__main__":
    base_dir = Path(__file__).parent

    # JSON으로 1000개 생성
    generate_dataset(1000, str(base_dir / "id_card_data.json"))

    # 샘플 출력
    print("\n샘플 데이터:")
    for i in range(3):
        record = generate_single_record()
        print(f"\n--- 레코드 {i+1} ---")
        for k, v in record.items():
            print(f"  {k}: {v}")
