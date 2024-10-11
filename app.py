import google.generativeai as genai 
import streamlit as st
from PIL import Image
import numpy as np
import requests
import xml.etree.ElementTree as ET
import json
from datetime import datetime
from ultralytics import YOLO  # YOLOv8 로드
from dotenv import load_dotenv
import os
from typing import Optional  # Optional 추가

# 환경 변수 로드
load_dotenv()
my_api_key = os.getenv('GENAI_API_KEY')
ttbkey = os.getenv('ALADIN_API_KEY')

# Gemini API 및 알라딘 API 설정
genai.configure(api_key=my_api_key)

# 알라딘 API에서 도서 정보를 가져오는 함수
def get_book_data_by_isbn(isbn: str) -> Optional[dict]:
    url = f"http://www.aladin.co.kr/ttb/api/ItemLookUp.aspx?ttbkey={ttbkey}&itemIdType=ISBN&ItemId={isbn}&output=xml&Version=20131101&OptResult=usedList"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # HTTP 에러 발생 시 예외 처리

        root = ET.fromstring(response.content)
        namespace = {'ns': 'http://www.aladin.co.kr/ttb/apiguide.aspx'}
        items = root.findall('.//ns:item', namespace)

        if not items:
            st.write("도서 정보를 찾을 수 없습니다.")
            return None

        item = items[0]  # 첫 번째 결과 사용
        book_data = {
            'title': item.find('ns:title', namespace).text or 'N/A',
            'author': item.find('ns:author', namespace).text or 'N/A',
            'pubDate': item.find('ns:pubDate', namespace).text or 'N/A',
            'description': item.find('ns:description', namespace).text or 'N/A',
            'isbn': item.find('ns:isbn', namespace).text or 'N/A',
            'isbn13': item.find('ns:isbn13', namespace).text or 'N/A',
            'priceSales': item.find('ns:priceSales', namespace).text or 'N/A',
            'priceStandard': item.find('ns:priceStandard', namespace).text or 'N/A',
            'publisher': item.find('ns:publisher', namespace).text or 'N/A',
            'cover': item.find('ns:cover', namespace).text or 'N/A',
            'salesPoint': item.find('ns:salesPoint', namespace).text or 'N/A',
            'customerReviewRank': item.find('ns:customerReviewRank', namespace).text or 'N/A'
        }
        return book_data
    except requests.RequestException as e:
        st.write(f"API 요청 중 오류가 발생했습니다: {e}")
        return None

# Gemini API를 사용해 책 설명을 생성하는 함수
def generate_book_description(book_data: dict) -> str:
    prompt = (
        f"책 제목: {book_data['title']}\n"
        f"저자: {book_data['author']}\n"
        f"출판일: {book_data['pubDate']}\n"
        f"책 설명: {book_data['description']}\n"
        f"판매 가격: {book_data['priceSales']}원\n"
        f"\n위 정보를 바탕으로 이 책에 대해 간단한 설명을 작성해 주세요."
    )

    model = genai.GenerativeModel('gemini-pro')  # Gemini 모델 사용
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(prompt)

    return response.text

# 도서 정보와 설명을 JSON 파일로 저장하는 함수
def save_description_to_json(book_data: dict, book_description: str) -> None:
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"book_description_{current_time}.json"

    book_data["generated_description"] = book_description

    with open(file_name, "w", encoding="utf-8") as file:
        json.dump(book_data, file, ensure_ascii=False, indent=4)
    print(f"설명과 책 정보가 {file_name} 파일에 저장되었습니다.")

# YOLOv8 모델 로드 함수
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

yolo_model = load_yolo_model()

# YOLOv8 객체 탐지 함수
def yoloinf(image: Image) -> Image:
    image = np.array(image)
    results = yolo_model(image)
    annotated_img = results[0].plot()  # 바운딩 박스 그리기
    return Image.fromarray(annotated_img[..., ::-1])  # RGB 포맷으로 변환하여 반환

# Gemini API를 사용하여 품질 평가 생성 함수
def generate_quality_evaluation(front_cover: Image, back_cover: Image, spine: Image, page_edges: Image) -> str:
    prompt = (
        f"위 정보를 바탕으로 책의 품질 등급을 최상, 상, 중, 매입불가 중 하나로 평가하고, 그 이유를 설명해주세요.\n"
        f"도서 품질 평가 기준:\n"
        f"- 최상: 새것에 가까운 책, 변색 없음, 찢어진 흔적 없음, 닳은 흔적 없음, 낙서 없음.\n"
        f"- 상: 약간의 사용감은 있으나 깨끗한 책, 희미한 변색, 작은 얼룩, 찢어진 흔적 없음, 약간의 모서리 해짐.\n"
        f"- 중: 전체적인 변색, 2cm 이하의 찢어짐, 오염 있음, 낙서 있음.\n"
        f"- 매입불가: 2cm 초과한 찢어짐, 심한 오염 및 낙서, 물에 젖은 흔적.\n"
    )

    model = genai.GenerativeModel('gemini-pro')
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(prompt)

    return response.text

# main() 함수로 코드를 구조화
def main():
    st.title("Gemini-Bot")

    col1, col2 = st.columns(2)

    # 책 정보 조회
    with col1:
        if st.button("📚 책 정보 조회", key="book_info_button"):
            st.session_state["show_isbn_input"] = True
            st.session_state["show_upload"] = False

    # 판매 등급 판정
    with col2:
        if st.button("⭐ 판매 등급 판정", key="grade_button"):
            st.session_state["show_upload"] = True
            st.session_state["show_isbn_input"] = False

    # ISBN 입력 필드
    if st.session_state.get("show_isbn_input", False):
        isbn = st.text_input("ISBN을 입력하세요")
        if isbn:
            book_data = get_book_data_by_isbn(isbn)
            if book_data:
                book_description = generate_book_description(book_data)
                st.write("생성된 설명:", book_description)
                save_description_to_json(book_data, book_description)

    # 이미지 업로드 필드
    if st.session_state.get("show_upload", False):
        st.write("판매 등급 판정을 위해 4장의 이미지를 업로드하세요.")
        front_cover = st.file_uploader("앞표지", type=["jpg", "png", "jpeg"])
        back_cover = st.file_uploader("뒷표지", type=["jpg", "png", "jpeg"])
        spine = st.file_uploader("책등", type=["jpg", "png", "jpeg"])
        page_edges = st.file_uploader("책배", type=["jpg", "png", "jpeg"])

        if front_cover and back_cover and spine and page_edges:
            st.success("이미지가 모두 업로드되었습니다.")
            detected_front = yoloinf(Image.open(front_cover))
            detected_back = yoloinf(Image.open(back_cover))
            detected_spine = yoloinf(Image.open(spine))
            detected_page_edges = yoloinf(Image.open(page_edges))

            st.image(detected_front, caption="앞표지 탐지 결과")
            st.image(detected_back, caption="뒷표지 탐지 결과")
            st.image(detected_spine, caption="책등 탐지 결과")
            st.image(detected_page_edges, caption="책배 탐지 결과")

            st.write("Gemini 품질 평가 요청 중...")
            quality_reason = generate_quality_evaluation(front_cover, back_cover, spine, page_edges)
            st.write("품질 평가 결과:", quality_reason)
        else:
            st.warning("4장의 이미지를 모두 업로드해 주세요.")

# main 함수 실행
if __name__ == "__main__":
    main()
