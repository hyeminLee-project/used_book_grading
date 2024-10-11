# 📚 Gemini-Bot: 중고 서적 상태 자동 평가 시스템

## 🛠 프로젝트 소개
**Gemini-Bot**은 YOLOv8 모델과 알라딘 API, Gemini API를 사용하여 중고 서적의 상태를 자동으로 평가하고, ISBN을 기반으로 책 정보를 조회하는 시스템입니다. 이 시스템은 AI 모델을 통해 책의 손상 상태를 자동 감지하고, 판매 등급을 판정하여 책의 품질에 대한 설명을 제공합니다.

## 📋 주요 기능

### 도서 정보 조회:
- 사용자가 ISBN을 입력하면, 알라딘 API를 통해 도서 정보를 가져옵니다.
- 책의 저자, 제목, 출판일, 서평 등을 자동으로 조회합니다.

### 도서 품질 평가:
- 현재 YOLOv8 모델을 통해 이미지를 분석하여 책을 탐지하는 기능이 구현되어 있습니다. 손상 상태 분석은 Roboflow를 사용하여 데이터에 주석을 다는 작업이 진행 중이며, 추후에 추가될 예정입니다.

- 앞표지, 뒷표지, 책등, 책배의 상태를 바탕으로 품질 등급을 자동으로 판정합니다(예정).

### AI 설명 생성:
- Gemini API를 사용하여 도서 정보 및 상태 평가 결과를 바탕으로 품질 설명을 자동 생성합니다.

📄 API 참고
알라딘 API: 도서 정보 조회를 위한 API입니다. 알라딘 API 가이드를 참조하세요.
(https://docs.google.com/document/d/1mX-WxuoGs8Hy-QalhHcvuV17n50uGI2Sg_GHofgiePE/edit)
Gemini API: AI 설명 생성을 위한 API입니다. Gemini API 문서를 확인하세요.
(https://ai.google.dev/gemini-api/docs?gad_source=1&gclid=CjwKCAjwmaO4BhAhEiwA5p4YL4Ut31B328H6AuFrlCmu12n3oIofU7CPj4Vo_5dH7OBMXDv39wyuixoCWGkQAvD_BwE&hl=ko)

📝 라이선스
이 프로젝트는 MIT 라이선스를 따릅니다. 자유롭게 수정 및 배포가 가능합니다.

🚀 향후 계획
추가적인 이미지 손상 유형 학습
챗봇 기능 강화 (Dialogflow 사용)

## 📦 설치 방법

### 코드 클론
```bash
git clone https://github.com/hyeminLee-project/used_book_grading.git
cd used_book_grading

# 📚 Gemini-Bot: Automated Book Condition Evaluation System

## 🛠 Project Overview
**Gemini-Bot** uses the YOLOv8 model, Aladin API, and Gemini API to automatically evaluate the condition of used books and retrieve book information based on ISBN. The system automatically detects the damage to a book using AI models, assigns a sales grade, and provides a detailed explanation of the book's condition.

## 📋 Key Features

### Book Information Retrieval:
- When the user inputs an ISBN, the system retrieves book information via the Aladin API.
- Automatically fetches the book's title, author, publication date, and reviews.

### Book Condition Evaluation:
- Upload images of the front cover, back cover, spine, and page edges, and then click the ⭐ Evaluate Condition button. Currently, the system detects books in the images using the YOLOv8 model. Full damage detection will be added in the future after further data annotation

- Automatically assigns a quality grade based on the condition of the front cover, back cover, spine, and page edges.

### AI Description Generation:
- The Gemini API generates a quality description based on the book's information and condition evaluation.

## 📄 API References
- **Aladin API**: This API is used for retrieving book information. Refer to the [Aladin API Guide](https://docs.google.com/document/d/1mX-WxuoGs8Hy-QalhHcvuV17n50uGI2Sg_GHofgiePE/edit).
- **Gemini API**: This API is used for generating AI descriptions. Check the [Gemini API Documentation](https://ai.google.dev/gemini-api/docs?gad_source=1&gclid=CjwKCAjwmaO4BhAhEiwA5p4YL4Ut31B328H6AuFrlCmu12n3oIofU7CPj4Vo_5dH7OBMXDv39wyuixoCWGkQAvD_BwE&hl=ko).

## 📝 License
This project is licensed under the MIT License. You are free to modify and distribute it.

## 🚀 Future Plans
- Additional training for image damage types
- Enhancing chatbot functionality (using Dialogflow)


## 📦 Installation

### Clone the Repository
```bash
git clone https://github.com/hyeminLee-project/used_book_grading.git
cd used_book_grading
