import streamlit as st
import cv2
import numpy as np
import easyocr
from PIL import Image
import io
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
import ssl
import os

# SSL 보안 우회
ssl._create_default_https_context = ssl._create_unverified_context

st.set_page_config(page_title="이미지 텍스트 제거기", layout="centered")
st.title("🎯 맞춤형 PPT 생성기")

# --- 모델 로딩 최적화 ---
@st.cache_resource
def load_ocr_reader():
    # 모델 저장 위치를 현재 폴더로 지정하여 권한 문제 방지
    model_path = os.path.join(os.getcwd(), 'models')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    # 서버 환경 최적화 설정
    return easyocr.Reader(['ko', 'en'], gpu=False, model_storage_directory=model_path)

# PPT 생성 함수
def create_pptx(clean_image_array, ocr_results, width_px, height_px):
    prs = Presentation()
    prs.slide_width = Emu(width_px * 9525)
    prs.slide_height = Emu(height_px * 9525)
    
    # 1페이지
    slide1 = prs.slides.add_slide(prs.slide_layouts[6])
    img_pil = Image.fromarray(clean_image_array)
    img_io = io.BytesIO()
    img_pil.save(img_io, format='PNG')
    img_io.seek(0)
    slide1.shapes.add_picture(img_io, 0, 0, width=prs.slide_width, height=prs.slide_height)
    
    # 2페이지
    slide2 = prs.slides.add_slide(prs.slide_layouts[6])
    for (bbox, text, prob) in ocr_results:
        x, y = bbox[0][0], bbox[0][1]
        w_box, h_box = bbox[2][0] - bbox[0][0], bbox[2][1] - bbox[0][1]
        txBox = slide2.shapes.add_textbox(Emu(x * 9525), Emu(y * 9525), Emu(w_box * 9525), Emu(h_box * 9525))
        tf = txBox.text_frame
        p = tf.add_paragraph()
        p.text = text
        p.font.size = Pt(max(6, h_box * 0.75))
        p.font.bold = True if prob > 0.5 else False
    
    ppt_io = io.BytesIO()
    prs.save(ppt_io)
    ppt_io.seek(0)
    return ppt_io

# 메인 화면
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = uploaded_file.read()
    if st.button("🚀 분석 및 PPT 생성"):
        progress_text = st.empty() # 진행 상황 표시용
        
        try:
            progress_text.info("1단계: AI 모델 불러오는 중... (최초 실행 시 1~2분 소요)")
            reader = load_ocr_reader()
            
            progress_text.info("2단계: 이미지 분석 및 텍스트 추출 중...")
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            h, w, _ = img.shape
            results = reader.readtext(img)
            
            progress_text.info("3단계: 배경 복원(Inpainting) 진행 중...")
            mask = np.zeros(img.shape[:2], dtype="uint8")
            for (bbox, text, prob) in results:
                top_left = tuple(map(int, bbox[0]))
                bottom_right = tuple(map(int, bbox[2]))
                cv2.rectangle(mask, top_left, bottom_right, 255, -1)
            
            clean_img = cv2.inpaint(img, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
            clean_img_rgb = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
            
            progress_text.info("4단계: PPT 파일 구성 중...")
            ppt_file = create_pptx(clean_img_rgb, results, w, h)
            
            progress_text.empty()
            st.success("🎉 모든 작업이 완료되었습니다!")
            st.download_button("📊 맞춤형 PPTX 다운로드", data=ppt_file, file_name="result.pptx")
            st.image(clean_img_rgb, caption="텍스트 제거 결과")
            
        except Exception as e:
            st.error(f"❌ 오류 발생: {str(e)}")
            st.warning("로그(Manage app)에서 더 자세한 에러 내용을 확인해 주세요.")
