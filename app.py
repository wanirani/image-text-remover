import streamlit as st
import cv2
import numpy as np
import easyocr
from PIL import Image
import io
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
import ssl

# SSL 인증서 보안 검사 우회 (서버 환경 필수)
ssl._create_default_https_context = ssl._create_unverified_context

# --- [추가] 모델 로딩 최적화: 한 번 로드하면 메모리에 고정 ---
@st.cache_resource
def load_ocr_reader():
    # cpu=True (gpu=False) 설정을 명시하여 서버 환경 오류 방지
    return easyocr.Reader(['ko', 'en'], gpu=False)

def process_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w, _ = img.shape
    
    # 캐시된 리더기 사용
    reader = load_ocr_reader()
    results = reader.readtext(img)
    
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for (bbox, text, prob) in results:
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))
        cv2.rectangle(mask, top_left, bottom_right, 255, -1)
    
    clean_img = cv2.inpaint(img, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    clean_img_rgb = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
    
    return clean_img_rgb, results, w, h

# (create_pptx 함수는 이전과 동일합니다)
def create_pptx(clean_image_array, ocr_results, width_px, height_px):
    prs = Presentation()
    prs.slide_width = Emu(width_px * 9525)
    prs.slide_height = Emu(height_px * 9525)
    
    slide1 = prs.slides.add_slide(prs.slide_layouts[6])
    img_pil = Image.fromarray(clean_image_array)
    img_io = io.BytesIO()
    img_pil.save(img_io, format='PNG')
    img_io.seek(0)
    slide1.shapes.add_picture(img_io, 0, 0, width=prs.slide_width, height=prs.slide_height)
    
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

# --- UI 부분 ---
st.set_page_config(page_title="고급 이미지-PPT 변환기", layout="centered")
st.title("🎯 맞춤형 PPT 생성기")

uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = uploaded_file.read()
    if st.button("🚀 분석 및 PPT 생성"):
        with st.spinner('AI 모델을 준비하고 이미지를 분석 중입니다... (처음엔 조금 오래 걸려요!)'):
            try:
                clean_img, ocr_results, w, h = process_image(file_bytes)
                ppt_file = create_pptx(clean_img, ocr_results, w, h)
                
                st.success("완료!")
                st.download_button("📊 PPTX 다운로드", data=ppt_file, file_name="result.pptx")
                st.image(clean_img, caption="텍스트 제거 결과")
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")
