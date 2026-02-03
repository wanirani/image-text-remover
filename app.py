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

# 1. 보안 설정 및 페이지 기본 설정
ssl._create_default_https_context = ssl._create_unverified_context
st.set_page_config(page_title="이미지 텍스트 제거기", layout="wide")

# 2. OCR 리더기 캐싱 (메모리 절약형)
@st.cache_resource
def get_reader():
    # 모델 저장 폴더 생성
    model_dir = os.path.join(os.getcwd(), "ocr_models")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # GPU 없이 CPU 전용으로 가볍게 실행
    return easyocr.Reader(['ko', 'en'], gpu=False, model_storage_directory=model_dir)

# 3. PPT 생성 함수
def create_pptx(img_array, results, w_px, h_px):
    prs = Presentation()
    prs.slide_width = Emu(w_px * 9525)
    prs.slide_height = Emu(h_px * 9525)
    
    # Slide 1: Clean Image
    slide1 = prs.slides.add_slide(prs.slide_layouts[6])
    img_pil = Image.fromarray(img_array)
    img_io = io.BytesIO()
    img_pil.save(img_io, format='PNG')
    img_io.seek(0)
    slide1.shapes.add_picture(img_io, 0, 0, width=prs.slide_width, height=prs.slide_height)
    
    # Slide 2: Text Overlay
    slide2 = prs.slides.add_slide(prs.slide_layouts[6])
    for (bbox, text, prob) in results:
        x, y = bbox[0][0], bbox[0][1]
        wb, hb = bbox[2][0] - bbox[0][0], bbox[2][1] - bbox[0][1]
        txBox = slide2.shapes.add_textbox(Emu(x * 9525), Emu(y * 9525), Emu(wb * 9525), Emu(hb * 9525))
        tf = txBox.text_frame
        p = tf.add_paragraph()
        p.text = text
        p.font.size = Pt(max(6, hb * 0.75))
        p.font.bold = True if prob > 0.5 else False
        
    ppt_io = io.BytesIO()
    prs.save(ppt_io)
    ppt_io.seek(0)
    return ppt_io

# --- 메인 화면 구성 ---
st.title("🎯 이미지 텍스트 제거 및 PPT 변환")
st.write("이미지를 업로드하면 배경 복원 후 텍스트가 분리된 PPT를 만듭니다.")

# 4. 모델 사전 로드 (앱 시작 시 미리 로드)
with st.sidebar:
    st.header("⚙️ 시스템 상태")
    with st.spinner("AI 엔진 준비 중..."):
        try:
            reader = get_reader()
            st.success("AI 엔진 준비 완료!")
        except Exception as e:
            st.error(f"엔진 로드 실패: {e}")

uploaded_file = st.file_uploader("이미지 파일 선택", type=["jpg", "png", "jpeg"])

if uploaded_file and 'reader' in locals():
    img_bytes = uploaded_file.read()
    if st.button("🚀 변환 시작"):
        status = st.status("작업 진행 중...", expanded=True)
        try:
            # Step 1: Image Processing
            status.write("1. 이미지 분석 중...")
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            
            # Step 2: OCR
            status.write("2. 텍스트 추출 중 (OCR)...")
            results = reader.readtext(img)
            
            # Step 3: Inpainting
            status.write("3. 배경 복원 중 (Inpainting)...")
            mask = np.zeros((h, w), dtype=np.uint8)
            for (bbox, text, prob) in results:
                pts = np.array(bbox, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
            
            clean_img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
            clean_img_rgb = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
            
            # Step 4: PPT
            status.write("4. PPT 생성 중...")
            ppt_out = create_pptx(clean_img_rgb, results, w, h)
            
            status.update(label="✅ 모든 작업 완료!", state="complete", expanded=False)
            
            # 결과 다운로드 및 표시
            st.divider()
            st.download_button("📊 PPTX 결과물 다운로드", data=ppt_out, file_name="output.pptx")
            st.image(clean_img_rgb, caption="텍스트가 제거된 이미지", use_container_width=True)
            
        except Exception as e:
            status.update(label="❌ 오류 발생", state="error")
            st.error(f"상세 에러: {e}")
