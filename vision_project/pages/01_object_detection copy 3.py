import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np

# 1. 페이지 설정 및 테마
st.set_page_config(page_title="AI 개/고양이 판독기", layout="wide")

# 커스텀 CSS로 스타일링 (글자 가독성 및 UI 정돈)
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #FF4B4B; color: white; }
    .stMetric { background-color: #999; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# 2. 모델 로드 (캐싱)
@st.cache_resource
def load_custom_model():
    model_path = "run/best.pt" 
    if os.path.exists(model_path):
        return YOLO(model_path)
    return None

model = load_custom_model()

# 3. 사이드바 - 설정 영역
with st.sidebar:
    st.header("⚙️ 설정")
    conf_threshold = st.slider("신뢰도 임계값 (Confidence)", 0.0, 1.0, 0.25, 0.05)
    st.info("임계값이 높을수록 확실한 결과만 표시합니다.")
    st.divider()
    st.markdown("### 모델 정보")
    st.write("📂 Path: `/run/best.pt` 가중치 사용")

# 4. 메인 화면 구성
st.title("🐾 Object Detection 체험하기")
st.subheader("YOLO 모델을 활용한 개/고양이 구분 서비스")

upload_file = st.file_uploader(
    "분석하고 싶은 개 또는 고양이 이미지를 업로드하세요", 
    type=["jpg", "jpeg", "png"]
)

st.divider()

# 5. 예측 및 결과 시각화 로직
if upload_file is not None:
    col1, col2 = st.columns(2)
    
    # 원본 이미지 로드
    original_img = Image.open(upload_file).convert("RGB")
    
    with col1:
        st.markdown("### 🖼️ 원본 이미지")
        st.image(original_img, use_container_width=False)

    # 예측 버튼
    if st.button(label="🔍 이미지 분석 시작"):
        if model is not None:
            with st.spinner('AI가 이미지를 분석하고 있습니다...'):
                # 모델 예측
                results = model.predict(original_img, conf=conf_threshold)
                
                # 시각화용 이미지 생성 (PIL 사용)
                draw_img = original_img.copy()
                draw = ImageDraw.Draw(draw_img)
                
                # 폰트 설정 (폰트 크기 14 강제 고정)
                try:
                    # Windows: 'arial.ttf', Linux: '/usr/share/fonts/...'
                    font = ImageFont.truetype("arial.ttf", 14)
                except:
                    font = ImageFont.load_default()

                # 결과 그리기 루프
                detections = []
                for result in results:
                    for box in result.boxes:
                        # 좌표 및 정보 추출
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cls_id = int(box.cls[0])
                        conf_val = float(box.conf[0])
                        label_name = model.names[cls_id]
                        
                        # 색상 지정 (고양이: 파랑, 개: 주황)
                        color = "#3498db" if label_name == "cat" else "#e67e22"
                        
                        # 1. 바운딩 박스 그리기
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                        
                        # 2. 라벨 텍스트 배경 그리기
                        display_text = f"{label_name} {conf_val:.2f}"
                        # textbbox를 사용하여 텍스트 영역 계산
                        t_x1, t_y1, t_x2, t_y2 = draw.textbbox((x1, y1), display_text, font=font)
                        draw.rectangle([t_x1, t_y1 - 2, t_x2 + 4, t_y2 + 2], fill=color)
                        
                        # 3. 텍스트 그리기 (폰트 사이즈 14 고정 적용)
                        draw.text((x1 + 2, y1 - 2), display_text, fill="white", font=font)
                        
                        detections.append((label_name, conf_val))

            with col2:
                st.markdown("### 🎯 분석 결과")
                st.image(draw_img, use_container_width=False)

            # 하단 요약 리포트
            st.divider()
            if detections:
                st.success(f"총 {len(detections)}마리의 동물을 탐지했습니다!")
                res_cols = st.columns(len(detections) if len(detections) < 4 else 4)
                for i, (name, prob) in enumerate(detections):
                    with res_cols[i % 4]:
                        st.metric(label=f"탐지된 {i+1}", value=name.upper(), delta=f"{prob:.1%}")
            else:
                st.warning("이미지에서 개나 고양이를 찾지 못했습니다. 신뢰도 임계값을 조절해 보세요.")
        else:
            st.error("모델 파일을 로드하지 못했습니다. 경로를 확인해 주세요.")