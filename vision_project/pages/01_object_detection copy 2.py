import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

# 1. 페이지 설정
st.set_page_config(page_title="Dog vs Cat Detector", layout="centered")
st.title("Object Detection 체험하기")
st.markdown("## YOLO 모델을 활용한 개/고양이 구분")
st.info("지정된 모델 경로: `/run/best.pt`를 사용하여 예측을 수행합니다.")

# 2. 모델 로드 함수
@st.cache_resource
def load_custom_model():
    model_path = "run/best.pt" # 사용자께서 지정하신 경로
    
    # 파일 존재 여부 확인 (에러 방지용)
    if not os.path.exists(model_path):
        st.error(f"모델 파일을 찾을 수 없습니다: {model_path}")
        return None
    
    return YOLO(model_path)

model = load_custom_model()

# 3. UI 구성
upload_file = st.file_uploader(
    "개 또는 고양이 이미지를 업로드하세요", 
    type=["jpg", "jpeg", "png"]
)

pred_button = st.button(label="예측하기")

# 4. 예측 로직
if upload_file is not None:
    image = Image.open(upload_file)
    st.image(image, caption="업로드된 이미지", use_container_width=False)

    if pred_button:
        if model is not None:
            with st.spinner('모델이 열심히 분석 중입니다...'):
                # 모델 예측 (고양이/개 클래스 필터링 및 시각화 포함)
                # results[0].plot()이 모델이 '어디를 보고 판단했는지' 상자를 그려줍니다.
                results = model.predict(image, conf=0.25)
                res_plotted = results[0].plot() 
                # font_size=14를 적용하여 상자 위의 글자 크기를 조절합니다.
                # line_width를 함께 조절하면 박스 선 두께도 맞출 수 있습니다.
                res_plotted = results[0].plot(
                    font_size=12, 
                    line_width=2,
                    labels=True, # 라벨 표시 여부
                    conf=True    # 신뢰도(conf) 표시 여부
                )
                
                # 결과 이미지 변환 (BGR -> RGB)
                res_image = Image.fromarray(res_plotted[:, :, ::-1])

            st.success("분석 완료!")
            st.image(res_image, caption="판단 근거 (Bounding Box)", use_container_width=False)
            
            # 탐지된 객체 정보 출력
            for box in results[0].boxes:
                label = model.names[int(box.cls[0])]
                prob = float(box.conf[0])
                st.write(f"💡 결과: 이 동물은 **{prob:.1%}**의 확률로 **[{label}]**입니다.")
        else:
            st.warning("모델이 로드되지 않아 예측을 수행할 수 없습니다.")