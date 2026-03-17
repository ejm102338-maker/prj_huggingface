import streamlit as st
from ultralytics import YOLO
from PIL import Image,ImageDraw,ImageFont

# 페이지 목표
# title : Object Detection 체험하기
# markdown ## YOLO
# YOLO 설명
# 파일 업로더
# 추출하기 버튼
# 버튼을 누르면 yolo 로 이미지 예측해서 결과반환
st.title("Object Detection 체험하기")
st.markdown("## Object Detection YOLO 모델")
st.markdown(
    """
         Object Detection YOLO 설명
    """
)

upload_file = st.file_uploader(
    "파일을 업로드 하세요",
    type=["jpg","gif","png","jpeg"]
)



pred_button = st.button(
    label="예측하기"
)

if pred_button:
    st.write("결과입니다.")