import streamlit as st 

# Step1. Session State 만들기 
# Step2. UI를 만든다. 
## title 
## 히스토리 메시지 출력 
## 사용자 입력 받고
## 답변을 출력

#=====================================================================
# 기본 뼈대 버전
#=====================================================================
import streamlit as st
from utils import mychat
#---------------------------------------------------------------------
# 프로필 설정
profile = {
    "user": "./resources/sample.jpeg",
    "ai": "./resources/chatbot.png"
}
#---------------------------------------------------------------------
# session state 정의: 히스토리 저장소 만들기
if not "messages" in st.session_state:
    st.session_state["messages"] = []

history = st.session_state["messages"]
#---------------------------------------------------------------------
# 챗봇 제목 
st.title("히스토리가 반영된 챗봇 만들기")

# 과거 메시지 출력: 히스토리에 뭐 담겼으면 출력해줘
if len(st.session_state["messages"]) > 0:
    for chat in st.session_state["messages"]:
        name = chat["role"]
        avatar = profile[name]
        st.chat_message(name=name,avatar=avatar).markdown(chat["content"])

# 사용자 입력
input_text = st.chat_input("메세지를 입력하세요...")

# 사용자 입력 이후
import time
if input_text:
    st.chat_message(name="user",avatar=profile["user"]).markdown(input_text)

    with st.chat_message(name="ai",avatar=profile["ai"]):
        container = st.empty() # 빈자리(자리 맡았다)
        with container:
            with st.spinner("생각하는 중이에요...."):
                answer = mychat(input_text, history)
                # answer = "답변이 담기는 곳입니다."
                
            st.markdown(answer)

    # 대화 저장: 저장소에 user-ai QA 저장    
    st.session_state["messages"].extend(
        [
            {"role": "user", "content": input_text},
            {"role": "ai", "content": answer}
        ]
    )