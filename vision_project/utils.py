from openai import OpenAI 
import json 

# STEP1. 환경변수 불러오기
from dotenv import load_dotenv 
load_dotenv()

# STEP2. OpenAI 객체 생성
client = OpenAI()

# STEP3. 시스템 프롬프트 만들기
system_prompt = """
너는 상상력이 풍부한 초등학생 아이야.
[대화 히스토리]는 사용자와 너가 나눈 대화야
이 대화를 참고해서 사용자의 질문에 답변해줘

[대화 히스토리]
"""

# STEP4. 채팅하는 함수 만들기
def mychat(input_text, history):
    # 현재 st.session_state["messages"]는 딕셔너리가 포함되어 있는 리스트
    # 문자열로 바꿔서 넣어주어야 한다. 
    history_str = json.dumps(history, ensure_ascii=False)

    messages = [
        {"role": "system", "content": system_prompt+history_str},
        {"role": "user", "content": input_text}
    ]
    # print("="* 50)
    # print(system_prompt+history_str)
    # print("="* 50)
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=messages
    )

    return response.choices[0].message.content