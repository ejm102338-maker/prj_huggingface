#https://fastapi.tiangolo.com/tutorial/#install-fastapi (uv add fastapi[standard])
# uvicorn main:app --port 8080 --reload
# http://127.0.0.1:8080/docs
from fastapi import FastAPI, UploadFile, File
import shutil
import os
from datetime import datetime
from ultralytics import YOLO
import io
from PIL import Image,ImageDraw,ImageFont
import requests

# 모델 불러오기
model = YOLO("./run/best.pt")
#model = YOLO("../models/yolo26n.pt")
print("모델을 불러 왔습니다.")


app = FastAPI()
@app.get("/")
async def root():
    return {"message": "Hello World!!!"}

@app.post("/upload_image")
def save_image(file: UploadFile = File(...)):
    # 파일명 설정
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"{now}-{file.filename}"
    #file_name = f"./images/{now}-{file.filename}"
    # 1. 저장할 폴더 이름 설정
    UPLOAD_DIR = "images"

    # 2. 폴더가 없다면 생성 (exist_ok=True는 이미 폴더가 있어도 에러를 내지 않아요)
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

    # 3. 파일 경로 결합 (images/파일명.jpg)
    # os.path.join을 써야 윈도우/리눅스 환경 상관없이 경로가 올바르게 합쳐집니다.
    file_path = os.path.join(UPLOAD_DIR, file_name)

    #업로드된 파일
    with open(file_path,"wb") as buffer:
        shutil.copyfileobj(file.file,buffer)

    return{
        "message" : "이미지를 저장 했습니다.",
        "time" : datetime.now().strftime("%Y%m%d%H%M%S")
    }

@app.post("/upload_image2")
async def predict_yolo(file:UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read()))

    # 예측하기
    results = model.predict(img)
    result = results[0]

    # 데이터 만들기
    detections = []
    names = result.names
    for box in result.boxes:
        x1, y1, x2, y2, conf, predict = box.data[0]
        detections.append(
            {
                "box" : [x1.item(),y1.item(),x2.item(),y2.item()],
                "conf" : conf.item(),
                "label" : names[int(predict)]
            }
        )

    # 파일 이름 설정
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"./images/{now}_{file.filename}"

    # 파일저장
    with open(file_name,"wb") as buffer:
        shutil.copyfileobj(file.file,buffer)

    return {
        "message" : "이미지를 저장 했습니다.",
        "filename" : file_name,
        "time" : datetime.now().strftime("%Y%m%d%H%M%S"),
        "results" : detections
    }

@app.post("/request")
def request() :
    file_path = "./images/cat1.jpeg"
    with open(file_path,"rb") as f:
        files = {"file":("cat1.jpeg",f,"image/jpeg")}

        print("서버로 요청중")
        response = requests.post("http://127.0.0.1:8080/upload_image2",files=files)

        # 결과확인
        if response.status_code == 200 :
            result = response.json()
            print("요청 성공")
            print(f"메시지 : {result.get('message')}")
            print(f"탐지된 객체 수 : {len(result.get('results'))}")
            for item in result.get('results') :
                print(f"탐지된 {item["label"]} {int(item["conf"]*100)}%")
                print(f"탐지된 {item["label"]} 의 좌표는 좌측 상단 x : {item["box"][0]},y : {item["box"][1]}")
                print(f"탐지된 {item["label"]} 의 좌표는 우측 하단 x : {item["box"][2]},y : {item["box"][3]}")
        else :
            print(f"요청 실패 (상태 코드 : {response.status_code})")
            print(f"에러 내용 : {response.text}")







# 오늘의 과제
# 1. streamlit 모델 설명 
# -object detection은 ~를 입력받아 ~를 하는 모델입니다.
# -~을 입력하면 ~를 내뱉습니다.
# -object detection을 학습 할 때에는 ~를 주의해야 합니다.
# -object detection으로 ~를 할 수 있습니다.
# 2.requests 라이브러리를 사용해서 내 서버 요청해 보기 캡처