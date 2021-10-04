import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from typing import List
import requests, json
from requests_toolbelt.multipart.encoder import MultipartEncoder
import io
import base64
import time
from io import BytesIO


# backend = "http://fastapi:8000/segmentation"
backend = "http://127.0.0.1:3677/onefile"

###################################################################################################################
# CSS Define
def fontSize(fontsize,content):
    html_str = f"""
                    <style>
                    p.a {{
                      font: bold {fontsize}px Courier;
                    }}
                    </style>
                    <p class="a">{content}</p>
                    """
    return html_str
###################################################################################################################
## 부가 기능 markdown 설정.

# def makeSendFiles():
#     uploaded_file = st.file_uploader("파일 업로드", accept_multiple_files=True)
#
#     col1,col2,col3 = st.columns([3,0.7,0.7])
#     start_button = col2.button("보내기")
#     cancel_button = col3.button("이전으로")
#     return uploaded_file, start_button,cancel_button


def makeSendFile():
    uploaded_file = st.file_uploader("파일 업로드") #accept_multiple_files=True)
    col1,col2,col3 = st.columns([3,0.7,0.7])
    start_button = col2.button("보내기")
    cancel_button = col3.button("이전으로")
    return uploaded_file, start_button,cancel_button

######################################################################################################################
# Dais Logo 생성 함수 -> 인자로 image 경로를 받는다.
def DaisLogoCreate(img_path):
    logo = Image.open(img_path)
    st.image(logo)
######################################################################################################################
# 메인 화면
def mainHomePage():

    # DaisLogoCreate(img_path = "logo1.jpg")
    st.write(" ")
    st.title("Anomaly Detection Sandbox")
    st.subheader("제조업의 AI 기반 이상탐지를 위한 알고리즘 및 플랫폼 개발")
    st.write("이미지와 시계열 데이터 알고리즘을 확인할 수 있음.")
    st.write(" ")
    st.subheader("Dais open lab.")
    st.write("DaiS stands for Data analytics and intelligent Systems.")


## 이미지 화면
### 이미지 사이드바
def ImageSidebar(Flag=True):
    if Flag:
        picture_check = st.sidebar.radio("업로드 한 사진은 정상인가요?", ( "정상", "비정상"))
    className = st.sidebar.selectbox("해당 클래스를 선택해주세요.", ("해당 클래스를 선택해주세요.", 'casting','bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper'))
    modelName = st.sidebar.selectbox("모델을 선택해주세요.", ("모델을 선택해주세요.", "resnet18", "wide_resnet50_2"))
    Threshold = st.sidebar.slider('Threshold를 설정하세요.', 0.00, 1.00,0.33)
    return [picture_check,className,modelName,Threshold] if Flag else [className,modelName,Threshold]

### 이미지 메인페이지
def ImageMainPage():
    # DaisLogoCreate(img_path = "logo1.jpg")
    st.title("이미지")
    st.subheader("   · 알고리즘 소개")
    st.write(" ");    st.write(" ");    st.write(" ")
    st.subheader("   · 기능에 대한 사용 설명서")
    st.write("      ▶ A 기능");    st.write("          ...")
    st.write(" ")
    st.write("      ▶ B 기능");    st.write("          ...")

### 이미지 A 기능 페이지
def ImageApartPage():
    st.header("단일 사진 정상유무 판단 테스트")
    st.markdown(fontSize(20, "사용방법"), True)
    st.markdown("이미지를 한장 업로드하고 그에 따른 정상 비정상 유무와 해당 사진의 클래스와 예측할 모델과 Threshold를 선택하시고 보내기 버튼을 눌러주세요."
                "그러면 자동으로 해당 사진에 대해 정상 비정상 유무를 알려줍니다.")
    st.text(" ")
    st.markdown("---")

    #st.markdown('<style>' + open('icons.css').read() + '</style>', unsafe_allow_html=True)
    #st.markdown('<i class="material-icons">face 시작하기 </i>', unsafe_allow_html=True)

    picture_check, className, modelName, threshold = ImageSidebar(Flag=True)
    uploaded_file, start_button, cancel_button = makeSendFile()
    st.markdown("---")
    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
        # st.write(file_details)
        # files = {"file": uploaded_file.getvalue() }
    if start_button:
        print("전송")
        picture_check = "0" if picture_check =="정상" else "1"

        files = {
                "file": uploaded_file.getvalue(),
                 }
        item = {
                        "y_labels": picture_check,
                        "threshold": str(threshold),
                        "className": className,
                        "modelName": modelName,
                        "fileName": uploaded_file.name,
                     }
        # print("Item ", item)
        response = requests.post(backend, files=files, data=item)

        # print(response.json())
        # print(type(response.json()))
        anomaly_info = response.json()["anomaly_score"][0]
        img = response.json()["result_Image"]["path"]
        # print("이미지 경로 : ",img)
        response1 = requests.get(backend +"/result_image", data={"result_filename":img})
        
        anomaly_content = "판독 결과 {}사진 입니다. Anomaly Score는 {} 입니다.".format(anomaly_info["real_class"], anomaly_info["anomaly_score"])
        result_image = Image.open(BytesIO(response1.content))
        st.markdown(fontSize(17,anomaly_content),True)
        st.image(result_image)
        st.markdown("---")
        
    
    elif cancel_button:
        print("비전송")
        M = st.markdown("---")
        M.empty()

### 이미지 B 기능 페이지
def imageBpartPage():
    #### 사이드바
    className, modelName, Threshold = ImageSidebar(Flag=False)

    ### 메인 컨텐츠
    st.header("다중 테스트 기능")
    st.markdown(fontSize(22, "다중 테스트 사용방법"), True)
    st.markdown("* 업로드할 폴더 구성")
    explanB = Image.open("./ImageFolder/mutiple_explan.PNG")
    st.image(explanB)
    st.text(" ")
    # st.markdown('<i class="material-icons">face</i>',unsafe_allow_html=True)
    #st.markdown('<style>' + open('icons.css').read() + '</style>', unsafe_allow_html=True)
    #st.markdown('<i class="material-icons">face 시작하기 </i>', unsafe_allow_html=True)
    uploaded_file, start_button, cancel_button = makeSendFile()
    st.markdown("---")

    if start_button:
        print("전송")
        print(className)
        print(modelName)
        print(Threshold)
        print(uploaded_file)

        st.markdown(fontSize(20, "실행결과"), unsafe_allow_html=True)
        # st.image(default_image)
        st.text(" ")
        st.text("성능 지표")
    elif cancel_button:
        print("이전으로 되돌리기")



def DaisWeb():
    sidebar = st.sidebar.radio("guide", ("Home", "Image"))   # 사이드바

    if sidebar == "Home":
        mainHomePage()
    elif sidebar == "Image":
        select_func = st.sidebar.selectbox("image", ("Image 메인 화면", "A 기능", "B 기능"))
        if select_func == "Image 메인 화면":
            ImageMainPage()
        elif select_func == "A 기능":
            ImageApartPage()
        elif select_func == "B 기능":
            imageBpartPage()


if __name__ =='__main__':
    main = DaisWeb()






























