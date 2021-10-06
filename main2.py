import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from typing import List
import requests, json
#from requests_toolbelt.multipart.encoder import MultipartEncoder
import io
import base64
import time
from io import BytesIO

# backend = "http://fastapi:8000/segmentation" http://ec55-116-121-38-22.ngrok.io
backendA = "http://127.0.0.1:3677/onefile"
backendB = "http://127.0.0.1:3677/somefile"
backendB_1 = "http://127.0.0.1:3677/result_File/"

# backendA = "http://ec55-116-121-38-22.ngrok.io/onefile"
# backendB = "http://ec55-116-121-38-22.ngrok.io/somefile"
# backendB_1 = "http://ec55-116-121-38-22.ngrok.io/result_File/"


###################################################################################################################
# CSS Define
def fontSize(fontsize, content):
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
def makeSendFile():
    uploaded_file = st.file_uploader(label="파일 업로드")  # accept_multiple_files=True)
    col1, col2, col3 = st.columns([3, 0.7, 0.7])
    start_button = col2.button("보내기")
    cancel_button = col3.button("이전으로")
    return uploaded_file, start_button, cancel_button


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
        picture_check = st.sidebar.radio("업로드 한 사진은 정상인가요?", ("정상", "비정상"))
    className = st.sidebar.selectbox("해당 클래스를 선택해주세요.",
                                     ("해당 클래스를 선택해주세요.", 'casting', 'bottle', 'cable', 'capsule', 'carpet', 'grid',
                                      'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                                      'tile', 'toothbrush', 'transistor', 'wood', 'zipper'))
    modelName = st.sidebar.selectbox("모델을 선택해주세요.", ("모델을 선택해주세요.", "resnet18", "wide_resnet50_2"))
    Threshold = st.sidebar.slider('Threshold를 설정하세요.', 0.00, 1.00, 0.33)
    return [picture_check, className, modelName, Threshold] if Flag else [className, modelName, Threshold]


### 이미지 메인페이지
def ImageMainPage():
    # DaisLogoCreate(img_path = "logo1.jpg")
    st.title("이미지 이상진단 테스트")
    st.write("\n")
    st.subheader("   · PaDim 알고리즘 소개")
    st.markdown(" **PaDim(a Patch Distribution Modeling Framework for Anomaly Detection and Localization)[1]**을 구현을 했습니다. \n"
                " 위 알고리즘은 One-Class Anomaly Detection 모델이며, 2021.10월까지 SOTA 부분에서 4위를 차지하고 있는 모델입니다.\n");
    st.markdown("* 알고리즘 특징 \n"
                "   * 정상샘플에 대해서 Gaussian distribution이 존재한다는 가정을 깔고 비정상 샘플과 비교합니다. \n"
                "   * 정상과 비정상을 구별하기 위해 **Mahalanobis distance[2]**를 사용해 이상진단을 진행합니다. \n"
                "   * 사전 학습된 CNN 모델을 사용하기에 학습을 진행하지 않습니다. \n"
                "   * Random Dimensionality reduction을 진행해 차원 축소를 합니다.")
    st.markdown("---")
    st.subheader("   · 서비스 기능 소개")
    st.write("      ▶ 단일 사진에 대해서 이상진단");
    st.markdown("-  단일 사진에 대해서 업로드하시면 서버에서 자동으로 이상진단 후 결과를 알려줍니다.")
    st.write(" ")
    st.write("      ▶ 다중 사진에 대해서 이상진단");
    st.write("- 다중 사진에 대해서 Zip 파일로 압축해서 업로드하시면 서버에서 자동으로 이상진단 후 결과를 알려줍니다.")
    st.write("\n\n")
    st.markdown("#### 참고문헌 \n")
    st.markdown(f'<a href= {"https://arxiv.org/pdf/2011.08785.pdf"}> [1] : (MLA) Defard, Thomas, et al. "PaDiM: a patch distribution modeling framework for anomaly detection and localization." International Conference on Pattern Recognition. Springer, Cham, 2021 </a>',True)
    st.markdown(f'<a href= {"https://darkpgmr.tistory.com/41"}> [2] : 다크 프로그래머. 평균,표준편차, 분산, 그리고 Mahalanobis 거리. 2013.02.18 </a>',True)
    st.markdown("---")

### 이미지 A 기능 페이지
def ImageApartPage():
    # picture_check, className, modelName, threshold = ImageSidebar(Flag=True)
    className, modelName, threshold = ImageSidebar(Flag=False)

    ## 메인 컨텐츠
    st.header("단일 사진에 대한 이상진단 테스트")
    st.write("\n")
    st.markdown(fontSize(20, "사용방법"), True)
    st.markdown("1. 왼쪽에 있는 사이드 바를 선택해주세요.\n"
                "   * 업로드할 사진에 대해 **해당 클래스를 선택**해주시고, 두 가지 **모델 중 하나를 선택**해주세요.\n"
                "   * 임계값이 **기본값(0.33)**입니다. `원하는 결과가 아니라면 해당 값을 낮추거나 올려주세요.`")
    st.markdown("2. 테스트할 사진을 업로드 해주세요.\n"
                "   * `Drag and drop file here` 라는 곳에 **Browse files** 라는 버튼이 있습니다.\n"
                "   * 해당 버튼을 눌러서 **테스트할 이미지 파일을 업로드** 해주세요.")
    st.markdown("3. 1번 2번 작업이 완료됐다면 보내기 버튼을 눌러주세요.")
    st.text(" ")
    st.markdown('<style>' + open('icons.css').read() + '</style>', unsafe_allow_html=True)
    st.markdown('<i class="material-icons">face 테스트 시작하기 </i>', unsafe_allow_html=True)

    uploaded_file, start_button, cancel_button = makeSendFile()
    st.markdown("---")
    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
        # st.write(file_details)
        # files = {"file": uploaded_file.getvalue() }
    if start_button:
        print("전송")
        waited = st.markdown("**이상진단 중이니 잠시만 기다려주세요.** ^____^")
        # picture_check = "0" if picture_check == "정상" else "1"

        files = {
            "file": uploaded_file.getvalue(),
        }
        item = {
            # "y_labels": picture_check,
            "threshold": str(threshold),
            "className": className,
            "modelName": modelName,
            "fileName": uploaded_file.name,
        }
        # print("Item ", item)
        response = requests.post(backendA, files=files, data=item)

        # print(response.json())
        # print(type(response.json()))
        anomaly_info = response.json()["anomaly_score"][0]
        img = response.json()["result_Image"]["path"]
        # print("이미지 경로 : ",img)
        response1 = requests.get(backendA + "/result_image", data={"result_filename": img})
        waited.empty()
        anomaly_content = " 판독결과는  {} 사진입니다.".format(anomaly_info["real_class"])
        st.markdown(fontSize(18, anomaly_content), True)
        st.write("\n")
        if anomaly_info["real_class"] =="비정상":
            st.markdown("* 해당 사진의 원인 결과는 다음과 같습니다.")
            result_image = Image.open(BytesIO(response1.content))
            st.image(result_image)
            st.markdown("---")


    elif cancel_button:
        print("비전송")
        M = st.markdown("---")
        M.empty()


### 이미지 B 기능 페이지
def imageBpartPage():
    #### 사이드바
    className, modelName, threshold = ImageSidebar(Flag=False)

    ### 메인 컨텐츠
    st.header("다중 사진에 대한 이상진단 테스트")
    st.write("\n")
    st.markdown(fontSize(20, "사용방법"), True)
    st.markdown("1. 테스트할 폴더 구성을 다음과 같이 구성해주세요.\n"
                "   * 파일 이름은 자유롭게 지정해도 괜찮습니다. **다만, 해당 zip폴더 안에는 다음 그림과 같이 구성해야합니다.**")
    explanB = Image.open("./ImageFolder/mutiple_explan.PNG")
    st.image(explanB)
    st.markdown("2. 왼쪽에 있는 사이드 바를 선택해주세요.\n"
                "   * 업로드할 사진들의 **해당 클래스를 선택**해주시고, 두 가지 **모델 중 하나를 선택**해주세요.\n"
                "   * 임계값이 **기본값(0.33)**입니다. `원하는 결과가 아니라면 해당 값을 낮추거나 올려주세요.`")
    st.markdown("3. 테스트할 파일을 업로드 해주세요.\n"
                "   * `Drag and drop file here` 라는 곳에 **Browse files** 라는 버튼이 있습니다.\n"
                "   * 해당 버튼을 눌러서 **테스트할 Zip 파일을 업로드** 해주세요.")
    st.markdown("3. 1번 2번 3번 작업이 완료됐다면 보내기 버튼을 눌러주세요.")

    st.text(" ")
    # st.markdown('<i class="material-icons">face</i>',unsafe_allow_html=True)
    st.markdown('<style>' + open('icons.css').read() + '</style>', unsafe_allow_html=True)
    st.markdown('<i class="material-icons">face 테스트 시작하기 </i>', unsafe_allow_html=True)
    uploaded_file, start_button, cancel_button = makeSendFile()
    st.markdown("---")

    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
        # st.write(file_details)
        # files = {"file": uploaded_file.getvalue() }
        # print(files)
    if start_button:
        waited = st.markdown("**이상진단 중이니 잠시만 기다려주세요.** ^____^")
        print("B 전송")

        files = {
            "file": uploaded_file.getvalue(),
        }
        item = {
            "threshold": str(threshold),
            "className": className,
            "modelName": modelName,
            "fileName": uploaded_file.name,
        }
        # print("Item ", item)
        response = requests.post(backendB, files=files, data=item)

        print(response.status_code)
        print(response.json())
        content = response.json()
        result_zip = content["input_path"].split("/")[3]
        Scores = content["score"][0]
        waited.empty() # 결과가 나왔으니 삭제.
        st.markdown("### 실행 결과 성능은 다음과 같습니다.\n")
        st.write("\n")
        for key in Scores.keys():
            Score_content = "`{}` : {} 입니다.".format(key,Scores[key])
            st.markdown(f"* {Score_content}")

        st.write("\n")
        href =f'<a href={backendB_1}{result_zip} download="result.zip">다운로드</a>'
        col1, col2, col3 = st.columns([3, 0.7, 1])
        col1.markdown("판독된 결과를 다운받기 원하시면 **다운로드 버튼**을 눌러주세요.")
        col2.markdown(href,True)
        st.markdown("----")


    elif cancel_button:
        print("비전송")
        M = st.markdown("---")

def DaisWeb():
    sidebar = st.sidebar.radio("guide", ("Home", "Image"))  # 사이드바
    sidebar_content = ["메인 화면", "단일 사진", "다중 사진"]

    if sidebar == "Home":
        mainHomePage()
    elif sidebar == "Image":
        select_func = st.sidebar.selectbox("이미지 이상진단 테스트", (sidebar_content[0],sidebar_content[1],sidebar_content[2]))
        if select_func == sidebar_content[0]:
            ImageMainPage()
        elif select_func == sidebar_content[1]:
            ImageApartPage()
        elif select_func == sidebar_content[2]:
            imageBpartPage()


if __name__ == '__main__':
    main = DaisWeb()
