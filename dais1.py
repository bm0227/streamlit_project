import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image


a1 = {1 : "클래스를 선택하세요.", 2 : "bottle", 3 : "casting", 4 : "가죽"}

def format_func(a_className):
    return a1[a_className]

##sidebar
sidebar = st.sidebar.radio("guide", ("Home", "Image", "Time series"))

#Home
if sidebar == "Home":

    ##text_home
    st.title("Anomaly Detection Sandbox")

elif sidebar == "Image":
    select_func = st.sidebar.selectbox("image",("Image 메인 화면", "A 기능", "B 기능"))
    
    ##sidebar_image
    if select_func == "Image 메인 화면":
        st.title("이미지")

    elif select_func == "A 기능":
        a_className = st.sidebar.selectbox("클래스를 선택하세요.", a_className=list(a1.keys()), format_func=format_func)
        a_modelName = st.sidebar.selectbox(" ", ("모델을 선택하세요.","resnet50", "wide_resnt50_2"))
        a_Threshold = st.sidebar.slider('Threshold를 설정하세요.', 0.00, 1.00)

        st.header("A 기능")
        if st.button("파일 업로드"):
          clean_sidebar()


    elif select_func == "B 기능":
        st.header("B 기능")

        b_picture_check = st.sidebar.radio(" ", ("업로드 한 사진은 정상인가요?","네", "아니오"))
        b_className = st.sidebar.selectbox(" ", ("클래스를 선택하세요.","bottle", "casting", "가죽"))
        b_modelName = st.sidebar.selectbox(" ", ("모델을 선택하세요.","resnet50", "wide_resnt50_2"))
        b_Threshold = st.sidebar.slider('Threshold를 설정하세요.', 0.00, 1.00)
