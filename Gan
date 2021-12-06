import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

##### 부가기능

# 이미지 업로드 함수
def BigImageCreate(img_path):
    Big = Image.open(img_path)
    st.image(Big, width=700)
    
def SmallImageCreate(img_path):
    Small = Image.open(img_path)
    st.image(Small, width=300)
    
###################################################################################################################


# 메인 화면
def HomePage():

    #ImageCreate(img_path = "SandboX-logo.png")
    st.write(" ")
    st.title("이모티콘 만들기")
    st.subheader("GAN을 활용한 이모티콘 생성")
    st.write(" ")
    
# StackGan 화면
def APage():

    st.write(" ")
    st.title("이모티콘 만들기")
    st.subheader("GAN을 활용한 이모티콘 생성")
    st.write(" ")
    BigImageCreate(img_path = "stack.png")
    title = st.text_input('텍스트를 입력해주세요.', ' ')
    st.write(" ")
    SmallImageCreate(img_path = "빈화면.jpg")
    st.write(" ")
    
# 메인 화면
def BPage():

    #ImageCreate(img_path = "SandboX-logo.png")
    st.write(" ")
    st.title("이모티콘 만들기")
    st.subheader("GAN을 활용한 이모티콘 생성")
    st.write(" ")
    BigImageCreate(img_path = "disco.png")

    
# 메인 화면
def CPage():

    #ImageCreate(img_path = "SandboX-logo.png")
    st.write(" ")
    st.title("이모티콘 만들기")
    st.subheader("GAN을 활용한 이모티콘 생성")
    st.write(" ")
    BigImageCreate(img_path = "cycle.png")    
    
def Web():
    sidebar = st.sidebar.radio(" ", ("Home", "Stack-Gan", "Disco-Gan", "Cycle-Gan"))   # 사이드바
    A_sidebar_content = ["a", "b", "c"]
    B_sidebar_content = ["a", "b", "c"]
    C_sidebar_content = ["a", "b", "c"]

    if sidebar == "Home":
        HomePage()
    elif sidebar == "Stack-Gan":
        select_func = st.sidebar.selectbox("Stack-Gan", (A_sidebar_content[0], A_sidebar_content[1], A_sidebar_content[2]))
        if select_func == A_sidebar_content[0]:
            APage()
        elif select_func == A_sidebar_content[1]:
            APage()
        elif select_func == A_sidebar_content[2]:
            APage()
    elif sidebar == "Disco-Gan":
        select_func = st.sidebar.selectbox("Disco-Gan", (B_sidebar_content[0], B_sidebar_content[1], B_sidebar_content[2]))
        if select_func == B_sidebar_content[0]:
            BPage()
        elif select_func == B_sidebar_content[1]:
            BPage()
        elif select_func == B_sidebar_content[2]:
            BPage()
    elif sidebar == "Cycle-Gan":
        select_func = st.sidebar.selectbox("Cycle-Gan", (C_sidebar_content[0], C_sidebar_content[1], C_sidebar_content[2]))
        if select_func == C_sidebar_content[0]:
            CPage()
        elif select_func == C_sidebar_content[1]:
            CPage()
        elif select_func == C_sidebar_content[2]:
            CPage()



if __name__ =='__main__':
    main = Web()
