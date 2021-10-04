import streamlit as st

choices = {1: "dataset a", 2: "dataset b", 3: "dataset c"}

def format_func(option):
    return choices[option]

option = st.selectbox("select option", options=list(choices.keys()), format_func=format_func)