import streamlit as st
import pandas as pd

st.title('데이터 업로드')
file = st.file_uploader('파일 업로드')


if file:
    df = pd.read_excel(file)
    st.dataframe(df)