import streamlit as st
import pandas as pd
from pycaret.classification import *

st.subheader('학습용 데이터 업로드')

# 학습용 데이터 업로드
st.title('데이터 업로드')
train_file = st.file_uploader('파일 업로드')

# 학습용 파일 업로드
if train_file:
    # pd.read_csv()
    train = # 코드 입력

# 예측 컬럼
target = st.text_input('예측 컬럼명 입력')

train_btn = st.button('학습 시작')

if train_btn:
    # setup
    clf = # 코드 입력
    
    # 학습 결과 출력
    best_model = # 코드 입력
    # 모델 블렌딩
    blended_models = # 코드 입력
    # 학습 결과 받아오기
    result = pull()
    st.dataframe(result)
    # 모델 저장
    save_model(# 코드 입력)

    
# CSV로 다운로드 
@st.cache_data
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')


# 예측용 데이터 업로드
st.title('예측용 데이터 업로드')
test_file = st.file_uploader('예측용 파일 업로드')

# 예측용 파일 업로드
if test_file:
    # 코드 입력
    test = 
    
# 예측 버튼
test_btn = st.button('예측')

# 예측 버튼 클릭시
if test_btn:
    # 모델 로드 load_model()
    loaded_model = # 코드 입력
    
    # 예측 predict_model()
    predictions = # 코드 입력
    
    # 결과 다운로드
    download_btn = st.download_button(
        "예측 결과 다운로드",
        convert_df(predictions),
        "prediction.csv",
        "text/csv",
        key='download-csv'
    )


    
