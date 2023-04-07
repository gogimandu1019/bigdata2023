import streamlit as st
import pandas as pd
from pycaret.classification import *

st.title('데이터 업로드')
file = st.file_uploader('파일 업로드')

# 저장한 모델 로드
loaded_model = load_model('mymodel-01')

# 파일 업로드
if file:
    df = pd.read_csv(file)
    
# CSV로 다운로드 
@st.cache_data
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

# 예측 버튼
btn = st.button('예측')

# 예측 버튼 클릭시
if btn:
    # 모델 로드
    predictions = predict_model(estimator=loaded_model, data=df)
    predictions[['survived', 'prediction_label']]
    
    # 결과 다운로드
    download_btn = st.download_button(
        "예측 결과 다운로드",
        convert_df(predictions),
        "prediction.csv",
        "text/csv",
        key='download-csv'
    )


    
