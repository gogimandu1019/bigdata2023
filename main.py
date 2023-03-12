import streamlit as st
import openai


openai.api_key = "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"

def generate_answer(prompt):
    # 모델 엔진 선택
    model_engine = "text-davinci-003"

    # 맥스 토큰
    max_tokens = 2048

    # ChatGPT 요청 
    completion = openai.Completion.create(
        engine=model_engine,   
        prompt=prompt,          # 프롬프트
        max_tokens=max_tokens,  # 최대 단어수
        temperature=0.3,        # creativity
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return completion

st.title('챗쥐피티')
f = st.file_uploader('파일 업로드')

prompt = st.text_input('프롬프트')
if prompt:
    response = generate_answer(prompt)
    # 생성된 글 출력
    st.write(response.choices[0].text)