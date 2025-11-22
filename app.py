"""Streamlit app for AI assistant chatbot based on resume."""
import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
            create_stuff_documents_chain,
        )
from langchain_classic.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

MY_NAME = "許皓翔"
st.title(f"🤖 與 {MY_NAME} 的 AI 分身聊天")
st.caption("您可以問我關於工作經歷、技能或專案的細節！")

# 1. 載入已經建立好的向量資料庫
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
)

# 2. 設定檢索器與 LLM
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(api_key=API_KEY, model_name="gpt-4.1-mini")

# 3. 設定 Prompt (人設非常重要！)
# 使用一般字串而非 f-string，保留 {context} 與 {question} 供 PromptTemplate 解析
template = """
你是 {MY_NAME}。請根據底下的資訊回答面試官的問題。
如果資訊中沒有答案，請誠實回答「這在履歷中沒有提到，但我可以補充...」
請保持專業、自信且友善的語氣。

相關履歷資訊：
{context}

面試官問題：
{input}
"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template).partial(MY_NAME=MY_NAME)

combine_docs_chain = create_stuff_documents_chain(
    llm, QA_CHAIN_PROMPT
)

qa_chain = create_retrieval_chain(retriever, combine_docs_chain)

# 4. Streamlit 聊天介面邏輯
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("請問您最擅長的技術是什麼？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = qa_chain.invoke({"input": prompt})
        
        st.markdown(response["answer"]) 
        st.session_state.messages.append(
            {"role": "assistant", "content": response["answer"]}
        )
