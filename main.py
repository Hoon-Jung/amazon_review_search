import pandas as pd

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

import openai
import streamlit as st
import os


from langchain import PromptTemplate

def prep_db(doc, key):

    df = pd.read_csv("./reviews/OFFICIAL_AMAZON_FASHION_TAGS.csv")

    same_products = df[df["asin"]==doc]["reviewText"]
    reviews_doc = same_products.tolist()
    reviews_doc = "\n".join(reviews_doc)

    text_splitter = CharacterTextSplitter(separator="\n",chunk_size=1000)
    texts = text_splitter.create_documents([reviews_doc])

    # db = FAISS.from_documents(texts, HuggingFaceEmbeddings())
    embedder = OpenAIEmbeddings(openai_api_key=key)
    db = FAISS.from_documents(texts, embedder)

    return texts



def get_answer(product_id, question, apikey, selected_db):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=apikey)

    # qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=selected_db.as_retriever())
    # qa_chain = RetrievalQA.from_chain_type(llm)
    # res = llm({"query": question})


    # prompt_template = PromptTemplate.from_template(
    # "Tell me a funny joke about {content}."
    # )
    # chain = LLMChain(llm=llm, prompt=prompt_template)
    # res = chain.run(content=question)

    return selected_db

if __name__ == "__main__":
    st.title("Review Search Engine")
    options = ["B00007GDFV", "B00008JOQI", "7106116521"]

    api_key_input = st.text_input("Enter OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY", ""))
    api_key_button = st.button("Add")
    if api_key_button:
        st.session_state["OPENAI_API_KEY"] = api_key_input

    openai_api_key = st.session_state.get("OPENAI_API_KEY")
    if openai_api_key:
        openai.api_key = openai_api_key
    else:
        st.warning("WARNING: Enter your OpenAI API key!")

    selected_option = st.selectbox("Select a product", options)
    st.write("Currently selected:", selected_option)
    question = st.text_input("Ask a question")
    if st.button("Get Answer"):
        # st.write(get_answer(selected_option, question, openai_api_key, prep_db(selected_option)))
        st.write(prep_db(selected_option, openai_api_key))