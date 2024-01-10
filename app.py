from PIL import Image
import io
import logging
import streamlit as st
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

temperature = 0.9

generation_config = {
    "temperature": temperature,
    "top_p": 0.95,
    "top_k": 1,
    "max_output_tokens": 99998,
}

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks  # list of strings

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "upload some pdfs and ask me a question"}]
    
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore

    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True, )

    print(response)
    return response


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   client=genai,
                                   temperature=0.3,
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

st.set_page_config(page_title="Gemini Chatbot", page_icon=":gem:")

with st.sidebar:
    st.title("Gemini Setting")

    api_key = st.text_input("API key", placeholder="if you have one.")
    if api_key:
        genai.configure(api_key=api_key)
    else:
        if "GOOGLE_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        else:
            st.error("Missing API key.")

    select_feature = st.selectbox(
        "Select feature", ["chatbot", "image-captioning", "document-chatbot"])
    
    if select_feature == "chatbot":
        temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.9, 0.1)
        st.caption(
            "Temperature controls the randomness of the model. Lower temperature results in less randomness.")

    if select_feature == "image-captioning":
        temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.9, 0.1)
        st.caption(
            "Temperature controls the randomness of the model. Lower temperature results in less randomness.")
        
        uploaded_image = st.file_uploader(
            "upload image",
            label_visibility="collapsed",
            accept_multiple_files=False,
            type=["png", "jpg"],
        )
        st.caption(
            "Note: The vision model gemini-pro-vision is not optimized for multi-turn chat."
        )
        if uploaded_image:
            image_bytes = uploaded_image.read()
    
    if select_feature == "document-chatbot":
        uploaded_file = st.file_uploader(
            "upload pdf",
            label_visibility="collapsed",
            accept_multiple_files=False,
            type=["pdf"],
        )
        st.caption(
            "Upload a pdf file to chat with the model. The model will read the pdf file and generate a response based on the content of the pdf file."
        )
        if uploaded_file:
            with st.spinner("Reading pdf..."):
                pdf_docs = [uploaded_file]
                text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(text)
                get_vector_store(chunks)
                st.success("Done!")
            
            st.sidebar.button("Clear chat history", on_click=clear_chat_history)


def get_response(messages, model="gemini-pro"):
    model = genai.GenerativeModel(model)
    res = model.generate_content(messages,
                                 generation_config=generation_config)
    return res


if "messages" not in st.session_state:
    st.session_state["messages"] = []
messages = st.session_state["messages"]

# The vision model gemini-pro-vision is not optimized for multi-turn chat.
st.header("Gemini Chatbot🎈")
st.markdown("""
    App built by [srikresna](https://github.com/srikresna) using [Streamlit](https://streamlit.io) and hosted on [Streamlit Share](https://share.streamlit.io).
""")

# Initialize session state for chat history if it doesn't exist
if messages and select_feature != "image-captioning" and select_feature != "document-chatbot":
    for item in messages:
        role, parts = item.values()
        if role == "user":
            st.chat_message("user").markdown(parts[0])
        elif role == "model":
            st.chat_message("assistant").markdown(parts[0])

chat_message = st.chat_input("Say something")

res = None
if chat_message:
    st.chat_message("user").markdown(chat_message)
    res_area = st.chat_message("assistant").markdown("...")

    if select_feature == "image-captioning":
        if "image_bytes" in globals():
            vision_message = [chat_message,
                              Image.open(io.BytesIO(image_bytes))]
            try:
                res = get_response(vision_message, model="gemini-pro-vision")
            except google_exceptions.InvalidArgument as e:
                if "API key not valid" in str(e):
                    st.error("API key not valid. Please pass a valid API key.")
                else:
                    st.error("An error occurred. Please try again.")
            except Exception as e:
                logging.error(e)
                st.error("Error occured. Please refresh your page and try again.")
        else:
            vision_message = [{"role": "user", "parts": [chat_message]}]
            st.warning(
                "Since there is no uploaded image, the result is generated by the default gemini-pro model.")
            try:
                res = get_response(vision_message)
            except google_exceptions.InvalidArgument as e:
                if "API key not valid" in str(e):
                    st.error("API key not valid. Please pass a valid API key.")
                else:
                    st.error("An error occurred. Please try again.")
            except Exception as e:
                logging.error(e)
                st.error("Error occured. Please refresh your page and try again.")
    
    elif select_feature == "document-chatbot":
        pass
        
    else:
        messages.append(
            {"role": "user", "parts":  [chat_message]},
        )
        try:
            res = get_response(messages)
        except google_exceptions.InvalidArgument as e:
            if "API key not valid" in str(e):
                st.error("API key not valid. Please pass a valid API key.")
            else:
                st.error("An error occurred. Please refresh your page and try again.")
        except Exception as e:
            logging.error(e)
            st.error("Error occured. Please refresh your page and try again.")
    
    if res is not None:
        res_text = ""
        for chunk in res:
            if chunk.candidates:
                res_text += chunk.text
            if res_text == "":
                res_text = "unappropriate words"
                st.error("Your words violate the rules that have been set. Please try again!")
        res_area.markdown(res_text)

        if select_feature != "gemini-pro-vision":
            messages.append({"role": "model", "parts": [res_text]})
