import streamlit as st
import os
from PyPDF2 import PdfReader
import docx
from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDF", page_icon="📄")
    st.header("Chatbot")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = False

    with st.sidebar:
        uploaded_files = st.file_uploader(
            "Upload your PDF/DOC/DOCX/CSV files",
            type=["pdf", "doc", "docx", "csv"],
            accept_multiple_files=True
        )
        process = st.button("Process")

    if process:
        if not uploaded_files:
            st.warning("Please upload at least one file.")
            return
        file_text = get_file_text(uploaded_files)
        text_chunks = get_text_chunks(file_text)
        vector_store = get_vector_store(text_chunks)
        st.session_state.conversation = get_conversation_chain(vector_store)
        st.session_state.processComplete = True
        st.success("Processing complete. Ask your questions!")

    if st.session_state.processComplete:
        user_question = st.chat_input("Ask me anything about your files")
        if user_question:
            handle_user_question(user_question)


def get_file_text(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension == ".pdf":
            text += get_pdf_text(uploaded_file)
        elif file_extension in [".doc", ".docx"]:
            text += get_docx_text(uploaded_file)
        elif file_extension == ".csv":
            text += get_csv_text(uploaded_file)
    return text


def get_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


def get_docx_text(file):
    doc = docx.Document(file)
    all_text = [para.text for para in doc.paragraphs]
    return '\n'.join(all_text)


def get_csv_text(file):
    # Stub for CSV reading – implement if needed
    return "CSV support not implemented yet."


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separator="\n",
        length_function=len
    )
    return text_splitter.split_text(text)


def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    return vector_store


def get_conversation_chain(vector_store):
    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-R1",
        task="text-generation",
        temperature=0.7
    )
    chat_model = ChatHuggingFace(llm=llm)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=vector_store.as_retriever(search_type="similarity"),
        memory=memory
    )
    return conversation_chain


def handle_user_question(user_question):
    response = st.session_state.conversation.run(user_question)
    st.session_state.chat_history = st.session_state.conversation.memory.chat_memory.messages

    with st.container():
        for i, message_obj in enumerate(st.session_state.chat_history):
            if message_obj.type == "human":
                message(message_obj.content, is_user=True, key=f"user-{i}")
            else:
                message(message_obj.content, key=f"ai-{i}")


if __name__ == '__main__':
    main()
