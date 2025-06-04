from flask import Flask, render_template, request, jsonify
import os
from PyPDF2 import PdfReader
import docx
import pandas as pd
from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

app = Flask(__name__)
load_dotenv()

conversation_chain = None
chat_history = []
custom_prompt_template = """
You are a helpful assistant. Use the context below to answer the question clearly and concisely.

Context:
{context}

Chat History:
{chat_history}

Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(
    template=custom_prompt_template,
    input_variables=["context", "question", "chat_history"]
)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_files():
    global conversation_chain, chat_history
    uploaded_files = request.files.getlist("files")

    if not uploaded_files:
        return jsonify({"status": "error", "message": "No files uploaded."})

    file_text = ""
    for uploaded_file in uploaded_files:
        ext = os.path.splitext(uploaded_file.filename)[1].lower()
        if ext == ".pdf":
            file_text += get_pdf_text(uploaded_file)
        elif ext in [".doc", ".docx"]:
            file_text += get_docx_text(uploaded_file)
        elif ext == ".csv":
            file_text += get_csv_text(uploaded_file)

    text_chunks = get_text_chunks(file_text)
    vector_store = get_vector_store(text_chunks)
    conversation_chain = get_conversation_chain(vector_store)
    chat_history = []

    return jsonify({"status": "success", "message": "Files processed."})


@app.route('/ask', methods=['POST'])
def ask():
    global conversation_chain, chat_history
    user_question = request.json.get("question")
    if not user_question or conversation_chain is None:
        return jsonify({"answer": "Please upload files first."})

    answer = conversation_chain.run(user_question)
    history = conversation_chain.memory.chat_memory.messages
    messages = [{"type": m.type, "content": m.content} for m in history]
    return jsonify({"answer": answer, "history": messages})


def get_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    return ''.join([page.extract_text() or '' for page in reader.pages])


def get_docx_text(file):
    doc = docx.Document(file)
    return '\n'.join([p.text for p in doc.paragraphs])


def get_csv_text(file):
    df = pd.read_csv(file)
    return df.to_string()


def get_text_chunks(text):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separator="\n", length_function=len)
    return splitter.split_text(text)


def get_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(chunks, embeddings)


def get_conversation_chain(vector_store):
    llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-R1",
        task="text-generation",
        temperature=0.7
    )
    chat_model = ChatHuggingFace(llm=llm)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # 👇 Use the prompt in the chain
    return ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=vector_store.as_retriever(search_type="similarity"),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )



if __name__ == '__main__':
    app.run(debug=True)
