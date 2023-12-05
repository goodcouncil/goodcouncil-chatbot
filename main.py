from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
# from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain

import os
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(temperature=0,model_name="gpt-4")

from langchain.document_loaders import DirectoryLoader
import json
from langchain.docstore.document import Document

def loadJSONFile(file_path):
    docs=[]
    # Load JSON file
    with open(file_path, encoding='UTF-8-sig') as file:
        data = json.load(file)

    for doc in data['docs']:
        law = {
            "의안ID" : doc['BILL_ID'] if doc['BILL_ID'] is not None else "",
            "의안번호" : doc['BILL_NO'] if doc['BILL_NO'] is not None else "",
            "대" : doc['AGE'] if doc['AGE'] is not None else "",
            "의안명" : doc['BILL_NAME'] if doc['BILL_NAME'] is not None else "",
            "제안자" : doc['PROPOSER'] if doc['PROPOSER'] is not None else "",
            "제안자구분" : doc['PROPOSER_KIND'] if doc['PROPOSER_KIND'] is not None else "",
            "제안일" : doc['PROPOSE_DT'] if doc['PROPOSE_DT'] is not None else "",
            "소관위코드" : doc['CURR_COMMITTEE_ID'] if doc['CURR_COMMITTEE_ID'] is not None else "",
            "소관위" : doc['CURR_COMMITTEE'] if doc['CURR_COMMITTEE'] is not None else "",
            "소관위회부일" : doc['COMMITTEE_DT'] if doc['COMMITTEE_DT'] is not None else "",
            "소관위심사처리일" : doc['COMMITTEE_PROC_DT'] if doc['COMMITTEE_PROC_DT'] is not None else "",
            "의안상세정보" : doc['LINK_URL'] if doc['LINK_URL'] is not None else "",
            "소관위상정일" : doc['CMT_PRESENT_DT'] if doc['CMT_PRESENT_DT'] is not None else "",
            "본회의심의결과" : doc['PROC_RESULT_CD'] if doc['PROC_RESULT_CD'] is not None else "",
            "소관위처리결과" : doc['CMT_PROC_RESULT_CD'] if doc['CMT_PROC_RESULT_CD'] is not None else "",
            "법사위회부일" : doc['LAW_SUBMIT_DT'] if doc['LAW_SUBMIT_DT'] is not None else "",
            "법사위상정일" : doc['LAW_PRESENT_DT'] if doc['LAW_PRESENT_DT'] is not None else "",
            "법사위처리일" : doc['LAW_PROC_DT'] if doc['LAW_PROC_DT'] is not None else "",
            "의결일" : doc['PROC_DT'] if doc['PROC_DT'] is not None else "",
            "법사위처리결과" : doc['LAW_PROC_RESULT_CD'] if doc['LAW_PROC_RESULT_CD'] is not None else "",
            "소관위처리일" : doc['CMT_PROC_DT'] if doc['CMT_PROC_DT'] is not None else "",
            "요약": doc.get('DETAIL_CONTENT')
        }

        docs.append(Document(page_content=str(law)))

    return docs 

import time

def init():
    data = loadJSONFile("./law.json")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings()
    
    total_length = len(documents)
    batch_size = 64 

    for batch_start in range(0, total_length, batch_size):
        batch_end = min(batch_start + batch_size, total_length)
        batch_texts = documents[batch_start:batch_end]
        Chroma.from_documents(documents=batch_texts, embedding=embeddings, persist_directory="./chromadb")
        time.sleep(0.2)
        print(f"Inserted {batch_end}/{total_length} chunks")
    
    vectorstore = Chroma(persist_directory="./chromadb", embedding_function=embeddings)

    # Initialise Langchain - Conversation Retrieval Chain
    qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0), vectorstore.as_retriever(),  get_chat_history=lambda h : h)
    return embeddings, vectorstore, qa

import gradio as gr
# Front end web app
with gr.Blocks() as demo:
    embeddings, vectorstore, qa = init()
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")
    chat_history = []
    def user(user_message, history):
        # Get response from QA chain
        response = qa({"question": user_message, "chat_history": history})
        # Append user message and response to chat history
        history.append((user_message, response["answer"]))
        return gr.update(value=""), history
    
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False)
    
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(debug=True)