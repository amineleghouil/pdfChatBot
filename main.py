import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter

def setup_qa_system(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embedding)

    retriever = vector_store.as_retriever()
    llm = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta", model_kwargs={"temperature": 0})

    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    return qa_chain

if __name__ == '__main__':
    qa_chain = setup_qa_system('a.pdf')

    while True:
        question = input('\nAsk question: ')
        if question.lower() == 'exit':
            break

        answer = qa_chain.invoke(question)

        print('Answer:')
        print(answer['result'])
