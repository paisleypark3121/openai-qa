import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# from utilities import *

load_dotenv()

# Configurazione iniziale
file_source = "./files/jokerbirot_space_musician.txt"
persist_directory = './chroma/jokerbirot_space_musician'
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
template = """Ti fornisco un breve [Context] che dovrai usare per rispondere alla [Question]. \
    Se non conosci la risposta relativamente al [Context] devi dire che non sai rispondere, non provare ad inventare la risposta. \
    [Context]: {context} \
    [Question]: {question} \
    Risposta:"""
model_name="gpt-3.5-turbo"
temperature=0.7

embedding = OpenAIEmbeddings()
loader = TextLoader(file_source)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
documents = text_splitter.split_documents(documents)

vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            persist_directory=persist_directory
        )

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
llm = ChatOpenAI(model_name=model_name, temperature=temperature)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectordb.as_retriever(), 
    memory=memory)

try:
    print("\n***WELCOME***\n")
    while True:
        question = input("\nTu: ")
        result = qa_chain({"question": question})
        print(f"Assistant: {result['answer']}")
except KeyboardInterrupt:
    print("ciao!!!")

# question = "Chi è Jokerbirot?"
# result = qa_chain({"question": question})
# print(result["answer"])
# question = "Mi puoi dare altre informazioni su di lui?"
# result = qa_chain({"question": question})
# print(result["answer"])
# question = "Chi è Jackie Chen?"
# result = qa_chain({"question": question})
# print(result["answer"])