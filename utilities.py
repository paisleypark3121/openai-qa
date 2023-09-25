import os

from langchain.embeddings.openai import OpenAIEmbeddings

import os
from urllib.parse import urlparse
import requests

def get_file_source(file_source, save_directory, overwrite=False):
    
    parsed = urlparse(file_source)
    if bool(parsed.netloc):
        local_filename = os.path.join(save_directory, os.path.basename(file_source))
        
        # Check if the file already exists on disk
        if not os.path.exists(local_filename) or overwrite:
            with requests.get(file_source, stream=True) as r:
                r.raise_for_status()
                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            #print("File downloaded and saved to disk.")  # Debug statement
        #else:
            #print("Il file è già presente e non sarà sovrascritto.")
        return local_filename
    else:
        if os.path.exists(file_source):
            return file_source
        else:
            raise FileNotFoundError(f"Il file '{file_source}' non è stato trovato.")


def get_embedding():
    embedding = OpenAIEmbeddings()
    return embedding

import numpy as np

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    cos_sim = dot_product / (norm_vec1 * norm_vec2)
    return cos_sim

def query_similarity_search(question, vectordb, embedding, k=5, cosine_similarity_fn=None):
        
    docs = vectordb.similarity_search(question, k=k)

    #print(docs)

    q_emb = embedding.embed_query(question)
    q_vec = np.array(q_emb)

    for d in docs:
        emb = embedding.embed_query(d.page_content)
        vec = np.array(emb)
        if cosine_similarity_fn:
            cosine = cosine_similarity_fn(q_vec, vec)
        else:
            cosine = cosine_similarity(q_vec, vec)
        print(cosine)

def test_similarity(embedding):
    sent1 = "I love dogs"
    sent2 = "I love cats"
    sent3 = "Yesterday I played basketball"
    sent4 = "Yesterday I played football"
    sent5 = "Leonardo Di Caprio is an underrated actor"
    embedding1 = embedding.embed_query(sent1)
    embedding2 = embedding.embed_query(sent2)
    embedding3 = embedding.embed_query(sent3)
    embedding4 = embedding.embed_query(sent4)
    embedding5 = embedding.embed_query(sent5)
    # Assuming vec1 and vec2 are your embeddings
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    vec3 = np.array(embedding3)
    vec4 = np.array(embedding4)
    vec5 = np.array(embedding5)
    # More similar
    print(cosine_similarity(vec1, vec2))
    print(cosine_similarity(vec3, vec4))
    print("---")
    # Less similar
    print(cosine_similarity(vec1, vec3))
    print(cosine_similarity(vec2, vec3))
    print(cosine_similarity(vec2, vec4))
    print(cosine_similarity(vec2, vec5))
    print(cosine_similarity(vec4, vec5))

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_splits(file_pdf, chunk_size=1200, chunk_overlap=200):

    if file_pdf.startswith("http"):
        response = requests.get(file_pdf)
        file_path = "temp.pdf"
        with open(file_path, 'wb') as file:
            file.write(response.content)
    else:
        file_path = file_pdf

    if not os.path.isfile(file_pdf):
        raise FileNotFoundError(f"Il file {file_pdf} non è stato trovato.")
    
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    except Exception as e:
        raise ValueError(f"Non è possibile caricare il file {file_path}: {e}")
    
    if not docs:
        raise ValueError(f"Il documento {file_path} è vuoto o non valido.")
    
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap, 
        length_function=len
    )
    
    splits = r_splitter.split_documents(docs)
    
    #print(len(docs))
    #print(type(docs[0]))
    #print(len(splits))
    #for doc in splits[:10]:
    #    print(len(doc.page_content))
    #test_similarity(embedding)

    return splits

from langchain.vectorstores import Chroma

def get_vectordb(persist_directory, embedding, file_source=None, chunk_size=1200, chunk_overlap=200, overwrite=False):

    if overwrite or (not os.path.isdir(persist_directory) or not os.listdir(persist_directory)):
        
        if file_source is None:
            raise ValueError("Il parametro 'file_source' è obbligatorio quando 'overwrite' è True o la directory è vuota.")
        
        # Ottieni gli splits dal file_source
        splits = get_splits(
            file_pdf=file_source,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    else:
        splits = None

    if overwrite or (splits is not None and (not os.path.isdir(persist_directory) or not os.listdir(persist_directory))):
        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embedding,
            persist_directory=persist_directory
        )
    else:
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

    #print(result["result"])
    #print(result["source_documents"])

    return vectordb

def get_vectordb_old(persist_directory, embedding, documents=None, overwrite=False):

    if overwrite:
        if documents is None:
            raise ValueError("Il parametro 'documents' è obbligatorio quando 'overwrite' è True.")
        
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            persist_directory=persist_directory
        )
    else:
        if not os.path.isdir(persist_directory) or not os.listdir(persist_directory):
            if documents is None:
                raise ValueError("Il parametro 'documents' è obbligatorio quando la directory è vuota.")
            
            vectordb = Chroma.from_documents(
                documents=documents,
                embedding=embedding,
                persist_directory=persist_directory
            )
        else:
            vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    
    #print(result["result"])
    #print(result["source_documents"])

    return vectordb

from langchain.chains import RetrievalQA,ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

def get_result(question, template, vectordb, model_name="gpt-3.5-turbo", temperature=0.7):
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    
    result = qa_chain({"query": question})
    # print(result["result"])
    # print(result["source_documents"]) # Scommenta questa linea se desideri stampare anche i documenti sorgente
    
    return result

def get_conversational_result(question, template, vectordb, model_name="gpt-3.5-turbo", temperature=0.7):
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_chain = ConversationalRetrievalChain.from_llm(llm,retriever=vectorstore.as_retriever(), memory=memory)


    # qa_chain = RetrievalQA.from_chain_type(
    #     llm,
    #     retriever=vectordb.as_retriever(),
    #     chain_type="stuff",
    #     return_source_documents=True,
    #     chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    #)
    
    result = qa_chain({"query": question})
    # print(result["result"])
    # print(result["source_documents"]) # Scommenta questa linea se desideri stampare anche i documenti sorgente
    
    return result