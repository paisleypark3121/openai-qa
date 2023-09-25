import os
from dotenv import load_dotenv
from utilities import *

def init():
    print("INIT")
    load_dotenv()
    embedding = get_embedding()
    return embedding

# Configurazione iniziale
file_source = "./files/PT691-Transcript.pdf"
persist_directory = 'chroma/sds/'
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
template = """I will provide you pieces of [Context] to answer the [Question]. \
    If you don't know the answer based on [Context] just say that you don't know, don't try to make up an answer. \
    [Context]: {context} \
    [Question]: {question} \
    Helpful Answer:"""
overwrite_file_source = False
overwrite_vectordb = False
save_directory = '.'
question = "Which is the difference between CPU and GPU?"

if __name__ == "__main__":

    embedding = init()    

    # Controllo e gestione del file_source
    file_source = get_file_source(file_source, save_directory, overwrite=overwrite_file_source)

    # Generazione o recupero del vectordb
    vectordb = get_vectordb(
        persist_directory=persist_directory,
        embedding=embedding,
        file_source=file_source,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        overwrite=overwrite_vectordb)

    # Chiedere una domanda e ottenere una risposta
    result = get_result(
        question=question,
        template=template,
        vectordb=vectordb
    )

    # Stampare il risultato
    print(result["result"])
