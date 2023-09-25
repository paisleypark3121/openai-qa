import streamlit as st
from main import *
from utilities import *

# Inizializza le variabili: controllo se vectordb è già nel session_state
if "embedding" not in st.session_state:
    st.session_state.embedding = init()

st.title('Il mio assistente basato su langchain')

st.write("""
    Questo è un assistente che usa langchain per rispondere alle domande in base al contenuto di un documento PDF.
""")

option = st.radio("Scegli un'opzione per fornire il PDF:", ('Carica PDF', 'Inserisci URL del PDF'))

if "file_source" not in st.session_state:
    st.session_state.file_source = None
    #st.write("Initialized file_source to None")  # Debug statement

#save_directory='.'
save_directory='./files'
if option == 'Carica PDF':
    uploaded_file = st.file_uploader("Scegli un file PDF", type=['pdf'])
    
    if uploaded_file:
        #st.write(f"Uploaded file: {uploaded_file.name}")  # Debug statement
        new_file_source = uploaded_file.name

        # Check if the new file source is different from the current file source in session state
        if new_file_source != st.session_state.get("file_source"):
            #st.write("New file detected. Resetting vectordb.")  # Debug statement
            st.session_state.vectordb = None
            st.session_state.file_source = new_file_source  # Update the file source in session state

        file_path = os.path.join(save_directory, st.session_state.file_source)
        #st.write(f"File path: {file_path}")  # Debug statement
        
        # Check if the file already exists on disk
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            #st.write("File written to disk.")  # Debug statement
        #else:
            #st.write("Il file è già presente e non sarà sovrascritto.")


elif option == 'Inserisci URL del PDF':
    file_source_url = st.text_input('Inserisci l\'URL del file PDF')
    
    if file_source_url:
        new_file_source = os.path.basename(file_source_url)
        # Check if the new file source is different from the current file source in session state
        if new_file_source != st.session_state.get("file_source"):
            # If different, reset the vectordb
            st.session_state.vectordb = None
            st.session_state.file_source = new_file_source  # Update the file source in session state
        
        file_path = os.path.join(save_directory, st.session_state.file_source)
        
        # Check if the file already exists on disk
        if not os.path.exists(file_path):
            st.session_state.file_source = get_file_source(file_source=file_source_url,save_directory=save_directory)
            # Note: Ensure that get_file_source function downloads the file and saves it to file_path
        #else:
            #st.write("Il file è già presente e non sarà sovrascritto.")


question = st.text_input('Inserisci la tua domanda:')

if st.button('Chiedi'):
    if question and st.session_state.file_source:

        file_source_full_path = os.path.join(save_directory, st.session_state.file_source)

        # Check if vectordb needs to be regenerated
        if st.session_state.get("vectordb") is None:
            vectordb = get_vectordb(
                persist_directory=persist_directory,
                embedding=st.session_state.embedding,
                file_source=file_source_full_path,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                overwrite=True)
            # Store the newly generated vectordb in the session state
            st.session_state.vectordb = vectordb
        else:
            # Use the existing vectordb from the session state
            vectordb = st.session_state.vectordb
        
        result = get_result(
            question=question,
            template=template,
            vectordb=vectordb
        )
        
        st.write('Risposta:', result['result'])
    else:
        st.write('Per favore inserisci una domanda e/o seleziona la fonte dati.')
