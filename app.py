# -------- Standard Libraries --------
import json
import os
import time
import logging
from pathlib import Path

# -------- Third-party Utilities --------
# from dotenv import load_dotenv
import requests
import yt_dlp
import streamlit as st

# -------- LangChain Core --------
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq
# -------- Logging --------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------- Load environment variables --------
api_token = st.secrets["ASSEMBLY_AI_KEY"]

base_url = "https://api.assemblyai.com/v2"
headers = {"authorization": api_token, "content-type": "application/json"}
upload_headers = {"authorization": api_token}


# -------- Audio Download --------
def save_audio(url):
    try:
        os.makedirs('temp', exist_ok=True)
        ydl_opts = {
            'format': 'bestaudio/best',
            'noplaylist': True,
            'quiet': True,
            'extractaudio': True,
            'audioformat': 'mp3',
            'outtmpl': 'temp/%(title)s.%(ext)s',
            'cookiefile': None,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0'
            }
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            audio_filename = Path(ydl.prepare_filename(info)).with_suffix('.mp3')
        logger.info(f"Successfully downloaded audio: {audio_filename}")
        return Path(audio_filename).name
    except Exception as e:
        logger.error(f"Error downloading audio: {str(e)}")
        st.error(f"Error downloading audio: {str(e)}")
        return None


# -------- AssemblyAI Transcription --------
def upload_audio_chunked(audio_path):
    upload_url = base_url + "/upload"

    def file_reader(file_path, chunk_size=5 * 1024 * 1024):
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    response = requests.post(upload_url, headers=upload_headers, data=file_reader(audio_path))
    response.raise_for_status()
    return response.json()['upload_url']


def assemblyai_stt(audio_filename):
    try:
        audio_path = os.path.join('temp', audio_filename)
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        st.info(f"Audio size: {os.path.getsize(audio_path)/(1024*1024):.2f} MB")
        upload_url = upload_audio_chunked(audio_path)

        response = requests.post(base_url + "/transcript", json={"audio_url": upload_url}, headers=headers)
        response.raise_for_status()

        transcript_id = response.json()['id']
        polling_endpoint = f"{base_url}/transcript/{transcript_id}"

        start_time = time.time()
        while True:
            result = requests.get(polling_endpoint, headers=headers).json()
            if result['status'] == 'completed':
                break
            elif result['status'] == 'error':
                raise RuntimeError(result['error'])
            elif time.time() - start_time > 300:
                raise TimeoutError("Transcription timed out.")
            time.sleep(3)

        return result['text'], result['words']
    except Exception as e:
        st.error(f"Error in speech-to-text conversion: {e}")
        return None, None


# -------- QA Chain Setup --------
@st.cache_resource
def setup_qa_chain():
    try:
        if not os.path.exists("docs/transcription.txt"):
            st.error("Transcription file not found.")
            return None, None

        loader = TextLoader('docs/transcription.txt')
        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(texts, embeddings)
        retriever = vectorstore.as_retriever()

        #chat = ChatOllama(model="mistral", temperature=0)
        chat = ChatGroq(
            groq_api_key=st.secrets["GROQ_API_KEY"],
            model_name="llama-3.1-8b-instant",
            temperature=0
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=chat,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        with open("docs/word_timestamps.json", "r") as file:
            word_timestamps = json.load(file)

        return qa_chain, word_timestamps
    except Exception as e:
        st.error(f"QA setup failed: {e}")
        return None, None


# -------- Timestamp Finder --------
def find_relevant_timestamps(answer, word_timestamps):
    relevant_timestamps = []
    answer_words = answer.lower().split()
    for word_info in word_timestamps:
        if word_info['text'].lower() in answer_words:
            relevant_timestamps.append(word_info['start'])
    return relevant_timestamps


# -------- Summary Generator --------
@st.cache_resource
def get_llm():
    #return ChatOllama(model="mistral", temperature=0.7)
    return ChatGroq(
        groq_api_key=st.secrets["GROQ_API_KEY"],
        model_name="llama-3.1-8b-instant",
        temperature=0.7
    )


def generate_summary(transcription):
    summary_prompt = PromptTemplate(
        input_variable=["transcription"],
        template="Summarize the following transcription in 3-5 sentences:\n\n{transcription}"
    )
    summary_chain = LLMChain(llm=get_llm(), prompt=summary_prompt)
    return summary_chain.run(transcription)


# -------- Streamlit App --------
st.set_page_config(layout="wide", page_title="ChatAudio", page_icon="🎧")
st.title("Chat with your Audio using Local LLM (Ollama)")

input_source = st.text_input("Enter the YouTube video URL")

qa_chain = None
word_timestamps = None

if input_source:
    col1, col2 = st.columns(2)

    with col1:
        st.info("Your uploaded video")
        st.video(input_source)
        audio_filename = save_audio(input_source)

        if audio_filename:
            transcription, word_timestamps = assemblyai_stt(audio_filename)
            if transcription:
                st.info("Transcription completed. You can now ask questions")
                st.text_area("Transcription", transcription, height=300)

                # Save transcription & timestamps
                os.makedirs("docs", exist_ok=True)
                with open("docs/transcription.txt", "w", encoding="utf-8") as f:
                    f.write(transcription)
                with open("docs/word_timestamps.json", "w", encoding="utf-8") as f:
                    json.dump(word_timestamps, f)

                # Setup QA chain
                qa_chain, word_timestamps = setup_qa_chain()

                if st.button("Generate summary"):
                    with st.spinner("Generating summary..."):
                        summary = generate_summary(transcription)
                        st.subheader("Summary:")
                        st.write(summary)

    with col2:
        st.info("Chat Below")
        query = st.text_input("Ask your question here..")
        if query:
            if qa_chain:
                with st.spinner("Generating answer..."):
                    result = qa_chain({"query": query})
                    answer = result['result']
                    st.success(answer)

                    relevant_timestamps = find_relevant_timestamps(answer, word_timestamps)
                    if relevant_timestamps:
                        st.subheader("Relevant timestamps:")
                        for timestamp in relevant_timestamps[:5]:
                            st.write(f"{timestamp // 60}:{timestamp % 60:02d}")
                    else:
                        st.error("QA system is not ready. Please complete transcription first.")


# -------- Cleanup Temporary Files --------
def cleanup_temp_files():
    if os.path.exists('temp'):
        for file in os.listdir('temp'):
            os.remove(os.path.join('temp', file))
