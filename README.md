# Chat with Audio using Local LLM (Ollama) & FAISS

A Streamlit web app that allows you to **ask questions and generate summaries from YouTube audio** using **local LLMs** and **FAISS-based semantic search**.

---

## 🚀 Features

- Download audio from YouTube videos (first 10 minutes).
- Transcribe audio using **AssemblyAI** (speech-to-text) with word-level timestamps.
- Save transcriptions and timestamps to disk for reuse.
- Perform **question answering** over the transcription using a **local LLM (Ollama)**.
- Generate **summaries** of the transcription in 3–5 sentences.
- Show **relevant timestamps** for each answer so you can jump back into the video.

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** – interactive web UI
- **LangChain** – QA chain & text splitting
- **HuggingFace Embeddings** – free text embeddings (`all-MiniLM-L6-v2`)
- **FAISS** – vector store for semantic search
- **Ollama (Local LLM)** – LLM inference locally (e.g. `mistral`)
- **AssemblyAI** – speech-to-text API
- **yt-dlp** – download audio from YouTube
- **python-dotenv** – environment variable management

---

## 📁 Project Structure


audioenhanced/
├── app.py # Main Streamlit app
├── requirements.txt # Python dependencies
├── .env.example # Environment variable template
├── README.md # Project documentation
├── docs/ # Transcriptions & word_timestamps.json
└── temp/ # Temporary audio files


---

## ⚡ Setup & Installation

1. **Clone the repo:**

>git clone https://github.com/yourusername/AudioQA-Ollama.git

>cd AudioQA-Ollama


2. **Create virtual environment (optional but recommended):**

>python -m venv .venv

>Linux/Mac:- 
source .venv/bin/activate

>Windows:- 
.venv\Scripts\activate


3. **Install dependencies:**

>pip install -r requirements.txt


4. **Create `.env` file** in the project root with your AssemblyAI key:

>ASSEMBLY_AI_KEY=your_assembly_ai_key_here


> Note: Ollama runs locally, so no OpenAI API key is needed.

5. **Install and prepare Ollama:**

- Install from https://ollama.ai
- Pull the model used in the app (e.g. `mistral`):


>ollama pull mistral

>ollama serve
6. **Run the Streamlit app:**

>streamlit run app.py

Then open the displayed URL (usually `http://localhost:8501`) in your browser.

---

## 📝 Usage

1. Enter a YouTube video URL in the input box.
2. Click **Download & Transcribe** and wait for:
   - Audio download (first 10 minutes, compressed).
   - Transcription via AssemblyAI.
3. View the full transcription in the text area.
4. Click **Generate Summary** to get a 3–5 sentence summary.
5. Ask questions in the chat box:
   - The app will answer using the local LLM.
   - It will display **relevant timestamps** for the answer based on word-level timestamps.

---

## ✅ Notes

- Only the first 10 minutes of the video are downloaded to keep file size small and avoid API limits.
- Transcription is done via AssemblyAI’s API; everything else (embeddings, FAISS, LLM) runs locally with Ollama.


