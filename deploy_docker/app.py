from flask import Flask, request, jsonify, render_template
import os
import time
from dotenv import load_dotenv
import yt_dlp
from moviepy.editor import VideoFileClip
import whisper
from pyannote.audio import Pipeline
import torch
import ffmpeg
from langchain_openai import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langgraph.checkpoint import MemorySaver
from langchain_core.tools import Tool
from langgraph.prebuilt import create_react_agent
from werkzeug.utils import secure_filename
import threading

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DOWNLOAD_FOLDER'] = 'downloads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}

# Load environment variables from .env file
load_dotenv()

# Retrieve the values and set them in os.environ
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', '').replace('\r', '').strip()
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY', '').replace('\r', '').strip()
os.environ['ELEVEN_API_KEY'] = os.getenv('ELEVEN_API_KEY', '').replace('\r', '').strip()
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN', '').replace('\r', '').strip()

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DOWNLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

class VideoChatbot:
    def __init__(self):
        self.retriever = None
        self.qa_chain = None
        self.memory = MemorySaver()
        self.model = ChatOpenAI(model="gpt-4o", temperature=0)
        self.language = 'en'  # Default language
        self.prompt = '''You are a chatbot that answers questions and performs tasks about a video that the user provides. 
                         Never ask the user to provide a video without first checking if there is one already.
                         If lacking context, assume the user is always talking about the video.
                         First, consider which tools you need to use, if any.
                         When retrieving information, consider that the transcription might not be perfect every time.
                         Then, if relevant, try to identify speakers by their names or usernames, using their dialogue and considering the available metadata.
                         Then use more steps when needed in order to get the right answer. 
                         Finally, you must always identify the language the user is utilizing in their last message and answer in that language, unless the user tells you otherwise.
                      '''
        self.agent = None

    def initialize_qa_chain(self):
        llm = ChatOpenAI(model="gpt-4o")
        try:
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.retriever
            )
            print("QA chain initialized successfully.")
            self.qa_chain = qa
        except Exception as e:
            print(f"Error initializing QA chains: {e}")
            self.qa_chain = None

    def create_agent(self):
        tools = [
            Tool(
                name='video_transcript_retriever',
                func=self.qa_chain.run,
                description=(
                    'Searches and returns excerpts from the transcript of the user uploaded video.'
                )
            ),
        ]
        self.agent = create_react_agent(self.model, tools=tools, messages_modifier=self.prompt, checkpointer=self.memory)
        print("Agent created successfully.")

    def process_query(self, query):
        if not self.agent:
            print("Agent not initialized.")
            return "Agent not initialized."

        # Detect language change commands
        if 'spanish' in query.lower():
            self.language = 'es'
            return "Idioma cambiado a español."
        elif 'english' in query.lower():
            self.language = 'en'
            return "Language switched to English."

        inputs = {"messages": [("user", query)]}
        config = {"configurable": {"thread_id": "2"}}
        stream = self.agent.stream(inputs, config=config, stream_mode="values")
        response = ""
        for s in stream:
            message = s["messages"][-1]
            if isinstance(message, tuple):
                response += str(message)
            elif hasattr(message, 'text'):
                response += message.text
            elif hasattr(message, 'content'):
                response += message.content

        # Ensure the response is in the preferred language
        if self.language == 'es':
            response = f"{response} (Traducido al español)"
        elif self.language == 'en':
            response = f"{response} (Translated to English)"

        return response

chatbot = VideoChatbot()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def save_file(file):
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    return file_path, filename

def download_youtube_video(url):
    start_time = time.time()
    ydl_opts = {
        'format': 'mp4',
        'outtmpl': os.path.join(app.config['DOWNLOAD_FOLDER'], 'video.%(ext)s'),
        'verbose': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            video_ext = info_dict.get('ext', 'mp4')
            initial_path = os.path.join(app.config['DOWNLOAD_FOLDER'], f'video.{video_ext}')
            final_path = os.path.join(app.config['DOWNLOAD_FOLDER'], f'video_{int(time.time())}.{video_ext}')
            os.rename(initial_path, final_path)
            duration = time.time() - start_time
            return final_path, info_dict, duration
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None, None, 0

def extract_audio(video_path):
    try:
        video = VideoFileClip(video_path)
        audio_path = video_path.replace('.mp4', '.wav')
        video.audio.write_audiofile(audio_path)
        return audio_path
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

def transcribe_audio(audio_path, duration, mode='Fast'):
    model_name = select_whisper_model(duration, mode)
    model = whisper.load_model(model_name)
    if torch.cuda.is_available():
        model = model.to("cuda")
    result = model.transcribe(audio_path)
    return result['text']

def select_whisper_model(duration, mode='Fast'):
    if mode == "Accurate":
        if duration > 3600:
            return "tiny"
        elif duration > 1800:
            return "base"
        elif duration > 600:
            return "small"
        else:
            return "medium"
    else:
        if duration > 1800:
            return "tiny"
        elif duration > 600:
            return "base"
        else:
            return "small"

def create_vectorstore(document):
    vectorstore = Chroma.from_documents(documents=[document], embedding=OpenAIEmbeddings())
    return vectorstore.as_retriever()

def extract_metadata_ffmpeg(video_path):
    try:
        probe = ffmpeg.probe(video_path)
        video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
        metadata = {
            "duration": float(video_info['duration']),
            "width": int(video_info['width']),
            "height": int(video_info['height']),
            "codec_name": video_info['codec_name'],
            "bit_rate": int(video_info['bit_rate'])
        }
        print(f"Extracted metadata: {metadata}")
        return metadata
    except Exception as e:
        print(f"Error extracting metadata with ffmpeg: {e}")
        return None

def process_video(source_type, source, mode='Fast', process_type='Transcription'):
    if source_type == 'upload':
        video_path = source
    elif source_type == 'youtube':
        video_path, metadata, duration = download_youtube_video(source)
    else:
        return None

    if not video_path:
        return None

    audio_path = extract_audio(video_path)
    metadata = extract_metadata_ffmpeg(video_path)
    if metadata is None:
        print("Failed to extract metadata.")
        return None
    
    duration = metadata["duration"]

    if process_type == 'Transcription':
        transcription = transcribe_audio(audio_path, duration, mode)
        document_text = f"Metadata:\n{metadata}\n\nTranscription:\n{transcription}"
    else:
        # Diarization handling can be added similarly
        document_text = f"Metadata:\n{metadata}\n\nTranscription with Diarization:\n{transcription}"

    document = Document(page_content=document_text, metadata=metadata)
    retriever = create_vectorstore(document)

    chatbot.retriever = retriever
    chatbot.initialize_qa_chain()
    chatbot.create_agent()

    return document_text

# Flask app initialization and routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        file_path, filename = save_file(file)
        thread = threading.Thread(target=process_video, args=('upload', file_path, 'Fast', 'Transcription'))
        thread.start()
        return jsonify({"message": "File uploaded and processing started"}), 202

@app.route('/download', methods=['POST'])
def download_file():
    data = request.json
    url = data.get('url')
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    thread = threading.Thread(target=process_video, args=('youtube', url, 'Fast', 'Transcription'))
    thread.start()
    return jsonify({"message": "Video download and processing started"}), 202

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    if not chatbot.agent:
        return jsonify({"response": "Agent not initialized."}), 400
    response = chatbot.process_query(query)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
