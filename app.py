import streamlit as st
import os
import sqlite3
from moviepy.editor import VideoFileClip
from faster_whisper import WhisperModel
from transformers import pipeline
from datetime import datetime
import pandas as pd

# 初始化資料庫
@st.cache_resource
def init_db():
    conn = sqlite3.connect('media_library.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS transcripts
                 (id INTEGER PRIMARY KEY, filename TEXT, category TEXT, transcript TEXT, timestamp TEXT)''')
    conn.commit()
    return conn

# 提取 MP3
def extract_mp3(mp4_path, mp3_path):
    video = VideoFileClip(mp4_path)
    video.audio.write_audiofile(mp3_path, codec='mp3')
    video.close()

# 轉逐字稿 (faster-whisper 3.13 兼容)
@st.cache_resource
def load_whisper():
    return WhisperModel("base", device="cpu", compute_type="int8")

def transcribe_mp3(mp3_path):
    model = load_whisper()
    segments, info = model.transcribe(mp3_path, beam_size=5)
    transcript = " ".join([segment.text for segment in segments])
    return transcript

# AI 分類
@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_text(text, categories=["技術", "AI新聞", "詐騙", "AI影音"]):
    classifier = load_classifier()
    result = classifier(text, candidate_labels=categories)
    return result['labels'][0]

# 處理 MP4
def process_mp4(uploaded_file):
    mp4_path = f"temp_{uploaded_file.name}"
    with open(mp4_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    mp3_path = mp4_path.replace('.mp4', '.mp3')
    extract_mp3(mp4_path, mp3_path)
    
    transcript = transcribe_mp3(mp3_path)
    category = classify_text(transcript)
    
    txt_path = mp4_path.replace('.mp4', f'_{category}.txt')
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(transcript)
    
    # 存 DB
    conn = init_db()
    c = conn.cursor()
    c.execute("INSERT INTO transcripts (filename, category, transcript, timestamp) VALUES (?, ?, ?, ?)",
              (uploaded_file.name, category, transcript, datetime.now().strftime("%Y-%m-%d %H:%M")))
    conn.commit()
    conn.close()
    
    # 清理
    os.remove(mp4_path)
    os.remove(mp3_path)
    
    return txt_path, category, transcript

# Streamlit 介面
st.title("AI 媒體處理庫 (NotebookLM 式)")
st.write("上傳 MP4，自動轉逐字稿 + 分類 + 存庫 (Python 3.13 優化版)")

uploaded_file = st.file_uploader("選擇 MP4 檔案", type="mp4")

if uploaded_file is not None:
    if st.button("處理檔案"):
        with st.spinner("提取音頻 + 轉譯 + 分類中..."):
            txt_path, category, transcript = process_mp4(uploaded_file)
            st.success(f"完成！類別：{category}")
            st.download_button("下載逐字稿", data=open(txt_path, "r", encoding="utf-8").read(), file_name=txt_path)
            st.text_area("逐字稿預覽", transcript, height=200)

# 資料庫瀏覽
st.header("資料庫瀏覽")
conn = init_db()
df = pd.read_sql_query("SELECT * FROM transcripts ORDER BY timestamp DESC", conn)
st.dataframe(df)
if not df.empty:
    selected = st.selectbox("選擇檔案", df['filename'])
    selected_transcript = df[df['filename'] == selected]['transcript'].iloc[0]
    st.text_area("完整內容", selected_transcript)

# 匯出
if st.button("匯出所有逐字稿 (CSV)"):
    df.to_csv("library.csv", index=False)
    st.download_button("下載 CSV", data=open("library.csv", "rb").read(), file_name="media_library.csv")
