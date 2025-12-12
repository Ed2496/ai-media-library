import streamlit as st
import os
import sqlite3
from moviepy.editor import VideoFileClip
from faster_whisper import WhisperModel
from transformers import pipeline
from datetime import datetime
import pandas as pd

st.set_page_config(page_title="AI 媒體處理庫", page_icon="🎬")

# 初始化資料庫
def init_db():
    conn = sqlite3.connect('media_library.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS files
                 (id INTEGER PRIMARY KEY, filename TEXT, category TEXT, transcript TEXT, date TEXT)''')
    conn.commit()
    return conn

conn = init_db()

# 載入模型（不 cache，避開 Streamlit bug）
@st.experimental_singleton
def load_models():
    whisper = WhisperModel("base", device="cpu", compute_type="int8")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return whisper, classifier

whisper_model, classifier = load_models()

categories = ["技術", "AI新聞", "詐騙", "AI影音"]

st.title("AI 媒體處理庫（NotebookLM 風格）")
st.write("上傳 MP4 → 自動轉逐字稿 + AI 分類 + 存庫")

uploaded = st.file_uploader("選擇 MP4 檔案", type=["mp4"])

if uploaded and st.button("開始處理"):
    with st.spinner("處理中..."):
        # 1. 存檔
        mp4_path = f"temp_{uploaded.name}"
        with open(mp4_path, "wb") as f:
            f.write(uploaded.getbuffer())

        # 2. 轉 MP3
        mp3_path = mp4_path.replace(".mp4", ".mp3")
        video = VideoFileClip(mp4_path)
        video.audio.write_audiofile(mp3_path, verbose=False, logger=None)
        video.close()

        # 3. 轉文字
        segments, _ = whisper_model.transcribe(mp3_path, language="zh")
        transcript = " ".join([s.text for s in segments])

        # 4. 分類
        result = classifier(transcript, categories)
        category = result["labels"][0]

        # 5. 存檔 + DB
        txt_name = f"{category}_{uploaded.name.replace('.mp4', '.txt')}"
        with open(txt_name, "w", encoding="utf-8") as f:
            f.write(transcript)

        c = conn.cursor()
        c.execute("INSERT INTO files (filename, category, transcript, date) VALUES (?, ?, ?, ?)",
                  (uploaded.name, category, transcript, datetime.now().strftime("%Y-%m-%d %H:%M")))
        conn.commit()

        # 清理
        os.remove(mp4_path)
        os.remove(mp3_path)

        st.success(f"完成！分類：**{category}**")
        st.download_button("下載逐字稿", transcript, file_name=txt_name)
        st.text_area("預覽", transcript, height=300)

# 資料庫瀏覽
st.divider()
st.subheader("我的媒體庫")
df = pd.read_sql_query("SELECT * FROM files ORDER BY date DESC", conn)
if not df.empty:
    st.dataframe(df, use_container_width=True)
    selected = st.selectbox("查看內容", df["filename"])
    content = df[df["filename"] == selected]["transcript"].iloc[0]
    st.text_area("完整逐字稿", content, height=400)
else:
    st.info("還沒有檔案，上傳第一個開始吧！")
