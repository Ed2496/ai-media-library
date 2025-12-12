import streamlit as st
import os
import sqlite3
from moviepy.editor import VideoFileClip
from openai import OpenAI
from datetime import datetime
import pandas as pd

st.set_page_config(page_title="AI 媒體處理庫", page_icon="🎬")

# OpenAI API key
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", "your-key-here"))  # 放 secrets

# 初始化資料庫
def init_db():
    conn = sqlite3.connect('media_library.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS files
                 (id INTEGER PRIMARY KEY, filename TEXT, category TEXT, transcript TEXT, date TEXT)''')
    conn.commit()
    return conn

conn = init_db()

categories = ["技術", "AI新聞", "詐騙", "AI影音"]

st.title("AI 媒體處理庫 (NotebookLM 風格)")
st.write("上傳 MP4 → 雲端轉逐字稿 + GPT 分類 + 存庫 (3.13 綠燈版)")

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

        # 3. 雲端轉文字 (OpenAI Whisper API)
        with open(mp3_path, "rb") as audio_file:
            transcript_response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="zh"
            )
        transcript = transcript_response.text

        # 4. GPT 分類
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是分類專家。根據內容分類為：技術、AI新聞、詐騙、AI影音。只回類別名稱。"},
                {"role": "user", "content": transcript}
            ]
        )
        category = response.choices[0].message.content.strip()

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
df = pd.read_sql_query("SELECT * FROM files ORDER BY date DESC LIMIT 10", conn)
if not df.empty:
    st.dataframe(df, use_container_width=True)
    selected = st.selectbox("查看內容", df["filename"])
    content = df[df["filename"] == selected]["transcript"].iloc[0]
    st.text_area("完整逐字稿", content, height=400)
else:
    st.info("還沒有檔案，上傳第一個開始吧！")

# 匯出
if st.button("匯出所有 (CSV)"):
    all_df = pd.read_sql_query("SELECT * FROM files ORDER BY date DESC", conn)
    csv = all_df.to_csv(index=False).encode('utf-8')
    st.download_button("下載 CSV", csv, "media_library.csv", "text/csv")
