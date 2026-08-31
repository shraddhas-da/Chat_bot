"""
YouTube RAG Chatbot - Streamlit app
Ask questions about any YouTube video's transcript.
"""

import os
import re

from dotenv import load_dotenv
import streamlit as st

load_dotenv()  # reads .env into os.environ when running locally
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


# Config

st.set_page_config(page_title="YouTube RAG Chatbot", page_icon="🎥", layout="wide")

def get_hf_token():
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if token:
        return token
    try:
        return st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    except (FileNotFoundError, KeyError):
        return None


HF_TOKEN = get_hf_token()
if not HF_TOKEN:
    st.error(
        "No Hugging Face token found. Add HUGGINGFACEHUB_API_TOKEN to a local "
        ".env file, or to Streamlit Cloud's Settings → Secrets."
    )
    st.stop()

CHUNK_CHAR_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_K = 4

PROMPT = PromptTemplate(
    template="""You are a helpful assistant answering questions about a YouTube video.
Answer ONLY from the provided transcript context. If the context is insufficient,
say you don't know — do not make anything up.
 
Formatting rule: whenever your answer includes a mathematical expression, equation,
or formula (fractions, \\boxed{{}}, exponents, etc.), you MUST wrap it using dollar
signs ONLY: $ ... $ for inline math, $$ ... $$ for a standalone/display expression.
Never use \\[ \\] or \\( \\) as delimiters. For example write:
$$\\frac{{6722}}{{9900}}$$
not:
\\[ \\frac{{6722}}{{9900}} \\]
 
Context:
{context}
 
Conversation so far:
{chat_history}
 
Question: {question}
""",
    input_variables=["context", "chat_history", "question"],
)

CONDENSE_PROMPT = PromptTemplate(
    template="""Given the conversation history and a follow-up question, rewrite the
follow-up as a standalone question that contains all context needed to answer it.
If the follow-up is already standalone, return it unchanged. Return ONLY the question.
 
Chat history:
{chat_history}
 
Follow-up question: {question}
Standalone question:""",
    input_variables=["chat_history", "question"],
)


# Cached resources (loaded once per session, not per rerun)

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatOpenAI(
        model="openai/gpt-oss-20b",
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN,
        temperature=0.2,
        streaming=True,
    )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def extract_video_id(url_or_id: str) -> str:
    url_or_id = url_or_id.strip()
    patterns = [
        r"(?:v=|/videos/|embed/|youtu\.be/|/v/|/e/|watch\?v=|&v=)([^#&?/\s]{11})",
        r"^([^#&?/\s]{11})$",  # bare 11-char ID
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    return url_or_id  # fall back, let the API error out clearly


def fetch_transcript_entries(video_id: str):
    ytt_api = YouTubeTranscriptApi()
    try:
        fetched = ytt_api.fetch(video_id, languages=["en"])
    except Exception:
        fetched = ytt_api.fetch(video_id)  # fall back to any available language

    entries = []
    for item in fetched:
        if isinstance(item, dict):
            entries.append({"text": item["text"], "start": item["start"]})
        else:
            entries.append({"text": item.text, "start": item.start})
    return entries


def chunk_with_timestamps(entries, chunk_size=CHUNK_CHAR_SIZE, overlap=CHUNK_OVERLAP):
    """Merge transcript entries into ~chunk_size character blocks, keeping the
    start timestamp of the first entry in each block for citation links."""
    chunks = []
    buf_text, buf_start = "", None
    for e in entries:
        if buf_start is None:
            buf_start = e["start"]
        buf_text = f"{buf_text} {e['text']}".strip()
        if len(buf_text) >= chunk_size:
            chunks.append({"text": buf_text, "start": buf_start})
            buf_text = buf_text[-overlap:]
            buf_start = e["start"]
    if buf_text:
        chunks.append({"text": buf_text, "start": buf_start})
    return chunks


def build_vector_store(video_id: str):
    entries = fetch_transcript_entries(video_id)
    chunks = chunk_with_timestamps(entries)
    docs = [
        Document(page_content=c["text"], metadata={"start": c["start"], "video_id": video_id})
        for c in chunks
    ]
    return FAISS.from_documents(docs, get_embeddings())


def normalize_latex_delimiters(text: str) -> str:
    """Convert \\[ \\] and \\( \\) LaTeX delimiters to $$ $$ / $ $, since
    Streamlit's markdown renderer only recognizes dollar-sign delimiters."""
    text = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", text, flags=re.DOTALL)
    text = re.sub(r"\\\((.*?)\\\)", r"$\1$", text, flags=re.DOTALL)
    return text


def fmt_timestamp(seconds: float) -> str:
    seconds = int(seconds)
    h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
    seconds = int(seconds)
    h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def format_chat_history(messages, max_turns=4):
    recent = messages[-(max_turns * 2):]
    return "\n".join(f"{m['role']}: {m['content']}" for m in recent) or "(none)"


# --------------------------------------------------------------------------
# Sidebar — video loading
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("🎥 Load a video")
    video_input = st.text_input("YouTube URL or video ID")
    load_clicked = st.button("Load transcript", type="primary", use_container_width=True)

    if load_clicked and video_input:
        video_id = extract_video_id(video_input)
        with st.spinner("Fetching transcript and building index..."):
            try:
                st.session_state.vector_store = build_vector_store(video_id)
                st.session_state.video_id = video_id
                st.session_state.messages = []
                st.success("Video loaded — ask away!")
            except TranscriptsDisabled:
                st.error("Captions are disabled for this video.")
            except NoTranscriptFound:
                st.error("No transcript could be found for this video.")
            except Exception as e:
                st.error(f"Couldn't load this video: {e}")

    if st.session_state.get("video_id"):
        st.divider()
        st.image(f"https://img.youtube.com/vi/{st.session_state.video_id}/mqdefault.jpg")
        st.caption(f"Loaded: `{st.session_state.video_id}`")
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()

# --------------------------------------------------------------------------
# Main chat area
# --------------------------------------------------------------------------
st.title("YouTube Transcript Q&A")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.info("Load a video from the sidebar to get started.")
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask something about the video...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    llm = get_llm()
    chat_history_str = format_chat_history(st.session_state.messages[:-1])

    # Rewrite follow-ups into standalone questions before retrieving
    if chat_history_str != "(none)":
        standalone_q = llm.invoke(
            CONDENSE_PROMPT.invoke({"chat_history": chat_history_str, "question": question})
        ).content.strip()
    else:
        standalone_q = question

    retriever = st.session_state.vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": RETRIEVER_K}
    )
    retrieved_docs = retriever.invoke(standalone_q)
    context_text = "\n\n".join(d.page_content for d in retrieved_docs)

    final_prompt = PROMPT.invoke(
        {"context": context_text, "chat_history": chat_history_str, "question": question}
    )

    with st.chat_message("assistant"):
        placeholder = st.empty()
        response = ""
        for chunk in llm.stream(final_prompt):
            if chunk.content:
                response += chunk.content
                placeholder.markdown(normalize_latex_delimiters(response))
        placeholder.markdown(normalize_latex_delimiters(response))

        video_id = st.session_state.video_id
        seen = set()
        links = []
        for d in retrieved_docs:
            start = d.metadata.get("start", 0)
            bucket = int(start // 15)  # collapse near-duplicate timestamps
            if bucket in seen:
                continue
            seen.add(bucket)
            ts = fmt_timestamp(start)
            url = f"https://youtu.be/{video_id}?t={int(start)}"
            links.append(f"[{ts}]({url})")
        if links:
            st.caption("Sources: " + " · ".join(links))

    st.session_state.messages.append({"role": "assistant", "content": normalize_latex_delimiters(response)})