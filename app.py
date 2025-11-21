# app.py
import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import PyPDF2, nltk, re, random, torch
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# ----------------- Utilities -----------------
def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)                   # remove weird glyphs
    text = re.sub(r'\bPage\s*\d+\b', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d+\s*/\s*\d+\b', ' ', text)
    text = re.sub(r'-\s*\n\s*', '', text)                         # join hyphenated words
    text = re.sub(r'\s*\n\s*', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    out = []
    for p in reader.pages:
        t = p.extract_text() or ""
        out.append(t)
    return " ".join(out)

def chunk_text_by_sentences(text, max_words=350):
    sents = nltk.sent_tokenize(text)
    chunks, cur, cur_words = [], [], 0
    for s in sents:
        wcount = len(s.split())
        if cur_words + wcount > max_words and cur:
            chunks.append(' '.join(cur))
            cur, cur_words = [], 0
        cur.append(s)
        cur_words += wcount
    if cur:
        chunks.append(' '.join(cur))
    return chunks

# ----------------- MCQ generator -----------------
def generate_mcq_quiz_from_chunks(chunks, num_questions=5):
    from nltk.corpus import wordnet
    quiz = []
    tried = set()
    flat_words = [w for c in chunks for w in nltk.word_tokenize(c) if w.isalpha()]

    for sentence in chunks:
        if len(quiz) >= num_questions:
            break
        if len(sentence.split()) < 6 or len(sentence.split()) > 25:
            continue
        words = [w for w in nltk.word_tokenize(sentence) if w.isalpha() and len(w) > 4]
        if not words:
            continue
        answer = random.choice(words)
        if answer.lower() in tried:
            continue
        tried.add(answer.lower())
        question = sentence.replace(answer, "_____")

        distractors = set()
        for syn in wordnet.synsets(answer):
            for lemma in syn.lemmas():
                w = lemma.name().replace('_', ' ')
                if w.lower() != answer.lower() and len(w) > 3:
                    distractors.add(w)
                if len(distractors) >= 3:
                    break
            if len(distractors) >= 3:
                break

        # fallback distractors
        while len(distractors) < 3 and flat_words:
            cand = random.choice(flat_words)
            if cand.lower() != answer.lower():
                distractors.add(cand)
            if len(distractors) > 200:
                break

        options = list(distractors)[:3] + [answer]
        random.shuffle(options)
        quiz.append({"question": question, "options": options, "answer": answer})
    return quiz

# ----------------- Load models (cached) -----------------
@st.cache_resource
def load_models():
    device = 0 if torch.cuda.is_available() else -1
    # smaller summarizer for hosted environments
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device)
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", device=device)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return summarizer, qa_pipeline, embedder

summarizer, qa_pipeline, embedder = load_models()

# ----------------- Retrieval + QA -----------------
def prepare_document_for_qa(text, max_words=350):
    text = clean_text(text)
    chunks = chunk_text_by_sentences(text, max_words=max_words)
    embeddings = embedder.encode(chunks, convert_to_tensor=True)
    return chunks, embeddings

def answer_question_retrieval(question, chunks, embeddings, top_k=3):
    q_emb = embedder.encode(question, convert_to_tensor=True)
    cos_scores = util.cos_sim(q_emb, embeddings)[0]
    topk = torch.topk(cos_scores, k=min(top_k, len(chunks)))
    selected_indices = topk[1].cpu().numpy().tolist()
    selected_ctxs = [chunks[i] for i in selected_indices]
    context = " ".join(selected_ctxs)
    result = qa_pipeline(question=question, context=context)
    return {"answer": result.get('answer'), "score": float(result.get('score', 0)), "contexts": selected_ctxs}

# ----------------- Summarization (hierarchical) -----------------
def summarize_text(text):
    text = clean_text(text)
    chunks = chunk_text_by_sentences(text, max_words=300)
    chunk_summaries = []
    for c in chunks:
        try:
            out = summarizer(c, max_length=120, min_length=20, do_sample=False)
            chunk_summaries.append(out[0]['summary_text'])
        except Exception:
            chunk_summaries.append(' '.join(c.split()[:120]))
    combined = " ".join(chunk_summaries)
    if len(combined.split()) < 30:
        return combined
    final = summarizer(combined, max_length=180, min_length=40, do_sample=False)
    return final[0]['summary_text']

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="AI Study Assistant", layout="wide")
st.title("📘 AI Study Assistant — Summarize • QA • MCQ")

with st.sidebar:
    st.header("Options")
    max_chunk = st.slider("Chunk size (words)", 200, 800, 350)
    top_k = st.slider("Top-k chunks for QA", 1, 5, 3)
    num_mcq = st.slider("Number of MCQs", 1, 10, 5)
    st.write("Local sample path (dev): /mnt/data/Gargi Soni_resume.docx.pdf")

uploaded = st.file_uploader("Upload PDF notes", type=["pdf"], help="Upload one PDF file (notes or resume).")
use_local = st.checkbox("Use local sample file (dev only)")

# local sample path you uploaded earlier
local_test_path = "/mnt/data/Gargi Soni_resume.docx.pdf"

raw_text = ""
if use_local and st.button("Load local sample (dev only)"):
    try:
        with open(local_test_path, "rb") as f:
            raw_text = extract_text_from_pdf(f)
        st.success("Loaded local test file.")
    except Exception as e:
        st.error("Local file not available in this environment.")
        raw_text = ""

if uploaded:
    raw_text = extract_text_from_pdf(uploaded)

if not raw_text:
    st.info("Upload a PDF to begin (or use local sample for dev).")
else:
    cleaned = clean_text(raw_text)
    st.write("Document length (chars):", len(cleaned))

    if st.button("Generate Summary"):
        with st.spinner("Summarizing..."):
            summary = summarize_text(cleaned)
        st.subheader("Summary")
        st.write(summary)

    # prepare for QA and MCQ
    chunks, embeddings = prepare_document_for_qa(cleaned, max_words=max_chunk)

    st.write("---")
    st.subheader("Ask a Question")
    question = st.text_input("Enter a question related to the uploaded notes:")
    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please type a question first.")
        else:
            res = answer_question_retrieval(question, chunks, embeddings, top_k=top_k)
            if res['score'] < 0.25:
                st.warning("No confident answer found in the notes. Try rephrasing or increase top-k.")
            else:
                st.success(f"Answer: {res['answer']}  (score: {res['score']:.2f})")
                for i, ctx in enumerate(res['contexts'], 1):
                    st.write(f"Source {i} (snippet): {ctx[:400]} ...")

    st.write("---")
    st.subheader("Generate MCQs")
    if st.button("Create MCQs"):
        quiz = generate_mcq_quiz_from_chunks(chunks, num_questions=num_mcq)
        for i, q in enumerate(quiz, 1):
            st.markdown(f"**Q{i}.** {q['question']}")
            for idx, opt in enumerate(q['options'], 1):
                st.write(f"{idx}. {opt}")
            with st.expander("Show answer"):
                st.write(f"Correct answer: **{q['answer']}**")
            st.write("---")
