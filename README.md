# 📘 AI Study Assistant  
An intelligent, all-in-one study assistant that can **summarize PDF notes**, answer **questions** based on the content, and generate **MCQ quizzes** — all for free using open-source NLP models.

Built using **Python, Streamlit, Hugging Face Transformers, and Sentence Transformers**.

---

## 🎯 Features  
### ✅ **1. Upload Any PDF**  
Upload your own notes, books, or study material. The app extracts clean, readable text using `PyPDF2` and a custom text cleaner.

### ✅ **2. Smart Summarization (Hierarchical)**  
- Breaks long PDFs into smaller sentence chunks  
- Summarizes each chunk  
- Then combines and summarizes again  
✔ Accurate  
✔ Stable  
✔ Works on large documents  

### Models used:  
- `sshleifer/distilbart-cnn-12-6` (fast distilled BART)

---

### ✅ **3. Retrieval-based Question Answering**  
Ask any question from your uploaded PDF.  
The app uses:

- `all-MiniLM-L6-v2` (embeddings)
- Cosine similarity to pick the best chunks  
- `deepset/roberta-base-squad2` for extractive QA

✔ Avoids hallucination  
✔ Shows source snippets  
✔ Confidence score included  

---

### ✅ **4. MCQ Quiz Generator**  
Automatically generates MCQs:  
- Extracts meaningful keywords  
- Creates a fill-in-the-blank question  
- Generates 3 distractor options using WordNet  
- Shuffles answers  
- Lets you create 1–10 questions instantly  

Great for revision and practice.

---

### ✅ **5. Clean, Modern Streamlit UI**  
- Upload PDF  
- Generate summary  
- Ask questions  
- Create MCQs  
- Adjustable options (chunk size, top-k, number of MCQs)

---

## 🧠 Tech Stack  
**Frontend / App:**  
- Streamlit

**NLP Models / ML:**  
- Hugging Face Transformers  
- Sentence Transformers  
- NLTK  
- PyTorch  

**Other:**  
- PyPDF2  
- Regular Expressions  
- Text cleaning

---

## 📦 Installation

### 1. Clone the repo  
```bash
git clone https://github.com/yourusername/AI-Study-Assistant.git
cd AI-Study-Assistant
