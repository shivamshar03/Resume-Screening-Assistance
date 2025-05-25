# 🤖 Resume Screening Assistance

A powerful Streamlit application to assist HR professionals in screening resumes against a job description using vector embeddings and semantic similarity.

---

## 🔍 Key Features

- 📄 Upload multiple resumes (PDF only)
- 🧠 Compare resumes semantically with a job description
- ⚡ Get the top N matching resumes based on relevance
- 📊 View match scores and AI-generated summaries
- 🧬 Uses vector embeddings (via FAISS) for similarity search

---

## 🚀 How It Works

1. HR uploads multiple resumes (PDF format)
2. Enters the job description
3. Specifies how many matching resumes to retrieve
4. The app:
   - Extracts text from resumes
   - Generates embeddings
   - Compares with job description
   - Returns most relevant resumes with a match score and summary

---

## 🛠️ Tech Stack

- 🧑‍💻 Python 3.10+
- [Streamlit](https://streamlit.io/) – UI
- [FAISS](https://github.com/facebookresearch/faiss) – Semantic similarity search
- [LangChain](https://www.langchain.com/) – LLM orchestration
- [HuggingFace / OpenAI / Similar] – Embeddings (pluggable)
- [UUID](https://docs.python.org/3/library/uuid.html) – Session-level isolation

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/resume-screening-app.git
cd resume-screening-app
```
