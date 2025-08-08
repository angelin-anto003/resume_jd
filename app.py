import streamlit as st
import re
import os
from transformers import pipeline
import PyPDF2
import docx

# Load the Hugging Face model for text generation
GEN_MODEL = "google/flan-t5-small"
generator = pipeline("text2text-generation", model=GEN_MODEL)

def extract_text_from_pdf_fileobj(file_obj):
    reader = PyPDF2.PdfReader(file_obj)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text

def extract_text_from_docx_fileobj(file_obj):
    tmp = "/tmp/_tmp_doc.docx"
    with open(tmp, "wb") as f:
        f.write(file_obj.read())
    doc = docx.Document(tmp)
    text = "\n".join([p.text for p in doc.paragraphs])
    try:
        os.remove(tmp)
    except:
        pass
    return text

def normalize_tokens(text):
    text = text.lower()
    tokens = re.findall(r"\b[a-zA-Z0-9\+#\.\-]+\b", text)
    stop = set(["and","the","for","with","that","this","will","has","have","are","is","in","on","to","of","a","an","be"])
    tokens = [t for t in tokens if t not in stop and len(t) >= 2]
    return tokens

def find_missing_keywords(resume_text, jd_text, extra_keywords=None):
    resume_set = set(normalize_tokens(resume_text))
    jd_tokens = normalize_tokens(jd_text)
    jd_freq = {}
    for t in jd_tokens:
        jd_freq[t] = jd_freq.get(t, 0) + 1
    if extra_keywords:
        for k in extra_keywords:
            k2 = k.lower().strip()
            jd_freq[k2] = jd_freq.get(k2, 0) + 1
    jd_keys = set(jd_freq.keys())
    missing = sorted([k for k in jd_keys if k not in resume_set], key=lambda x: -jd_freq[x])
    return missing, jd_freq, resume_set

def generate_suggestions(resume_text, jd_text):
    prompt = f"""You are a professional resume coach. Given the resume text and a job description, produce:
1) 6 concise, formal bullet suggestions (skills/phrasing/projects) that the candidate can add to their resume without fabricating job history.
2) One improved 1-line professional summary that includes relevant missing skills.
Resume:
{resume_text}

Job Description:
{jd_text}
"""
    out = generator(prompt, max_length=256, do_sample=False, num_return_sequences=1)
    return out[0]['generated_text']

st.set_page_config(layout="wide")
st.title("AI Resume Keyword Analyzer (Hugging Face)")

col1, col2 = st.columns([2,1])
with col1:
    uploaded = st.file_uploader("Upload resume (PDF / DOCX / TXT)", type=['pdf','docx','txt'])
    jd_input = st.text_area("Paste Job Description here", height=250)
    custom_keywords = st.text_input("Extra keywords to consider (comma-separated)", "")
    run = st.button("Analyze")

with col2:
    st.info("How it works: uploads resume → extracts text → finds missing JD keywords → generates safe suggestions (no fabricated experiences).")

if run:
    if not uploaded:
        st.warning("Upload a resume file first.")
    elif not jd_input or len(jd_input.strip()) < 10:
        st.warning("Paste the job description (at least 10 chars).")
    else:
        uploaded.seek(0)
        name = uploaded.name.lower()
        if name.endswith(".pdf"):
            resume_text = extract_text_from_pdf_fileobj(uploaded)
        elif name.endswith(".docx"):
            resume_text = extract_text_from_docx_fileobj(uploaded)
        else:
            resume_text = uploaded.read().decode('utf-8', errors='ignore')

        extra = [k.strip() for k in custom_keywords.split(",")] if custom_keywords else None
        missing, jd_freq, resume_tokens = find_missing_keywords(resume_text, jd_input, extra)
        jd_total = len(set(normalize_tokens(jd_input)))
        present = len([t for t in normalize_tokens(jd_input) if t in resume_tokens])
        match_score = (present / jd_total * 100) if jd_total else 0

        st.subheader("Results")
        st.metric("Keyword Match Score", f"{match_score:.1f}%")
        st.write("**Missing keywords (prioritized):**", ", ".join(missing[:40]) if missing else "None — good match!")

        st.write("---")
        st.subheader("AI suggestions (safe, non-fabricated)")
        with st.spinner("Generating suggestions..."):
            suggestions = generate_suggestions(resume_text[:4000], jd_input[:2000])
        st.write(suggestions)

        st.download_button("Download suggestions (TXT)", data=suggestions, file_name="resume_suggestions.txt", mime="text/plain")
