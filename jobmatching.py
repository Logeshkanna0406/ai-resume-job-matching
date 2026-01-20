import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

# OCR imports
from pdf2image import convert_from_bytes
import pytesseract

# ---------------- LOAD EMBEDDING MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_model()

# ---------------- CANONICAL SKILL EXTRACTION ----------------
def extract_canonical_skills(text):
    text = text.lower()
    skills = set()

    # 🔹 Digital Marketing
    if any(k in text for k in [
        "seo", "sem", "digital marketing", "content marketing",
        "google ads", "facebook ads", "social media marketing"
    ]):
        skills.add("digital marketing")

    # 🔹 Software Development
    if any(k in text for k in [
        "software", "developer", "engineering",
        "java", "python", "c++", "javascript",
        "html", "css", "sql", "spring", "django", "flask"
    ]):
        skills.add("software development")

    # 🔹 Web Development
    if any(k in text for k in [
        "web developer", "frontend", "backend", "full stack",
        "react", "angular", "vue", "node", "express"
    ]):
        skills.add("web development")

    # 🔹 UI / UX Design
    if any(k in text for k in [
        "ui", "ux", "user interface", "user experience",
        "figma", "adobe xd", "wireframe", "prototype"
    ]):
        skills.add("ui ux design")

    # 🔹 Machine Learning / AI
    if any(k in text for k in [
        "ai", "ml", "machine learning", "deep learning",
        "cnn", "rnn", "nlp", "computer vision"
    ]):
        skills.add("machine learning")

    # 🔹 Data Science / Data Analytics
    if any(k in text for k in [
        "data scientist", "data science", "data analyst",
        "pandas", "numpy", "matplotlib", "statistics",
        "power bi", "tableau", "excel"
    ]):
        skills.add("data science")

    # 🔹 Cloud & DevOps
    if any(k in text for k in [
        "aws", "azure", "gcp", "cloud computing",
        "devops", "docker", "kubernetes", "ci/cd"
    ]):
        skills.add("cloud devops")

    # 🔹 Cybersecurity
    if any(k in text for k in [
        "cyber security", "cybersecurity", "ethical hacking",
        "penetration testing", "network security", "soc"
    ]):
        skills.add("cybersecurity")

    # 🔹 Mobile App Development
    if any(k in text for k in [
        "android", "ios", "mobile app", "flutter",
        "react native", "kotlin", "swift"
    ]):
        skills.add("mobile app development")

    # 🔹 Testing / QA
    if any(k in text for k in [
        "software testing", "qa", "quality assurance",
        "manual testing", "automation testing",
        "selenium", "junit"
    ]):
        skills.add("software testing")

    # 🔹 Business / Management
    if any(k in text for k in [
        "business analyst", "project manager",
        "product manager", "agile", "scrum"
    ]):
        skills.add("business management")

    # 🔹 Finance / Accounting
    if any(k in text for k in [
        "finance", "accounting", "tally", "gst",
        "auditing", "financial analysis"
    ]):
        skills.add("finance accounting")

    # 🔹 HR / Recruitment
    if any(k in text for k in [
        "human resources", "hr", "recruitment",
        "talent acquisition", "payroll"
    ]):
        skills.add("human resources")

    return skills

# ---------------- OCR FUNCTION ----------------
def extract_text_with_ocr(pdf_file):
    images = convert_from_bytes(pdf_file.read())
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img) + " "
    return clean_text(text)

# ---------------- PDF + TEXT UTILS ----------------
def extract_text_from_pdf(pdf_file):
    text = ""

    # 1️⃣ Try PyPDF first
    try:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + " "
    except:
        pass

    # 2️⃣ Fallback to pdfplumber
    if not text.strip():
        try:
            import pdfplumber
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    if page.extract_text():
                        text += page.extract_text() + " "
        except:
            pass

    # 3️⃣ OCR fallback for scanned PDFs
    if len(text.strip()) < 100:
        try:
            pdf_file.seek(0)
            text = extract_text_with_ocr(pdf_file)
        except:
            pass

    return clean_text(text)

def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

# ---------------- SCORING FUNCTIONS ----------------
def semantic_match_score(resume_skills, jd_skills):
    embeddings = embed_model.encode([
        " ".join(resume_skills),
        " ".join(jd_skills)
    ])
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return round(score * 100, 2)

def keyword_match_score(resume_skills, jd_skills):
    if not jd_skills:
        return 0.0
    matched = resume_skills.intersection(jd_skills)
    return round((len(matched) / len(jd_skills)) * 100, 2)

# ---------------- FEEDBACK ----------------
def generate_ai_feedback(final_score, resume_skills, jd_skills):
    matched = resume_skills & jd_skills
    missing = jd_skills - resume_skills
    extra = resume_skills - jd_skills

    return f"""
📊 Match Score: {final_score}%

✅ Matching JD Skills:
- {', '.join(matched) if matched else 'No direct skill overlap detected'}

⭐ Additional Skills (Not Required but Valuable):
- {', '.join(extra) if extra else 'None'}

❌ Missing JD-Required Skills:
- {', '.join(missing) if missing else 'No required skills missing'}

💡 Resume Improvement Suggestions:
- Emphasize experience related to JD-required skills
- Highlight how additional skills add value
- Use JD terminology in resume descriptions
"""

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="AI Resume Matcher", layout="centered")
st.title("🤖 AI-Powered Resume Screening System")

st.write(
    "Upload a resume and provide a job description to get an "
    "**AI-powered match score**, explanation, and improvement suggestions."
)

resume_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("📝 Paste Job Description", height=200)
manual_resume_text = st.text_area(
    "✍️ Or paste resume text manually (optional)",
    height=200
)


if st.button("🔍 Analyze Resume"):
    if (resume_file or manual_resume_text.strip()) and job_description.strip():
        with st.spinner("Analyzing resume..."):

            # 1️⃣ Decide where resume text comes from
            if manual_resume_text.strip():
                resume_text = manual_resume_text
            else:
                resume_text = extract_text_from_pdf(resume_file)

            # 2️⃣ Job description text
            jd_text = clean_text(job_description)

            # 3️⃣ Safety check
            if len(resume_text.strip()) < 100:
                st.warning(
                    "⚠️ Unable to read resume text properly. "
                    "Please upload a text-based PDF or paste the resume text manually."
                )
                st.stop()


            if len(resume_text.strip()) < 50:
                st.warning("⚠️ Unable to read text from this resume PDF. "
                "This usually happens if the file is scanned, image-based, or low quality.\n\n"
                "👉 Please upload a text-based PDF (exported from Word/Google Docs), "
                "or copy–paste the resume text manually."
                )


                st.stop()

            resume_skills = extract_canonical_skills(resume_text)
            jd_skills = extract_canonical_skills(jd_text)

            if not jd_skills:
                jd_skills = resume_skills.copy()

            semantic_score = semantic_match_score(resume_skills, jd_skills)
            keyword_score = keyword_match_score(resume_skills, jd_skills)

            final_score = round(
                (semantic_score * 0.7) + (keyword_score * 0.3), 2
            )

            feedback = generate_ai_feedback(
                final_score, resume_skills, jd_skills
            )

        st.success(f"✅ Match Score: **{final_score}%**")
        st.write(f"🔍 Semantic Score: {semantic_score}%")
        st.write(f"🧩 Keyword Match Score: {keyword_score}%")

        st.subheader("📊 AI Analysis")
        st.write(feedback)
    else:
        st.warning("Please upload a resume and enter a job description.")

st.markdown("---")
st.caption("Built using Generative AI, Semantic Embeddings & Streamlit")
