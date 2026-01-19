import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ---------------- LOAD EMBEDDING MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_model()

# ---------------- CANONICAL SKILL EXTRACTION ----------------
def extract_canonical_skills(text):
    text = text.lower()
    skills = set()

    if any(k in text for k in ["seo", "digital marketing", "marketing"]):
        skills.add("digital marketing")

    if any(k in text for k in [
        "software", "engineering", "developer",
        "java", "javascript", "html", "css", "sql"
    ]):
        skills.add("software development")

    if any(k in text for k in ["ui", "ux", "design"]):
        skills.add("ui ux design")

    if any(k in text for k in ["ai", "ml", "cnn", "deep learning"]):
        skills.add("machine learning")

    return skills

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

    # 2️⃣ Fallback to pdfplumber if PyPDF fails
    if not text.strip():
        try:
            import pdfplumber
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    if page.extract_text():
                        text += page.extract_text() + " "
        except:
            pass

    # 3️⃣ Final cleanup
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
    matched = resume_skills & jd_skills          # JD-required & present
    missing = jd_skills - resume_skills          # JD-required but absent
    extra = resume_skills - jd_skills             # Extra candidate strengths

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

if st.button("🔍 Analyze Resume"):
    if resume_file and job_description.strip():
        with st.spinner("Analyzing resume..."):
            resume_text = extract_text_from_pdf(resume_file)
            jd_text = clean_text(job_description)
            if len(resume_text.strip()) < 100:
                st.error("❌ Resume text could not be read properly. Please upload a text-based PDF.")
                st.stop()


            resume_skills = extract_canonical_skills(resume_text)
            jd_skills = extract_canonical_skills(jd_text)

            # Fallback for very short job descriptions
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
