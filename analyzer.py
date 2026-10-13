"""
Core NLP + ML logic for resume analysis.
"""

import re
import string
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx
import io

# Download required NLTK data
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

from nltk.corpus import stopwords

# ── Load spaCy model ──────────────────────────────────────────────────────────
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")

STOP_WORDS = set(stopwords.words("english"))

# ── Common skill keywords ─────────────────────────────────────────────────────
TECH_SKILLS = {
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "sql",
    "react", "angular", "vue", "node.js", "django", "flask", "fastapi", "spring",
    "docker", "kubernetes", "aws", "azure", "gcp", "git", "linux", "bash",
    "machine learning", "deep learning", "nlp", "tensorflow", "pytorch", "scikit-learn",
    "pandas", "numpy", "spark", "hadoop", "kafka", "postgresql", "mongodb", "redis",
    "rest", "graphql", "microservices", "ci/cd", "agile", "scrum", "html", "css",
    "tableau", "power bi", "excel", "r", "scala", "swift", "kotlin", "flutter",
}

EDUCATION_KEYWORDS = {
    "bachelor", "master", "phd", "b.s", "m.s", "b.e", "m.e", "b.tech", "m.tech",
    "mba", "degree", "university", "college", "institute", "diploma",
}

SECTION_HEADERS = {
    "experience", "work experience", "employment", "education", "skills",
    "projects", "certifications", "summary", "objective", "achievements",
}


# ── File parsing ──────────────────────────────────────────────────────────────
def extract_text(file) -> str:
    """Extract plain text from PDF, DOCX, or TXT file."""
    if file is None:
        return ""
    name = file.name.lower()
    if name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif name.endswith(".docx"):
        doc = docx.Document(io.BytesIO(file.read()))
        return "\n".join(p.text for p in doc.paragraphs)
    else:  # plain text
        return file.read().decode("utf-8", errors="ignore")


# ── Text preprocessing ────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9\s\.\+#]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Entity extraction ─────────────────────────────────────────────────────────
def extract_entities(text: str) -> dict:
    doc = nlp(text[:100_000])  # spaCy limit guard
    entities = {"names": [], "organizations": [], "locations": []}
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            entities["names"].append(ent.text)
        elif ent.label_ == "ORG":
            entities["organizations"].append(ent.text)
        elif ent.label_ in ("GPE", "LOC"):
            entities["locations"].append(ent.text)
    # Deduplicate
    return {k: list(dict.fromkeys(v)) for k, v in entities.items()}


def extract_email(text: str) -> str:
    match = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else ""


def extract_phone(text: str) -> str:
    match = re.search(r"(\+?\d[\d\s\-\(\)]{8,}\d)", text)
    return match.group(0).strip() if match else ""


def extract_skills(text: str) -> list:
    lower = text.lower()
    found = [skill for skill in TECH_SKILLS if skill in lower]
    return sorted(found)


def extract_education(text: str) -> list:
    lines = text.split("\n")
    edu_lines = []
    for line in lines:
        low = line.lower()
        if any(kw in low for kw in EDUCATION_KEYWORDS):
            cleaned = line.strip()
            if cleaned:
                edu_lines.append(cleaned)
    return edu_lines[:5]  # top 5 relevant lines


def extract_experience_years(text: str) -> float:
    """Heuristic: sum years mentioned near 'experience' or 'year' patterns."""
    patterns = [
        r"(\d+)\+?\s*years?\s+of\s+experience",
        r"(\d+)\+?\s*years?\s+experience",
        r"experience\s+of\s+(\d+)\+?\s*years?",
    ]
    years = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            years.append(int(m.group(1)))
    return max(years) if years else 0


# ── ML: TF-IDF similarity scoring ────────────────────────────────────────────
def compute_match_score(resume_text: str, jd_text: str) -> float:
    """Cosine similarity between resume and job description using TF-IDF."""
    if not resume_text.strip() or not jd_text.strip():
        return 0.0
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform([clean_text(resume_text), clean_text(jd_text)])
    score = cosine_similarity(tfidf[0], tfidf[1])[0][0]
    return round(float(score) * 100, 2)


def get_missing_keywords(resume_text: str, jd_text: str, top_n: int = 15) -> list:
    """Return important JD keywords missing from the resume."""
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=100)
    vectorizer.fit([clean_text(jd_text)])
    jd_terms = set(vectorizer.get_feature_names_out())
    resume_lower = clean_text(resume_text)
    missing = [term for term in jd_terms if term not in resume_lower]
    # Sort by TF-IDF weight in JD
    jd_vec = vectorizer.transform([clean_text(jd_text)]).toarray()[0]
    feature_names = vectorizer.get_feature_names_out()
    term_weights = {feature_names[i]: jd_vec[i] for i in range(len(feature_names))}
    missing_sorted = sorted(missing, key=lambda t: term_weights.get(t, 0), reverse=True)
    return missing_sorted[:top_n]


# ── Resume strength rating ────────────────────────────────────────────────────
def rate_resume(resume_text: str) -> dict:
    """Rule-based resume quality assessment."""
    scores = {}
    lower = resume_text.lower()

    # Contact info
    has_email = bool(extract_email(resume_text))
    has_phone = bool(extract_phone(resume_text))
    scores["contact_info"] = (has_email + has_phone) * 50  # 0, 50, or 100

    # Skills section
    skills = extract_skills(resume_text)
    scores["skills"] = min(len(skills) * 10, 100)

    # Length / detail (word count)
    word_count = len(resume_text.split())
    scores["content_length"] = min(int(word_count / 5), 100)

    # Section structure
    section_count = sum(1 for h in SECTION_HEADERS if h in lower)
    scores["structure"] = min(section_count * 20, 100)

    # Action verbs
    action_verbs = ["developed", "built", "led", "managed", "designed", "implemented",
                    "improved", "created", "analyzed", "delivered", "increased", "reduced"]
    verb_count = sum(1 for v in action_verbs if v in lower)
    scores["action_verbs"] = min(verb_count * 15, 100)

    overall = round(sum(scores.values()) / len(scores), 1)
    return {"breakdown": scores, "overall": overall}


# ── Main analysis function ────────────────────────────────────────────────────
def analyze_resume(resume_text: str, jd_text: str = "") -> dict:
    entities = extract_entities(resume_text)
    result = {
        "email": extract_email(resume_text),
        "phone": extract_phone(resume_text),
        "name": entities["names"][0] if entities["names"] else "Not detected",
        "organizations": entities["organizations"],
        "locations": entities["locations"],
        "skills": extract_skills(resume_text),
        "education": extract_education(resume_text),
        "experience_years": extract_experience_years(resume_text),
        "resume_rating": rate_resume(resume_text),
        "match_score": 0.0,
        "missing_keywords": [],
    }
    if jd_text.strip():
        result["match_score"] = compute_match_score(resume_text, jd_text)
        result["missing_keywords"] = get_missing_keywords(resume_text, jd_text)
    return result
