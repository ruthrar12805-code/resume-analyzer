"""
AI Resume Analyzer — Streamlit UI
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from analyzer import extract_text, analyze_resume

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
)

st.title("📄 AI Resume Analyzer")
st.caption("Upload your resume and optionally paste a job description to get an AI-powered analysis.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload Resume")
    resume_file = st.file_uploader(
        "Supports PDF, DOCX, TXT",
        type=["pdf", "docx", "txt"],
    )
    st.header("Job Description (optional)")
    jd_text = st.text_area(
        "Paste the job description here",
        height=250,
        placeholder="Paste job description to get a match score and missing keywords...",
    )
    analyze_btn = st.button("Analyze Resume", type="primary", use_container_width=True)

# ── Main area ─────────────────────────────────────────────────────────────────
if analyze_btn:
    if not resume_file:
        st.warning("Please upload a resume first.")
        st.stop()

    with st.spinner("Analyzing your resume..."):
        resume_text = extract_text(resume_file)
        if not resume_text.strip():
            st.error("Could not extract text from the file. Try a different format.")
            st.stop()
        result = analyze_resume(resume_text, jd_text)

    # ── Row 1: Contact info ───────────────────────────────────────────────
    st.subheader("👤 Contact Information")
    c1, c2, c3 = st.columns(3)
    c1.metric("Name", result["name"])
    c2.metric("Email", result["email"] or "Not found")
    c3.metric("Phone", result["phone"] or "Not found")

    st.divider()

    # ── Row 2: Scores ─────────────────────────────────────────────────────
    st.subheader("📊 Scores")
    s1, s2, s3 = st.columns(3)
    overall = result["resume_rating"]["overall"]
    s1.metric("Resume Strength", f"{overall} / 100")
    s2.metric("Experience", f"{result['experience_years']} yrs" if result["experience_years"] else "N/A")
    if jd_text.strip():
        match = result["match_score"]
        color = "normal" if match >= 50 else "inverse"
        s3.metric("JD Match Score", f"{match}%", delta=f"{'Good' if match >= 60 else 'Needs work'}")

    # Resume strength breakdown chart
    breakdown = result["resume_rating"]["breakdown"]
    fig, ax = plt.subplots(figsize=(5, 2.5))
    labels = [k.replace("_", " ").title() for k in breakdown]
    values = list(breakdown.values())
    bars = ax.barh(labels, values, color=["#4CAF50" if v >= 70 else "#FF9800" if v >= 40 else "#F44336" for v in values])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Score")
    ax.set_title("Resume Quality Breakdown")
    ax.bar_label(bars, fmt="%d", padding=3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # ── Row 3: Skills ─────────────────────────────────────────────────────
    st.subheader("🛠️ Detected Skills")
    if result["skills"]:
        cols = st.columns(4)
        for i, skill in enumerate(result["skills"]):
            cols[i % 4].success(skill)
    else:
        st.info("No common technical skills detected. Make sure skills are explicitly listed.")

    st.divider()

    # ── Row 4: Education & Entities ───────────────────────────────────────
    col_edu, col_org = st.columns(2)

    with col_edu:
        st.subheader("🎓 Education")
        if result["education"]:
            for line in result["education"]:
                st.write(f"• {line}")
        else:
            st.info("No education details detected.")

    with col_org:
        st.subheader("🏢 Organizations Mentioned")
        if result["organizations"]:
            for org in result["organizations"][:8]:
                st.write(f"• {org}")
        else:
            st.info("No organizations detected.")

    # ── Row 5: Missing keywords (JD mode) ────────────────────────────────
    if jd_text.strip() and result["missing_keywords"]:
        st.divider()
        st.subheader("🔍 Missing Keywords (from Job Description)")
        st.caption("Adding these terms could improve your match score.")
        kw_cols = st.columns(4)
        for i, kw in enumerate(result["missing_keywords"]):
            kw_cols[i % 4].warning(kw)

    # ── Row 6: Suggestions ────────────────────────────────────────────────
    st.divider()
    st.subheader("💡 Suggestions")
    bd = result["resume_rating"]["breakdown"]
    suggestions = []
    if bd.get("contact_info", 0) < 100:
        suggestions.append("Add both email and phone number to your contact section.")
    if bd.get("skills", 0) < 60:
        suggestions.append("Expand your skills section with more specific technologies.")
    if bd.get("structure", 0) < 60:
        suggestions.append("Add clear section headers: Experience, Education, Skills, Projects.")
    if bd.get("action_verbs", 0) < 60:
        suggestions.append("Use strong action verbs (developed, led, improved) to describe achievements.")
    if bd.get("content_length", 0) < 50:
        suggestions.append("Your resume may be too short. Add more detail to your experience and projects.")
    if result["experience_years"] == 0:
        suggestions.append("Mention total years of experience explicitly (e.g., '5 years of experience in...').")
    if not suggestions:
        suggestions.append("Your resume looks solid. Keep tailoring it for each job application.")
    for s in suggestions:
        st.write(f"✅ {s}")

    # ── Raw text expander ─────────────────────────────────────────────────
    with st.expander("View extracted resume text"):
        st.text(resume_text[:3000] + ("..." if len(resume_text) > 3000 else ""))

elif not resume_file:
    st.info("Upload a resume using the sidebar to get started.")
