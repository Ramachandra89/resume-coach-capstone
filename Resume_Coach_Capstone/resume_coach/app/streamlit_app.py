"""
ResumeCoach AI - Streamlit Frontend
Apple-inspired minimalist design: clean whites, subtle grays, SF-style typography.
"""

import streamlit as st
import requests
import json
import time
import base64
from typing import Optional
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ResumeCoach AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# ─────────────────────────────────────────────
# CUSTOM CSS — Apple-style aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Reset & Base ── */
* { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif;
    background-color: #f5f5f7;
    color: #1d1d1f;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}

/* ── Custom Header ── */
.rc-header {
    text-align: center;
    padding: 3rem 0 2rem;
}
.rc-header .logo {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #86868b;
    margin-bottom: 0.5rem;
}
.rc-header h1 {
    font-size: 48px;
    font-weight: 700;
    color: #1d1d1f;
    letter-spacing: -0.02em;
    line-height: 1.05;
    margin: 0 0 0.75rem;
}
.rc-header p {
    font-size: 17px;
    font-weight: 400;
    color: #6e6e73;
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.5;
}

/* ── Cards ── */
.rc-card {
    background: #ffffff;
    border-radius: 18px;
    padding: 28px 32px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06), 0 0 1px rgba(0,0,0,0.04);
    margin-bottom: 16px;
}
.rc-card-sm {
    background: #ffffff;
    border-radius: 14px;
    padding: 20px 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    margin-bottom: 12px;
}

/* ── Section labels ── */
.rc-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #86868b;
    margin-bottom: 8px;
}

/* ── Score Ring ── */
.score-ring-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.score-ring {
    width: 96px;
    height: 96px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0 auto 8px;
}
.score-ring.green  { background: #e8f9f0; color: #1d7a47; border: 3px solid #34c759; }
.score-ring.yellow { background: #fffbe6; color: #7a6300; border: 3px solid #ffcc00; }
.score-ring.orange { background: #fff3e0; color: #8a3500; border: 3px solid #ff9500; }
.score-ring.red    { background: #fff0f0; color: #8a0000; border: 3px solid #ff3b30; }

/* ── Verdict badge ── */
.verdict-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.verdict-strong { background: #e8f9f0; color: #1d7a47; }
.verdict-good   { background: #e8f0ff; color: #1a3a8a; }
.verdict-partial { background: #fff3e0; color: #8a3500; }
.verdict-weak   { background: #fff0f0; color: #8a0000; }

/* ── Progress bars ── */
.ats-bar-container {
    margin: 8px 0;
}
.ats-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    color: #1d1d1f;
    margin-bottom: 4px;
}
.ats-bar-track {
    background: #f0f0f5;
    border-radius: 4px;
    height: 7px;
    overflow: hidden;
}
.ats-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.8s ease;
}

/* ── Tag pills ── */
.tag-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 500;
    margin: 2px 3px;
}
.tag-present  { background: #e8f9f0; color: #1d7a47; }
.tag-missing  { background: #fff0f0; color: #c0392b; }
.tag-added    { background: #e8f0ff; color: #1a3a8a; }

/* ── Delta badge ── */
.delta-up   { color: #34c759; font-weight: 700; font-size: 18px; }
.delta-down { color: #ff3b30; font-weight: 700; font-size: 18px; }

/* ── Strength / Gap cards ── */
.strength-item {
    border-left: 3px solid #34c759;
    padding: 10px 14px;
    margin: 8px 0;
    background: #f8fffe;
    border-radius: 0 8px 8px 0;
}
.gap-item-critical {
    border-left: 3px solid #ff3b30;
    padding: 10px 14px;
    margin: 8px 0;
    background: #fff8f8;
    border-radius: 0 8px 8px 0;
}
.gap-item-moderate {
    border-left: 3px solid #ff9500;
    padding: 10px 14px;
    margin: 8px 0;
    background: #fffbf5;
    border-radius: 0 8px 8px 0;
}
.gap-item-minor {
    border-left: 3px solid #ffcc00;
    padding: 10px 14px;
    margin: 8px 0;
    background: #fffef0;
    border-radius: 0 8px 8px 0;
}

/* ── Chat ── */
.chat-bubble-user {
    background: #1d1d1f;
    color: #f5f5f7;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 16px;
    margin: 6px 0 6px 60px;
    font-size: 14px;
    line-height: 1.5;
}
.chat-bubble-ai {
    background: #ffffff;
    color: #1d1d1f;
    border-radius: 18px 18px 18px 4px;
    padding: 12px 16px;
    margin: 6px 60px 6px 0;
    font-size: 14px;
    line-height: 1.5;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}
.chat-name {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.04em;
    color: #86868b;
    margin-bottom: 4px;
}

/* ── Divider ── */
.rc-divider {
    border: none;
    border-top: 1px solid #e5e5ea;
    margin: 24px 0;
}

/* ── Buttons ── */
div.stButton > button {
    border-radius: 980px;
    font-size: 14px;
    font-weight: 600;
    padding: 0.55rem 1.6rem;
    border: none;
    transition: all 0.2s ease;
}
div.stButton > button:first-child {
    background: #1d1d1f;
    color: #ffffff;
}
div.stButton > button:first-child:hover {
    background: #3a3a3c;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #f5f5f7;
    border: 1.5px dashed #c7c7cc;
    border-radius: 12px;
    padding: 12px;
}

/* ── Text areas and inputs ── */
textarea, input[type="text"] {
    border-radius: 10px !important;
    border: 1.5px solid #d2d2d7 !important;
    font-size: 14px !important;
    background: #ffffff !important;
}
textarea:focus, input:focus {
    border-color: #1d1d1f !important;
    box-shadow: 0 0 0 3px rgba(29,29,31,0.08) !important;
}

/* ── Tab styling ── */
[data-baseweb="tab-list"] {
    background: #f5f5f7;
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
}
[data-baseweb="tab"] {
    border-radius: 8px;
    font-size: 13px;
    font-weight: 500;
    color: #6e6e73;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: #ffffff !important;
    color: #1d1d1f !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
}

/* ── Spinner ── */
.stSpinner > div { border-color: #1d1d1f transparent transparent; }

/* ── Recommendation card ── */
.rec-card {
    background: #f5f5f7;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 13.5px;
    line-height: 1.5;
}
.rec-priority-high   { border-left: 3px solid #ff3b30; }
.rec-priority-medium { border-left: 3px solid #ff9500; }
.rec-priority-low    { border-left: 3px solid #34c759; }

/* ── Section title ── */
.section-title {
    font-size: 20px;
    font-weight: 700;
    color: #1d1d1f;
    letter-spacing: -0.01em;
    margin-bottom: 4px;
}
.section-subtitle {
    font-size: 14px;
    color: #6e6e73;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def score_color(score: int) -> str:
    if score >= 75: return "green"
    if score >= 55: return "yellow"
    if score >= 35: return "orange"
    return "red"


def verdict_class(verdict: str) -> str:
    mapping = {
        "Strong Match": "verdict-strong",
        "Good Match": "verdict-good",
        "Partial Match": "verdict-partial",
        "Weak Match": "verdict-weak",
    }
    return mapping.get(verdict, "verdict-good")


def bar_color(score: int) -> str:
    if score >= 75: return "#34c759"
    if score >= 55: return "#ffcc00"
    if score >= 35: return "#ff9500"
    return "#ff3b30"


def render_ats_bar(label: str, score: int):
    color = bar_color(score)
    st.markdown(f"""
    <div class="ats-bar-container">
        <div class="ats-bar-label"><span>{label}</span><span>{score}</span></div>
        <div class="ats-bar-track">
            <div class="ats-bar-fill" style="width:{score}%;background:{color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def call_api(method: str, endpoint: str, **kwargs):
    try:
        url = f"{API_BASE}{endpoint}"
        response = getattr(requests, method)(url, timeout=120, **kwargs)
        response.raise_for_status()
        return response
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to the backend API. Make sure the FastAPI server is running.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API Error: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

def init_session():
    defaults = {
        "session_id": None,
        "report": None,
        "context_summary": None,
        "candidate_name": "Candidate",
        "optimized_resume": None,
        "optimization_result": None,
        "cover_letter": None,
        "interview_prep": None,
        "chat_history": [],
        "active_tab": "report",
        "backend": "nebius",
        "job_description": "",
        "resume_text_input": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div class="rc-header">
    <div class="logo">✦ Powered by Generative AI</div>
    <h1>ResumeCoach AI</h1>
    <p>Upload your resume and a job description. Get an instant ATS analysis, gap report, and optimized resume.</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────

with st.container():
    st.markdown('<div class="rc-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Start Your Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Provide your resume and the job you\'re targeting</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="rc-label">Your Resume</div>', unsafe_allow_html=True)
        resume_tab1, resume_tab2 = st.tabs(["📎 Upload PDF", "✏️ Paste Text"])
        with resume_tab1:
            uploaded_file = st.file_uploader(
                "Drop your resume PDF here",
                type=["pdf"],
                label_visibility="collapsed",
            )
        with resume_tab2:
            resume_text_input = st.text_area(
                "Paste resume text",
                height=220,
                placeholder="Paste your resume text here...",
                label_visibility="collapsed",
                key="resume_text_area",
            )

    with col2:
        st.markdown('<div class="rc-label">Job Description</div>', unsafe_allow_html=True)
        job_desc = st.text_area(
            "Job description",
            height=250,
            placeholder="Paste the full job description here (include requirements, responsibilities, qualifications)...",
            label_visibility="collapsed",
            key="jd_input",
        )

    # Settings row
    st.markdown('<hr class="rc-divider">', unsafe_allow_html=True)
    col_s1, col_s2, col_s3 = st.columns([1, 1, 2])
    with col_s1:
        st.markdown('<div class="rc-label">Inference Backend</div>', unsafe_allow_html=True)
        backend_choice = st.selectbox(
            "Backend",
            options=["nebius", "local_hf", "sagemaker"],
            index=0,
            label_visibility="collapsed",
            help="nebius=API testing | local_hf=Llama-3.1-8B on EC2 | sagemaker=Llama-3.2-3B JumpStart"
        )
    with col_s2:
        st.markdown('<div class="rc-label">&nbsp;</div>', unsafe_allow_html=True)
        analyze_btn = st.button("✦ Analyze My Resume", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ANALYSIS TRIGGER
# ─────────────────────────────────────────────

if analyze_btn:
    has_resume = uploaded_file is not None or (resume_text_input and resume_text_input.strip())
    has_jd = job_desc and job_desc.strip()

    if not has_resume:
        st.warning("Please upload a resume PDF or paste your resume text.")
    elif not has_jd:
        st.warning("Please paste the job description.")
    else:
        with st.spinner("Analyzing your resume against the job description…"):
            if uploaded_file:
                files = {"resume_file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                data = {"job_description": job_desc, "backend": backend_choice}
                resp = call_api("post", "/analyze", files=files, data=data)
            else:
                data = {
                    "resume_text": resume_text_input,
                    "job_description": job_desc,
                    "backend": backend_choice,
                }
                resp = call_api("post", "/analyze", data=data)

            if resp:
                result = resp.json()
                st.session_state.session_id = result["session_id"]
                st.session_state.report = result["report"]
                st.session_state.candidate_name = result.get("candidate_name", "Candidate")
                st.session_state.backend = backend_choice
                st.session_state.job_description = job_desc
                st.session_state.chat_history = []
                st.session_state.optimized_resume = None
                st.session_state.optimization_result = None
                st.session_state.cover_letter = None
                st.session_state.interview_prep = None
                st.rerun()


# ─────────────────────────────────────────────
# RESULTS SECTION
# ─────────────────────────────────────────────

if st.session_state.report:
    report = st.session_state.report
    candidate = st.session_state.candidate_name

    st.markdown("---")

    # ── Score Overview ──
    st.markdown('<div class="rc-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">Analysis for {candidate}</div>', unsafe_allow_html=True)

    ov_col1, ov_col2, ov_col3, ov_col4 = st.columns(4)

    with ov_col1:
        fit = report.get("overall_fit_score", 0)
        c = score_color(fit)
        st.markdown(f"""
        <div style="text-align:center;">
            <div class="rc-label" style="text-align:center;">Fit Score</div>
            <div class="score-ring {c}">{fit}</div>
        </div>""", unsafe_allow_html=True)

    with ov_col2:
        ats = report.get("ats_score", 0)
        c = score_color(ats)
        st.markdown(f"""
        <div style="text-align:center;">
            <div class="rc-label" style="text-align:center;">ATS Score</div>
            <div class="score-ring {c}">{ats}</div>
        </div>""", unsafe_allow_html=True)

    with ov_col3:
        verdict = report.get("fit_verdict", "")
        vc = verdict_class(verdict)
        st.markdown(f"""
        <div style="text-align:center;">
            <div class="rc-label" style="text-align:center;">Verdict</div>
            <div style="padding:12px 0;"><span class="verdict-badge {vc}">{verdict}</span></div>
        </div>""", unsafe_allow_html=True)

    with ov_col4:
        app_rec = report.get("application_recommendation", "")
        rec_color = "#34c759" if "Strongly" in app_rec or (app_rec == "Recommend Applying") else "#ff9500"
        st.markdown(f"""
        <div style="text-align:center;">
            <div class="rc-label" style="text-align:center;">Recommendation</div>
            <div style="padding-top:14px;font-size:12px;font-weight:600;color:{rec_color};line-height:1.3;">{app_rec}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="rc-divider">', unsafe_allow_html=True)
    fit_summary = report.get("fit_summary", "")
    st.markdown(f'<div style="font-size:15px;color:#3a3a3c;line-height:1.6;">{fit_summary}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── TABS ──
    tab_report, tab_optimize, tab_chat, tab_cover, tab_interview = st.tabs([
        "📋 Full Report", "⚡ Optimize Resume", "💬 Chat Coach", "✉️ Cover Letter", "🎯 Interview Prep"
    ])

    # ══════════════════════════════════════
    # TAB 1: FULL REPORT
    # ══════════════════════════════════════
    with tab_report:
        r_col1, r_col2 = st.columns([1, 1], gap="large")

        with r_col1:
            # ATS Breakdown
            st.markdown('<div class="rc-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title" style="font-size:16px;">ATS Score Breakdown</div>', unsafe_allow_html=True)
            breakdown = report.get("ats_score_breakdown", {})
            label_map = {
                "keyword_match": "Keyword Match",
                "format_compatibility": "Format Compatibility",
                "skills_alignment": "Skills Alignment",
                "experience_relevance": "Experience Relevance",
                "education_match": "Education Match",
            }
            for key, label in label_map.items():
                render_ats_bar(label, breakdown.get(key, 0))
            st.markdown('</div>', unsafe_allow_html=True)

            # Keywords
            st.markdown('<div class="rc-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title" style="font-size:16px;">Keyword Analysis</div>', unsafe_allow_html=True)
            present = report.get("present_keywords", [])
            missing = report.get("missing_keywords", [])
            if present:
                st.markdown('<div class="rc-label">Present ✓</div>', unsafe_allow_html=True)
                pills = " ".join([f'<span class="tag-pill tag-present">{k}</span>' for k in present[:15]])
                st.markdown(pills, unsafe_allow_html=True)
            if missing:
                st.markdown('<div class="rc-label" style="margin-top:12px;">Missing ✗</div>', unsafe_allow_html=True)
                pills = " ".join([f'<span class="tag-pill tag-missing">{k}</span>' for k in missing[:15]])
                st.markdown(pills, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with r_col2:
            # Strengths
            strengths = report.get("strengths", [])
            if strengths:
                st.markdown('<div class="rc-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title" style="font-size:16px;">Your Strengths for This Role</div>', unsafe_allow_html=True)
                for s in strengths:
                    st.markdown(f"""
                    <div class="strength-item">
                        <div style="font-weight:600;font-size:14px;margin-bottom:4px;">✓ {s['title']}</div>
                        <div style="font-size:13px;color:#3a3a3c;margin-bottom:4px;">{s['detail']}</div>
                        <div style="font-size:12px;color:#6e6e73;font-style:italic;">💡 {s['how_to_emphasize']}</div>
                    </div>""", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Gaps
            gaps = report.get("gaps", [])
            if gaps:
                st.markdown('<div class="rc-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title" style="font-size:16px;">Gaps to Address</div>', unsafe_allow_html=True)
                for g in gaps:
                    severity = g.get("severity", "Minor")
                    css_class = {
                        "Critical": "gap-item-critical",
                        "Moderate": "gap-item-moderate",
                        "Minor": "gap-item-minor",
                    }.get(severity, "gap-item-minor")
                    icon = {"Critical": "🔴", "Moderate": "🟠", "Minor": "🟡"}.get(severity, "🟡")
                    st.markdown(f"""
                    <div class="{css_class}">
                        <div style="font-weight:600;font-size:14px;margin-bottom:4px;">{icon} {g['title']} <span style="font-size:11px;font-weight:500;color:#86868b;">({severity})</span></div>
                        <div style="font-size:13px;color:#3a3a3c;margin-bottom:4px;">{g['detail']}</div>
                        <div style="font-size:12px;color:#6e6e73;">🛠 {g['mitigation']}</div>
                    </div>""", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # Recommendations
        recs = report.get("coaching_recommendations", [])
        if recs:
            st.markdown('<div class="rc-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title" style="font-size:16px;">Coaching Recommendations</div>', unsafe_allow_html=True)
            cat_cols = st.columns(min(len(set(r.get("category","") for r in recs)), 3))
            by_cat = {}
            for r in recs:
                cat = r.get("category", "General")
                by_cat.setdefault(cat, []).append(r)
            for i, (cat, cat_recs) in enumerate(by_cat.items()):
                col_i = i % 3
                with cat_cols[col_i] if len(cat_cols) > 1 else st.container():
                    st.markdown(f'<div class="rc-label">{cat}</div>', unsafe_allow_html=True)
                    for rec in cat_recs:
                        pri = rec.get("priority", "Medium")
                        pri_class = {"High": "rec-priority-high", "Medium": "rec-priority-medium", "Low": "rec-priority-low"}.get(pri, "rec-priority-medium")
                        st.markdown(f"""
                        <div class="rec-card {pri_class}">
                            <div style="font-size:11px;font-weight:600;color:#86868b;margin-bottom:4px;">{pri} Priority</div>
                            <div style="font-weight:500;margin-bottom:3px;">{rec['recommendation']}</div>
                            <div style="font-size:12px;color:#6e6e73;">{rec['rationale']}</div>
                        </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Red flags + talking points
        rf_col, tp_col = st.columns(2)
        with rf_col:
            red_flags = report.get("red_flags", [])
            if red_flags:
                st.markdown('<div class="rc-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title" style="font-size:15px;">⚠️ Potential Red Flags</div>', unsafe_allow_html=True)
                for flag in red_flags:
                    st.markdown(f'<div style="font-size:13px;padding:6px 0;border-bottom:1px solid #f0f0f5;color:#c0392b;">• {flag}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        with tp_col:
            talking_pts = report.get("interview_talking_points", [])
            if talking_pts:
                st.markdown('<div class="rc-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title" style="font-size:15px;">💡 Interview Talking Points</div>', unsafe_allow_html=True)
                for tp in talking_pts:
                    st.markdown(f'<div style="font-size:13px;padding:6px 0;border-bottom:1px solid #f0f0f5;">• {tp}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════
    # TAB 2: OPTIMIZE RESUME
    # ══════════════════════════════════════
    with tab_optimize:
        st.markdown('<div class="rc-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">One-Click Resume Optimizer</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Rewrites your resume to maximize ATS compatibility for this specific role — without fabricating experience.</div>', unsafe_allow_html=True)

        if not st.session_state.optimization_result:
            opt_btn = st.button("⚡ Optimize My Resume", use_container_width=False)
            if opt_btn:
                with st.spinner("Optimizing your resume for maximum ATS impact…"):
                    resp = call_api("post", "/optimize", json={"session_id": st.session_state.session_id})
                    if resp:
                        result = resp.json()
                        st.session_state.optimization_result = result
                        st.session_state.optimized_resume = result.get("optimized_resume", "")
                        st.rerun()
        else:
            result = st.session_state.optimization_result
            orig_ats = result.get("original_ats_score", report.get("ats_score", 0))
            new_ats = result.get("new_ats_score", 0)
            delta = result.get("score_delta", new_ats - orig_ats)

            # Score comparison
            sc1, sc2, sc3 = st.columns([1, 0.5, 1])
            with sc1:
                c = score_color(orig_ats)
                st.markdown(f"""
                <div style="text-align:center;">
                    <div class="rc-label" style="text-align:center;">Before</div>
                    <div class="score-ring {c}" style="margin:auto;">{orig_ats}</div>
                </div>""", unsafe_allow_html=True)
            with sc2:
                delta_class = "delta-up" if delta >= 0 else "delta-down"
                delta_sign = "+" if delta >= 0 else ""
                st.markdown(f'<div class="{delta_class}" style="text-align:center;padding-top:30px;">{delta_sign}{delta}</div>', unsafe_allow_html=True)
            with sc3:
                c = score_color(new_ats)
                st.markdown(f"""
                <div style="text-align:center;">
                    <div class="rc-label" style="text-align:center;">After</div>
                    <div class="score-ring {c}" style="margin:auto;">{new_ats}</div>
                </div>""", unsafe_allow_html=True)

            if result.get("improvement_summary"):
                st.markdown(f'<div style="margin:16px 0;font-size:14px;color:#3a3a3c;text-align:center;">{result["improvement_summary"]}</div>', unsafe_allow_html=True)

            # New keywords added
            new_kw = result.get("new_keywords_added", [])
            if new_kw:
                st.markdown('<div class="rc-label">Keywords Added</div>', unsafe_allow_html=True)
                pills = " ".join([f'<span class="tag-pill tag-added">{k}</span>' for k in new_kw])
                st.markdown(pills, unsafe_allow_html=True)

            st.markdown('<hr class="rc-divider">', unsafe_allow_html=True)

            # ATS breakdown comparison
            if result.get("new_ats_breakdown"):
                orig_bd = report.get("ats_score_breakdown", {})
                new_bd = result["new_ats_breakdown"]
                label_map = {
                    "keyword_match": "Keyword Match",
                    "format_compatibility": "Format Compatibility",
                    "skills_alignment": "Skills Alignment",
                    "experience_relevance": "Experience Relevance",
                    "education_match": "Education Match",
                }
                st.markdown('<div style="font-weight:600;font-size:14px;margin-bottom:12px;">Breakdown Comparison</div>', unsafe_allow_html=True)
                for key, label in label_map.items():
                    orig_v = orig_bd.get(key, 0)
                    new_v = new_bd.get(key, 0)
                    d = new_v - orig_v
                    d_str = f"({'+' if d >= 0 else ''}{d})" if d != 0 else ""
                    d_color = "#34c759" if d > 0 else ("#ff3b30" if d < 0 else "#86868b")
                    st.markdown(f"""
                    <div class="ats-bar-container">
                        <div class="ats-bar-label">
                            <span>{label}</span>
                            <span>{orig_v} → <b>{new_v}</b> <span style="color:{d_color};font-size:11px;">{d_str}</span></span>
                        </div>
                        <div class="ats-bar-track">
                            <div class="ats-bar-fill" style="width:{new_v}%;background:{bar_color(new_v)};"></div>
                        </div>
                    </div>""", unsafe_allow_html=True)

            st.markdown('<hr class="rc-divider">', unsafe_allow_html=True)

            # Optimized resume text
            st.markdown('<div class="rc-label">Optimized Resume</div>', unsafe_allow_html=True)
            st.text_area(
                "Optimized Resume",
                value=st.session_state.optimized_resume,
                height=400,
                label_visibility="collapsed",
            )

            # Download button
            dl_resp = call_api("get", f"/download-resume/{st.session_state.session_id}")
            if dl_resp:
                st.download_button(
                    label="⬇ Download as PDF",
                    data=dl_resp.content,
                    file_name=f"{candidate.replace(' ', '_')}_Optimized_Resume.pdf",
                    mime="application/pdf",
                )

            if st.button("🔄 Re-optimize"):
                st.session_state.optimization_result = None
                st.session_state.optimized_resume = None
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════
    # TAB 3: CHAT COACH
    # ══════════════════════════════════════
    with tab_chat:
        st.markdown('<div class="rc-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Chat with Your Career Coach</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Ask anything about your resume, this role, interview prep, or negotiation strategy.</div>', unsafe_allow_html=True)

        # Chat display area
        chat_container = st.container()
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown(f"""
                <div class="chat-bubble-ai">
                    <div class="chat-name">RESUMECOACH AI</div>
                    Hi {candidate}! I've finished analyzing your resume. Your overall fit score is <b>{report.get("overall_fit_score", 0)}/100</b> and ATS score is <b>{report.get("ats_score", 0)}/100</b> — verdict: <b>{report.get("fit_verdict", "")}</b>.<br><br>
                    I can help you with:<br>
                    • Improving specific resume sections<br>
                    • Interview preparation and likely questions<br>
                    • How to address the gaps identified<br>
                    • Salary negotiation strategy<br><br>
                    What would you like to focus on?
                </div>""", unsafe_allow_html=True)
            else:
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        st.markdown(f"""
                        <div style="text-align:right;">
                            <div class="chat-bubble-user">{msg["content"]}</div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="chat-bubble-ai">
                            <div class="chat-name">RESUMECOACH AI</div>
                            {msg["content"]}
                        </div>""", unsafe_allow_html=True)

        # Suggested prompts
        if not st.session_state.chat_history:
            st.markdown('<div class="rc-label" style="margin-top:16px;">Suggested Questions</div>', unsafe_allow_html=True)
            sugg_col1, sugg_col2, sugg_col3 = st.columns(3)
            suggestions = [
                "How do I address the experience gaps?",
                "What salary should I negotiate for?",
                "What are the toughest interview questions I'll face?",
                "How should I rewrite my summary section?",
                "What skills should I learn to be a stronger candidate?",
                "Write a 30-second elevator pitch for this role",
            ]
            for i, sugg in enumerate(suggestions):
                col = [sugg_col1, sugg_col2, sugg_col3][i % 3]
                with col:
                    if st.button(sugg, key=f"sugg_{i}", use_container_width=True):
                        st.session_state._pending_chat = sugg
                        st.rerun()

        # Chat input
        st.markdown('<hr class="rc-divider">', unsafe_allow_html=True)
        user_msg = st.chat_input("Ask your career coach anything...")

        # Handle pending suggestion
        if hasattr(st.session_state, '_pending_chat'):
            user_msg = st.session_state._pending_chat
            del st.session_state._pending_chat

        if user_msg:
            st.session_state.chat_history.append({"role": "user", "content": user_msg})
            with st.spinner(""):
                resp = call_api("post", "/chat", json={
                    "session_id": st.session_state.session_id,
                    "message": user_msg,
                })
                if resp:
                    ai_response = resp.json().get("response", "")
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
            st.rerun()

        if st.session_state.chat_history:
            if st.button("🗑 Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════
    # TAB 4: COVER LETTER
    # ══════════════════════════════════════
    with tab_cover:
        st.markdown('<div class="rc-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Generate Cover Letter</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">A tailored cover letter using your strengths and the specific job requirements.</div>', unsafe_allow_html=True)

        if not st.session_state.cover_letter:
            if st.button("✉️ Generate Cover Letter"):
                with st.spinner("Writing your personalized cover letter…"):
                    resp = call_api("post", "/cover-letter", json={"session_id": st.session_state.session_id})
                    if resp:
                        st.session_state.cover_letter = resp.json().get("cover_letter", "")
                        st.rerun()
        else:
            cl_text = st.text_area(
                "Cover Letter",
                value=st.session_state.cover_letter,
                height=450,
                label_visibility="collapsed",
            )
            st.download_button(
                "⬇ Download Cover Letter",
                data=cl_text,
                file_name="Cover_Letter.txt",
                mime="text/plain",
            )
            if st.button("🔄 Regenerate"):
                st.session_state.cover_letter = None
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════
    # TAB 5: INTERVIEW PREP
    # ══════════════════════════════════════
    with tab_interview:
        st.markdown('<div class="rc-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Interview Preparation Guide</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Likely questions, suggested answers, and questions to ask the interviewer.</div>', unsafe_allow_html=True)

        if not st.session_state.interview_prep:
            if st.button("🎯 Generate Interview Prep Guide"):
                with st.spinner("Preparing your personalized interview guide…"):
                    resp = call_api("post", "/interview-prep", json={"session_id": st.session_state.session_id})
                    if resp:
                        st.session_state.interview_prep = resp.json()
                        st.rerun()
        else:
            prep = st.session_state.interview_prep

            questions = prep.get("likely_interview_questions", [])
            if questions:
                st.markdown('<div class="section-title" style="font-size:16px;margin-bottom:12px;">Likely Interview Questions</div>', unsafe_allow_html=True)
                for i, q in enumerate(questions):
                    with st.expander(f"{i+1}. {q['question']} [{q['type']}]"):
                        st.markdown(f"**Why they'll ask this:** {q['why_theyll_ask']}")
                        st.markdown(f"**Answer framework:** {q['suggested_answer_framework']}")
                        st.markdown("**Key points to hit:**")
                        for pt in q.get("key_points_to_hit", []):
                            st.markdown(f"- {pt}")

            ask_qs = prep.get("questions_to_ask_interviewer", [])
            if ask_qs:
                st.markdown('<hr class="rc-divider">', unsafe_allow_html=True)
                st.markdown('<div class="section-title" style="font-size:16px;margin-bottom:12px;">Questions to Ask the Interviewer</div>', unsafe_allow_html=True)
                for q in ask_qs:
                    st.markdown(f"""
                    <div class="rc-card-sm">
                        <div style="font-weight:600;font-size:14px;">"{q['question']}"</div>
                        <div style="font-size:12px;color:#6e6e73;margin-top:4px;">💡 {q['purpose']}</div>
                    </div>""", unsafe_allow_html=True)

            rebuttals = prep.get("red_flag_rebuttals", [])
            if rebuttals:
                st.markdown('<hr class="rc-divider">', unsafe_allow_html=True)
                st.markdown('<div class="section-title" style="font-size:16px;margin-bottom:12px;">How to Address Tough Questions</div>', unsafe_allow_html=True)
                for r in rebuttals:
                    with st.expander(f"⚠️ {r['potential_concern']}"):
                        st.markdown(r['rebuttal'])

            if st.button("🔄 Regenerate Guide"):
                st.session_state.interview_prep = None
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

else:
    # Empty state
    st.markdown("""
    <div style="text-align:center;padding:60px 0;color:#86868b;">
        <div style="font-size:48px;margin-bottom:16px;">✦</div>
        <div style="font-size:17px;font-weight:500;color:#3a3a3c;margin-bottom:8px;">Ready when you are</div>
        <div style="font-size:14px;">Upload your resume and paste a job description above to get started.</div>
    </div>
    """, unsafe_allow_html=True)
