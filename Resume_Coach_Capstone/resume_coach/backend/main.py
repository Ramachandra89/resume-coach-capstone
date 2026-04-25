"""
Resume Coach - FastAPI Backend
Provides REST endpoints consumed by the Streamlit frontend.
"""

import logging
import uuid
import json
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.settings import APP_TITLE, APP_VERSION, S3_BUCKET_NAME
from utils.document_processor import (
    extract_text_from_pdf,
    prepare_resume_for_llm,
    resume_text_to_pdf_bytes,
    extract_candidate_name,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description="AI-powered resume coaching API backed by Llama-3.1-8B / SageMaker",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (use Redis in production)
sessions: dict = {}


# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message: str

class OptimizeRequest(BaseModel):
    session_id: str

class CoverLetterRequest(BaseModel):
    session_id: str

class InterviewPrepRequest(BaseModel):
    session_id: str


# ─────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────

@app.get("/health")
async def health_check():
    from config.settings import INFERENCE_BACKEND
    return {
        "status": "healthy",
        "version": APP_VERSION,
        "inference_backend": INFERENCE_BACKEND,
    }


# ─────────────────────────────────────────────
# ANALYZE: Resume + JD → Coaching Report
# ─────────────────────────────────────────────

@app.post("/analyze")
async def analyze(
    background_tasks: BackgroundTasks,
    resume_file: Optional[UploadFile] = File(None),
    resume_text: Optional[str] = Form(None),
    job_description: str = Form(...),
    backend: Optional[str] = Form(None),
):
    """
    Accepts resume as PDF upload OR plain text.
    Returns full coaching report JSON + session_id for subsequent calls.
    """
    if not resume_file and not resume_text:
        raise HTTPException(status_code=400, detail="Provide resume_file or resume_text")

    session_id = str(uuid.uuid4())
    logger.info(f"New analysis session: {session_id}")

    # Extract resume text
    if resume_file:
        file_bytes = await resume_file.read()
        try:
            extracted_text = extract_text_from_pdf(file_bytes)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"PDF extraction failed: {e}")
    else:
        extracted_text = resume_text

    # Prepare for LLM (handle long resumes)
    processed_text, was_chunked = prepare_resume_for_llm(extracted_text)
    if was_chunked:
        logger.warning(f"Session {session_id}: Resume was chunked due to length")

    # Run coaching report chain
    try:
        from backend.chains.coaching_chains import CoachingReportChain
        chain = CoachingReportChain(backend=backend)
        report, context_summary = chain.run(processed_text, job_description)
    except Exception as e:
        logger.error(f"Chain error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    # Store session
    sessions[session_id] = {
        "resume_text": extracted_text,
        "processed_resume_text": processed_text,
        "job_description": job_description,
        "context_summary": context_summary,
        "report": report,
        "optimized_resume": None,
        "backend": backend,
    }

    # Upload to S3 in background
    if resume_file:
        background_tasks.add_task(
            _upload_to_s3_background,
            session_id=session_id,
            file_bytes=file_bytes,
            filename=resume_file.filename,
            report=report,
        )

    return {
        "session_id": session_id,
        "report": report,
        "was_chunked": was_chunked,
        "candidate_name": extract_candidate_name(extracted_text),
    }


async def _upload_to_s3_background(session_id, file_bytes, filename, report):
    """Background task: upload resume + report to S3."""
    try:
        from utils.s3_storage import upload_resume, upload_report
        upload_resume(file_bytes, filename, session_id)
        upload_report(report, session_id)
    except Exception as e:
        logger.warning(f"S3 upload failed (non-critical): {e}")


# ─────────────────────────────────────────────
# ONE-CLICK OPTIMIZE
# ─────────────────────────────────────────────

@app.post("/optimize")
async def optimize_resume(request: OptimizeRequest):
    """
    Rewrites resume for ATS optimization.
    Returns optimized text, new ATS score, and delta.
    """
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        from backend.chains.coaching_chains import ResumeOptimizerChain
        chain = ResumeOptimizerChain(backend=session.get("backend"))
        result = chain.run(
            resume_text=session["processed_resume_text"],
            job_description=session["job_description"],
            context_summary=session["context_summary"],
            original_report=session["report"],
        )
    except Exception as e:
        logger.error(f"Optimization error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

    sessions[request.session_id]["optimized_resume"] = result["optimized_resume"]

    return result


# ─────────────────────────────────────────────
# DOWNLOAD OPTIMIZED RESUME AS PDF
# ─────────────────────────────────────────────

@app.get("/download-resume/{session_id}")
async def download_resume(session_id: str):
    """Download the optimized resume as a formatted PDF."""
    session = sessions.get(session_id)
    if not session or not session.get("optimized_resume"):
        raise HTTPException(status_code=404, detail="No optimized resume found for this session")

    candidate_name = extract_candidate_name(session["optimized_resume"])
    try:
        pdf_bytes = resume_text_to_pdf_bytes(
            session["optimized_resume"],
            candidate_name=candidate_name,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{candidate_name.replace(" ", "_")}_Optimized_Resume.pdf"'
        },
    )


# ─────────────────────────────────────────────
# CHAT COACH
# ─────────────────────────────────────────────

@app.post("/chat")
async def chat(request: ChatRequest):
    """Multi-turn coaching chat with context memory."""
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Lazy-init chat chain per session
    if "chat_chain" not in session:
        from backend.chains.coaching_chains import ChatCoachChain
        chat_chain = ChatCoachChain(backend=session.get("backend"))
        chat_chain.initialize(
            context_summary=session["context_summary"],
            report=session["report"],
        )
        session["chat_chain"] = chat_chain

    try:
        response = session["chat_chain"].chat(request.message)
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

    return {
        "response": response,
        "history": session["chat_chain"].get_history(),
    }


# ─────────────────────────────────────────────
# COVER LETTER
# ─────────────────────────────────────────────

@app.post("/cover-letter")
async def generate_cover_letter(request: CoverLetterRequest):
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        from backend.chains.coaching_chains import CoverLetterChain
        chain = CoverLetterChain(backend=session.get("backend"))
        cover_letter = chain.run(
            context_summary=session["context_summary"],
            report=session["report"],
            job_description=session["job_description"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cover letter generation failed: {e}")

    return {"cover_letter": cover_letter}


# ─────────────────────────────────────────────
# INTERVIEW PREP
# ─────────────────────────────────────────────

@app.post("/interview-prep")
async def generate_interview_prep(request: InterviewPrepRequest):
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        from backend.chains.coaching_chains import InterviewPrepChain
        chain = InterviewPrepChain(backend=session.get("backend"))
        prep = chain.run(
            context_summary=session["context_summary"],
            report=session["report"],
            job_description=session["job_description"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Interview prep generation failed: {e}")

    return prep


# ─────────────────────────────────────────────
# SESSION INFO
# ─────────────────────────────────────────────

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "has_report": session.get("report") is not None,
        "has_optimized_resume": session.get("optimized_resume") is not None,
    }


if __name__ == "__main__":
    import uvicorn
    from config.settings import APP_HOST, APP_PORT
    uvicorn.run("main:app", host=APP_HOST, port=APP_PORT, reload=True)
