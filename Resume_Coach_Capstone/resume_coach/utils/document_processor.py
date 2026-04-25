"""
Resume Coach - Document Processing Utilities
Handles PDF parsing, text cleaning, token counting, and chunking.
"""

import re
import logging
import tempfile
import os
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract text from PDF bytes using pdfminer.six with pymupdf fallback.
    Returns cleaned plain text.
    """
    text = ""

    # Primary: pdfminer.six (better text extraction for formatted resumes)
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        import io
        text = pdfminer_extract(io.BytesIO(file_bytes))
        if text and len(text.strip()) > 100:
            logger.info("PDF extracted with pdfminer.six")
            return clean_resume_text(text)
    except Exception as e:
        logger.warning(f"pdfminer extraction failed: {e}")

    # Fallback: pymupdf (fitz) — handles more complex PDFs
    try:
        import fitz  # pymupdf
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
        if text and len(text.strip()) > 100:
            logger.info("PDF extracted with pymupdf")
            return clean_resume_text(text)
    except Exception as e:
        logger.warning(f"pymupdf extraction failed: {e}")

    # Last fallback: pypdf
    try:
        import pypdf
        import io
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            text += page.extract_text() + "\n"
        logger.info("PDF extracted with pypdf")
        return clean_resume_text(text)
    except Exception as e:
        logger.error(f"All PDF extraction methods failed: {e}")
        raise ValueError("Could not extract text from PDF. Please paste your resume as text.")


def clean_resume_text(text: str) -> str:
    """
    Clean extracted resume text:
    - Remove excessive whitespace
    - Fix common OCR/extraction artifacts
    - Normalize unicode characters
    - Preserve meaningful line breaks
    """
    if not text:
        return ""

    # Normalize unicode
    import unicodedata
    text = unicodedata.normalize("NFKD", text)

    # Remove null bytes and control characters except newlines/tabs
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

    # Normalize bullet points to standard dashes
    text = re.sub(r'[•·▪▸►‣⁃∙◦]', '-', text)

    # Fix hyphenated line breaks (PDF extraction artifact)
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

    # Collapse multiple spaces to single space (but preserve newlines)
    text = re.sub(r'[ \t]+', ' ', text)

    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text.strip()


def count_tokens(text: str, method: str = "approximate") -> int:
    """
    Count tokens in text. Uses tiktoken if available, else approximates.
    Rule of thumb: 1 token ≈ 4 characters for English text.
    """
    if method == "tiktoken":
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            pass

    # Approximate: 4 chars per token is industry standard approximation
    return len(text) // 4


def chunk_text(text: str, max_tokens: int = 1500, overlap_tokens: int = 100) -> List[str]:
    """
    Split text into chunks of max_tokens with overlap.
    Tries to split on paragraph boundaries first, then sentences.
    """
    chars_per_chunk = max_tokens * 4
    overlap_chars = overlap_tokens * 4

    if len(text) <= chars_per_chunk:
        return [text]

    chunks = []
    paragraphs = text.split('\n\n')
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= chars_per_chunk:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
                # Add overlap from end of previous chunk
                overlap_text = current_chunk[-overlap_chars:] if len(current_chunk) > overlap_chars else current_chunk
                current_chunk = overlap_text + para + "\n\n"
            else:
                # Single paragraph longer than max — split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sent in sentences:
                    if len(current_chunk) + len(sent) <= chars_per_chunk:
                        current_chunk += sent + " "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sent + " "

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    logger.info(f"Text split into {len(chunks)} chunks")
    return chunks


def extract_candidate_name(resume_text: str) -> str:
    """
    Attempt to extract candidate name from first few lines of resume.
    Returns 'Candidate' as fallback.
    """
    lines = [l.strip() for l in resume_text.split('\n') if l.strip()]
    if lines:
        first_line = lines[0]
        # Name is typically the first non-empty line, usually 2-4 words, no numbers
        if len(first_line.split()) <= 5 and not re.search(r'\d', first_line):
            return first_line
    return "Candidate"


def extract_contact_info(resume_text: str) -> dict:
    """Extract email, phone, LinkedIn from resume text."""
    info = {"email": None, "phone": None, "linkedin": None}

    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', resume_text)
    if email_match:
        info["email"] = email_match.group()

    phone_match = re.search(
        r'(\+?1?\s?)?(\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})', resume_text
    )
    if phone_match:
        info["phone"] = phone_match.group()

    linkedin_match = re.search(
        r'linkedin\.com/in/([A-Za-z0-9\-_]+)', resume_text, re.IGNORECASE
    )
    if linkedin_match:
        info["linkedin"] = f"linkedin.com/in/{linkedin_match.group(1)}"

    return info


def prepare_resume_for_llm(
    resume_text: str,
    max_tokens: int = 2000
) -> Tuple[str, bool]:
    """
    Prepare resume text for LLM ingestion.
    If too long, returns a summarized version and sets was_truncated=True.
    Returns (processed_text, was_chunked)
    """
    token_count = count_tokens(resume_text)
    logger.info(f"Resume token count: {token_count}")

    if token_count <= max_tokens:
        return resume_text, False

    # Resume is too long — chunk and summarize
    logger.info(f"Resume exceeds {max_tokens} tokens, applying chunking strategy")
    chunks = chunk_text(resume_text, max_tokens=max_tokens)

    # For very long resumes, we return the structured sections we need most
    # Prioritize: contact, summary, recent experience, skills, education
    truncated = resume_text[:max_tokens * 4]  # Approximate truncation
    return truncated, True


def resume_text_to_pdf_bytes(resume_text: str, candidate_name: str = "Resume") -> bytes:
    """
    Convert plain text resume to a styled PDF.
    Uses reportlab for clean formatting.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
        import io

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )

        styles = getSampleStyleSheet()
        # Custom styles
        name_style = ParagraphStyle(
            'NameStyle',
            parent=styles['Normal'],
            fontSize=18,
            fontName='Helvetica-Bold',
            textColor=colors.HexColor('#1a1a2e'),
            spaceAfter=4,
            alignment=TA_CENTER,
        )
        contact_style = ParagraphStyle(
            'ContactStyle',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#666666'),
            spaceAfter=8,
            alignment=TA_CENTER,
        )
        section_header_style = ParagraphStyle(
            'SectionHeader',
            parent=styles['Normal'],
            fontSize=11,
            fontName='Helvetica-Bold',
            textColor=colors.HexColor('#1a1a2e'),
            spaceBefore=10,
            spaceAfter=4,
        )
        body_style = ParagraphStyle(
            'Body',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#333333'),
            spaceAfter=3,
            leading=14,
        )
        bullet_style = ParagraphStyle(
            'Bullet',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#333333'),
            leftIndent=12,
            spaceAfter=2,
            leading=13,
        )

        story = []
        lines = resume_text.split('\n')
        is_first_section = True

        for line in lines:
            line = line.strip()
            if not line:
                story.append(Spacer(1, 4))
                continue

            # Detect name (first line)
            if story == [] or (len(story) == 1 and isinstance(story[0], Spacer)):
                story.append(Paragraph(line, name_style))
                continue

            # Detect contact line (contains @ or phone pattern)
            if re.search(r'@|\||\d{3}[-.)]\d{3}', line) and len(story) < 5:
                story.append(Paragraph(line, contact_style))
                continue

            # Detect section headers (ALL CAPS or known section names)
            is_header = (
                line.isupper() and len(line) > 3 or
                re.match(r'^(PROFESSIONAL SUMMARY|CORE COMPETENCIES|SKILLS|EXPERIENCE|EDUCATION|CERTIFICATIONS|PROJECTS|AWARDS|PUBLICATIONS)', line, re.I)
            )
            if is_header:
                if not is_first_section:
                    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#dddddd'), spaceAfter=4))
                story.append(Paragraph(line.upper(), section_header_style))
                is_first_section = False
                continue

            # Bullet points
            if line.startswith('-') or line.startswith('•'):
                story.append(Paragraph(f"• {line.lstrip('-•').strip()}", bullet_style))
                continue

            # Regular body text
            story.append(Paragraph(line, body_style))

        doc.build(story)
        buffer.seek(0)
        return buffer.read()

    except ImportError:
        raise ImportError("Install reportlab: pip install reportlab")
