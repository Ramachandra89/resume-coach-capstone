"""
Resume Coach - LangChain Chain Definitions
All LLM chain logic lives here:
  - CoachingReportChain
  - ResumeOptimizerChain
  - ChatCoachChain
  - CoverLetterChain
  - InterviewPrepChain
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple

from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import HumanMessage, AIMessage

logger = logging.getLogger(__name__)


def _parse_json_response(response: str) -> dict:
    """
    Robustly parse JSON from LLM response.
    Handles markdown code fences, trailing commas, and minor formatting issues.
    """
    # Strip markdown code fences
    response = re.sub(r'```json\s*', '', response)
    response = re.sub(r'```\s*', '', response)
    response = response.strip()

    # Find JSON object boundaries
    start = response.find('{')
    end = response.rfind('}')
    if start != -1 and end != -1:
        response = response[start:end+1]

    # Remove trailing commas before } or ]
    response = re.sub(r',\s*([}\]])', r'\1', response)

    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse failed: {e}\nRaw response:\n{response[:500]}")
        raise ValueError(f"LLM returned invalid JSON: {e}")


class CoachingReportChain:
    """
    Generates the full coaching report:
    1. Compress resume + JD to context summary
    2. Generate structured JSON coaching report
    3. Returns parsed dict + raw response for audit
    """

    def __init__(self, backend: Optional[str] = None):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        from backend.llm_client import get_llm
        from backend.prompts.templates import (
            CONTEXT_COMPRESSION_PROMPT,
            COACHING_REPORT_PROMPT,
        )

        self.llm_analytical = get_llm(backend=backend, temperature=0.2, max_tokens=2048)
        self.compression_chain = LLMChain(
            llm=self.llm_analytical,
            prompt=CONTEXT_COMPRESSION_PROMPT,
            verbose=True,
        )
        self.report_chain = LLMChain(
            llm=self.llm_analytical,
            prompt=COACHING_REPORT_PROMPT,
            verbose=True,
        )

    def run(self, resume_text: str, job_description: str) -> Tuple[Dict[str, Any], str]:
        """
        Run the full coaching report pipeline.
        Returns (parsed_report_dict, context_summary)
        """
        logger.info("Step 1: Compressing context...")
        context_summary = self.compression_chain.run(
            resume_text=resume_text,
            job_description=job_description,
        )
        logger.info(f"Context summary ({len(context_summary)} chars) generated.")

        logger.info("Step 2: Generating coaching report...")
        raw_response = self.report_chain.run(
            resume_text=resume_text,
            job_description=job_description,
            context_summary=context_summary,
        )

        report = _parse_json_response(raw_response)
        logger.info(f"Coaching report generated. ATS Score: {report.get('ats_score')}, Fit: {report.get('overall_fit_score')}")
        return report, context_summary


class ResumeOptimizerChain:
    """
    One-click resume optimizer:
    1. Rewrites resume to maximize ATS score
    2. Re-scores optimized resume
    3. Returns optimized text + before/after ATS delta
    """

    def __init__(self, backend: Optional[str] = None):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        from backend.llm_client import get_llm
        from backend.prompts.templates import RESUME_OPTIMIZATION_PROMPT, ATS_RESCORE_PROMPT

        self.llm_generative = get_llm(backend=backend, temperature=0.4, max_tokens=2048)
        self.llm_analytical = get_llm(backend=backend, temperature=0.1, max_tokens=512)

        self.optimize_chain = LLMChain(
            llm=self.llm_generative,
            prompt=RESUME_OPTIMIZATION_PROMPT,
            verbose=True,
        )
        self.rescore_chain = LLMChain(
            llm=self.llm_analytical,
            prompt=ATS_RESCORE_PROMPT,
            verbose=True,
        )

    def run(
        self,
        resume_text: str,
        job_description: str,
        context_summary: str,
        original_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Returns:
        {
          "optimized_resume": str,
          "new_ats_score": int,
          "new_ats_breakdown": dict,
          "original_ats_score": int,
          "score_delta": int,
          "new_keywords_added": list,
          "improvement_summary": str
        }
        """
        gaps = json.dumps([g["title"] for g in original_report.get("gaps", [])[:5]])
        missing_keywords = json.dumps(original_report.get("missing_keywords", [])[:20])
        original_ats = original_report.get("ats_score", 0)

        logger.info("Generating optimized resume...")
        optimized_resume = self.optimize_chain.run(
            resume_text=resume_text,
            job_description=job_description,
            context_summary=context_summary,
            gaps=gaps,
            missing_keywords=missing_keywords,
        )
        optimized_resume = optimized_resume.strip()

        logger.info("Re-scoring optimized resume...")
        rescore_raw = self.rescore_chain.run(
            optimized_resume=optimized_resume,
            job_description=job_description,
        )
        rescore = _parse_json_response(rescore_raw)

        new_ats = rescore.get("ats_score", original_ats + 10)

        return {
            "optimized_resume": optimized_resume,
            "new_ats_score": new_ats,
            "new_ats_breakdown": rescore.get("ats_score_breakdown", {}),
            "original_ats_score": original_ats,
            "score_delta": new_ats - original_ats,
            "new_keywords_added": rescore.get("new_keywords_added", []),
            "remaining_gaps": rescore.get("remaining_gaps", []),
            "improvement_summary": rescore.get("improvement_summary", ""),
        }


class ChatCoachChain:
    """
    Multi-turn conversational coaching chain with sliding window memory.
    Uses ConversationSummaryBufferMemory to handle Llama-3.1's 128k context.
    """

    def __init__(self, backend: Optional[str] = None):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        from backend.llm_client import get_llm
        from config.settings import MEMORY_MAX_TOKEN_LIMIT
        from backend.prompts.templates import CHAT_SYSTEM_PROMPT, CHAT_HUMAN_PROMPT

        self.llm = get_llm(backend=backend, temperature=0.5, max_tokens=1024)
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=MEMORY_MAX_TOKEN_LIMIT,
            return_messages=True,
            memory_key="chat_history",
        )
        self.context_summary = ""
        self.fit_score = 0
        self.ats_score = 0
        self.fit_verdict = ""
        self.top_gaps = ""
        self.top_strengths = ""

    def initialize(
        self,
        context_summary: str,
        report: Dict[str, Any],
    ):
        """Set coaching context from the generated report."""
        self.context_summary = context_summary
        self.fit_score = report.get("overall_fit_score", 0)
        self.ats_score = report.get("ats_score", 0)
        self.fit_verdict = report.get("fit_verdict", "")
        self.top_gaps = ", ".join([g["title"] for g in report.get("gaps", [])[:3]])
        self.top_strengths = ", ".join([s["title"] for s in report.get("strengths", [])[:3]])

    def chat(self, user_message: str) -> str:
        """Process a user message and return the coach's response."""
        from backend.prompts.templates import CHAT_SYSTEM_PROMPT

        system_content = CHAT_SYSTEM_PROMPT.format(
            context_summary=self.context_summary,
            fit_score=self.fit_score,
            ats_score=self.ats_score,
            fit_verdict=self.fit_verdict,
            top_gaps=self.top_gaps,
            top_strengths=self.top_strengths,
        )

        # Build messages list with memory
        messages = [{"role": "system", "content": system_content}]

        # Add memory history
        for msg in self.memory.chat_memory.messages:
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})

        messages.append({"role": "user", "content": user_message})

        # Invoke LLM
        response = self.llm.invoke(messages)

        # Handle different response types
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)

        # Save to memory
        self.memory.chat_memory.add_user_message(user_message)
        self.memory.chat_memory.add_ai_message(response_text)

        return response_text

    def get_history(self) -> List[Dict[str, str]]:
        """Return chat history as list of dicts."""
        history = []
        for msg in self.memory.chat_memory.messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
        return history

    def clear_memory(self):
        """Reset conversation memory."""
        self.memory.clear()


class CoverLetterChain:
    """Generates a tailored cover letter."""

    def __init__(self, backend: Optional[str] = None):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        from backend.llm_client import get_llm
        from backend.prompts.templates import COVER_LETTER_PROMPT

        self.llm = get_llm(backend=backend, temperature=0.6, max_tokens=1024)
        self.chain = LLMChain(llm=self.llm, prompt=COVER_LETTER_PROMPT, verbose=True)

    def run(self, context_summary: str, report: Dict[str, Any], job_description: str) -> str:
        top_strengths = "\n".join([
            f"- {s['title']}: {s['detail']}"
            for s in report.get("strengths", [])[:3]
        ])
        return self.chain.run(
            context_summary=context_summary,
            fit_score=report.get("overall_fit_score", 0),
            top_strengths=top_strengths,
            job_description=job_description,
        ).strip()


class InterviewPrepChain:
    """Generates targeted interview preparation guide."""

    def __init__(self, backend: Optional[str] = None):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        from backend.llm_client import get_llm
        from backend.prompts.templates import INTERVIEW_PREP_PROMPT

        self.llm = get_llm(backend=backend, temperature=0.4, max_tokens=2048)
        self.chain = LLMChain(llm=self.llm, prompt=INTERVIEW_PREP_PROMPT, verbose=True)

    def run(self, context_summary: str, report: Dict[str, Any], job_description: str) -> Dict[str, Any]:
        strengths_text = "\n".join([
            f"- {s['title']}: {s['detail']}"
            for s in report.get("strengths", [])[:4]
        ])
        gaps_text = "\n".join([
            f"- {g['title']} ({g['severity']}): {g['detail']}"
            for g in report.get("gaps", [])[:4]
        ])
        raw = self.chain.run(
            context_summary=context_summary,
            strengths=strengths_text,
            gaps=gaps_text,
            job_description=job_description,
        )
        return _parse_json_response(raw)
