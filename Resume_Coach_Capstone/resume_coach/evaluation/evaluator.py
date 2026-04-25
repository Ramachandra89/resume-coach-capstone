"""
Resume Coach - Model Evaluation Framework
Compares responses from multiple LLM backends on standardized test cases.
Results are stored in S3 for inclusion in the project report.

Evaluation dimensions:
1. JSON validity — does output parse correctly?
2. Schema completeness — are all required keys present?
3. Score calibration — are scores reasonable for given inputs?
4. Specificity — does advice reference actual resume/JD details?
5. Latency — response time per backend
6. Coherence — human-readable quality score (manual)
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# TEST CASES
# ─────────────────────────────────────────────

TEST_CASES = [
    {
        "id": "tc_001",
        "name": "Senior Engineer — Strong Match",
        "resume": """
Alex Johnson | alex.johnson@email.com | linkedin.com/in/alexj | San Francisco, CA

PROFESSIONAL SUMMARY
Senior Software Engineer with 7 years building distributed systems at scale.
Passionate about cloud-native architectures and developer tooling.

CORE COMPETENCIES
Python | AWS | Kubernetes | Terraform | PostgreSQL | Redis | Kafka | CI/CD | Go | React

PROFESSIONAL EXPERIENCE

Senior Software Engineer | Stripe | San Francisco, CA | Mar 2021 – Present
• Architected event-driven microservices processing $2B+ in daily transactions
• Reduced infrastructure costs by 35% through Kubernetes optimization and spot instances
• Led team of 5 engineers to deliver new payment gateway API 2 weeks ahead of schedule
• Improved P99 latency from 450ms to 80ms via Redis caching and query optimization

Software Engineer | Airbnb | San Francisco, CA | Jun 2018 – Mar 2021
• Built real-time pricing engine serving 200M+ search requests/day using Python and Kafka
• Migrated 15 microservices from on-premise to AWS EKS, reducing downtime by 99%
• Mentored 3 junior engineers; 2 promoted to mid-level within 18 months

EDUCATION
M.S. Computer Science | Stanford University | 2018
B.S. Computer Science | UC Berkeley | 2016

CERTIFICATIONS
AWS Solutions Architect Professional | 2022
Certified Kubernetes Administrator (CKA) | 2021
""",
        "job_description": """
Staff Software Engineer — Platform Infrastructure
TechCorp Inc. | San Francisco, CA | $180k-$250k

About the Role:
We're looking for a Staff Engineer to lead the design and implementation of our next-generation
platform infrastructure. You'll work closely with our VP of Engineering and architect solutions
that scale to 10M+ users.

Requirements:
• 7+ years software engineering experience
• Deep expertise in AWS (EC2, EKS, RDS, Lambda, CloudFront)
• Strong Python and/or Go proficiency
• Experience with Kubernetes at scale (100+ node clusters)
• Infrastructure as Code (Terraform or Pulumi)
• Distributed systems design experience
• Strong communication and leadership skills

Nice to have:
• Experience at high-growth fintech or marketplace
• Published technical blog posts or OSS contributions
• Experience with eBPF or service mesh (Istio, Linkerd)

Responsibilities:
• Own the platform infrastructure roadmap
• Lead a team of 6 platform engineers
• Drive adoption of internal developer platforms
• Reduce infrastructure costs while maintaining 99.99% SLA
""",
        "expected_fit": "Strong Match",
        "expected_score_range": (80, 100),
    },
    {
        "id": "tc_002",
        "name": "Career Changer — Partial Match",
        "resume": """
Maria Santos | maria.santos@email.com | New York, NY

PROFESSIONAL SUMMARY
Marketing Manager with 5 years of experience in digital marketing and brand strategy.
Looking to transition into product management.

SKILLS
Digital Marketing | SEO/SEM | Google Analytics | A/B Testing | Excel | SQL (basic)
Customer Research | Campaign Management | Salesforce | Jira | Agile (certified)

PROFESSIONAL EXPERIENCE

Senior Marketing Manager | CPG Brand | New York, NY | 2020 – Present
• Led digital marketing campaigns generating $15M in annual revenue
• Managed $2M budget across 10+ channels with 340% average ROAS
• Used Google Analytics and SQL to analyze user behavior and optimize conversion funnels
• Collaborated with product team on feature launch campaigns (GTM strategy)
• Conducted 50+ customer interviews to inform campaign messaging

Marketing Manager | Agency | New York | 2018 – 2020
• Managed 12 client accounts across e-commerce, SaaS, and consumer brands
• Built reporting dashboards in Excel and Google Data Studio

EDUCATION
B.A. Marketing | NYU Stern | 2018

CERTIFICATIONS
CSPO (Certified Scrum Product Owner) | 2023
Google Analytics 4 | 2022
""",
        "job_description": """
Associate Product Manager
Fintech Startup | New York, NY | $90k-$110k

We're looking for an APM to join our growing product team. This role is ideal for someone
with 2-3 years of experience who wants to own features end-to-end.

Requirements:
• 2+ years in product management, UX research, or related field
• Experience with agile/scrum methodology
• Ability to write clear product requirements (PRDs)
• SQL proficiency for data analysis
• Strong communication and stakeholder management

Nice to have:
• Experience in fintech or financial products
• Technical background (engineering or CS degree)
• Experience with product analytics tools (Amplitude, Mixpanel)

Responsibilities:
• Define product requirements for new features
• Work with engineering and design teams
• Analyze user data to identify opportunities
• Run A/B tests and measure feature impact
""",
        "expected_fit": "Partial Match",
        "expected_score_range": (45, 70),
    },
]

# ─────────────────────────────────────────────
# EVALUATOR
# ─────────────────────────────────────────────

@dataclass
class EvaluationResult:
    test_case_id: str
    backend: str
    model_name: str
    latency_seconds: float
    json_valid: bool
    schema_complete: bool
    fit_score: Optional[int]
    ats_score: Optional[int]
    fit_verdict: Optional[str]
    expected_verdict: str
    verdict_correct: bool
    score_in_expected_range: bool
    num_gaps: int
    num_strengths: int
    num_missing_keywords: int
    has_specificity: bool  # Does response reference actual content?
    error: Optional[str]
    raw_response_preview: str


REQUIRED_SCHEMA_KEYS = [
    "overall_fit_score", "ats_score", "fit_verdict", "fit_summary",
    "strengths", "gaps", "missing_keywords", "present_keywords",
    "coaching_recommendations", "application_recommendation"
]


class ModelEvaluator:

    def __init__(self, backends_to_test: List[Dict[str, str]]):
        """
        backends_to_test: list of {"backend": "nebius", "model": "meta-llama/..."}
        """
        self.backends = backends_to_test
        self.results: List[EvaluationResult] = []

    def run_all(self) -> List[EvaluationResult]:
        """Run all test cases against all backends."""
        for backend_config in self.backends:
            backend = backend_config["backend"]
            model = backend_config.get("model", "")
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating backend: {backend} | model: {model}")
            logger.info(f"{'='*60}")

            import sys, os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            import os
            if model:
                os.environ["NEBIUS_MODEL"] = model

            for tc in TEST_CASES:
                logger.info(f"Running test case: {tc['name']}")
                result = self._evaluate_single(tc, backend, model)
                self.results.append(result)
                self._print_result(result)

        return self.results

    def _evaluate_single(self, tc: dict, backend: str, model: str) -> EvaluationResult:
        """Evaluate a single test case on a single backend."""
        from backend.chains.coaching_chains import CoachingReportChain

        start = time.time()
        error = None
        report = None

        try:
            chain = CoachingReportChain(backend=backend)
            report, _ = chain.run(tc["resume"], tc["job_description"])
            latency = time.time() - start
        except Exception as e:
            latency = time.time() - start
            error = str(e)
            logger.error(f"Error on {backend}/{model}: {e}")

        if report is None:
            return EvaluationResult(
                test_case_id=tc["id"], backend=backend, model_name=model,
                latency_seconds=latency, json_valid=False, schema_complete=False,
                fit_score=None, ats_score=None, fit_verdict=None,
                expected_verdict=tc["expected_fit"], verdict_correct=False,
                score_in_expected_range=False, num_gaps=0, num_strengths=0,
                num_missing_keywords=0, has_specificity=False,
                error=error, raw_response_preview=str(error)[:200]
            )

        fit_score = report.get("overall_fit_score")
        ats_score = report.get("ats_score")
        fit_verdict = report.get("fit_verdict", "")
        schema_keys_present = all(k in report for k in REQUIRED_SCHEMA_KEYS)

        # Check specificity: does any gap/strength mention actual resume content?
        resume_keywords = ["Python", "AWS", "Stripe", "Airbnb", "Marketing", "SQL", "Agile"]
        response_text = json.dumps(report).lower()
        has_specificity = any(kw.lower() in response_text for kw in resume_keywords)

        low, high = tc["expected_score_range"]
        score_in_range = (fit_score is not None and low <= fit_score <= high)

        return EvaluationResult(
            test_case_id=tc["id"],
            backend=backend,
            model_name=model,
            latency_seconds=round(latency, 2),
            json_valid=True,
            schema_complete=schema_keys_present,
            fit_score=fit_score,
            ats_score=ats_score,
            fit_verdict=fit_verdict,
            expected_verdict=tc["expected_fit"],
            verdict_correct=(fit_verdict == tc["expected_fit"]),
            score_in_expected_range=score_in_range,
            num_gaps=len(report.get("gaps", [])),
            num_strengths=len(report.get("strengths", [])),
            num_missing_keywords=len(report.get("missing_keywords", [])),
            has_specificity=has_specificity,
            error=error,
            raw_response_preview=json.dumps(report)[:200],
        )

    def _print_result(self, r: EvaluationResult):
        status = "✅" if (r.json_valid and r.schema_complete and r.verdict_correct) else "⚠️"
        logger.info(f"""
  {status} Test: {r.test_case_id} | {r.backend}/{r.model_name}
    Latency:     {r.latency_seconds}s
    JSON Valid:  {r.json_valid} | Schema Complete: {r.schema_complete}
    Fit Score:   {r.fit_score} | ATS: {r.ats_score}
    Verdict:     {r.fit_verdict} (expected: {r.expected_verdict}) | Correct: {r.verdict_correct}
    Gaps: {r.num_gaps} | Strengths: {r.num_strengths} | Specific: {r.has_specificity}
    Error:       {r.error or 'None'}
""")

    def generate_report(self, output_path: str = "./evaluation/eval_results.json"):
        """Generate evaluation summary report."""
        results_data = [asdict(r) for r in self.results]

        # Summary stats by backend
        summary = {}
        for r in self.results:
            key = f"{r.backend}/{r.model_name}"
            if key not in summary:
                summary[key] = {
                    "total": 0, "json_valid": 0, "schema_complete": 0,
                    "verdict_correct": 0, "score_in_range": 0,
                    "avg_latency": 0, "total_latency": 0,
                }
            s = summary[key]
            s["total"] += 1
            s["json_valid"] += int(r.json_valid)
            s["schema_complete"] += int(r.schema_complete)
            s["verdict_correct"] += int(r.verdict_correct)
            s["score_in_range"] += int(r.score_in_expected_range)
            s["total_latency"] += r.latency_seconds

        for key, s in summary.items():
            if s["total"] > 0:
                s["avg_latency"] = round(s["total_latency"] / s["total"], 2)
                s["json_valid_rate"] = f"{s['json_valid']/s['total']*100:.0f}%"
                s["schema_complete_rate"] = f"{s['schema_complete']/s['total']*100:.0f}%"
                s["verdict_accuracy"] = f"{s['verdict_correct']/s['total']*100:.0f}%"

        full_report = {
            "evaluation_id": str(uuid.uuid4()),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "summary": summary,
            "results": results_data,
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(full_report, f, indent=2)
        logger.info(f"Evaluation report saved to {output_path}")

        # Print summary table
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        for key, s in summary.items():
            print(f"\n{key}")
            print(f"  JSON Valid Rate:    {s.get('json_valid_rate','N/A')}")
            print(f"  Schema Complete:    {s.get('schema_complete_rate','N/A')}")
            print(f"  Verdict Accuracy:   {s.get('verdict_accuracy','N/A')}")
            print(f"  Avg Latency:        {s.get('avg_latency','N/A')}s")
        print("="*70)

        return full_report


if __name__ == "__main__":
    # Backends to compare — update model names as desired
    backends = [
        {"backend": "nebius", "model": "meta-llama/Meta-Llama-3-8B-Instruct"},
        {"backend": "nebius", "model": "meta-llama/Meta-Llama-3.1-70B-Instruct"},
        {"backend": "nebius", "model": "meta-llama/Llama-3.1-8B-Instruct"},
        # {"backend": "local_hf", "model": "meta-llama/Llama-3.1-8B-Instruct"},
        # {"backend": "sagemaker", "model": "meta-textgeneration-llama-3-8b-instruct"},
    ]

    evaluator = ModelEvaluator(backends_to_test=backends)
    evaluator.run_all()
    report = evaluator.generate_report()

    # Upload to S3
    try:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from utils.s3_storage import upload_evaluation
        s3_key = upload_evaluation(report)
        logger.info(f"Evaluation uploaded to S3: {s3_key}")
    except Exception as e:
        logger.warning(f"S3 upload skipped: {e}")
