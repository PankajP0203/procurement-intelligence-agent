import os
from typing import Literal
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors as genai_errors
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

load_dotenv()

gemini = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


class VendorRiskReport(BaseModel):
    vendor_name: str
    search_summary: str
    risk_indicators: list[str]
    overall_sentiment: Literal["positive", "neutral", "concerning"]


@retry(
    retry=retry_if_exception(lambda e: isinstance(e, genai_errors.ClientError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=10, max=60),
    reraise=True,
)
def _llm_call(**kwargs):
    return gemini.models.generate_content(**kwargs)


def lookup_vendor_risk(vendor_name: str) -> VendorRiskReport:
    """
    Two-step vendor research:
    1. Grounded Google Search call — retrieves fresh news about the vendor.
    2. Structured extraction call — parses the raw findings into VendorRiskReport.

    The two-step split is required because Gemini does not allow google_search
    and response_schema in the same request.
    """
    search_prompt = (
        f'Search for recent news about the company "{vendor_name}". '
        "Focus specifically on: lawsuits or legal disputes, data breaches or "
        "security incidents, financial distress or bankruptcy risk, regulatory "
        "penalties or compliance failures, and any other negative press relevant "
        "to a procurement team evaluating this vendor. "
        "If no negative signals are found, state that clearly."
    )

    search_response = _llm_call(
        model="gemini-2.5-flash",
        contents=search_prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )
    raw_text = (search_response.text or "").strip()

    structure_prompt = f"""You are a procurement risk analyst. Based on the research below about "{vendor_name}", produce a structured risk assessment.

Research findings:
{raw_text}

Instructions:
- vendor_name: the exact company name as searched
- search_summary: 2-3 sentences summarising the most relevant procurement risk findings; if nothing concerning was found say so directly
- risk_indicators: list of specific, dated risk signals (e.g. "Class-action lawsuit filed Jan 2024", "FTC fine of $4M for data misuse Q2 2023"); empty list if none found
- overall_sentiment: "positive" if the vendor looks clean, "neutral" if signals are minor or ambiguous, "concerning" if clear risk signals are present"""

    structured_response = _llm_call(
        model="gemini-2.5-flash",
        contents=structure_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=VendorRiskReport,
        ),
    )

    result = structured_response.parsed
    if result is None:
        return VendorRiskReport(
            vendor_name=vendor_name,
            search_summary="Structured extraction failed. Raw findings available in risk indicators.",
            risk_indicators=[raw_text[:400]] if raw_text else ["Search returned no results."],
            overall_sentiment="neutral",
        )
    return result
