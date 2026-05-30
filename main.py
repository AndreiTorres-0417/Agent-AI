import json
import logging
import os
import re
from io import BytesIO
from typing import Any, Dict, Generator, List, Optional

import requests
from docx import Document
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("academic-review-mvp")

app = FastAPI(title="Academic Review MVP", version="0.1.0")

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled API error at %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Unhandled server error",
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        },
    )


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Client-provided chat session ID")
    message: str = Field(..., min_length=1, description="Latest user message")
    model: Optional[str] = Field(None, description="Optional Ollama model override")
    format_mode: Optional[str] = Field("general", description="Review style: general, apa7, or ieee")


class ChatResponse(BaseModel):
    session_id: str
    phase: str
    questions: List[str]
    next_prompt: str
    context: Dict[str, Any]
    analysis: Optional[Dict[str, Any]] = None


class AnalyzeTextRequest(BaseModel):
    text: str = Field(..., min_length=20)
    metadata: Optional[Dict[str, Any]] = None
    model: Optional[str] = Field(None, description="Optional Ollama model override")
    format_mode: Optional[str] = Field("general", description="Review style: general, apa7, or ieee")


class AnalyzeTextStreamRequest(AnalyzeTextRequest):
    pass


class IssueItem(BaseModel):
    issue: str
    severity: str
    evidence: str
    recommendation: str


class SuggestionItem(BaseModel):
    priority: str
    suggestion: str
    rationale: str
    expected_impact: str


class RewriteItem(BaseModel):
    original_excerpt: str
    rewritten_excerpt: str
    reason: str


class HighlightItem(BaseModel):
    excerpt: str
    message: str
    severity: str
    category: str


class FormatChangeItem(BaseModel):
    issue: str
    severity: str
    category: str = "IEEE formatting"
    original: str
    replacement: str
    note: str
    sub_issues: List[str] = Field(default_factory=list)


class FormatTextRequest(BaseModel):
    text: str = Field(..., min_length=20)


class ExportDocxRequest(BaseModel):
    text: str = Field(..., min_length=1)
    filename: str = "ieee_formatted.docx"


class FormatResponse(BaseModel):
    summary: str
    fixed_text: str
    changes: List[FormatChangeItem]
    highlights: List[HighlightItem]
    transformations_applied: bool = False
    transformation_count: int = 0
    source: str = "deterministic_ieee_formatter"


class AnalysisResponse(BaseModel):
    summary: str
    structure_format_issues: List[IssueItem]
    academic_quality_issues: List[IssueItem]
    citation_consistency_issues: List[IssueItem]
    prioritized_suggestions: List[SuggestionItem]
    optional_rewrite_suggestions: List[RewriteItem]
    highlights: List[HighlightItem]
    reviewed_text: Optional[str] = None
    source: str
    fallback_used: bool


INTAKE_QUESTIONS = [
    "What is your course or discipline?",
    "What assignment type is this (essay, report, literature review, etc.)?",
    "What citation style is required (APA, MLA, Chicago, Harvard, other)?",
    "What grading criteria or rubric should I prioritize?",
    "Do you want strict critique only, or critique plus rewrite suggestions?",
]

chat_sessions: Dict[str, Dict[str, Any]] = {}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse("static/index.html")


def get_ollama_url() -> str:
    return os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")


ALLOWED_MODELS = {"deepseek-r1:8b", "gemma3:1b", "groq:llama-3.1-8b-instant"}
ALLOWED_FORMAT_MODES = {"general", "apa7", "ieee"}


def get_ollama_model(model_override: Optional[str] = None) -> str:
    model = model_override or os.getenv("OLLAMA_MODEL", "deepseek-r1:8b")
    if model not in ALLOWED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model '{model}'. Choose one of: {', '.join(sorted(ALLOWED_MODELS))}.",
        )
    return model


def is_groq_model(model: str) -> bool:
    return model.startswith("groq:")


def groq_model_id(model: str) -> str:
    return model.removeprefix("groq:")


def get_groq_api_key() -> str:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY is not configured. Add it to .env to use Groq hosted models.")
    return key


def normalize_format_mode(format_mode: Optional[str] = None) -> str:
    mode = (format_mode or "general").lower()
    if mode not in ALLOWED_FORMAT_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format mode '{format_mode}'. Choose one of: {', '.join(sorted(ALLOWED_FORMAT_MODES))}.",
        )
    return mode


def int_to_roman(value: int) -> str:
    pairs = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    result = ""
    for number, roman in pairs:
        while value >= number:
            result += roman
            value -= number
    return result


def reference_number_from_text(value: str) -> Optional[str]:
    match = re.search(r"\[(\d+)\]", value)
    return match.group(1) if match else None


def clean_ieee_findings(changes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    allowed_categories = {
        "IEEE formatting",
        "IEEE citation issues",
        "citation consistency",
        "citation numbering issues",
        "research methodology",
        "academic tone",
        "reference heading",
        "figure label",
        "table label",
    }

    def severity_rank(value: str) -> int:
        return {"low": 1, "medium": 2, "high": 3}.get(value, 2)

    def add_group(
        key: str,
        issue: str,
        severity: str,
        category: str,
        original: str,
        replacement: str,
        note: str,
        sub_issue: Optional[str] = None,
    ) -> None:
        current = grouped.get(key)
        if not current:
            grouped[key] = {
                "issue": issue,
                "severity": severity,
                "category": category,
                "original": original,
                "replacement": replacement,
                "note": note,
                "sub_issues": [],
            }
            current = grouped[key]
        elif severity_rank(severity) > severity_rank(current["severity"]):
            current["severity"] = severity
        if sub_issue and sub_issue not in current["sub_issues"]:
            current["sub_issues"].append(sub_issue)

    for item in changes:
        issue = item.get("issue", "")
        category = item.get("category", "IEEE formatting")
        if category not in allowed_categories:
            continue
        original = item.get("original", "")
        replacement = item.get("replacement", "")
        note = item.get("note", "")
        severity = item.get("severity", "medium")
        ref_no = reference_number_from_text(original) or reference_number_from_text(replacement)

        if ref_no and (
            "APA-style" in issue
            or "author order" in issue
            or "ampersand" in issue
            or "year placement" in issue
            or "multiple-author" in issue
        ):
            add_group(
                f"ref-{ref_no}-ieee-reference",
                f"Reference [{ref_no}] does not comply with IEEE reference format",
                "high",
                "IEEE citation issues",
                original,
                "Rewrite using IEEE reference structure, e.g., [n] J. Smith, \"Article Title,\" Source, year.",
                "Multiple formatting patterns were detected in the same reference entry.",
                issue,
            )
            continue

        if ref_no and (
            "incomplete" in issue.lower()
            or "raw url" in issue.lower()
            or "web reference" in issue.lower()
            or "et al." in issue.lower()
            or "quotation marks" in issue.lower()
        ):
            add_group(
                f"ref-{ref_no}-ieee-reference",
                f"Reference [{ref_no}] does not comply with IEEE reference format",
                severity,
                "IEEE citation issues",
                original,
                "Add complete IEEE metadata: initials and surnames, quoted title, source/site, year, URL/DOI if applicable, and access date for web sources.",
                "The reference is missing required IEEE metadata or structure.",
                issue,
            )
            continue

        if "Citation order" in issue or "skipped" in issue.lower() or "not tied" in issue:
            add_group(
                "citation-numbering-sync",
                "Citation numbering and reference list are not synchronized",
                "high",
                "citation numbering issues",
                original,
                "Review citation order manually; do not renumber automatically unless each source identity is verified.",
                "IEEE numbering must follow first appearance and every listed reference should match an in-text citation.",
                issue,
            )
            continue

        if "Mixed citation systems" in issue or "Narrative source mention" in issue:
            add_group(
                "mixed-citation-systems",
                "The paper mixes IEEE numbered citations with author-name citation style",
                "high",
                "citation consistency",
                original,
                "Use numbered IEEE citations consistently, e.g., replace author-name source mentions with the correct bracket number after verifying the source.",
                "The text contains both numbered citations and author-name references.",
                issue,
            )
            continue

        if "causal" in issue.lower() or "overstated" in issue.lower():
            add_group(
                "survey-causation",
                "Survey-based methodology does not support causal conclusions",
                "high",
                "research methodology",
                original,
                "Use wording such as 'is associated with' or 'shows a relationship with.'",
                "The paper uses causal language even though the method appears to be a survey.",
                issue,
            )
            continue

        key = f"{category}-{issue}-{original[:40]}"
        add_group(key, issue, severity, category, original, replacement, note)

    cleaned = list(grouped.values())
    cleaned.sort(key=lambda item: ({"high": 0, "medium": 1, "low": 2}.get(item["severity"], 1), item["category"], item["issue"]))
    return cleaned


def split_applied_transformations(changes: List[Dict[str, Any]], fixed_text: str) -> tuple[List[Dict[str, Any]], int]:
    transformation_keywords = [
        "converted",
        "changed",
        "normalized",
        "numbered",
        "revised",
    ]
    unresolved: List[Dict[str, Any]] = []
    transformation_count = 0
    allowed_categories = {
        "IEEE formatting",
        "IEEE citation issues",
        "citation consistency",
        "citation numbering issues",
        "research methodology",
        "academic tone",
        "reference heading",
        "figure label",
        "table label",
    }

    for item in changes:
        if item.get("category", "IEEE formatting") not in allowed_categories:
            continue
        if item.get("category") in {"academic tone", "research methodology"}:
            unresolved.append(item)
            continue
        issue = item.get("issue", "").lower()
        original = item.get("original", "")
        looks_like_applied_fix = any(keyword in issue for keyword in transformation_keywords)

        if looks_like_applied_fix and (not original or original not in fixed_text):
            transformation_count += 1
            continue

        if item.get("issue") == "Reference entry numbered" and re.match(r"^\[\d+\]", item.get("replacement", "")):
            transformation_count += 1
            continue

        unresolved.append(item)

    return unresolved, transformation_count


def ieee_format_text(text: str) -> Dict[str, Any]:
    fixed = text
    changes: List[Dict[str, str]] = []
    highlights: List[Dict[str, str]] = []
    citation_map: Dict[str, int] = {}

    def add_change(issue: str, severity: str, original: str, replacement: str, note: str, category: str = "IEEE formatting") -> None:
        changes.append(
            {
                "issue": issue,
                "severity": severity,
                "category": category,
                "original": original,
                "replacement": replacement,
                "note": note,
            }
        )
        if original:
            highlights.append(
                {
                    "excerpt": original,
                    "message": note,
                    "severity": severity,
                    "category": category,
                }
            )

    def citation_number(key: str) -> int:
        if key not in citation_map:
            citation_map[key] = len(citation_map) + 1
        return citation_map[key]

    def replace_apa_citation(match: re.Match[str]) -> str:
        original = match.group(0)
        author = match.group(1).strip()
        year = match.group(2).strip()
        number = citation_number(f"{author.lower()}-{year}")
        replacement = f"[{number}]"
        add_change(
            "APA author-date citation converted to IEEE numeric citation",
            "high",
            original,
            replacement,
            "IEEE uses numbered bracket citations in order of first appearance.",
                    "IEEE citation issues",
                )
        return replacement

    fixed = re.sub(r"\(([A-Z][A-Za-z\s&.-]{1,60}),\s*(\d{4}[a-z]?)\)", replace_apa_citation, fixed)

    def replace_bare_author_citation(match: re.Match[str]) -> str:
        original = match.group(0)
        author = match.group(1).strip()
        number = citation_number(author.lower())
        replacement = f"[{number}]"
        add_change(
            "Bare author parenthetical citation converted to IEEE numeric citation",
            "high",
            original,
            replacement,
            "IEEE does not use parenthetical author-only citations such as (Smith). Use a numbered citation.",
            "in-text citation",
        )
        return replacement

    fixed = re.sub(r"\(([A-Z][A-Za-z.-]{2,30}(?:\s+et\s+al\.)?)\)", replace_bare_author_citation, fixed)

    def replace_author_year_bracket(match: re.Match[str]) -> str:
        original = match.group(0)
        author = match.group(1).strip()
        year = match.group(2).strip()
        number = citation_number(f"{author.lower()}-{year}")
        replacement = f"[{number}]"
        add_change(
            "Non-IEEE author-year bracket citation converted",
            "high",
            original,
            replacement,
            "IEEE does not use author [year] citation style.",
            "IEEE citation issues",
        )
        return replacement

    fixed = re.sub(r"\b([A-Z][A-Za-z.-]{2,30})\s*\[(\d{4}[a-z]?)\]", replace_author_year_bracket, fixed)

    narrative_uncited = re.findall(
        r"\b([A-Z][A-Za-z.-]{2,30}(?:\s+et\s+al\.)?)\s+(?:argues?|states?|found|finds|suggests?|indicates?|demonstrates?|claims?|reports?|concludes?)\b(?!\s*\[\d+\])",
        fixed,
    )
    trailing_author_after_citation = re.findall(
        r"(?:\[\d+\](?:,\s*)?)+(?:\s*,?\s*and\s+|\s*,\s*)([A-Z][A-Za-z.-]{2,30}(?:\s+et\s+al\.?)?)",
        fixed,
    )
    narrative_uncited.extend(trailing_author_after_citation)
    for author in sorted(set(narrative_uncited), key=str.lower):
        add_change(
            "Narrative source mention may be missing IEEE bracket citation",
            "high",
            author,
            f"{author} [n]",
            "IEEE narrative citations still need a numbered bracket citation, e.g., Smith [1] states...",
            "in-text citation",
        )

    has_author_parenthetical = bool(re.search(r"\([A-Z][A-Za-z.-]{2,30}(?:,\s*\d{4})?\)", text))
    has_author_year_bracket = bool(re.search(r"\b[A-Z][A-Za-z.-]{2,30}\s*\[\d{4}\]", text))
    has_ieee_brackets = bool(re.search(r"\[\d+\]", text))
    has_narrative_source = bool(narrative_uncited)
    mixed_count = sum([has_author_parenthetical, has_author_year_bracket, has_ieee_brackets, has_narrative_source])
    if mixed_count >= 2:
        add_change(
            "Mixed citation systems detected",
            "high",
            "mixed citation styles",
            "IEEE numbered citations only",
            "The draft appears to mix author-date/author-only/narrative citations with IEEE bracket citations.",
            "citation consistency",
        )

    survey_design = bool(re.search(r"\bsurvey\b|\bquestionnaire\b|\brespondents?\b", fixed, re.IGNORECASE))
    causal_overclaim = re.findall(
        r"\b(directly causes?|causes?|determine whether .{0,80}? causes?|directly improves?|causes better .{0,40})\b",
        fixed,
        re.IGNORECASE,
    )
    if survey_design and causal_overclaim:
        add_change(
            "Causal claim is not supported by survey design",
            "high",
            causal_overclaim[0],
            "is associated with",
            "A survey can usually support association, not direct causation.",
            "research methodology",
        )

    overstated_findings = re.findall(
        r"\b(coffee directly improves [^.]+|coffee consumption causes [^.]+|directly improves [^.]+|causes better [^.]+)\b",
        fixed,
        re.IGNORECASE,
    )
    for phrase in sorted(set(overstated_findings), key=str.lower):
        add_change(
            "Overstated causal finding",
            "high",
            phrase,
            phrase.replace("directly improves", "is associated with").replace("causes", "is associated with"),
            "Survey results should be framed as association or relationship unless the design supports causality.",
            "research methodology",
        )

    for wrong_heading in ["Bibliography", "Works Cited", "Reference List"]:
        pattern = re.compile(rf"(?im)^\s*{re.escape(wrong_heading)}\s*$")
        if pattern.search(fixed):
            fixed = pattern.sub("References", fixed)
            add_change(
                "Reference list title changed to IEEE style",
                "medium",
                wrong_heading,
                "References",
                "IEEE papers use the heading 'References'.",
                "reference heading",
            )

    fixed = re.sub(
        r"(?im)^Figure\s+(\d+)\s*[:.-]\s*(.+)$",
        lambda m: f"Fig. {m.group(1)}. {m.group(2).strip()}",
        fixed,
    )
    if re.search(r"(?im)^Figure\s+\d+\s*[:.-]", text):
        add_change(
            "Figure label normalized",
            "medium",
            "Figure n:",
            "Fig. n.",
            "IEEE figure captions use 'Fig. 1. Caption text.'",
            "figure label",
        )

    fixed = re.sub(
        r"(?im)^Table\s+(\d+)\s*[:.-]\s*(.+)$",
        lambda m: f"Table {int_to_roman(int(m.group(1)))}. {m.group(2).strip()}",
        fixed,
    )
    if re.search(r"(?im)^Table\s+\d+\s*[:.-]", text):
        add_change(
            "Table label normalized",
            "medium",
            "Table 1:",
            "Table I.",
            "IEEE table captions use Roman numerals such as 'Table I.'",
            "table label",
        )

    tone_replacements: List[tuple[str, str, str]] = [
        (r"\bproves\b", "suggests", "IEEE/academic reporting should avoid 'proves' when the method only supports association."),
        (r"\bdefinitely\b", "strongly suggests", "Use cautious academic wording."),
        (r"\bvery big impact\b", "substantial association", "Use precise academic phrasing."),
        (r"\ba lot of studies say\b", "prior studies suggest", "Use formal academic wording."),
    ]
    for pattern, replacement, note in tone_replacements:
        if re.search(pattern, fixed, flags=re.IGNORECASE):
            original = re.search(pattern, fixed, flags=re.IGNORECASE).group(0)
            fixed = re.sub(pattern, replacement, fixed, flags=re.IGNORECASE)
            add_change("Informal or overstrong wording revised", "medium", original, replacement, note, "academic tone")

    lines = fixed.splitlines()
    ref_start = None
    for idx, line in enumerate(lines):
        if re.match(r"^\s*References\s*$", line, re.IGNORECASE):
            ref_start = idx
            break

    if ref_start is not None:
        body_text = "\n".join(lines[:ref_start])
        body_citations = [int(x) for x in re.findall(r"\[(\d+)\]", body_text)]
        body_citation_set = set(body_citations)
        reference_numbers_seen: List[int] = []
        ref_number = 1
        for idx in range(ref_start + 1, len(lines)):
            line = lines[idx].strip()
            if not line:
                continue
            ref_match_before = re.match(r"^\[(\d+)\]", line)
            if ref_match_before:
                reference_numbers_seen.append(int(ref_match_before.group(1)))
            if not re.match(r"^\[\d+\]", line):
                lines[idx] = f"[{ref_number}] {line}"
                add_change(
                    "Reference entry numbered",
                    "high",
                    line[:80],
                    lines[idx][:80],
                    "Each IEEE reference entry should start with a bracketed number.",
                    "IEEE citation issues",
                )
            else:
                current = re.match(r"^\[(\d+)\]", line)
                if current and int(current.group(1)) != ref_number:
                    new_line = re.sub(r"^\[\d+\]", f"[{ref_number}]", line)
                    add_change(
                        "Reference number order corrected",
                        "high",
                        line[:80],
                        new_line[:80],
                        "IEEE references should be numbered in citation order.",
                        "citation numbering issues",
                    )
                    lines[idx] = new_line
            author_order_match = re.match(r"^(\[\d+\]\s+)([A-Z][A-Za-z-]+),\s+([A-Z])\.\s*(.*)$", lines[idx])
            if author_order_match:
                original_line = lines[idx]
                lines[idx] = (
                    f"{author_order_match.group(1)}"
                    f"{author_order_match.group(3)}. {author_order_match.group(2)}, "
                    f"{author_order_match.group(4)}"
                )
                add_change(
                    "Reference author order converted to IEEE initials-before-surname format",
                    "high",
                    original_line[:80],
                    lines[idx][:80],
                    "A safe author-order transformation was applied for a single surname/initial pattern.",
                    "IEEE citation issues",
                )
            if "&" in lines[idx]:
                original_line = lines[idx]
                lines[idx] = lines[idx].replace(" & ", " and ")
                add_change(
                    "APA ampersand converted in reference entry",
                    "medium",
                    original_line[:80],
                    lines[idx][:80],
                    "IEEE references should not use APA-style ampersands between authors.",
                    "IEEE citation issues",
                )
            if re.match(r"^[A-Z][a-z]+,\s+[A-Z][a-z]+", re.sub(r"^\[\d+\]\s*", "", lines[idx])):
                add_change(
                    "Author name may not follow IEEE initials-before-surname format",
                    "medium",
                    lines[idx][:80],
                    lines[idx][:80],
                    "IEEE references usually use initials before surname, e.g., J. Smith.",
                    "IEEE citation issues",
                )
            if re.search(r"^\[\d+\]\s+[A-Z][A-Za-z-]+,\s+[A-Z]\.", lines[idx]):
                add_change(
                    "Reference author order is APA-like, not IEEE",
                    "high",
                    lines[idx][:80],
                    "Use initials before surname, e.g., [1] J. Smith, ...",
                    "IEEE uses initials before surname, e.g., J. Smith, not Smith, J.",
                    "IEEE citation issues",
                )
            if re.search(r"\bet\s+al\.?,", lines[idx], re.IGNORECASE):
                add_change(
                    "Reference entry uses incomplete 'et al.' author listing",
                    "high",
                    lines[idx][:80],
                    lines[idx][:80],
                    "IEEE references should provide the available author names in the required reference format, not a vague 'et al.' placeholder.",
                    "IEEE citation issues",
                )
            if re.search(r"\(\d{4}\)", lines[idx]):
                add_change(
                    "APA-style year placement detected in IEEE reference",
                    "high",
                    lines[idx][:80],
                    lines[idx][:80],
                    "IEEE references usually place the year near the end, not in APA-style parentheses after the author.",
                    "IEEE citation issues",
                )
            if re.search(r"^\[\d+\]\s+[A-Z][A-Za-z-]+,\s+[A-Z]\.\s*&", lines[idx]) or re.search(r"&\s+[A-Z][A-Za-z-]+,\s+[A-Z]\.", lines[idx]):
                add_change(
                    "APA-style multiple-author reference detected",
                    "high",
                    lines[idx][:80],
                    lines[idx][:80],
                    "IEEE references should use IEEE author formatting, not APA surname-initial plus ampersand formatting.",
                    "IEEE citation issues",
                )
            if re.search(r"&", lines[idx]):
                add_change(
                    "APA-style ampersand detected in reference authors",
                    "medium",
                    lines[idx][:80],
                    lines[idx][:80],
                    "IEEE references use commas and 'and' conventions rather than APA-style ampersands.",
                    "IEEE citation issues",
                )
            stripped_ref = re.sub(r"^\[\d+\]\s*", "", lines[idx]).strip()
            has_title_quotes = bool(re.search(r"[\"“”].+[\"“”]", stripped_ref))
            looks_like_article = bool(re.search(r"\b(journal|proceedings|conference|transactions|vol\.|no\.|pp\.)\b", stripped_ref, re.IGNORECASE))
            year_present = bool(re.search(r"\b(19|20)\d{2}\b", stripped_ref))
            has_publication_detail = bool(
                re.search(r"\b(journal|conference|proceedings|transactions|vol\.|no\.|pp\.|doi|press|publisher|university)\b", stripped_ref, re.IGNORECASE)
            )
            if not has_title_quotes and looks_like_article:
                add_change(
                    "Article title may need quotation marks",
                    "medium",
                    lines[idx][:80],
                    lines[idx][:80],
                    "IEEE article titles are commonly placed in quotation marks.",
                    "IEEE citation issues",
                )
            if len(stripped_ref.split()) < 7 or not year_present or not has_publication_detail:
                add_change(
                    "Reference entry appears incomplete for IEEE",
                    "high" if len(stripped_ref.split()) < 7 else "medium",
                    lines[idx][:80],
                    lines[idx][:80],
                    "IEEE references need enough metadata: authors, title, source/publication, year, and location details such as volume/pages/DOI/URL when applicable.",
                    "IEEE citation issues",
                )
            if re.match(r"^[A-Z][A-Za-z\s]+ Organization\.", stripped_ref) or re.match(r"^World Health Organization\.", stripped_ref):
                if not year_present or not re.search(r"https?://|Accessed:|accessed", stripped_ref):
                    add_change(
                        "Organization web reference is incomplete",
                        "medium",
                        lines[idx][:80],
                        lines[idx][:80],
                        "For an online organization source, include organization, page title, site name, date if available, URL, and accessed date.",
                        "IEEE citation issues",
                    )
            if re.search(r"https?://\S+\s*$", lines[idx]) and not re.search(r"Accessed:|accessed", lines[idx]):
                add_change(
                    "Raw URL reference needs full IEEE web reference details",
                    "medium",
                    lines[idx][:80],
                    lines[idx][:80],
                    "Include organization/author, page title, site name, date if available, URL, and accessed date.",
                    "IEEE citation issues",
                )
            ref_number += 1

        if body_citations:
            first_seen: List[int] = []
            for number in body_citations:
                if number not in first_seen:
                    first_seen.append(number)
            expected = list(range(1, len(first_seen) + 1))
            if first_seen != expected:
                add_change(
                    "Citation order inconsistency",
                    "high",
                    ", ".join(f"[{n}]" for n in first_seen),
                    "Manual verification required",
                    "IEEE citation numbers should follow first appearance order without skipping earlier numbers.",
                    "citation numbering issues",
                )
            skipped = [n for n in range(1, max(body_citations) + 1) if n not in body_citation_set]
            if skipped:
                add_change(
                    "Citation number skipped in body text",
                    "high",
                    ", ".join(f"[{n}]" for n in body_citations),
                    "Manual verification required",
                    f"The body cites up to [{max(body_citations)}] but never cites {', '.join(f'[{n}]' for n in skipped)}.",
                    "citation numbering issues",
                )

        reference_set = set(reference_numbers_seen or range(1, ref_number))
        uncited_refs = sorted(reference_set - body_citation_set)
        if uncited_refs and body_citations:
            add_change(
                "References not tied to in-text citations",
                "high",
                ", ".join(f"[{n}]" for n in uncited_refs),
                "Cite each listed reference or remove unused entries",
                "IEEE reference numbers should correspond to citations in the body.",
                "citation numbering issues",
            )
        fixed = "\n".join(lines)
    else:
        add_change(
            "Missing IEEE References section",
            "medium",
            "",
            "References",
            "A complete IEEE paper should end with a 'References' section.",
            "IEEE citation issues",
        )

    ieee_citations = [int(x) for x in re.findall(r"\[(\d+)\]", fixed)]
    if ieee_citations:
        first_seen: List[int] = []
        for number in ieee_citations:
            if number not in first_seen:
                first_seen.append(number)
        expected = list(range(1, len(first_seen) + 1))
        if first_seen != expected:
            add_change(
                "Citation order may be incorrect",
                "high",
                ", ".join(f"[{n}]" for n in first_seen),
                ", ".join(f"[{n}]" for n in expected),
                "IEEE citations should be numbered by first appearance.",
                "citation numbering issues",
            )

    unresolved_changes, transformation_count = split_applied_transformations(changes, fixed)
    cleaned_changes = clean_ieee_findings(unresolved_changes)
    high_count = sum(1 for item in cleaned_changes if item["severity"] == "high")
    medium_count = sum(1 for item in cleaned_changes if item["severity"] == "medium")
    summary = (
        f"IEEE formatter completed with {len(cleaned_changes)} grouped issue(s): "
        f"{high_count} high priority and {medium_count} medium priority. "
        f"{transformation_count} deterministic fix(es) were applied before reporting remaining issues. "
        "Raw rule hits were grouped and deduplicated into root problems."
    )
    return {
        "summary": summary,
        "fixed_text": fixed,
        "changes": cleaned_changes,
        "highlights": highlights,
        "transformations_applied": transformation_count > 0,
        "transformation_count": transformation_count,
        "source": "deterministic_ieee_formatter",
    }


def build_docx_from_text(text: str) -> BytesIO:
    doc = Document()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            doc.add_paragraph("")
            continue
        if stripped.lower() == "references":
            doc.add_heading("References", level=1)
        else:
            doc.add_paragraph(stripped)
    output = BytesIO()
    doc.save(output)
    output.seek(0)
    return output


def extract_docx_text(file_bytes: bytes) -> str:
    from io import BytesIO

    try:
        doc = Document(BytesIO(file_bytes))
        parts: List[str] = []
        for p in doc.paragraphs:
            t = p.text.strip()
            if t:
                parts.append(t)
        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
        full_text = "\n".join(parts).strip()
        if not full_text:
            raise ValueError("No readable text found in .docx file.")
        return full_text
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse .docx: {exc}") from exc


def fallback_analysis(text: str, reason: str, format_mode: str = "general") -> Dict[str, Any]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    word_count = len(re.findall(r"\b\w+\b", text))
    has_citation_like = bool(re.search(r"\([A-Za-z]+,\s*\d{4}\)|\[[0-9]+\]", text))
    causal_language = bool(
        re.search(
            r"\b(impact|influence|effect|affect|determine|directly affects|causes?|leads to)\b",
            text,
            re.IGNORECASE,
        )
    )
    gap_asserted = bool(re.search(r"\b(limited understanding|gap|despite|however|therefore)\b", text, re.IGNORECASE))

    data = {
        "summary": f"Fallback analysis used because LLM was unavailable ({reason}). "
        f"Text appears to have about {word_count} words and {len(paragraphs)} non-empty paragraphs.",
        "structure_format_issues": [
            {
                "issue": "Paragraph organization may be weak",
                "severity": "medium",
                "evidence": f"Detected {len(paragraphs)} non-empty paragraphs; manual review needed for section flow.",
                "recommendation": "Add clear introduction, body sections with topic sentences, and a conclusion.",
            }
        ],
        "academic_quality_issues": [
            {
                "issue": "Possible causal overclaiming" if causal_language else "Argument strength requires manual review",
                "severity": "high" if causal_language else "medium",
                "evidence": "The draft uses causal language such as impact, influence, affect, or determine."
                if causal_language
                else "Fallback mode cannot fully verify thesis-evidence alignment.",
                "recommendation": "If the study is not experimental or longitudinal, revise causal claims into association language such as 'relationship between' or 'is associated with.'"
                if causal_language
                else "Make sure each major claim is supported by evidence and explicit reasoning.",
            },
            {
                "issue": "Research gap may be asserted rather than demonstrated",
                "severity": "high" if gap_asserted else "medium",
                "evidence": "The draft appears to claim a gap using phrases like limited understanding, despite, however, or therefore."
                if gap_asserted
                else "No clear evidence summary was detected in fallback mode.",
                "recommendation": "Briefly summarize what prior studies found, identify the specific unresolved point, explain why it matters, and state how this study addresses it.",
            },
        ],
        "citation_consistency_issues": [
            {
                "issue": "Citation presence appears limited or inconsistent"
                if not has_citation_like
                else "Citation style consistency not verifiable in fallback mode",
                "severity": "high" if not has_citation_like else "medium",
                "evidence": "No obvious in-text citations detected." if not has_citation_like else "Some citation-like patterns detected.",
                "recommendation": "Apply one citation style consistently and verify references match in-text citations.",
            }
        ],
        "prioritized_suggestions": [
            {
                "priority": "high",
                "suggestion": "Align the claim strength with the likely research design.",
                "rationale": "Academic panels often challenge causal words when the method can only support correlation or association.",
                "expected_impact": "Reduces a major validity objection before defense or grading.",
            },
            {
                "priority": "high",
                "suggestion": "Make the research gap evidence-based.",
                "rationale": "A gap statement is weak if it is only asserted and not grounded in prior findings.",
                "expected_impact": "Creates a stronger justification for the study.",
            },
        ],
        "optional_rewrite_suggestions": [
            {
                "original_excerpt": "This essay talks about many things.",
                "rewritten_excerpt": "This study examines the relationship between social media use and academic performance among college students.",
                "reason": "Uses association language instead of unsupported causal language.",
            }
        ],
        "highlights": [],
        "reviewed_text": text,
        "source": "fallback",
        "fallback_used": True,
    }
    return merge_deterministic_findings(data, text, format_mode=format_mode)


def deterministic_findings(text: str, format_mode: str = "general") -> Dict[str, List[Dict[str, str]]]:
    academic_quality: List[Dict[str, str]] = []
    citation_issues: List[Dict[str, str]] = []
    suggestions: List[Dict[str, str]] = []
    rewrites: List[Dict[str, str]] = []
    highlights: List[Dict[str, str]] = []

    causal_match = re.search(
        r"\b(determine the impact|direct(?:ly)? affects?|impact|influence|effect|affects?|causes?)\b",
        text,
        re.IGNORECASE,
    )
    design_support = re.search(
        r"\b(experiment|experimental|random(?:ized)?|control group|longitudinal|causal design)\b",
        text,
        re.IGNORECASE,
    )
    if causal_match and not design_support:
        academic_quality.append(
            {
                "issue": "Causal claim is stronger than the apparent study design supports",
                "severity": "high",
                "evidence": f"The draft uses causal wording such as '{causal_match.group(0)}' without describing a causal research design.",
                "recommendation": "Use association language unless the methodology can establish causation.",
            }
        )
        suggestions.append(
            {
                "priority": "high",
                "suggestion": "Replace causal framing with relationship/correlation framing.",
                "rationale": "A panelist can challenge causal language if the study only observes social media use and grades.",
                "expected_impact": "Improves methodological accuracy and reduces a major defense risk.",
            }
        )
        rewrites.append(
            {
                "original_excerpt": causal_match.group(0),
                "rewritten_excerpt": "examine the relationship between social media usage and academic performance",
                "reason": "This wording avoids claiming causation before the research design supports it.",
            }
        )
        highlights.append(
            {
                "excerpt": causal_match.group(0),
                "message": "This causal wording may be too strong unless your method can prove cause and effect.",
                "severity": "high",
                "category": "causal claim",
            }
        )

    gap_match = re.search(r"\b(limited understanding|research gap|gap)\b", text, re.IGNORECASE)
    study_claims = re.findall(r"\b(recent studies|researchers argue|studies indicate|research suggests)\b", text, re.IGNORECASE)
    citation_like = re.findall(r"\([A-Za-z][A-Za-z\s&.-]*,\s*\d{4}\)|\[[0-9]+\]", text)
    if gap_match and len(study_claims) < 3:
        academic_quality.append(
            {
                "issue": "Research gap is asserted rather than demonstrated",
                "severity": "high",
                "evidence": f"The draft says '{gap_match.group(0)}' but does not clearly synthesize prior findings to prove the gap.",
                "recommendation": "Add a brief literature progression: what is known, what remains unresolved, why it matters, and how this study responds.",
            }
        )
        suggestions.append(
            {
                "priority": "high",
                "suggestion": "Rebuild the gap paragraph around specific prior findings.",
                "rationale": "A justified gap is stronger than a general claim that understanding is limited.",
                "expected_impact": "Makes the study rationale more defensible.",
            }
        )
        highlights.append(
            {
                "excerpt": gap_match.group(0),
                "message": "This gap is asserted; strengthen it by showing what prior studies have not resolved.",
                "severity": "high",
                "category": "research gap",
            }
        )

    if study_claims and not citation_like:
        first_claim = re.search(
            r"\b(recent studies have suggested|recent studies|some researchers argue|researchers argue|studies indicate|research suggests)\b",
            text,
            re.IGNORECASE,
        )
        citation_issues.append(
            {
                "issue": "Literature claims need citations",
                "severity": "high",
                "evidence": f"The draft refers to {', '.join(sorted(set(study_claims), key=str.lower))} but includes no visible in-text citations.",
                "recommendation": "Add citations for each claim about prior research and make sure every in-text citation appears in the reference list.",
            }
        )
        if first_claim:
            highlights.append(
                {
                    "excerpt": first_claim.group(0),
                    "message": "This claim refers to prior research but is not cited properly.",
                    "severity": "high",
                    "category": "citation",
                }
            )

    apa_citations = re.findall(r"\([A-Za-z][A-Za-z\s&.-]*,\s*\d{4}[a-z]?\)", text)
    ieee_citations = re.findall(r"\[[0-9]+(?:,\s*[0-9]+|-+[0-9]+)?\]", text)
    has_reference_heading = bool(re.search(r"(?im)^\s*(references|reference list|works cited)\s*$", text))

    if format_mode == "apa7":
        if ieee_citations and not apa_citations:
            citation_issues.append(
                {
                    "issue": "IEEE-style numeric citations do not match APA 7 format",
                    "severity": "high",
                    "evidence": f"Detected numeric citation marker '{ieee_citations[0]}'.",
                    "recommendation": "Use APA 7 author-date citations such as (Smith, 2023) and a References section.",
                }
            )
            highlights.append(
                {
                    "excerpt": ieee_citations[0],
                    "message": "APA 7 uses author-date citations, not numeric bracket citations.",
                    "severity": "high",
                    "category": "APA 7 format",
                }
            )
        if study_claims and not apa_citations:
            citation_issues.append(
                {
                    "issue": "APA 7 requires author-date citations for research claims",
                    "severity": "high",
                    "evidence": "The text discusses prior studies but no APA-style author-date citation is visible.",
                    "recommendation": "Add citations like (Author, Year) or Author (Year) for each research claim.",
                }
            )
        if not has_reference_heading:
            citation_issues.append(
                {
                    "issue": "APA 7 reference section is not visible",
                    "severity": "medium",
                    "evidence": "No clear References heading was detected.",
                    "recommendation": "Include a References section if this is a complete paper draft.",
                }
            )

    if format_mode == "ieee":
        if apa_citations and not ieee_citations:
            citation_issues.append(
                {
                    "issue": "APA-style author-date citations do not match IEEE format",
                    "severity": "high",
                    "evidence": f"Detected author-date citation '{apa_citations[0]}'.",
                    "recommendation": "Use IEEE numeric citations such as [1] and list references in numerical order.",
                }
            )
            highlights.append(
                {
                    "excerpt": apa_citations[0],
                    "message": "IEEE uses numbered bracket citations, not author-date citations.",
                    "severity": "high",
                    "category": "IEEE format",
                }
            )
        if study_claims and not ieee_citations:
            citation_issues.append(
                {
                    "issue": "IEEE requires numbered citations for research claims",
                    "severity": "high",
                    "evidence": "The text discusses prior studies but no IEEE-style numeric citation is visible.",
                    "recommendation": "Add citations like [1] near each research claim and order references numerically.",
                }
            )
        if not has_reference_heading:
            citation_issues.append(
                {
                    "issue": "IEEE references section is not visible",
                    "severity": "medium",
                    "evidence": "No clear References heading was detected.",
                    "recommendation": "Include a References section with entries numbered to match in-text citations.",
                }
            )

    return {
        "academic_quality_issues": academic_quality,
        "citation_consistency_issues": citation_issues,
        "prioritized_suggestions": suggestions,
        "optional_rewrite_suggestions": rewrites,
        "highlights": highlights,
    }


def merge_deterministic_findings(data: Dict[str, Any], text: str, format_mode: str = "general") -> Dict[str, Any]:
    findings = deterministic_findings(text, format_mode=format_mode)
    for field, items in findings.items():
        existing = data.get(field, [])
        if not isinstance(existing, list):
            existing = []
        existing_titles = {
            str(item.get("issue") or item.get("suggestion") or item.get("reason") or "").lower()
            for item in existing
            if isinstance(item, dict)
        }
        merged = []
        for item in items:
            key = str(item.get("issue") or item.get("suggestion") or item.get("reason") or "").lower()
            if key and key not in existing_titles:
                merged.append(item)
        data[field] = merged + existing
    return data


def coerce_analysis_shape(data: Dict[str, Any], text: str, format_mode: str = "general") -> Dict[str, Any]:
    required_list_fields = [
        "structure_format_issues",
        "academic_quality_issues",
        "citation_consistency_issues",
        "prioritized_suggestions",
        "optional_rewrite_suggestions",
        "highlights",
    ]
    if "summary" not in data or not isinstance(data["summary"], str):
        data["summary"] = "Automated analysis completed."
    for field in required_list_fields:
        if field not in data or not isinstance(data[field], list):
            data[field] = []
    data["source"] = "ollama"
    data["fallback_used"] = False
    data["reviewed_text"] = text

    if not data["prioritized_suggestions"]:
        data["prioritized_suggestions"] = fallback_analysis(text, "empty suggestions", format_mode=format_mode)["prioritized_suggestions"]
    return merge_deterministic_findings(data, text, format_mode=format_mode)


def build_analysis_prompt(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    format_mode: str = "general",
) -> str:
    style_label = {"general": "General academic review", "apa7": "APA 7", "ieee": "IEEE"}[format_mode]
    return f"""
You are a strict academic panel reviewer for student research papers.
Selected review format: {style_label}.
Return ONLY valid JSON, no markdown.
JSON keys required:
- summary (string)
- structure_format_issues (array of objects: issue, severity, evidence, recommendation)
- academic_quality_issues (array of objects: issue, severity, evidence, recommendation)
- citation_consistency_issues (array of objects: issue, severity, evidence, recommendation)
- prioritized_suggestions (array of objects: priority, suggestion, rationale, expected_impact)
- optional_rewrite_suggestions (array of objects: original_excerpt, rewritten_excerpt, reason)
- highlights (array of objects: excerpt, message, severity, category)

Constraints:
- Focus on high-value critique a thesis/research panelist would actually raise.
- Do not invent weak issues just to fill categories.
- Prefer fewer, stronger issues over many generic comments.
- Every issue must cite concrete evidence from the draft.
- If a category has no meaningful issue, return an empty array for that category.
- Severity must reflect academic risk: high, medium, or low.
- Keep summary under 100 words.

Reviewer priorities:
1. Causal overclaiming:
   - Flag words such as "impact", "influence", "effect", "affects", "directly affects", "determine", or "causes" when the text does not establish an experimental, longitudinal, or otherwise causal design.
   - Recommend association/correlation wording when causality is unsupported.
2. Research gap quality:
   - Check whether the gap is demonstrated or merely asserted.
   - A weak gap says "limited understanding" without summarizing prior findings, identifying the specific unresolved point, explaining why it matters, and stating how this study addresses it.
3. Literature support:
   - Flag missing citations where the introduction makes claims about "recent studies", "researchers argue", or broad scholarly findings.
   - Do not treat missing citations as the deepest issue if a stronger conceptual issue exists.
4. Format mode:
   - If selected format is APA 7, check for author-date in-text citations, References heading, citation/reference consistency, and APA-style academic formatting risks.
   - If selected format is IEEE, check for numbered bracket citations like [1], numerical reference ordering, References heading, and citation/reference consistency.
   - If selected format is General academic review, do not enforce APA 7 or IEEE-specific rules.
4. Scope and specificity:
   - Flag vague scope only when it affects research design clarity, such as unclear population, variables, grade measure, or social media usage measure.
5. Avoid low-value filler:
   - Do not criticize ordinary words like "distraction" unless the term is central to the research variables and needs operational definition.
   - Do not claim terminology inconsistency unless genuinely different concepts are being mixed.
   - Do not penalize repeated words unless repetition creates real ambiguity.

Output expectations:
- academic_quality_issues should include the strongest conceptual problems first.
- citation_consistency_issues should focus on missing source support and citation/reference matching.
- prioritized_suggestions should rank the most important fixes, not restate every minor issue.
- optional_rewrite_suggestions should rewrite only the riskiest sentence(s), especially causal claims or weak gap statements.
- highlights should contain exact short excerpts copied from the text that deserve inline marking.
- highlight messages should be concise hover text, for example: "This claim needs a citation."
- Only highlight meaningful problems; do not highlight harmless wording.

Context metadata:
{json.dumps(metadata or {}, ensure_ascii=True)}

Text to review:
{text}
"""


def parse_ollama_analysis(raw: str, text: str, format_mode: str = "general") -> Dict[str, Any]:
    if not raw:
        raise ValueError("Empty response from Ollama.")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Ollama output is not a JSON object.")
    return coerce_analysis_shape(data, text, format_mode=format_mode)


def call_ollama_for_analysis(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    model_override: Optional[str] = None,
    format_mode: str = "general",
) -> Dict[str, Any]:
    model = get_ollama_model(model_override)
    format_mode = normalize_format_mode(format_mode)
    prompt = build_analysis_prompt(text=text, metadata=metadata, format_mode=format_mode)

    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.2},
    }
    url = f"{get_ollama_url()}/api/generate"

    resp = requests.post(url, json=body, timeout=90)
    if resp.status_code >= 400:
        detail = resp.text.strip()
        if resp.status_code == 404:
            raise RuntimeError(
                f"Ollama returned 404. The model '{model}' may not be installed, "
                f"or OLLAMA_URL may not point to Ollama. Response: {detail}"
            )
        raise RuntimeError(f"Ollama request failed with HTTP {resp.status_code}: {detail}")
    payload = resp.json()
    raw = payload.get("response", "").strip()
    return parse_ollama_analysis(raw=raw, text=text, format_mode=format_mode)


def analyze_text_with_resilience(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    model_override: Optional[str] = None,
    format_mode: str = "general",
) -> Dict[str, Any]:
    try:
        return call_ollama_for_analysis(
            text=text,
            metadata=metadata,
            model_override=model_override,
            format_mode=format_mode,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Ollama analysis failed")
        return fallback_analysis(text=text, reason=str(exc), format_mode=format_mode)


def json_line(event: str, payload: Dict[str, Any]) -> str:
    return json.dumps({"event": event, **payload}, ensure_ascii=True) + "\n"


def analysis_event_stream(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    model_override: Optional[str] = None,
    format_mode: str = "general",
) -> Generator[str, None, None]:
    try:
        model = get_ollama_model(model_override)
        format_mode = normalize_format_mode(format_mode)
        yield json_line(
            "status",
            {
                "step": "Validating input",
                "detail": "Checking draft length and selected model.",
            },
        )
        if len(text.strip()) < 20:
            yield json_line("error", {"detail": "Text too short for meaningful analysis."})
            return

        yield json_line(
            "status",
            {
                "step": "Preparing prompt",
                "detail": f"Building the {format_mode.upper()} review request for {model}.",
            },
        )
        yield json_line(
            "status",
            {
                "step": "Contacting Ollama",
                "detail": f"Sending the draft to {model}.",
            },
        )
        body = {
            "model": model,
            "prompt": build_analysis_prompt(text=text, metadata=metadata, format_mode=format_mode),
            "stream": True,
            "format": "json",
            "options": {"temperature": 0.2},
        }
        raw_parts: List[str] = []
        chunk_count = 0
        with requests.post(f"{get_ollama_url()}/api/generate", json=body, timeout=90, stream=True) as resp:
            if resp.status_code >= 400:
                detail = resp.text.strip()
                if resp.status_code == 404:
                    raise RuntimeError(
                        f"Ollama returned 404. The model '{model}' may not be installed, "
                        f"or OLLAMA_URL may not point to Ollama. Response: {detail}"
                    )
                raise RuntimeError(f"Ollama request failed with HTTP {resp.status_code}: {detail}")

            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                payload = json.loads(line)
                piece = payload.get("response", "")
                if piece:
                    raw_parts.append(piece)
                    chunk_count += 1
                    if chunk_count == 1 or chunk_count % 12 == 0:
                        yield json_line(
                            "status",
                            {
                                "step": "Receiving model output",
                                "detail": f"Ollama has streamed {chunk_count} response chunks.",
                            },
                        )
                if payload.get("done"):
                    break

        yield json_line(
            "status",
            {
                "step": "Parsing model output",
                "detail": "Normalizing the structured feedback response.",
            },
        )
        analysis = parse_ollama_analysis(raw="".join(raw_parts).strip(), text=text, format_mode=format_mode)
        yield json_line("analysis", {"analysis": analysis})
    except HTTPException as exc:
        yield json_line("error", {"detail": exc.detail, "error_type": exc.__class__.__name__})
    except Exception as exc:
        logger.exception("Ollama streaming analysis failed")
        yield json_line(
            "status",
            {
                "step": "Using fallback",
                "detail": "Ollama failed, so the backend is generating a safe fallback review.",
            },
        )
        yield json_line(
            "analysis",
            {
                "analysis": fallback_analysis(
                    text=text,
                    reason=f"{exc.__class__.__name__}: {exc}",
                    format_mode=format_mode,
                )
            },
        )


@app.get("/health")
def health() -> Dict[str, Any]:
    ollama_ok = True
    ollama_error = None
    try:
        resp = requests.get(f"{get_ollama_url()}/api/tags", timeout=5)
        resp.raise_for_status()
    except Exception as exc:
        ollama_ok = False
        ollama_error = str(exc)
    return {
        "status": "ok",
        "service": "academic-review-mvp",
        "ollama_model": get_ollama_model(),
        "ollama_url": get_ollama_url(),
        "ollama_reachable": ollama_ok,
        "ollama_error": ollama_error,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    state = chat_sessions.setdefault(
        req.session_id,
        {"phase": "intake", "answers": {}, "question_index": 0, "history": []},
    )
    message = req.message.strip()
    state["history"].append({"role": "user", "message": message})

    likely_draft = len(message.split()) >= 60 or len(message) >= 350
    if likely_draft:
        metadata = {
            "source_type": "chat_message",
            "intake_answers": state.get("answers", {}),
            "instruction": "Analyze this academic draft directly.",
            "format_mode": req.format_mode,
        }
        analysis = analyze_text_with_resilience(
            text=message,
            metadata=metadata,
            model_override=req.model,
            format_mode=req.format_mode or "general",
        )
        state["phase"] = "analyzed"
        return ChatResponse(
            session_id=req.session_id,
            phase="analyzed",
            questions=[],
            next_prompt="I analyzed the draft and organized the feedback into the review panel.",
            context={"intake_answers": state.get("answers", {})},
            analysis=analysis,
        )

    direct_review_phrases = [
        "analyze",
        "review",
        "feedback",
        "thoughts",
        "check this",
        "critique",
    ]
    if any(phrase in message.lower() for phrase in direct_review_phrases):
        return ChatResponse(
            session_id=req.session_id,
            phase="ready_for_text",
            questions=[],
            next_prompt="Paste the academic text you want reviewed, and I will analyze it directly.",
            context={"intake_answers": state.get("answers", {})},
        )

    if state["phase"] == "intake":
        state["phase"] = "ready_for_text"
        return ChatResponse(
            session_id=req.session_id,
            phase="ready_for_text",
            questions=[],
            next_prompt="Paste your academic text here and I will return structured feedback. You can also use the DOCX upload panel.",
            context={"intake_answers": state["answers"]},
        )

    return ChatResponse(
        session_id=req.session_id,
        phase=state["phase"],
        questions=[],
        next_prompt="Session active. Submit text to /analyze_text or .docx to /analyze_docx.",
        context={"intake_answers": state["answers"]},
    )


@app.post("/analyze_text", response_model=AnalysisResponse)
def analyze_text(req: AnalyzeTextRequest) -> AnalysisResponse:
    text = req.text.strip()
    if len(text) < 20:
        raise HTTPException(status_code=422, detail="Text too short for meaningful analysis.")
    data = analyze_text_with_resilience(
        text=text,
        metadata=req.metadata,
        model_override=req.model,
        format_mode=req.format_mode or "general",
    )
    return AnalysisResponse(**data)


@app.post("/analyze_text_stream")
def analyze_text_stream(req: AnalyzeTextStreamRequest) -> StreamingResponse:
    return StreamingResponse(
        analysis_event_stream(
            text=req.text.strip(),
            metadata=req.metadata,
            model_override=req.model,
            format_mode=req.format_mode or "general",
        ),
        media_type="application/x-ndjson",
    )


@app.post("/analyze_docx", response_model=AnalysisResponse)
async def analyze_docx(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    format_mode: str = Form("general"),
) -> AnalysisResponse:
    if not file.filename.lower().endswith(".docx"):
        raise HTTPException(status_code=400, detail="Only .docx files are supported.")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    text = extract_docx_text(content)
    metadata = {"filename": file.filename, "source_type": "docx_upload"}
    data = analyze_text_with_resilience(
        text=text,
        metadata=metadata,
        model_override=model,
        format_mode=format_mode,
    )
    return AnalysisResponse(**data)


@app.post("/analyze_docx_stream")
async def analyze_docx_stream(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    format_mode: str = Form("general"),
) -> StreamingResponse:
    if not file.filename.lower().endswith(".docx"):
        raise HTTPException(status_code=400, detail="Only .docx files are supported.")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    def stream() -> Generator[str, None, None]:
        yield json_line(
            "status",
            {
                "step": "Parsing DOCX",
                "detail": f"Extracting readable text from {file.filename}.",
            },
        )
        try:
            text = extract_docx_text(content)
        except HTTPException as exc:
            yield json_line("error", {"detail": exc.detail, "error_type": exc.__class__.__name__})
            return
        metadata = {"filename": file.filename, "source_type": "docx_upload"}
        yield from analysis_event_stream(
            text=text,
            metadata=metadata,
            model_override=model,
            format_mode=format_mode,
        )

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/format_ieee_text", response_model=FormatResponse)
def format_ieee_text(req: FormatTextRequest) -> FormatResponse:
    return FormatResponse(**ieee_format_text(req.text.strip()))


@app.post("/format_ieee_docx", response_model=FormatResponse)
async def format_ieee_docx(file: UploadFile = File(...)) -> FormatResponse:
    if not file.filename.lower().endswith(".docx"):
        raise HTTPException(status_code=400, detail="Only .docx files are supported.")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    text = extract_docx_text(content)
    return FormatResponse(**ieee_format_text(text))


@app.post("/export_ieee_docx")
def export_ieee_docx(req: ExportDocxRequest) -> StreamingResponse:
    filename = req.filename if req.filename.lower().endswith(".docx") else f"{req.filename}.docx"
    safe_filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", filename)
    output = build_docx_from_text(req.text)
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{safe_filename}"'},
    )


@app.post("/format_ieee_docx_export")
async def format_ieee_docx_export(file: UploadFile = File(...)) -> StreamingResponse:
    if not file.filename.lower().endswith(".docx"):
        raise HTTPException(status_code=400, detail="Only .docx files are supported.")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    text = extract_docx_text(content)
    formatted = ieee_format_text(text)["fixed_text"]
    base = re.sub(r"\.docx$", "", file.filename, flags=re.IGNORECASE)
    safe_filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{base}_ieee_formatted.docx")
    output = build_docx_from_text(formatted)
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{safe_filename}"'},
    )


# Provider-aware overrides. These are intentionally defined after the routes because
# route handlers resolve these globals at request time.
def call_ollama_for_analysis(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    model_override: Optional[str] = None,
    format_mode: str = "general",
) -> Dict[str, Any]:
    model = get_ollama_model(model_override)
    format_mode = normalize_format_mode(format_mode)
    prompt = build_analysis_prompt(text=text, metadata=metadata, format_mode=format_mode)

    if is_groq_model(model):
        body = {
            "model": groq_model_id(model),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "max_tokens": 2500,
        }
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {get_groq_api_key()}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=90,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"Groq request failed with HTTP {resp.status_code}: {resp.text.strip()}")
        payload = resp.json()
        raw = payload["choices"][0]["message"]["content"].strip()
        return parse_ollama_analysis(raw=raw, text=text, format_mode=format_mode)

    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.2},
    }
    resp = requests.post(f"{get_ollama_url()}/api/generate", json=body, timeout=90)
    if resp.status_code >= 400:
        detail = resp.text.strip()
        if resp.status_code == 404:
            raise RuntimeError(
                f"Ollama returned 404. The model '{model}' may not be installed, "
                f"or OLLAMA_URL may not point to Ollama. Response: {detail}"
            )
        raise RuntimeError(f"Ollama request failed with HTTP {resp.status_code}: {detail}")
    payload = resp.json()
    raw = payload.get("response", "").strip()
    return parse_ollama_analysis(raw=raw, text=text, format_mode=format_mode)


def analyze_text_with_resilience(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    model_override: Optional[str] = None,
    format_mode: str = "general",
) -> Dict[str, Any]:
    try:
        return call_ollama_for_analysis(
            text=text,
            metadata=metadata,
            model_override=model_override,
            format_mode=format_mode,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Hosted/local model analysis failed")
        return fallback_analysis(text=text, reason=str(exc), format_mode=format_mode)


def analysis_event_stream(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    model_override: Optional[str] = None,
    format_mode: str = "general",
) -> Generator[str, None, None]:
    try:
        model = get_ollama_model(model_override)
        format_mode = normalize_format_mode(format_mode)
        yield json_line("status", {"step": "Validating input", "detail": "Checking draft length and selected model."})
        if len(text.strip()) < 20:
            yield json_line("error", {"detail": "Text too short for meaningful analysis."})
            return

        provider = "Groq" if is_groq_model(model) else "Ollama"
        yield json_line(
            "status",
            {
                "step": "Preparing prompt",
                "detail": f"Building the {format_mode.upper()} review request for {provider}.",
            },
        )
        yield json_line("status", {"step": f"Contacting {provider}", "detail": f"Sending the draft to {model}."})

        raw_parts: List[str] = []
        chunk_count = 0

        if is_groq_model(model):
            body = {
                "model": groq_model_id(model),
                "messages": [{"role": "user", "content": build_analysis_prompt(text=text, metadata=metadata, format_mode=format_mode)}],
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
                "max_tokens": 2500,
                "stream": True,
            }
            with requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {get_groq_api_key()}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=90,
                stream=True,
            ) as resp:
                if resp.status_code >= 400:
                    raise RuntimeError(f"Groq request failed with HTTP {resp.status_code}: {resp.text.strip()}")
                for line in resp.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    data = line.removeprefix("data: ").strip()
                    if data == "[DONE]":
                        break
                    payload = json.loads(data)
                    piece = payload.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if piece:
                        raw_parts.append(piece)
                        chunk_count += 1
                        if chunk_count == 1 or chunk_count % 12 == 0:
                            yield json_line(
                                "status",
                                {
                                    "step": "Receiving hosted model output",
                                    "detail": f"Groq has streamed {chunk_count} response chunks.",
                                },
                            )
        else:
            body = {
                "model": model,
                "prompt": build_analysis_prompt(text=text, metadata=metadata, format_mode=format_mode),
                "stream": True,
                "format": "json",
                "options": {"temperature": 0.2},
            }
            with requests.post(f"{get_ollama_url()}/api/generate", json=body, timeout=90, stream=True) as resp:
                if resp.status_code >= 400:
                    detail = resp.text.strip()
                    if resp.status_code == 404:
                        raise RuntimeError(
                            f"Ollama returned 404. The model '{model}' may not be installed, "
                            f"or OLLAMA_URL may not point to Ollama. Response: {detail}"
                        )
                    raise RuntimeError(f"Ollama request failed with HTTP {resp.status_code}: {detail}")

                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    payload = json.loads(line)
                    piece = payload.get("response", "")
                    if piece:
                        raw_parts.append(piece)
                        chunk_count += 1
                        if chunk_count == 1 or chunk_count % 12 == 0:
                            yield json_line(
                                "status",
                                {
                                    "step": "Receiving local model output",
                                    "detail": f"Ollama has streamed {chunk_count} response chunks.",
                                },
                            )
                    if payload.get("done"):
                        break

        yield json_line("status", {"step": "Parsing model output", "detail": "Normalizing the structured feedback response."})
        analysis = parse_ollama_analysis(raw="".join(raw_parts).strip(), text=text, format_mode=format_mode)
        yield json_line("analysis", {"analysis": analysis})
    except HTTPException as exc:
        yield json_line("error", {"detail": exc.detail, "error_type": exc.__class__.__name__})
    except Exception as exc:
        logger.exception("Streaming analysis failed")
        yield json_line(
            "status",
            {
                "step": "Using fallback",
                "detail": "The selected model failed, so the backend is generating a safe fallback review.",
            },
        )
        yield json_line(
            "analysis",
            {
                "analysis": fallback_analysis(
                    text=text,
                    reason=f"{exc.__class__.__name__}: {exc}",
                    format_mode=format_mode,
                )
            },
        )
