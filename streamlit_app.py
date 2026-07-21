"""SourceGround — source-grounded PDF question answering for Streamlit."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import faiss
import fitz
import ftfy
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from fastembed import TextEmbedding
from llama_cpp import Llama
from PIL import Image


# -----------------------------------------------------------------------------
# App and model configuration
# -----------------------------------------------------------------------------

APP_NAME = "SourceGround"
EMBEDDING_MODEL_ID = "BAAI/bge-small-en-v1.5"
LLM_REPO_ID = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
LLM_FILENAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
OCR_MIN_CHARACTERS = 45
MIN_SIMILARITY = 0.30
DEFAULT_TOP_K = 4
MAX_CONTEXT_CHARACTERS = 10_000

SOURCE_TABLE_COLUMNS = [
    "rank",
    "source",
    "document_type",
    "pages",
    "similarity",
    "chunk_id",
]

DEFAULT_SUGGESTED_QUERIES = [
    "What is the main purpose of this document?",
    "What are the most important requirements or findings?",
    "What dates, deadlines, or milestones are mentioned?",
    "Which organizations, people, or programs are identified?",
]


st.set_page_config(
    page_title=f"{APP_NAME} | Document Intelligence",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Uses Streamlit's own theme variables, so the interface stays coherent when
# the visitor selects light, dark, or system theme in Streamlit settings.
APP_CSS = """
<style>
    :root {
        color-scheme: light dark;
        --primary-color: #17A9DC !important;
        --primary-color-rgb: 23, 169, 220 !important;
    }

    .stApp {
        background:
            radial-gradient(circle at 78% 0%, color-mix(in srgb, var(--primary-color) 9%, transparent), transparent 31rem),
            var(--background-color);
    }

    [data-testid="stHeader"] {
        background: color-mix(in srgb, var(--background-color) 84%, transparent);
        backdrop-filter: blur(12px);
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid color-mix(in srgb, var(--text-color) 12%, transparent);
    }

    .block-container {
        max-width: 1480px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }

    .sg-header {
        position: relative;
        overflow: hidden;
        padding: 1.25rem 1.4rem;
        margin-bottom: 1rem;
        border: 1px solid color-mix(in srgb, var(--primary-color) 32%, transparent);
        border-radius: 18px;
        background:
            linear-gradient(115deg,
                color-mix(in srgb, var(--primary-color) 15%, var(--secondary-background-color)),
                var(--secondary-background-color) 62%);
        box-shadow: 0 14px 42px color-mix(in srgb, var(--text-color) 8%, transparent);
    }

    .sg-header::after {
        content: "";
        position: absolute;
        width: 230px;
        height: 230px;
        right: -105px;
        top: -128px;
        border-radius: 50%;
        border: 1px solid color-mix(in srgb, var(--primary-color) 45%, transparent);
        box-shadow: 0 0 70px color-mix(in srgb, var(--primary-color) 20%, transparent);
    }

    .sg-brand-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        flex-wrap: wrap;
    }

    .sg-brand {
        display: flex;
        align-items: center;
        gap: .9rem;
    }

    .sg-logo {
        display: grid;
        place-items: center;
        width: 44px;
        height: 44px;
        border-radius: 13px;
        font-weight: 800;
        color: white;
        background: linear-gradient(145deg, #23c6e8, #1678ef);
        box-shadow: 0 8px 24px rgba(22, 120, 239, .28);
    }

    .sg-title {
        margin: 0;
        font-size: clamp(1.25rem, 2vw, 1.65rem);
        line-height: 1.1;
        letter-spacing: -.025em;
    }

    .sg-subtitle {
        margin-top: .28rem;
        color: color-mix(in srgb, var(--text-color) 67%, transparent);
        font-size: .88rem;
    }

    .sg-badges {
        display: flex;
        gap: .45rem;
        flex-wrap: wrap;
    }

    .sg-badge {
        padding: .34rem .58rem;
        border-radius: 999px;
        border: 1px solid color-mix(in srgb, var(--text-color) 13%, transparent);
        background: color-mix(in srgb, var(--background-color) 50%, transparent);
        color: color-mix(in srgb, var(--text-color) 78%, transparent);
        font-size: .68rem;
        font-weight: 750;
        letter-spacing: .045em;
        text-transform: uppercase;
    }

    .sg-online {
        border-color: color-mix(in srgb, #21c9a4 42%, transparent);
        color: #16a985;
    }

    .sg-pipeline {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: .55rem;
        margin-bottom: 1.15rem;
    }

    .sg-stage {
        min-height: 67px;
        padding: .66rem .72rem;
        border-radius: 12px;
        border: 1px solid color-mix(in srgb, var(--text-color) 11%, transparent);
        background: var(--secondary-background-color);
    }

    .sg-stage-number {
        color: var(--primary-color);
        font-size: .67rem;
        font-weight: 800;
        letter-spacing: .08em;
    }

    .sg-stage-name {
        margin-top: .2rem;
        font-size: .82rem;
        font-weight: 750;
    }

    .sg-stage-detail {
        margin-top: .13rem;
        color: color-mix(in srgb, var(--text-color) 57%, transparent);
        font-size: .67rem;
    }

    .sg-section-kicker {
        margin-bottom: .2rem;
        color: var(--primary-color);
        font-size: .67rem;
        font-weight: 800;
        letter-spacing: .11em;
        text-transform: uppercase;
    }

    .sg-ready, .sg-waiting, .sg-warning {
        margin: .7rem 0 1rem;
        padding: .7rem .78rem;
        border-radius: 11px;
        font-size: .83rem;
        border: 1px solid;
    }

    .sg-ready {
        color: color-mix(in srgb, #12a982 75%, var(--text-color));
        border-color: color-mix(in srgb, #12a982 34%, transparent);
        background: color-mix(in srgb, #12a982 10%, transparent);
    }

    .sg-waiting {
        color: color-mix(in srgb, var(--text-color) 66%, transparent);
        border-color: color-mix(in srgb, var(--text-color) 12%, transparent);
        background: var(--secondary-background-color);
    }

    .sg-warning {
        color: color-mix(in srgb, #ef9f22 78%, var(--text-color));
        border-color: color-mix(in srgb, #ef9f22 34%, transparent);
        background: color-mix(in srgb, #ef9f22 9%, transparent);
    }

    [data-testid="stMetric"] {
        padding: .82rem .9rem;
        border-radius: 13px;
        border: 1px solid color-mix(in srgb, var(--text-color) 10%, transparent);
        background: var(--secondary-background-color);
    }

    [data-testid="stChatMessage"] {
        border: 1px solid color-mix(in srgb, var(--text-color) 10%, transparent);
        border-radius: 14px;
        background: color-mix(in srgb, var(--secondary-background-color) 86%, transparent);
        padding: .25rem .4rem;
        margin-bottom: .65rem;
    }

    [data-testid="stFileUploaderDropzone"] {
        border-color: color-mix(in srgb, var(--primary-color) 38%, transparent);
        background: color-mix(in srgb, var(--primary-color) 5%, var(--secondary-background-color));
    }

    .stButton > button, .stDownloadButton > button {
        border-radius: 10px;
        font-weight: 700;
    }

    .sg-footnote {
        margin-top: .6rem;
        color: color-mix(in srgb, var(--text-color) 57%, transparent);
        font-size: .75rem;
        line-height: 1.45;
    }

    @media (max-width: 900px) {
        .sg-pipeline { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }

    @media (max-width: 560px) {
        .sg-pipeline { grid-template-columns: 1fr; }
        .sg-badges { display: none; }
    }
</style>
"""

st.markdown(APP_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Document metadata and routing
# -----------------------------------------------------------------------------


@dataclass
class ChunkMetadata:
    text: str
    doc_type: str
    page_start: int
    page_end: int
    source_id: str
    source_name: str
    document_id: str
    chunk_id: str
    ocr_used: bool = False


DOCUMENT_TYPE_RULES: Dict[str, List[str]] = {
    "Certificate of Quality": [
        "certificate of quality",
        "certificate of analysis",
        "lot number",
        "batch number",
        "test result",
        "quality control",
        "conforms",
        "release date",
    ],
    "Packaging Specification": [
        "packaging specification",
        "packaging component",
        "container closure",
        "stopper",
        "vial",
        "bottle",
        "carton",
        "seal",
        "label specification",
    ],
    "BSE/TSE Declaration": [
        "bse/tse",
        "bovine spongiform",
        "transmissible spongiform",
        "animal-derived material",
    ],
    "Safety Data Sheet": [
        "safety data sheet",
        "hazard statement",
        "first aid measures",
        "first-aid measures",
        "handling and storage",
        "cas number",
        "personal protective equipment",
    ],
    "Product Specification": [
        "product specification",
        "acceptance criteria",
        "specification limit",
        "assay",
        "purity",
        "appearance",
    ],
    "Cover Letter": [
        "dear sir",
        "dear madam",
        "to whom it may concern",
        "please find enclosed",
        "sincerely",
        "subject:",
    ],
    "Supplier Qualification Record": [
        "supplier qualification record",
        "supplier name",
        "approved supplier",
        "quality agreement",
        "performance metrics",
        "on-time delivery",
        "incoming quality",
        "capa response time",
        "audit rights",
    ],
    "Chain of Custody": [
        "global chain of custody",
        "chain of custody",
        "list of assemblies",
        "traceability",
        "raw materials received",
        "in-process checks",
        "distribution center",
        "distributed to end customer",
        "supply chain quality",
    ],
}


QUERY_ROUTE_RULES: Dict[str, List[str]] = {
    "Certificate of Quality": [
        "lot",
        "batch",
        "test result",
        "release",
        "quality certificate",
        "conform",
        "quality requirement",
    ],
    "Packaging Specification": [
        "packaging",
        "container",
        "closure",
        "stopper",
        "vial",
        "carton",
        "seal",
        "label",
    ],
    "BSE/TSE Declaration": ["bse", "tse", "animal-derived", "spongiform"],
    "Safety Data Sheet": [
        "hazard",
        "first aid",
        "spill",
        "ppe",
        "safety data",
        "cas number",
    ],
    "Product Specification": [
        "product specification",
        "acceptance criteria",
        "assay",
        "purity",
        "appearance",
    ],
    "Supplier Qualification Record": [
        "supplier",
        "quality agreement",
        "on-time delivery",
        "incoming quality",
        "capa response",
        "audit",
        "approved supplier",
    ],
    "Chain of Custody": [
        "chain of custody",
        "traceability",
        "assembly",
        "raw material",
        "distribution center",
        "manufacturing facility",
        "eysins",
    ],
}


def clean_text(text: str) -> str:
    """Normalize extracted or OCR text without discarding useful punctuation."""

    text = ftfy.fix_text(text or "")
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", " ")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def classify_document_type(text: str, filename: str = "") -> str:
    """Classify a PDF page from its content and filename."""

    normalized_filename = re.sub(r"[_\-]+", " ", filename.lower())
    searchable_text = f"{normalized_filename} {text[:8000].lower()}"
    scores: Dict[str, int] = {}

    for doc_type, keywords in DOCUMENT_TYPE_RULES.items():
        score = 0
        for keyword in keywords:
            if keyword in searchable_text:
                score += 1
            if keyword in normalized_filename:
                score += 2
        scores[doc_type] = score

    best_type = max(scores, key=scores.get)
    return best_type if scores[best_type] else "Other"


def infer_query_doc_type(query: str) -> Optional[str]:
    """Infer a metadata route only when the query has clear type-specific intent."""

    normalized_query = query.lower()
    scores = {
        doc_type: sum(keyword in normalized_query for keyword in keywords)
        for doc_type, keywords in QUERY_ROUTE_RULES.items()
    }
    best_type = max(scores, key=scores.get)
    return best_type if scores[best_type] else None


# -----------------------------------------------------------------------------
# PDF extraction, OCR, chunking, and metadata
# -----------------------------------------------------------------------------


class DocumentProcessor:
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        ocr_min_characters: int = OCR_MIN_CHARACTERS,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ocr_min_characters = ocr_min_characters

    @staticmethod
    def render_page_for_ocr(page: fitz.Page, dpi: int = 220) -> Image.Image:
        pixmap = page.get_pixmap(dpi=dpi, alpha=False)
        return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)

    def extract_pages(self, file_path: str) -> List[Dict[str, Any]]:
        extracted_pages: List[Dict[str, Any]] = []

        with fitz.open(file_path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                native_text = clean_text(page.get_text("text"))
                meaningful_characters = len(re.sub(r"\s", "", native_text))
                use_ocr = meaningful_characters < self.ocr_min_characters

                if use_ocr:
                    page_image = self.render_page_for_ocr(page)
                    extracted_text = pytesseract.image_to_string(
                        page_image,
                        config="--psm 6",
                    )
                    final_text = clean_text(extracted_text)
                else:
                    final_text = native_text

                extracted_pages.append(
                    {
                        "page": page_number,
                        "text": final_text,
                        "ocr_used": use_ocr,
                    }
                )

        return extracted_pages

    def split_page(self, text: str) -> List[str]:
        text = clean_text(text)
        if not text:
            return []

        chunks: List[str] = []
        start = 0

        while start < len(text):
            hard_end = min(start + self.chunk_size, len(text))
            end = hard_end

            if hard_end < len(text):
                search_floor = start + int(self.chunk_size * 0.60)
                possible_boundaries = [
                    text.rfind("\n\n", search_floor, hard_end),
                    text.rfind(". ", search_floor, hard_end),
                    text.rfind("; ", search_floor, hard_end),
                    text.rfind(" ", search_floor, hard_end),
                ]
                best_boundary = max(possible_boundaries)
                if best_boundary > start:
                    end = best_boundary + 1

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(chunk_text)
            if end >= len(text):
                break
            start = max(end - self.chunk_overlap, start + 1)

        return chunks

    def process_pdf(
        self,
        file_path: str,
        source_name: Optional[str] = None,
    ) -> Tuple[List[ChunkMetadata], Dict[str, Any]]:
        path = Path(file_path)
        display_name = source_name or path.name
        pages = self.extract_pages(str(path))

        file_hash = hashlib.sha1()
        with path.open("rb") as pdf_file:
            for block in iter(lambda: pdf_file.read(1024 * 1024), b""):
                file_hash.update(block)
        source_id = file_hash.hexdigest()[:10]

        previous_type = "Other"
        active_group_type: Optional[str] = None
        group_number = 0

        for page_record in pages:
            detected_type = classify_document_type(page_record["text"], display_name)
            if detected_type == "Other" and previous_type != "Other":
                detected_type = previous_type
            if detected_type != active_group_type:
                group_number += 1
                active_group_type = detected_type

            page_record["doc_type"] = detected_type
            page_record["document_id"] = f"{source_id}-d{group_number}"
            if detected_type != "Other":
                previous_type = detected_type

        chunks: List[ChunkMetadata] = []
        for page_record in pages:
            page_chunks = self.split_page(page_record["text"])
            for chunk_number, chunk_text in enumerate(page_chunks, start=1):
                chunk_id = (
                    f"{page_record['document_id']}-p{page_record['page']}-c{chunk_number}"
                )
                chunks.append(
                    ChunkMetadata(
                        text=chunk_text,
                        doc_type=page_record["doc_type"],
                        page_start=page_record["page"],
                        page_end=page_record["page"],
                        source_id=source_id,
                        source_name=display_name,
                        document_id=page_record["document_id"],
                        chunk_id=chunk_id,
                        ocr_used=page_record["ocr_used"],
                    )
                )

        detected_types = sorted({page["doc_type"] for page in pages})
        summary = {
            "source": display_name,
            "source_id": source_id,
            "document_types": ", ".join(detected_types),
            "document_groups": group_number,
            "pages": len(pages),
            "ocr_pages": sum(bool(page["ocr_used"]) for page in pages),
            "chunks": len(chunks),
        }
        return chunks, summary


# -----------------------------------------------------------------------------
# Cached models and FAISS retrieval
# -----------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def load_embedding_model() -> TextEmbedding:
    """Load one shared ONNX BGE embedding model for the Streamlit process."""

    return TextEmbedding(model_name=EMBEDDING_MODEL_ID)


@st.cache_resource(show_spinner=False)
def load_language_model() -> Llama:
    """Load a quantized open-source Qwen model suited to free CPU hosting."""

    cpu_count = os.cpu_count() or 2
    return Llama.from_pretrained(
        repo_id=LLM_REPO_ID,
        filename=LLM_FILENAME,
        n_ctx=4096,
        n_batch=128,
        n_threads=max(1, min(cpu_count, 4)),
        n_threads_batch=max(1, min(cpu_count, 4)),
        verbose=False,
    )


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize rows so FAISS inner product is cosine similarity."""

    embeddings = np.asarray(embeddings, dtype="float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, 1e-12, None)


class RAGIndex:
    def __init__(self, embedding_model: TextEmbedding) -> None:
        self.embedding_model = embedding_model
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[ChunkMetadata] = []
        self.last_route: Optional[str] = None
        self.indexing_seconds = 0.0

    def build(self, chunks: Sequence[ChunkMetadata]) -> None:
        if not chunks:
            raise ValueError("No chunks were provided for indexing.")

        started = time.perf_counter()
        embeddings = normalize_embeddings(
            np.asarray(
                list(
                    self.embedding_model.passage_embed(
                        [chunk.text for chunk in chunks],
                        batch_size=32,
                    )
                )
            )
        )

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.chunks = list(chunks)
        self.indexing_seconds = time.perf_counter() - started

    def retrieve(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        filter_doc_type: Optional[str] = None,
        auto_route: bool = True,
    ) -> List[Tuple[ChunkMetadata, float]]:
        if self.index is None or not self.chunks:
            raise RuntimeError("The FAISS index has not been built.")

        available_types = {chunk.doc_type for chunk in self.chunks}
        route = filter_doc_type

        if route is None and auto_route:
            inferred_type = infer_query_doc_type(query)
            if inferred_type in available_types:
                route = inferred_type
        self.last_route = route

        query_embedding = normalize_embeddings(
            np.asarray(list(self.embedding_model.query_embed(query)))
        )

        if route:
            candidate_k = len(self.chunks)
        else:
            candidate_k = min(len(self.chunks), max(k * 8, 24))

        scores, indices = self.index.search(query_embedding, candidate_k)
        results: List[Tuple[ChunkMetadata, float]] = []

        for index_position, score in zip(indices[0], scores[0]):
            if index_position < 0:
                continue
            chunk = self.chunks[int(index_position)]
            if route is not None and chunk.doc_type != route:
                continue
            if float(score) < MIN_SIMILARITY:
                continue
            results.append((chunk, float(score)))
            if len(results) == k:
                break

        return results


# -----------------------------------------------------------------------------
# Grounded generation, citations, and suggested queries
# -----------------------------------------------------------------------------


def generate_with_qwen(prompt: str, max_tokens: int = 350) -> str:
    llm = load_language_model()
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful document-analysis assistant. Use only the "
                    "supplied document context. Never invent facts or citations."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        repeat_penalty=1.05,
    )
    return str(response["choices"][0]["message"]["content"]).strip()


def page_label(chunk: ChunkMetadata) -> str:
    if chunk.page_start == chunk.page_end:
        return str(chunk.page_start)
    return f"{chunk.page_start}-{chunk.page_end}"


def enforce_valid_citations(answer: str, sources: Sequence[Dict[str, Any]]) -> str:
    """Remove unsupported citation labels and guarantee at least one valid citation."""

    allowed = {(source["source"], str(source["pages"])) for source in sources}
    pattern = re.compile(r"\[Source:\s*(.+?),\s*p\.\s*([^\]]+)\]")
    valid_citation_present = False

    def validate(match: re.Match[str]) -> str:
        nonlocal valid_citation_present
        citation_pair = (match.group(1).strip(), match.group(2).strip())
        if citation_pair in allowed:
            valid_citation_present = True
            return match.group(0)
        return ""

    answer = pattern.sub(validate, answer).strip()

    if sources and not valid_citation_present:
        unique_citations: List[str] = []
        seen: set[Tuple[str, str]] = set()
        for source in sources:
            pair = (str(source["source"]), str(source["pages"]))
            if pair not in seen:
                unique_citations.append(f"[Source: {pair[0]}, p. {pair[1]}]")
                seen.add(pair)
        answer += "\n\nSupporting evidence: " + " ".join(unique_citations[:3])

    return answer


def answer_query(
    rag_index: RAGIndex,
    query: str,
    k: int,
    filter_doc_type: Optional[str],
    auto_route: bool,
) -> Dict[str, Any]:
    started = time.perf_counter()
    retrieved_chunks = rag_index.retrieve(
        query=query,
        k=k,
        filter_doc_type=filter_doc_type,
        auto_route=auto_route,
    )

    if not retrieved_chunks:
        return {
            "answer": "I couldn't find sufficiently relevant information in the indexed documents.",
            "sources": [],
            "confidence": 0.0,
            "chunks_used": 0,
            "route": rag_index.last_route or "All document types",
            "latency_seconds": round(time.perf_counter() - started, 2),
        }

    context_parts: List[str] = []
    sources: List[Dict[str, Any]] = []

    for rank, (chunk, similarity_score) in enumerate(retrieved_chunks, start=1):
        pages = page_label(chunk)
        context_parts.append(
            f"[SOURCE {rank} | {chunk.source_name} | {chunk.doc_type} | Page {pages}]\n"
            f"{chunk.text}"
        )
        sources.append(
            {
                "rank": rank,
                "source": chunk.source_name,
                "source_id": chunk.source_id,
                "document_id": chunk.document_id,
                "document_type": chunk.doc_type,
                "pages": pages,
                "chunk_id": chunk.chunk_id,
                "similarity": round(similarity_score, 3),
            }
        )

    context = "\n\n".join(context_parts)[:MAX_CONTEXT_CHARACTERS]
    prompt = f"""
Answer the question using only the document context below.

Rules:
1. If the answer is not contained in the context, say exactly:
   "The provided documents do not contain enough information to answer this question."
2. Do not use outside knowledge.
3. Do not guess or invent missing information.
4. Cite factual statements inline as [Source: filename, p. X].
5. If sources disagree, explain the disagreement and cite both.
6. Give a direct and concise answer first.
7. Use bullet points only when they improve clarity.

DOCUMENT CONTEXT
----------------
{context}

QUESTION
--------
{query}
""".strip()

    answer = generate_with_qwen(prompt)
    answer = enforce_valid_citations(answer, sources)
    average_similarity = float(np.mean([score for _, score in retrieved_chunks]))

    return {
        "answer": answer,
        "sources": sources,
        "confidence": round(average_similarity, 3),
        "chunks_used": len(retrieved_chunks),
        "route": rag_index.last_route or "All document types",
        "latency_seconds": round(time.perf_counter() - started, 2),
    }


def select_representative_chunks(
    chunks: Sequence[ChunkMetadata],
    limit: int = 6,
) -> List[ChunkMetadata]:
    if not chunks:
        return []

    selected_indices: List[int] = []
    seen_document_types: set[str] = set()
    for index, chunk in enumerate(chunks):
        if chunk.doc_type not in seen_document_types:
            selected_indices.append(index)
            seen_document_types.add(chunk.doc_type)
        if len(selected_indices) >= limit:
            break

    target_count = min(limit, len(chunks))
    distributed_indices = (
        [0]
        if target_count == 1
        else [
            round(position * (len(chunks) - 1) / (target_count - 1))
            for position in range(target_count)
        ]
    )
    for index in distributed_indices:
        if index not in selected_indices:
            selected_indices.append(index)
        if len(selected_indices) >= limit:
            break

    return [chunks[index] for index in selected_indices[:limit]]


def heuristic_suggestions(chunks: Sequence[ChunkMetadata]) -> List[str]:
    """Document-aware fallback if the small LLM does not return valid JSON."""

    all_text = " ".join(chunk.text.lower() for chunk in chunks[:20])
    document_types = {chunk.doc_type for chunk in chunks}
    suggestions: List[str] = []

    if "entry to a major" in all_text or "etam" in all_text:
        suggestions.extend(
            [
                "How does the Entry to a Major placement process work?",
                "How many eligible applicants were placed in each engineering major?",
                "What is the difference between auto-entry and holistic review placement?",
                "Which majors received the most first-choice placements?",
            ]
        )
    if "Certificate of Quality" in document_types:
        suggestions.extend(
            [
                "What is the product lot number?",
                "Which quality tests were performed and what were the results?",
            ]
        )
    if "Packaging Specification" in document_types:
        suggestions.append("What are the packaging requirements and specifications?")
    if "Supplier Qualification Record" in document_types:
        suggestions.append("What are the supplier's performance metrics?")
    if "Chain of Custody" in document_types:
        suggestions.append("How are the assemblies traced through the chain of custody?")
    if "BSE/TSE Declaration" in document_types:
        suggestions.append("What does the document state about BSE/TSE risk?")

    for question in DEFAULT_SUGGESTED_QUERIES:
        if question not in suggestions:
            suggestions.append(question)
    return suggestions[:4]


def generate_suggested_queries(
    chunks: Sequence[ChunkMetadata],
    max_questions: int = 4,
) -> List[str]:
    if not chunks:
        return DEFAULT_SUGGESTED_QUERIES[:max_questions]

    excerpts = []
    for number, chunk in enumerate(select_representative_chunks(chunks), start=1):
        excerpts.append(
            f"[Excerpt {number} | Source: {chunk.source_name} | "
            f"Type: {chunk.doc_type} | Page: {chunk.page_start}]\n{chunk.text[:700]}"
        )

    excerpt_text = "\n\n".join(excerpts)

    prompt = f"""
Create exactly {max_questions} concise questions that can be answered from the
document excerpts below. Cover different topics, do not answer the questions,
and do not mention excerpts or context. Return only a valid JSON array of
strings. Every item must end with a question mark.

DOCUMENT EXCERPTS
-----------------
{excerpt_text}
""".strip()

    try:
        raw_response = generate_with_qwen(prompt, max_tokens=180)
        json_match = re.search(r"\[[\s\S]*?\]", raw_response)
        parsed = json.loads(json_match.group(0)) if json_match else []
        cleaned: List[str] = []
        for question in parsed if isinstance(parsed, list) else []:
            if not isinstance(question, str):
                continue
            normalized = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", question)
            normalized = normalized.strip().strip('"')
            if normalized and not normalized.endswith("?"):
                normalized += "?"
            if normalized and len(normalized) <= 180 and normalized not in cleaned:
                cleaned.append(normalized)
        if len(cleaned) >= max_questions:
            return cleaned[:max_questions]
    except Exception as error:
        print(f"Suggested-query generation fallback: {error!r}")

    return heuristic_suggestions(chunks)[:max_questions]


# -----------------------------------------------------------------------------
# Upload lifecycle and per-session isolation
# -----------------------------------------------------------------------------


def initialize_session_state() -> None:
    defaults: Dict[str, Any] = {
        "rag_index": None,
        "indexed_signature": None,
        "summaries": [],
        "messages": [],
        "suggested_queries": DEFAULT_SUGGESTED_QUERIES,
        "last_result": None,
        "processing_seconds": 0.0,
        "processing_error": None,
        "document_type_filter": "All",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def uploaded_signature(uploaded_files: Optional[Iterable[Any]]) -> Optional[str]:
    files = list(uploaded_files or [])
    if not files:
        return None

    digest = hashlib.sha1()
    for uploaded_file in files:
        digest.update(str(uploaded_file.name).encode("utf-8"))
        digest.update(uploaded_file.getvalue())
    return digest.hexdigest()


def reset_knowledge_base() -> None:
    """Deactivate the current corpus before attempting a replacement."""

    st.session_state.rag_index = None
    st.session_state.indexed_signature = None
    st.session_state.summaries = []
    st.session_state.messages = []
    st.session_state.suggested_queries = DEFAULT_SUGGESTED_QUERIES
    st.session_state.last_result = None
    st.session_state.processing_seconds = 0.0
    st.session_state.processing_error = None
    st.session_state.document_type_filter = "All"


def process_uploaded_documents(uploaded_files: Sequence[Any]) -> None:
    """Build a brand-new index using only the current session's uploaded PDFs."""

    reset_knowledge_base()
    if not uploaded_files:
        raise ValueError("Upload at least one PDF before processing.")

    started = time.perf_counter()
    processor = DocumentProcessor()
    all_chunks: List[ChunkMetadata] = []
    summaries: List[Dict[str, Any]] = []

    try:
        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.getvalue()
            if not file_bytes:
                raise ValueError(f"{uploaded_file.name} is empty.")

            temporary_path: Optional[Path] = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as handle:
                    handle.write(file_bytes)
                    temporary_path = Path(handle.name)

                chunks, summary = processor.process_pdf(
                    str(temporary_path),
                    source_name=uploaded_file.name,
                )
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

            if not chunks:
                raise ValueError(f"No readable text was extracted from {uploaded_file.name}.")
            all_chunks.extend(chunks)
            summaries.append(summary)

        with st.spinner("Loading the embedding model and building the FAISS index…"):
            new_index = RAGIndex(load_embedding_model())
            new_index.build(all_chunks)

        expected_sources = sorted(uploaded_file.name for uploaded_file in uploaded_files)
        active_sources = sorted({chunk.source_name for chunk in new_index.chunks})
        if set(active_sources) != set(expected_sources):
            raise RuntimeError(
                f"Index source verification failed: expected {expected_sources}, "
                f"found {active_sources}."
            )
        if new_index.index is None or new_index.index.ntotal != len(all_chunks):
            raise RuntimeError("FAISS vector-count verification failed.")

        with st.spinner("Generating questions from the newly indexed documents…"):
            suggestions = generate_suggested_queries(new_index.chunks)

        st.session_state.rag_index = new_index
        st.session_state.indexed_signature = uploaded_signature(uploaded_files)
        st.session_state.summaries = summaries
        st.session_state.suggested_queries = suggestions
        st.session_state.processing_seconds = time.perf_counter() - started
        st.session_state.processing_error = None
    except Exception as error:
        reset_knowledge_base()
        st.session_state.processing_error = f"{type(error).__name__}: {error}"
        raise


def empty_source_table() -> pd.DataFrame:
    return pd.DataFrame(columns=SOURCE_TABLE_COLUMNS)


def source_table_for_result(result: Optional[Dict[str, Any]]) -> pd.DataFrame:
    if not result or not result.get("sources"):
        return empty_source_table()
    return pd.DataFrame(result["sources"])[SOURCE_TABLE_COLUMNS]


initialize_session_state()


# -----------------------------------------------------------------------------
# Interface
# -----------------------------------------------------------------------------


st.markdown(
    """
<section class="sg-header">
  <div class="sg-brand-row">
    <div class="sg-brand">
      <div class="sg-logo">S</div>
      <div>
        <h1 class="sg-title">SourceGround</h1>
        <div class="sg-subtitle">Source-grounded document intelligence with verifiable evidence</div>
      </div>
    </div>
    <div class="sg-badges">
      <span class="sg-badge sg-online">● System online</span>
      <span class="sg-badge">Qwen 2.5</span>
      <span class="sg-badge">BGE embeddings</span>
      <span class="sg-badge">FAISS index</span>
    </div>
  </div>
</section>

<section class="sg-pipeline">
  <div class="sg-stage"><div class="sg-stage-number">01</div><div class="sg-stage-name">Ingest</div><div class="sg-stage-detail">Digital PDF or OCR</div></div>
  <div class="sg-stage"><div class="sg-stage-number">02</div><div class="sg-stage-name">Structure</div><div class="sg-stage-detail">Chunk + metadata</div></div>
  <div class="sg-stage"><div class="sg-stage-number">03</div><div class="sg-stage-name">Retrieve</div><div class="sg-stage-detail">BGE + FAISS</div></div>
  <div class="sg-stage"><div class="sg-stage-number">04</div><div class="sg-stage-name">Generate</div><div class="sg-stage-detail">Open-source Qwen</div></div>
  <div class="sg-stage"><div class="sg-stage-number">05</div><div class="sg-stage-name">Verify</div><div class="sg-stage-detail">Citations + diagnostics</div></div>
</section>
""",
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown('<div class="sg-section-kicker">Knowledge base</div>', unsafe_allow_html=True)
    st.subheader("Document ingestion")
    st.caption("Upload digital or scanned PDFs. A new process run completely replaces the active index.")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
    )
    current_signature = uploaded_signature(uploaded_files)
    signature_matches = (
        current_signature is not None
        and current_signature == st.session_state.indexed_signature
        and st.session_state.rag_index is not None
    )

    process_clicked = st.button(
        "Process and index documents",
        type="primary",
        use_container_width=True,
        disabled=not bool(uploaded_files),
    )

    if process_clicked:
        try:
            with st.spinner("Extracting, classifying, and chunking documents…"):
                process_uploaded_documents(uploaded_files)
            signature_matches = True
            st.success("Verified knowledge base created.")
        except Exception as error:
            st.error(f"Document processing failed: {error}")

    if signature_matches:
        active_index: RAGIndex = st.session_state.rag_index
        total_pages = sum(int(item["pages"]) for item in st.session_state.summaries)
        total_ocr_pages = sum(int(item["ocr_pages"]) for item in st.session_state.summaries)
        st.markdown(
            f'<div class="sg-ready"><strong>Knowledge base ready</strong><br>'
            f'{len(st.session_state.summaries)} file(s) · {total_pages} pages · '
            f'{total_ocr_pages} OCR pages · {len(active_index.chunks)} chunks</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Processing and indexing: {st.session_state.processing_seconds:.2f} seconds")
        with st.expander("Indexed document details"):
            st.dataframe(
                pd.DataFrame(st.session_state.summaries),
                hide_index=True,
                use_container_width=True,
            )
    elif uploaded_files and st.session_state.rag_index is not None:
        st.markdown(
            '<div class="sg-warning"><strong>Uploads changed.</strong><br>'
            'Process the selected PDFs to replace the previous index.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="sg-waiting"><strong>Knowledge base waiting</strong><br>'
            'Upload one or more PDF documents to begin.</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown('<div class="sg-section-kicker">Retrieval engine</div>', unsafe_allow_html=True)
    st.subheader("Search controls")

    if signature_matches:
        available_types = sorted({chunk.doc_type for chunk in st.session_state.rag_index.chunks})
    else:
        available_types = []

    selected_document_type = st.selectbox(
        "Document-type filter",
        options=["All", *available_types],
        key="document_type_filter",
        disabled=not signature_matches,
    )
    auto_route = st.toggle(
        "Enable metadata-aware routing",
        value=True,
        disabled=not signature_matches or selected_document_type != "All",
    )
    top_k = st.slider(
        "Top retrieved chunks",
        min_value=1,
        max_value=8,
        value=DEFAULT_TOP_K,
        disabled=not signature_matches,
    )
    st.caption(
        "A manual document-type filter overrides automatic routing. "
        "Clear queries are routed to the most relevant detected type."
    )


st.markdown('<div class="sg-section-kicker">Research console</div>', unsafe_allow_html=True)
st.subheader("Evidence-backed document Q&A")
st.caption("Ask questions grounded exclusively in the active, verified document index.")

if signature_matches:
    index = st.session_state.rag_index
    metric_columns = st.columns(4)
    metric_columns[0].metric("Active files", len(st.session_state.summaries))
    metric_columns[1].metric("Indexed chunks", len(index.chunks))
    metric_columns[2].metric("Document types", len({chunk.doc_type for chunk in index.chunks}))
    metric_columns[3].metric("OCR pages", sum(int(item["ocr_pages"]) for item in st.session_state.summaries))
else:
    st.info("Upload PDFs in the sidebar and select **Process and index documents** to activate the research console.")

st.markdown("#### Document conversation")

if not st.session_state.messages:
    st.markdown(
        '<div class="sg-waiting">Your grounded answers and source citations will appear here.</div>',
        unsafe_allow_html=True,
    )

for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🔎"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        result = message.get("result")
        if result:
            st.caption(
                f"Confidence {result['confidence']:.1%} · "
                f"{result['chunks_used']} chunks · {result['route']} · "
                f"{result['latency_seconds']:.2f}s"
            )
            if result.get("sources"):
                with st.expander("View supporting evidence"):
                    for source in result["sources"]:
                        st.markdown(
                            f"- **{source['source']}**, p. {source['pages']} · "
                            f"{source['document_type']} · similarity `{source['similarity']:.3f}`"
                        )


st.markdown("#### Suggested queries")
suggestion_columns = st.columns(2)
selected_suggestion: Optional[str] = None

for index, suggestion in enumerate(st.session_state.suggested_queries):
    if suggestion_columns[index % 2].button(
        suggestion,
        key=f"suggestion_{index}_{hashlib.sha1(suggestion.encode()).hexdigest()[:8]}",
        use_container_width=True,
        disabled=not signature_matches,
    ):
        selected_suggestion = suggestion

typed_query = st.chat_input(
    "Ask a precise question about the indexed documents…",
    disabled=not signature_matches,
)
pending_query = typed_query or selected_suggestion

if pending_query and signature_matches:
    st.session_state.messages.append({"role": "user", "content": pending_query})
    try:
        with st.spinner("Retrieving evidence and generating a grounded answer…"):
            result = answer_query(
                rag_index=st.session_state.rag_index,
                query=pending_query,
                k=top_k,
                filter_doc_type=None if selected_document_type == "All" else selected_document_type,
                auto_route=auto_route and selected_document_type == "All",
            )
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "result": result,
            }
        )
        st.session_state.last_result = result
    except Exception as error:
        error_message = f"I could not complete this request: {type(error).__name__}: {error}"
        st.session_state.messages.append(
            {"role": "assistant", "content": error_message, "result": None}
        )
    st.rerun()


clear_column, spacer_column = st.columns([1, 4])
if clear_column.button(
    "Clear conversation",
    use_container_width=True,
    disabled=not bool(st.session_state.messages),
):
    st.session_state.messages = []
    st.session_state.last_result = None
    st.rerun()


st.divider()
st.markdown('<div class="sg-section-kicker">Retrieval diagnostics</div>', unsafe_allow_html=True)
st.subheader("Evidence and source traceability")
st.caption("Inspect routing, similarity, source pages, chunk usage, and end-to-end latency.")

last_result = st.session_state.last_result
diagnostic_columns = st.columns(4)

if last_result:
    diagnostic_columns[0].metric("Confidence", f"{last_result['confidence']:.1%}")
    diagnostic_columns[1].metric("Chunks used", last_result["chunks_used"])
    diagnostic_columns[2].metric("Route", last_result["route"])
    diagnostic_columns[3].metric("Latency", f"{last_result['latency_seconds']:.2f}s")
else:
    diagnostic_columns[0].metric("Confidence", "—")
    diagnostic_columns[1].metric("Chunks used", "—")
    diagnostic_columns[2].metric("Route", "—")
    diagnostic_columns[3].metric("Latency", "—")

st.dataframe(
    source_table_for_result(last_result),
    hide_index=True,
    use_container_width=True,
    column_config={
        "similarity": st.column_config.NumberColumn("Similarity", format="%.3f"),
        "rank": st.column_config.NumberColumn("Rank", format="%d"),
    },
)
st.markdown(
    '<div class="sg-footnote"><strong>Confidence interpretation:</strong> the displayed '
    'value is the average semantic similarity of the retrieved chunks. It measures retrieval '
    'relevance; it is not a calibrated probability that the generated answer is correct.</div>',
    unsafe_allow_html=True,
)

st.caption(
    "SourceGround · Open-source Qwen generation · BGE semantic retrieval · FAISS vector search · Tesseract OCR"
)
