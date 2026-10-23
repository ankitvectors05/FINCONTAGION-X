
from collections import defaultdict
from math import isfinite
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.responses import RedirectResponse

from pydantic import BaseModel, Field

from fincontagion.rag.ingestion import extract_pdf_pages
from fincontagion.rag.retrieval import FinancialRetriever
from fincontagion.rag.generation import generate_answer
from threading import Lock

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fincontagion.rag.document_store import (
    calculate_sha256,
    get_document_directory,
)


# Locate the project root independently of the terminal's working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

app = FastAPI(
    title="FINCONTAGION-X API",
    description="API for financial network and systemic-risk analytics.",
    version="0.1.0",
)


import hashlib
import os
import tempfile


MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB
USER_DOCUMENT_ROOT = (
    PROJECT_ROOT / "data" / "processed" / "user_documents"
)

OLLAMA_CHAT_URL = os.getenv(
    "OLLAMA_CHAT_URL",
    "http://127.0.0.1:11434/api/chat",
)

OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "qwen3:1.7b")



@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Validate and persist PDF, CSV, and XLSX uploads using SHA-256 IDs."""

    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()

    allowed = {".pdf", ".csv", ".xlsx"}
    if not filename or suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, CSV, and XLSX files are supported.",
        )

    USER_DOCUMENT_ROOT.mkdir(parents=True, exist_ok=True)
    sha256 = hashlib.sha256()
    total_bytes = 0
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=USER_DOCUMENT_ROOT,
            suffix=".upload",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)

            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break

                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="File exceeds the 20 MB upload limit.",
                    )

                sha256.update(chunk)
                temp_file.write(chunk)

        if total_bytes == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        document_id = sha256.hexdigest()
        document_dir = get_document_directory(
            document_id, USER_DOCUMENT_ROOT
        )
        stored_file = document_dir / f"document{suffix}"

        if stored_file.is_file():
            return {
                "document_id": document_id,
                "filename": filename,
                "sha256": document_id,
                "file_type": suffix[1:],
                "status": "already_uploaded",
            }

        # Validate content before permanently storing the upload.
        try:
            if suffix == ".pdf":
                from pypdf import PdfReader

                reader = PdfReader(temp_path)
                if len(reader.pages) == 0:
                    raise ValueError("PDF contains no pages.")

            elif suffix == ".csv":
                # Read the complete CSV to detect malformed or empty files.
                df = pd.read_csv(temp_path, nrows=1000)
                if len(df.columns) == 0:
                    raise ValueError("CSV has no columns.")

            else:
                sheets = pd.read_excel(
                    temp_path, sheet_name=None, engine="openpyxl"
                )
                if not sheets:
                    raise ValueError("Workbook contains no worksheets.")

        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid {suffix[1:].upper()} file. "
                    "Check that the file is readable and correctly formatted."
                ),
            ) from exc

        document_dir.mkdir(parents=True, exist_ok=True)
        os.replace(temp_path, stored_file)
        temp_path = None

        response = {
            "document_id": document_id,
            "filename": filename,
            "sha256": document_id,
            "file_type": suffix[1:],
            "status": "uploaded",
        }

        if suffix == ".xlsx":
            response["sheets"] = list(sheets.keys())

        return response

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        await file.close()



# Keep your existing imports and cache variables.
_rbi_retriever = None
_rbi_retriever_lock = Lock()

RBI_CACHE_DIR = (
    PROJECT_ROOT / "data" / "processed" / "rag_cache_test"
)


def get_rbi_retriever() -> FinancialRetriever:
    """Load a saved RBI index or build and save it once."""

    global _rbi_retriever

    if _rbi_retriever is None:
        with _rbi_retriever_lock:
            if _rbi_retriever is None:
                retriever = FinancialRetriever()

                if (RBI_CACHE_DIR / "metadata.json").exists():
                    print("RBI CACHE: Loading saved index", flush=True)
                    retriever.load_index(RBI_CACHE_DIR)
                else:
                    print("RBI CACHE: Building new index", flush=True)
                    pages = extract_pdf_pages("rbi_fsr_june_2025.pdf")
                    retriever.index_pages(pages)
                    retriever.save_index(RBI_CACHE_DIR)

                _rbi_retriever = retriever

    else:
        print("RBI CACHE: Reusing in-memory index", flush=True)

    return _rbi_retriever


# Separate in-memory cache for uploaded documents.
_document_retrievers = {}
_document_retrievers_lock = Lock()


def get_document_retriever(document_id: str) -> FinancialRetriever:
    """Load or build a persistent search index for one uploaded PDF."""

    global _document_retrievers

    # Validate the ID before constructing any document paths.
    try:
        document_dir = get_document_directory(
            document_id,
            USER_DOCUMENT_ROOT,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid document ID.",
        ) from exc

    stored_pdf = document_dir / "document.pdf"
    index_dir = document_dir / "index"

    if not stored_pdf.is_file():
        raise HTTPException(
            status_code=404,
            detail="Uploaded document not found.",
        )

    with _document_retrievers_lock:
        # Reuse the already-loaded index for this API process.
        if document_id in _document_retrievers:
            print(
                f"DOCUMENT CACHE: Reusing index {document_id[:12]}",
                flush=True,
            )
            return _document_retrievers[document_id]

        retriever = FinancialRetriever()

        if (index_dir / "metadata.json").is_file():
            print(
                f"DOCUMENT CACHE: Loading saved index {document_id[:12]}",
                flush=True,
            )
            retriever.load_index(index_dir)
        else:
            print(
                f"DOCUMENT CACHE: Building index {document_id[:12]}",
                flush=True,
            )

            pages = extract_pdf_pages(stored_pdf.resolve())

            if not pages or not any(
                page.get("text", "").strip() for page in pages
            ):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "No extractable text was found in this PDF. "
                        "Scanned PDFs may require OCR."
                    ),
                )

            retriever.index_pages(pages)
            retriever.save_index(index_dir)

        _document_retrievers[document_id] = retriever
        return retriever



class ResearchQuestion(BaseModel):
    question: str = Field(min_length=1, max_length=1000)

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    history: list[dict] = Field(default_factory=list)


@app.post("/api/research/ask")
def ask_financial_research(request: ResearchQuestion):
    """Answer a financial research question using the indexed RBI report."""

    try:
        retriever = get_rbi_retriever()

        results = retriever.search(request.question.strip(), top_k=3)
        output = generate_answer(request.question.strip(), results)

        return {
            "question": request.question.strip(),
            "answer": output["answer"],
            "sources": output["sources"],
        }

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="The configured financial research PDF is unavailable.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Financial research request failed: {exc}",
        ) from exc


def answer_tabular_question(
    file_path: Path,
    suffix: str,
    question: str,
) -> dict:
    """Profile CSV/XLSX data and generate a grounded answer using Ollama."""

    try:
        if suffix == ".csv":
            sheets = {"CSV": pd.read_csv(file_path)}
        else:
            sheets = pd.read_excel(
                file_path, sheet_name=None, engine="openpyxl"
            )
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail="The stored tabular file could not be parsed.",
        ) from exc

    if not sheets or all(df.empty for df in sheets.values()):
        raise HTTPException(
            status_code=422,
            detail="The uploaded file contains no data rows.",
        )

    # Limit the context sent to the local model.
    profiles = []
    for sheet_name, df in sheets.items():
        if df.empty:
            continue

        profile = {
            "sheet": sheet_name,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "column_names": [str(c) for c in df.columns],
            "data_types": {
                str(c): str(dtype) for c, dtype in df.dtypes.items()
            },
            "missing_values": {
                str(c): int(n)
                for c, n in df.isna().sum().items()
            },
            "sample_rows": df.head(5).astype(object).where(
                pd.notna(df.head(5)), None
            ).to_dict(orient="records"),
        }

        numeric = df.select_dtypes(include="number")
        if not numeric.empty:
            profile["numeric_summary"] = (
                numeric.describe().round(3).to_dict()
            )

        profiles.append(profile)

    system_prompt = (
        "You are the FINCONTAGION-X data analysis assistant. "
        "Answer using only the supplied table profiles and statistics. "
        "Never invent computed results. The sample rows are examples, "
        "not the complete dataset. If a question requires an exact "
        "calculation not present in the supplied statistics, say that "
        "you cannot determine the exact result from this summary alone. "
        "Be concise and identify the worksheet or column when relevant."
    )

    payload = {
        "model": OLLAMA_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Computed dataset profiles:\n"
                    f"{json.dumps(profiles, default=str)}"
                ),
            },
        ],
        "stream": False,
        "think": False,
    }

    request = Request(
        OLLAMA_CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))

        answer = result.get("message", {}).get("content", "").strip()
        if not answer:
            raise HTTPException(
                status_code=502,
                detail="The local model returned an empty answer.",
            )

        return {
            "answer": answer,
            "sheets": [
                {
                    "name": item["sheet"],
                    "rows": item["rows"],
                    "columns": item["columns"],
                }
                for item in profiles
            ],
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Tabular analysis failed: {type(exc).__name__}: {exc}",
        ) from exc




@app.post("/api/documents/{document_id}/ask")
def ask_uploaded_document(
    document_id: str,
    request: ResearchQuestion,
):
    """Answer questions about an uploaded PDF, CSV, or XLSX file."""

    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    try:
        try:
            document_dir = get_document_directory(
                document_id, USER_DOCUMENT_ROOT
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail="Invalid document ID."
            ) from exc

        # Preserve the existing PDF RAG behaviour.
        stored_pdf = document_dir / "document.pdf"
        if stored_pdf.is_file():
            retriever = get_document_retriever(document_id)
            results = retriever.search(question, top_k=3)
            output = generate_answer(question, results)

            return {
                "document_id": document_id,
                "question": question,
                "answer": output["answer"],
                "sources": output["sources"],
                "file_type": "pdf",
            }

        # Route CSV and Excel uploads to structured analysis.
        for suffix in (".csv", ".xlsx"):
            stored_file = document_dir / f"document{suffix}"
            if stored_file.is_file():
                output = answer_tabular_question(
                    stored_file, suffix, question
                )
                return {
                    "document_id": document_id,
                    "question": question,
                    "answer": output["answer"],
                    "sources": [],
                    "file_type": suffix[1:],
                    "sheets": output["sheets"],
                }

        raise HTTPException(
            status_code=404,
            detail="Uploaded document not found.",
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Document analysis failed: {exc}",
        ) from exc




@app.post("/api/chat")
def general_finance_chat(request: ChatRequest):
    """Answer general finance questions using the local Ollama model."""

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are FINCONTAGION-X, a financial research assistant. "
                "Explain financial concepts clearly and distinguish facts "
                "from assumptions. Never invent live market data, citations, "
                "or sources. State uncertainty when appropriate. "
                "You may explain financial concepts and perform calculations "
                "when the necessary information is provided. "
                "You do not have access to live market data or project files "
                "unless they are explicitly supplied."
            ),
        }
    ]

    # Include only valid recent conversation messages.
    for item in request.history[-12:]:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")

        if (
            role in {"user", "assistant"}
            and isinstance(content, str)
            and content.strip()
        ):
            messages.append({
                "role": role,
                "content": content[:4000],
            })

    messages.append({
        "role": "user",
        "content": question,
    })

    payload = {
        "model": OLLAMA_MODEL_NAME,
        "messages": messages,
        "stream": False,
        "think": False,
    }

    ollama_request = Request(
        OLLAMA_CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(ollama_request, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))

        answer = result.get("message", {}).get("content", "").strip()

        if not answer:
            raise HTTPException(
                status_code=502,
                detail="The local model returned an empty response.",
            )

        return {
            "question": question,
            "answer": answer,
            "model": "qwen3:1.7b",
            "mode": "general_chat",
        }

    except HTTPException:
        raise

    except HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned HTTP {exc.code}.",
        ) from exc

    except HTTPException:
            raise

    except Exception as exc:
        print(
            f"OLLAMA CHAT ERROR: {type(exc).__name__}: {exc}",
            flush=True,
        )
        raise HTTPException(
            status_code=502,
            detail=f"Local LLM request failed: {type(exc).__name__}: {exc}",
        ) from exc

    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise HTTPException(
            status_code=502,
            detail="The local model returned an invalid response.",
        ) from exc

    
@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse(url="/dashboard")

@app.get("/dashboard", include_in_schema=False)
def dashboard():
    """Serve the financial network dashboard."""
    dashboard_file = (
        PROJECT_ROOT / "src" / "fincontagion" / "static" / "index.html"
    )

    if not dashboard_file.is_file():
        raise HTTPException(
            status_code=404,
            detail="Dashboard file not found.",
        )

    return FileResponse(dashboard_file)


def load_csv(filename: str, required_columns: set[str]) -> pd.DataFrame:
    """Load a processed CSV and verify its required columns."""
    path = DATA_DIR / filename

    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Required data file is missing: {filename}",
        )

    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read {filename}: {exc}",
        ) from exc

    missing = required_columns - set(df.columns)

    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"{filename} is missing columns: {sorted(missing)}",
        )

    return df


@app.get("/health")
def health_check():
    """Confirm that the API process is responding."""
    return {"status": "ok", "service": "FINCONTAGION-X"}


@app.get("/api/network")
def get_financial_network():
    """Return financial entities and exposure relationships for graphing."""

    entities = load_csv(
        "entities.csv",
        {"entity_id", "name", "entity_type", "country"},
    )

    exposures = load_csv(
        "exposures.csv",
        {
            "exposure_id",
            "creditor_id",
            "debtor_id",
            "outstanding_amount",
            "currency",
        },
    )

    # Entity IDs must be unique for graph nodes to be unambiguous.
    if entities["entity_id"].isna().any():
        raise HTTPException(
            status_code=500,
            detail="entities.csv contains missing entity IDs.",
        )

    if entities["entity_id"].duplicated().any():
        raise HTTPException(
            status_code=500,
            detail="entities.csv contains duplicate entity IDs.",
        )

    entity_ids = set(entities["entity_id"].astype(str))

    # Build the nodes expected by a graph-visualization library.
    nodes = []

    for row in entities.to_dict(orient="records"):
        nodes.append(
            {
                "id": str(row["entity_id"]),
                "label": str(row["name"]),
                "type": str(row["entity_type"]),
                "country": str(row["country"]),
            }
        )

    # Validate and convert exposure amounts before returning them.
    amounts = pd.to_numeric(
        exposures["outstanding_amount"],
        errors="coerce",
    )

    if amounts.isna().any() or not amounts.map(
        lambda value: isfinite(float(value))
    ).all():
        raise HTTPException(
            status_code=500,
            detail="Exposure amounts contain missing or invalid numbers.",
        )

    if (amounts < 0).any():
        raise HTTPException(
            status_code=500,
            detail="Exposure amounts contain negative values.",
        )

    # A creditor lends or has an exposure to a debtor.
    edges = []
    exposure_by_currency = defaultdict(float)

    for row, amount in zip(
        exposures.to_dict(orient="records"),
        amounts.tolist(),
    ):
        source = str(row["creditor_id"])
        target = str(row["debtor_id"])

        if source not in entity_ids or target not in entity_ids:
            raise HTTPException(
                status_code=500,
                detail=(
                    "An exposure references an entity missing "
                    "from entities.csv."
                ),
            )

        currency = str(row["currency"])

        edges.append(
            {
                "id": str(row["exposure_id"]),
                "source": source,
                "target": target,
                "amount": float(amount),
                "currency": currency,
            }
        )

        exposure_by_currency[currency] += float(amount)

    return {
        "summary": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "exposure_by_currency": dict(exposure_by_currency),
        },
        "nodes": nodes,
        "edges": edges,
    }


@app.get("/api/analytics")
def get_financial_analytics():
    """Return existing stress-test, systemic-risk, and ML results."""

    stress = load_csv(
        "integrated_stress_results.csv",
        {
            "entity_id",
            "entity_type",
            "scenario_id",
            "credit_loss",
            "market_loss",
            "interest_rate_loss",
            "funding_withdrawal",
            "liquidity_shortfall",
            "total_accounting_loss",
            "baseline_equity",
            "stressed_equity",
        },
    )

    risk = load_csv(
        "systemic_risk_assessment.csv",
        {
            "systemic_risk_rank",
            "entity_id",
            "entity_type",
            "systemic_risk_score",
            "risk_tier",
            "primary_risk_drivers",
            "total_connected_exposure",
        },
    )

    ml = load_csv(
        "ml_early_warning_predictions.csv",
        {
            "entity_id",
            "scenario_id",
            "actual_distress",
            "logistic_regression_probability",
            "random_forest_probability",
            "random_forest_warning_band",
        },
    )

    # Convert numeric columns safely and reject invalid values.
    stress_numeric = [
        "credit_loss",
        "market_loss",
        "interest_rate_loss",
        "funding_withdrawal",
        "liquidity_shortfall",
        "total_accounting_loss",
        "baseline_equity",
        "stressed_equity",
    ]

    for column in stress_numeric:
        stress[column] = pd.to_numeric(stress[column], errors="coerce")

        if stress[column].isna().any():
            raise HTTPException(
                status_code=500,
                detail=f"Invalid numeric values in stress results: {column}",
            )

    for column in [
        "systemic_risk_rank",
        "systemic_risk_score",
        "total_connected_exposure",
    ]:
        risk[column] = pd.to_numeric(risk[column], errors="coerce")

        if risk[column].isna().any():
            raise HTTPException(
                status_code=500,
                detail=f"Invalid numeric values in systemic risk: {column}",
            )

    for column in [
        "actual_distress",
        "logistic_regression_probability",
        "random_forest_probability",
    ]:
        ml[column] = pd.to_numeric(ml[column], errors="coerce")

        if ml[column].isna().any():
            raise HTTPException(
                status_code=500,
                detail=f"Invalid numeric values in ML predictions: {column}",
            )

    
    # Summarize each integrated stress scenario separately.
    stress["scenario_id"] = stress["scenario_id"].astype(str)

    scenario_summaries = []

    for scenario_id, scenario_df in stress.groupby("scenario_id", sort=True):
        scenario_summaries.append({
            "scenario_id": scenario_id,
            "entity_count": int(scenario_df["entity_id"].nunique()),
            "credit_loss": float(scenario_df["credit_loss"].sum()),
            "market_loss": float(scenario_df["market_loss"].sum()),
            "interest_rate_loss": float(scenario_df["interest_rate_loss"].sum()),
            "funding_withdrawal": float(scenario_df["funding_withdrawal"].sum()),
            "liquidity_shortfall": float(scenario_df["liquidity_shortfall"].sum()),
            "total_accounting_loss": float(
                scenario_df["total_accounting_loss"].sum()
            ),
            "entities_with_negative_stressed_equity": int(
                (scenario_df["stressed_equity"] < 0).sum()
            ),
            "assumption_basis": "Illustrative synthetic assumptions",
        })

    stress_summary = {
        "scenario_ids": [item["scenario_id"] for item in scenario_summaries],
        "entity_count": int(stress["entity_id"].nunique()),
        "scenario_count": len(scenario_summaries),
        "scenarios": scenario_summaries,
        "assumption_basis": "Illustrative synthetic assumptions",
    }


    # Return the highest systemic-risk rankings first.
    risk = risk.sort_values("systemic_risk_rank").head(10)

    top_risk_entities = risk[
        [
            "systemic_risk_rank",
            "entity_id",
            "entity_type",
            "systemic_risk_score",
            "risk_tier",
            "primary_risk_drivers",
            "total_connected_exposure",
        ]
    ].to_dict(orient="records")

    # Return ML warning records without recalculating or retraining models.
    ml_records = ml[
        [
            "entity_id",
            "scenario_id",
            "actual_distress",
            "logistic_regression_probability",
            "random_forest_probability",
            "random_forest_warning_band",
        ]
    ].to_dict(orient="records")

    return {
        "summary": stress_summary,
        "top_risk_entities": top_risk_entities,
        "ml_warnings": ml_records,
        "notes": [
            "Results are based on synthetic financial data.",
            "Systemic risk scores are monitoring-priority indicators, not real-world forecasts.",
            "ML probabilities are not calibrated real-world distress probabilities.",
            "Stress totals are sums of recorded entity-level outputs and may not represent unique system-wide economic losses.",
        ],
    }


@app.get("/api/contagion/scenarios")
def get_scenario_contagion():
    """Return validated contagion propagation results for all scenarios."""
    path = DATA_DIR / "scenario_contagion_results.csv"

    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail="Scenario contagion results are not available. Run the project pipeline first.",
        )

    try:
        df = pd.read_csv(path)

        if df.empty:
            raise HTTPException(
                status_code=503,
                detail="Scenario contagion results are empty.",
            )

        required_columns = {"scenario_id", "entity_id"}
        missing = required_columns - set(df.columns)

        if missing:
            raise HTTPException(
                status_code=500,
                detail=f"Scenario contagion file is missing columns: {sorted(missing)}",
            )

        # Keep every scenario and propagation round; do not combine
        # these incremental contagion results with integrated stress losses.
        return {
            "scenario_count": int(df["scenario_id"].nunique()),
            "row_count": len(df),
            "scenarios": sorted(df["scenario_id"].dropna().unique().tolist()),
            "columns": df.columns.tolist(),
            "results": df.astype(object).where(pd.notna(df), None).to_dict(
                orient="records"
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to load scenario contagion results: {exc}",
        ) from exc

print("\nFINCONTAGION-X dashboard: http://127.0.0.1:8000/dashboard")
print("API documentation:       http://127.0.0.1:8000/docs\n")