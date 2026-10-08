"""FastAPI app. Run: uvicorn app.main:app --reload -> http://localhost:8000"""

import io
import csv
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .engine import Engine
from app.data_quality import clean_and_validate


app, E = FastAPI(
    title="AdPilot — Autonomous D2C Ad Decision Engine"
), Engine()

STATIC = Path(__file__).resolve().parent.parent / "static"


class Scenario(BaseModel):
    scenario: int


class Budget(BaseModel):
    total: float | None = None


class Question(BaseModel):
    question: str


class ExcelCampaign(BaseModel):
    cid: str
    budget: float


def guard(fn, *a):
    try:
        return fn(*a)

    except KeyError:
        raise HTTPException(
            404,
            "Recommendation not found (state may have changed)"
        )

    except PermissionError as ex:
        raise HTTPException(409, str(ex))

    except ValueError as ex:
        raise HTTPException(400, str(ex))


# ---------------------------------------------------------
# CSV Upload
# ---------------------------------------------------------

@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            400,
            "Please upload a CSV file"
        )

    try:
        text = (await file.read()).decode("utf-8-sig")

    except UnicodeDecodeError:
        raise HTTPException(
            400,
            "The file must use UTF-8 text encoding"
        )

    reader = csv.DictReader(io.StringIO(text))

    if not reader.fieldnames:
        raise HTTPException(
            400,
            "The CSV file is empty"
        )

    required = {
        "campaign",
        "spend",
        "revenue",
        "clicks",
        "conversions",
        "inventory",
        "margin",
    }

    headers = {
        header.strip().lower(): header
        for header in reader.fieldnames
        if header
    }

    missing = sorted(required - set(headers))

    if missing:
        raise HTTPException(
            400,
            f"Missing columns: {', '.join(missing)}"
        )

    rows = []

    for row in reader:
        campaign = {
            name: (row.get(headers[name]) or "").strip()
            for name in required
        }

        if any(campaign.values()):
            rows.append(campaign)

    if not rows:
        raise HTTPException(
            400,
            "The CSV has no campaign rows"
        )

    # -----------------------------------------------------
    # CLEAN + VALIDATE + AUTO-CORRECT
    # -----------------------------------------------------

    quality_result = clean_and_validate(rows)

    cleaned_rows = quality_result["data"]
    quality_report = quality_result["validation"]

    # -----------------------------------------------------
    # Reject only unfixable errors
    # -----------------------------------------------------

    if quality_report["error_count"] > 0:

        errors = [
            issue
            for issue in quality_report["issues"]
            if issue["severity"] == "error"
        ]

        warnings = [
            issue
            for issue in quality_report["issues"]
            if issue["severity"] == "warning"
        ]

        raise HTTPException(
            status_code=422,
            detail={
                "message": "Data quality validation failed",
                "errors": errors,
                "warnings": warnings,
                "corrections": quality_report["corrections"],
            }
        )

    # -----------------------------------------------------
    # Successful cleaned upload
    # -----------------------------------------------------

    warnings = [
        issue
        for issue in quality_report["issues"]
        if issue["severity"] == "warning"
    ]

    return {
        "ok": True,
        "filename": file.filename,
        "rows_loaded": len(cleaned_rows),
        "preview": cleaned_rows[:5],

        "data_quality": {
            "valid": quality_report["valid"],
            "errors": quality_report["error_count"],
            "warnings": quality_report["warning_count"],
            "corrections": quality_report["correction_count"],
        },

        "correction_log": quality_report["corrections"],
        "warnings": warnings,
    }


# ---------------------------------------------------------
# Dashboard
# ---------------------------------------------------------

@app.get("/api/dashboard")
def dashboard():
    return E.dashboard()


# ---------------------------------------------------------
# Campaigns
# ---------------------------------------------------------

@app.get("/api/campaigns")
def campaigns():
    return E.campaigns()


# ---------------------------------------------------------
# Excel Update
# ---------------------------------------------------------

@app.post("/api/excel/update")
def excel_update(data: ExcelCampaign):

    cid = data.cid

    if cid not in E.camps.index:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown campaign: {cid}"
        )

    # -----------------------------------------------------
    # Remove entire campaign if budget is negative
    # -----------------------------------------------------

    if data.budget < 0:

        # Remove the complete campaign row
        E.camps = E.camps.drop(index=cid)

        # Recalculate engine state after removing the campaign
        E.refresh()

        return {
            "success": True,
            "campaign": cid,
            "removed": True,
            "new_budget": data.budget,
            "message": (
                f"Campaign {cid} was removed because "
                "its budget was negative"
            )
        }

    # -----------------------------------------------------
    # Normal budget update
    # -----------------------------------------------------

    E.camps.loc[cid, "budget"] = data.budget

    E.refresh()

    return {
        "success": True,
        "campaign": cid,
        "removed": False,
        "new_budget": data.budget,
        "message": "Excel input applied successfully"
    }

# ---------------------------------------------------------
# Products  
# ---------------------------------------------------------

@app.get("/api/products")
def products():
    return E.products()


# ---------------------------------------------------------
# Anomalies
# ---------------------------------------------------------

@app.get("/api/anomalies")
def anomalies():
    return E.anomalies


@app.get("/api/anomalies/{cid}/root-cause")
def root_cause(cid: str):

    if cid not in E.camps.index:
        raise HTTPException(
            404,
            "Unknown campaign"
        )

    return E.rca(cid)


# ---------------------------------------------------------
# Recommendations
# ---------------------------------------------------------

@app.get("/api/recommendations")
def recommendations():
    return E.recs


@app.get("/api/opportunities")
def opportunities():
    return E.opportunities()


@app.post("/api/recommendations/{rid}/approve")
def approve(rid: str):
    return guard(E.approve, rid)


@app.post("/api/recommendations/{rid}/execute")
def execute(rid: str):
    return guard(E.execute, rid)


# ---------------------------------------------------------
# Decisions
# ---------------------------------------------------------

@app.get("/api/decisions")
def decisions():
    return E.decisions


# ---------------------------------------------------------
# Feedback
# ---------------------------------------------------------

@app.get("/api/feedback")
def feedback():
    return {
        "confidence": E.conf,
        "platform_boost": E.boost,
        "outcomes": E.fb
    }


# ---------------------------------------------------------
# Demo Scenarios
# ---------------------------------------------------------

@app.post("/api/demo/scenario")
def scenario(s: Scenario):
    return {
        "scenario": guard(E.scenario, s.scenario)
    }


@app.get("/api/demo/scenarios")
def scenarios():
    return E.SCENARIOS


@app.post("/api/demo/reset")
def reset():
    E.reset()
    return {
        "ok": True
    }


# ---------------------------------------------------------
# Optimizer
# ---------------------------------------------------------

@app.post("/api/optimizer")
def optimizer(b: Budget):
    return E.optimizer(b.total)


# ---------------------------------------------------------
# Assistant
# ---------------------------------------------------------

@app.post("/api/assistant")
def assistant(q: Question):
    return {
        "answer": E.ask(q.question)
    }


# ---------------------------------------------------------
# Frontend
# ---------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(
        STATIC / "index.html"
    )


app.mount(
    "/static",
    StaticFiles(directory=STATIC),
    name="static"
)