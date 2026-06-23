"""
Drift Monitoring Module
Tracks query patterns and embedding drift using Evidently.
Logs queries and generates drift reports for monitoring in production.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently import ColumnMapping

LOGS_PATH = "data/query_logs.jsonl"
REPORTS_PATH = "data/reports"


def log_query(question: str, answer: str, sources: List[Dict], latency_ms: float):
    """
    Log a query and its response for drift monitoring.
    Appends to a JSONL file for streaming log analysis.
    """
    os.makedirs("data", exist_ok=True)
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": question,
        "question_length": len(question),
        "word_count": len(question.split()),
        "answer_length": len(answer),
        "num_sources": len(sources),
        "latency_ms": latency_ms,
        "sources": [s["file"] for s in sources]
    }

    with open(LOGS_PATH, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def load_query_logs() -> pd.DataFrame:
    """Load query logs into a DataFrame for analysis."""
    if not os.path.exists(LOGS_PATH):
        return pd.DataFrame()

    logs = []
    with open(LOGS_PATH, "r") as f:
        for line in f:
            logs.append(json.loads(line.strip()))

    return pd.DataFrame(logs)


def generate_drift_report(reference_size: int = 50) -> Optional[str]:
    """
    Generate an Evidently data drift report comparing:
    - Reference: first N queries (baseline)
    - Current: most recent N queries

    Returns path to the HTML report.
    """
    df = load_query_logs()

    if len(df) < reference_size * 2:
        print(f"⚠️  Not enough queries for drift analysis. Need {reference_size * 2}, have {len(df)}")
        return None

    # Split into reference and current windows
    reference = df.iloc[:reference_size][["question_length", "word_count", "answer_length", "latency_ms", "num_sources"]]
    current = df.iloc[-reference_size:][["question_length", "word_count", "answer_length", "latency_ms", "num_sources"]]

    # Generate Evidently report
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=reference, current_data=current)

    # Save report
    os.makedirs(REPORTS_PATH, exist_ok=True)
    report_path = f"{REPORTS_PATH}/drift_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html"
    report.save_html(report_path)

    print(f"✅ Drift report saved to {report_path}")
    return report_path


def get_query_stats() -> Dict:
    """Return summary statistics about logged queries."""
    df = load_query_logs()

    if df.empty:
        return {"total_queries": 0}

    return {
        "total_queries": len(df),
        "avg_latency_ms": round(df["latency_ms"].mean(), 2),
        "avg_question_length": round(df["question_length"].mean(), 2),
        "avg_answer_length": round(df["answer_length"].mean(), 2),
        "most_queried_source": (
            pd.Series([s for sources in df["sources"] for s in sources]).value_counts().index[0]
            if not df.empty else "N/A"
        )
    }


if __name__ == "__main__":
    stats = get_query_stats()
    print("📊 Query Stats:", json.dumps(stats, indent=2))
