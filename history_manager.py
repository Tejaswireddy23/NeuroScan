"""
NeuroScan Analysis History Manager
Provides persistent JSON-based history management with unique Analysis IDs,
full record retrieval, search, comparison, and deletion.
"""

import os
import json
import uuid
from datetime import datetime

DATA_DIR = "data"
HISTORY_FILE = os.path.join(DATA_DIR, "history.json")
os.makedirs(DATA_DIR, exist_ok=True)

def generate_analysis_id() -> str:
    """Generates a unique, professional clinical Analysis ID: NS-YYYYMMDD-XXXXXX"""
    date_str = datetime.now().strftime("%Y%m%d")
    short_uuid = uuid.uuid4().hex[:6].upper()
    return f"NS-{date_str}-{short_uuid}"

def load_history() -> list:
    """Loads all historical analysis records ordered newest first."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except Exception:
        return []

def save_history(records: list):
    """Saves records safely to disk."""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

def add_analysis_record(record: dict) -> dict:
    """Adds a new analysis record to history."""
    records = load_history()
    records.insert(0, record)  # Newest first
    # Keep up to 100 recent analyses
    if len(records) > 100:
        records = records[:100]
    save_history(records)
    return record

def get_analysis_by_id(analysis_id: str) -> dict:
    """Retrieves a single record by its Analysis ID."""
    records = load_history()
    for rec in records:
        if rec.get("analysis_id") == analysis_id or rec.get("id") == analysis_id:
            return rec
    return None

def delete_analysis_by_id(analysis_id: str) -> bool:
    """Deletes an analysis record by its ID."""
    records = load_history()
    initial_len = len(records)
    records = [r for r in records if r.get("analysis_id") != analysis_id and r.get("id") != analysis_id]
    if len(records) < initial_len:
        save_history(records)
        return True
    return False
