from typing import TypedDict, Optional, List
from enum import Enum

class RiskLevel(Enum):
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"

class PipelineState(TypedDict):
    # --- Input ---
    engine_id: int
    cycle: int
    sensor_readings: dict[str, float]
    rul_actual: Optional[int]        # known RUL, used only for evaluation

    # --- Validation ---
    is_valid: bool
    validation_errors: List[str]

    # --- Anomaly Detection ---
    anomalies: List[str]             # list of flagged sensor names
    anomaly_summary: str             # human readable summary

    # --- Diagnosis ---
    diagnosis: str                   # confirmed fault description
    diagnosis_confidence: str        # HIGH / MEDIUM / LOW
    debate_transcript: str  # full debate transcript between mechanical and electrical agents

    # --- Maintenance Plan ---
    maintenance_plan: str            # recommended actions
    risk_level: RiskLevel            # HEALTHY / WARNING / CRITICAL

    # --- Output ---
    final_report: str                # markdown report for operator
    alert_operator: bool             # should operator be alerted?