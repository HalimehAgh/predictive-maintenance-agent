import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent_modules.diagnosis_agents import run_diagnosis
from typing import Literal
from langgraph.graph import StateGraph, END
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from schemas.pipeline_state import PipelineState, RiskLevel
from agent_modules.knowledge_crew import run_knowledge_crew
from dotenv import load_dotenv
from agent_modules.anomaly_detector import run_anomaly_detection

load_dotenv(override=True)

langfuse= Langfuse()
langfuse_handler = CallbackHandler()
# Load sensor config
config_path = os.path.join(os.path.dirname(__file__), '../schemas/sensor_config.json')
with open(config_path) as f:
    SENSOR_CONFIG = json.load(f)

# ─────────────────────────────────────────────
# NODE 1 — Input Validator
# ─────────────────────────────────────────────
def input_validator(state: PipelineState) -> PipelineState:
    print("[Node 1] Validating input...")
    errors = []
    readings = state["sensor_readings"]

    # Check all useful sensors are present
    for sensor in SENSOR_CONFIG["useful_sensors"]:
        if sensor not in readings:
            errors.append(f"Missing sensor: {sensor}")

    # Check no NaN or None values
    for sensor, value in readings.items():
        if value is None or (isinstance(value, float) and value != value):
            errors.append(f"Invalid value for {sensor}: {value}")

    state["is_valid"] = len(errors) == 0
    state["validation_errors"] = errors

    if not state["is_valid"]:
        print(f"  ❌ Validation failed: {errors}")
    else:
        print("  ✅ Validation passed")

    return state

# ─────────────────────────────────────────────
# NODE 2 — Anomaly Detector (placeholder)
# ─────────────────────────────────────────────
def anomaly_detector(state: PipelineState) -> PipelineState:
    print("[Node 2] Detecting anomalies...")
    result = run_anomaly_detection(
        engine_id=state["engine_id"],
        cycle=state["cycle"],
        sensor_readings=state["sensor_readings"],
        rul=state.get("rul_actual", 0)
    )
    state["anomalies"] = result["anomalies"]
    state["anomaly_summary"] = result["anomaly_summary"]
    print(f"  ✅ {len(result['anomalies'])} anomalies detected, severity={result['overall_severity']}")
    return state

# ─────────────────────────────────────────────
# NODE 3 — Diagnosis Debate
# ─────────────────────────────────────────────
def diagnosis_debate(state: PipelineState) -> PipelineState:
    print("[Node 3] Running diagnosis debate...")
    result = run_diagnosis(
        engine_id=state["engine_id"],
        cycle=state["cycle"],
        sensor_readings=state["sensor_readings"],
        anomalies=state["anomalies"],
        rul_actual=state.get("rul_actual", 0)
    )
    state["diagnosis"] = f"{result['diagnosis']} | Next step: {result['recommended_next_step']}"
    state["diagnosis_confidence"] = result["confidence"]
    state["debate_transcript"] = result["debate_transcript"]
    print(f"  ✅ Diagnosis: {result['diagnosis']} (Confidence: {result['confidence']})")
    return state

# ─────────────────────────────────────────────
# NODE 4 — Knowledge & Risk Crew
# ─────────────────────────────────────────────
def knowledge_crew(state: PipelineState) -> PipelineState:
    print("[Node 4] Running knowledge crew...")
    maintenance_plan = run_knowledge_crew(
        engine_id=state["engine_id"],
        cycle=state["cycle"],
        rul=state.get("rul_actual", 0),
        diagnosis=state["diagnosis"],
        risk_level=state["risk_level"].value,
        anomalies=state["anomalies"]
    )
    state["maintenance_plan"] = maintenance_plan
    print("  ✅ Maintenance plan generated")
    return state

# ─────────────────────────────────────────────
# NODE 5 — Guardrail Check
# ─────────────────────────────────────────────
def guardrail_check(state: PipelineState) -> PipelineState:
    print("[Node 5] Running guardrail check...")
    rul = state.get("rul_actual")
    warning_threshold = SENSOR_CONFIG["rul_thresholds"]["healthy_above"]
    critical_threshold = SENSOR_CONFIG["rul_thresholds"]["critical_below"]

    if rul is not None:
        if rul <= critical_threshold:
            state["risk_level"] = RiskLevel.CRITICAL
            state["alert_operator"] = True
            print("  🔴 CRITICAL — operator alert forced by guardrail")
        elif rul <= warning_threshold:
            state["risk_level"] = RiskLevel.WARNING
            state["alert_operator"] = True
            print("  🟡 WARNING — operator alert triggered")
        else:
            state["risk_level"] = RiskLevel.HEALTHY
            state["alert_operator"] = False
            print("  🟢 HEALTHY — no alert needed")

    return state

# ─────────────────────────────────────────────
# NODE 6 — Report Generator
# ─────────────────────────────────────────────
def report_generator(state: PipelineState) -> PipelineState:
    print("[Node 6] Generating report...")
    risk = state["risk_level"].value
    report = f"""
# Predictive Maintenance Report
**Engine ID:** {state['engine_id']}
**Cycle:** {state['cycle']}
**Risk Level:** {risk}
**RUL:** {state.get('rul_actual', 'Unknown')} cycles

## Anomalies Detected
{state['anomaly_summary']}

## Agent Debate Transcript
{state.get('debate_transcript', 'Not available')}

## Diagnosis
{state['diagnosis']} (Confidence: {state['diagnosis_confidence']})

## Maintenance Plan
{state['maintenance_plan']}

## Action Required
{'⚠️ ALERT OPERATOR IMMEDIATELY' if state['alert_operator'] else '✅ Continue monitoring'}
"""
    state["final_report"] = report
    print("  ✅ Report generated")
    return state

# ─────────────────────────────────────────────
# CONDITIONAL EDGE — after input validation
# ─────────────────────────────────────────────
def should_continue(state: PipelineState) -> Literal["anomaly_detector", "report_generator"]:
    if not state["is_valid"]:
        print("Skipping to report due to validation failure")
        return "report_generator"
    return "anomaly_detector"

# ─────────────────────────────────────────────
# BUILD THE GRAPH
# ─────────────────────────────────────────────
def build_pipeline() -> StateGraph:
    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("input_validator", input_validator)
    graph.add_node("anomaly_detector", anomaly_detector)
    graph.add_node("diagnosis_debate", diagnosis_debate)
    graph.add_node("knowledge_crew", knowledge_crew)
    graph.add_node("guardrail_check", guardrail_check)
    graph.add_node("report_generator", report_generator)

    # Entry point
    graph.set_entry_point("input_validator")

    # Edges
    graph.add_conditional_edges("input_validator", should_continue)
    graph.add_edge("anomaly_detector", "diagnosis_debate")
    graph.add_edge("diagnosis_debate", "knowledge_crew")
    graph.add_edge("knowledge_crew", "guardrail_check")
    graph.add_edge("guardrail_check", "report_generator")
    graph.add_edge("report_generator", END)

    return graph.compile()

pipeline = build_pipeline()