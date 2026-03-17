import json
import os
from typing import Optional
from pydantic import BaseModel, Field
from agents import Agent, Runner
from langfuse import get_client
from dotenv import load_dotenv
import asyncio

load_dotenv(override=True)

langfuse = get_client()

# ─────────────────────────────────────────────
# Load configs
# ─────────────────────────────────────────────
config_path = os.path.join(os.path.dirname(__file__), '../schemas/sensor_config.json')
with open(config_path) as f:
    SENSOR_CONFIG = json.load(f)

baselines_path = os.path.join(os.path.dirname(__file__), '../schemas/sensor_baselines.json')
with open(baselines_path) as f:
    BASELINES = json.load(f)

# ─────────────────────────────────────────────
# Structured Output Schema (Pydantic)
# ─────────────────────────────────────────────
class SensorAnomaly(BaseModel):
    sensor_name: str = Field(description="Name of the anomalous sensor")
    current_value: float = Field(description="Current sensor reading")
    baseline_mean: float = Field(description="Expected healthy baseline mean")
    deviation_std: float = Field(description="How many standard deviations from baseline")
    severity: str = Field(description="LOW, MEDIUM, or HIGH")

class AnomalyDetectionResult(BaseModel):
    flagged_sensors: list[SensorAnomaly] = Field(
        description="List of sensors with anomalous readings"
    )
    overall_severity: str = Field(
        description="Overall severity: NONE, LOW, MEDIUM, or HIGH"
    )
    pattern_description: str = Field(
        description="Description of the overall degradation pattern observed"
    )
    anomaly_count: int = Field(
        description="Total number of anomalous sensors detected"
    )


# ─────────────────────────────────────────────
# Step 1 — Statistical Pre-filter
# ─────────────────────────────────────────────
def statistical_prefilter(
    sensor_readings: dict,
    std_threshold: float = 2.0
) -> list[dict]:
    """Flag sensors deviating more than std_threshold standard deviations from baseline."""
    flagged = []
    rising = SENSOR_CONFIG["degradation_sensors"]["rising"]
    falling = SENSOR_CONFIG["degradation_sensors"]["falling"]

    for sensor in SENSOR_CONFIG["useful_sensors"]:
        if sensor not in sensor_readings or sensor not in BASELINES:
            continue

        value = sensor_readings[sensor]
        baseline = BASELINES[sensor]
        mean = baseline["mean"]
        std = baseline["std"]

        if std == 0:
            continue

        deviation = (value - mean) / std

        # For rising sensors: flag if significantly above baseline
        # For falling sensors: flag if significantly below baseline
        is_anomalous = False
        if sensor in rising and deviation > std_threshold:
            is_anomalous = True
        elif sensor in falling and deviation < -std_threshold:
            is_anomalous = True

        if is_anomalous:
            # Determine severity based on deviation magnitude
            abs_dev = abs(deviation)
            if abs_dev > 5:
                severity = "HIGH"
            elif abs_dev > 3:
                severity = "MEDIUM"
            else:
                severity = "LOW"

            flagged.append({
                "sensor": sensor,
                "value": round(value, 4),
                "mean": mean,
                "std": std,
                "deviation": round(deviation, 2),
                "severity": severity
            })

    return flagged


# ─────────────────────────────────────────────
# Step 2 — OpenAI Agents SDK LLM Reasoning
# ─────────────────────────────────────────────
def build_agent_prompt(
    sensor_readings: dict,
    flagged_sensors: list[dict],
    rul: int
) -> str:
    """Build the prompt for the anomaly detection agent."""
    rising = SENSOR_CONFIG["degradation_sensors"]["rising"]
    falling = SENSOR_CONFIG["degradation_sensors"]["falling"]

    flagged_text = ""
    if flagged_sensors:
        flagged_text = "\n## Statistically Flagged Sensors\n"
        for f in flagged_sensors:
            direction = "↑ rising" if f["sensor"] in rising else "↓ falling"
            flagged_text += (
                f"- {f['sensor']}: value={f['value']}, "
                f"baseline={f['mean']} ± {f['std']}, "
                f"deviation={f['deviation']}σ, "
                f"severity={f['severity']} ({direction})\n"
            )
    else:
        flagged_text = "\n## Statistically Flagged Sensors\nNone flagged by statistical filter.\n"

    readings_text = "\n## All Useful Sensor Readings\n"
    for sensor in SENSOR_CONFIG["useful_sensors"]:
        if sensor in sensor_readings:
            direction = "↑ rising" if sensor in rising else "↓ falling"
            baseline = BASELINES.get(sensor, {})
            readings_text += (
                f"- {sensor}: {sensor_readings[sensor]:.4f} "
                f"(baseline: {baseline.get('mean', 'N/A')}, pattern: {direction})\n"
            )

    return f"""You are an expert anomaly detection engineer for industrial turbofan engines.

**Engine Status:**
- RUL: {rul} cycles remaining
- Thresholds: Critical < {SENSOR_CONFIG['rul_thresholds']['critical_below']}, 
  Warning < {SENSOR_CONFIG['rul_thresholds']['warning_above']}

{flagged_text}
{readings_text}

Analyze the sensor readings and the statistically flagged sensors.
Consider combinations of sensor anomalies — sometimes a pattern of multiple 
sensors together is more significant than individual flags.

Return a structured anomaly detection result with:
- All flagged sensors with their severity
- Overall severity assessment
- A clear pattern description explaining what the sensor combination suggests
  about the engine's degradation state
"""


# ─────────────────────────────────────────────
# Main anomaly detection function
# ─────────────────────────────────────────────
def run_anomaly_detection(
    engine_id: int,
    cycle: int,
    sensor_readings: dict,
    rul: int
) -> dict:
    """Run statistical pre-filter then LLM reasoning for anomaly detection."""

    with langfuse.start_as_current_observation(
        as_type="span",
        name="anomaly_detection"
    ) as span:
        span.update(input={
            "engine_id": engine_id,
            "cycle": cycle,
            "rul": rul
        })

        # ── Step 1: Statistical pre-filter ──
        with langfuse.start_as_current_observation(
            as_type="span",
            name="statistical_prefilter"
        ) as stats_span:
            flagged = statistical_prefilter(sensor_readings)
            stats_span.update(output={
                "flagged_count": len(flagged),
                "flagged_sensors": [f["sensor"] for f in flagged]
            })
            print(f"Statistical filter: {len(flagged)} sensors flagged")

        # ── Step 2: LLM reasoning ──
        with langfuse.start_as_current_observation(
            as_type="span",
            name="llm_anomaly_reasoning"
        ) as llm_span:
            prompt = build_agent_prompt(sensor_readings, flagged, rul)

            anomaly_agent = Agent(
                name="AnomalyDetector",
                instructions="""You are an expert anomaly detection engineer 
                for industrial turbofan engines. Analyze sensor readings and 
                identify degradation patterns. Always return structured output.""",
                output_type=AnomalyDetectionResult,
                model="gpt-4o-mini"
            )

            
            result = asyncio.run(
                Runner.run(anomaly_agent, prompt)
            )

            detection: AnomalyDetectionResult = result.final_output

            llm_span.update(output={
                "overall_severity": detection.overall_severity,
                "anomaly_count": detection.anomaly_count,
                "pattern": detection.pattern_description
            })
            print(f"LLM reasoning: {detection.anomaly_count} anomalies, severity={detection.overall_severity}")

        # Build summary for pipeline state
        anomaly_names = [a.sensor_name for a in detection.flagged_sensors]
        summary = f"**Overall Severity:** {detection.overall_severity}\n\n"
        summary += f"**Pattern:** {detection.pattern_description}\n\n"
        if detection.flagged_sensors:
            summary += "**Flagged Sensors:**\n"
            for a in detection.flagged_sensors:
                summary += f"- {a.sensor_name}: {a.severity} severity ({a.deviation_std:.1f}σ from baseline)\n"

        span.update(output={
            "anomalies": anomaly_names,
            "overall_severity": detection.overall_severity
        })
        langfuse.flush()

    return {
        "anomalies": anomaly_names,
        "anomaly_summary": summary,
        "overall_severity": detection.overall_severity,
        "pattern_description": detection.pattern_description
    }