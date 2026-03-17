import asyncio
import json
import os
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from langfuse import get_client
from dotenv import load_dotenv

load_dotenv(override=True)

# Initialize Langfuse
langfuse = get_client()

# Load sensor config
config_path = os.path.join(os.path.dirname(__file__), '../schemas/sensor_config.json')
with open(config_path) as f:
    SENSOR_CONFIG = json.load(f)


def build_sensor_context(sensor_readings: dict, anomalies: list) -> str:
    """Build a human readable sensor context string for agents."""
    rising = SENSOR_CONFIG["degradation_sensors"]["rising"]
    falling = SENSOR_CONFIG["degradation_sensors"]["falling"]

    lines = ["## Current Sensor Readings\n"]
    for sensor, value in sensor_readings.items():
        if sensor in SENSOR_CONFIG["useful_sensors"]:
            direction = "↑ rising" if sensor in rising else "↓ falling"
            flagged = "⚠️ ANOMALY" if sensor in anomalies else ""
            lines.append(f"  - {sensor}: {value:.4f} (degradation pattern: {direction}) {flagged}")

    if anomalies:
        lines.append(f"\n## Flagged Anomalies\n  {', '.join(anomalies)}")
    else:
        lines.append("\n## No anomalies flagged yet")

    return "\n".join(lines)


async def run_diagnosis_debate(
    engine_id: int,
    cycle: int,
    sensor_readings: dict,
    anomalies: list,
    rul_actual: int
) -> dict:
    """Run a multi-round AutoGen debate between mechanical and electrical agents."""

    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
    sensor_context = build_sensor_context(sensor_readings, anomalies)

    # ─────────────────────────────────────────────
    # Define the three agents
    # ─────────────────────────────────────────────
    mechanical_agent = AssistantAgent(
        name="MechanicalAgent",
        model_client=model_client,
        system_message="""You are a senior mechanical engineer specializing in turbofan engine diagnostics.
You analyze sensor patterns related to physical wear, pressure, and temperature degradation.
Key sensors you focus on: sensor_3 (LPC outlet temp), sensor_4 (HPC outlet temp), 
sensor_9 (physical fan speed), sensor_14 (corrected fan speed), sensor_17 (bleed enthalpy).
Rising temperature and pressure sensors indicate HPC or LPC degradation.
Be specific about which sensors concern you and why.
Keep your response concise — 3 to 5 sentences maximum."""
    )

    electrical_agent = AssistantAgent(
        name="ElectricalAgent",
        model_client=model_client,
        system_message="""You are a senior electrical and control systems engineer specializing in turbofan engines.
You analyze sensor patterns related to efficiency, speed ratios, and control system performance.
Key sensors you focus on: sensor_7 (fan efficiency), sensor_11 (corrected core speed), 
sensor_12 (bypass ratio), sensor_20 (fuel flow ratio), sensor_21 (efficiency metric).
Falling efficiency sensors combined with rising speed sensors indicate control system stress.
Be specific about which sensors concern you and why.
Keep your response concise — 3 to 5 sentences maximum."""
    )

    judge_agent = AssistantAgent(
        name="JudgeAgent",
        model_client=model_client,
        system_message="""You are a chief diagnostic engineer making the final maintenance decision.
You have read the debate between the mechanical and electrical engineers.
Your job is to synthesize their findings into a final verdict.
You MUST respond in this exact JSON format:
{
  "diagnosis": "brief description of the confirmed fault",
  "confidence": "HIGH or MEDIUM or LOW",
  "recommended_next_step": "specific action the operator should take"
}
Do not include any text outside the JSON."""
    )

    # ─────────────────────────────────────────────
    # Debate task
    # ─────────────────────────────────────────────
    debate_task = f"""
## Engine Diagnostic Request
**Engine ID:** {engine_id}
**Current Cycle:** {cycle}
**RUL:** {rul_actual} cycles remaining

{sensor_context}

Please analyze the sensor readings and debate the likely fault mode.
Mechanical engineer goes first, then electrical engineer responds.
Focus on what the sensor patterns tell you about engine health.
"""

    debate_team = RoundRobinGroupChat(
        participants=[mechanical_agent, electrical_agent],
        termination_condition=MaxMessageTermination(max_messages=5)
    )

    # ─────────────────────────────────────────────
    # Run with Langfuse tracing
    # ─────────────────────────────────────────────
    with langfuse.start_as_current_observation(
        as_type="span",
        name="diagnosis_debate",
    ) as trace:
        trace.update(input={
            "engine_id": engine_id,
            "cycle": cycle,
            "rul_actual": rul_actual,
            "anomalies": anomalies
        })

        # ── Debate span ──
        with langfuse.start_as_current_observation(
            as_type="span",
            name="mechanical_electrical_debate"
        ) as debate_span:
            print("Starting mechanical vs electrical debate...")
            debate_result = await debate_team.run(task=debate_task)
            debate_span.update(output={"message_count": len(debate_result.messages)})

        # Collect the full debate transcript
        debate_transcript = "\n\n".join([
            f"**{msg.source}:** {msg.content}"
            for msg in debate_result.messages
            if hasattr(msg, 'source') and msg.source != "user"
        ])

        # ── Judge span ──
        judge_prompt = f"""
## Debate Transcript
{debate_transcript}

Based on this debate, provide your final diagnostic verdict in the required JSON format.
"""
        with langfuse.start_as_current_observation(
            as_type="span",
            name="judge_verdict"
        ) as judge_span:
            print("Judge synthesizing verdict...")
            judge_response = await judge_agent.on_messages(
                [TextMessage(content=judge_prompt, source="user")],
                CancellationToken()
            )
            try:
                verdict = json.loads(judge_response.chat_message.content)
            except json.JSONDecodeError:
                verdict = {
                    "diagnosis": judge_response.chat_message.content,
                    "confidence": "LOW",
                    "recommended_next_step": "Manual inspection required"
                }
            judge_span.update(output=verdict)

        trace.update(output=verdict)

    langfuse.flush()

    return {
        "debate_transcript": debate_transcript,
        "diagnosis": verdict.get("diagnosis", "Unknown"),
        "confidence": verdict.get("confidence", "LOW"),
        "recommended_next_step": verdict.get("recommended_next_step", "Manual inspection required")
    }


def run_diagnosis(
    engine_id: int,
    cycle: int,
    sensor_readings: dict,
    anomalies: list,
    rul_actual: int
) -> dict:
    """Synchronous wrapper for the async debate function."""
    return asyncio.run(run_diagnosis_debate(
        engine_id, cycle, sensor_readings, anomalies, rul_actual
    ))