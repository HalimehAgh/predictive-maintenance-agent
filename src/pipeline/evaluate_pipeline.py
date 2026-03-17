import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import pandas as pd
import json
import time
from datetime import datetime
from pipeline import pipeline
from schemas.pipeline_state import RiskLevel
from langfuse.langchain import CallbackHandler
from langfuse import Langfuse
from dotenv import load_dotenv

load_dotenv(override=True)

langfuse = Langfuse()

# ─────────────────────────────────────────────
# Load dataset
# ─────────────────────────────────────────────
columns = ['unit', 'cycle', 'os1', 'os2', 'os3'] + \
          [f'sensor_{i}' for i in range(1, 22)]
train = pd.read_csv('data/train_FD001.txt', sep=r'\s+', header=None, names=columns)

def get_snapshot(engine_id: int, iloc_index: int) -> dict:
    """Get a sensor snapshot for a given engine at a given position."""
    engine_data = train[train['unit'] == engine_id]
    snapshot = engine_data.iloc[iloc_index]
    max_cycle = engine_data['cycle'].max()
    current_cycle = int(snapshot['cycle'])
    rul = max_cycle - current_cycle
    sensor_readings = {f'sensor_{i}': snapshot[f'sensor_{i}'] for i in range(1, 22)}
    return {
        "engine_id": engine_id,
        "cycle": current_cycle,
        "sensor_readings": sensor_readings,
        "rul_actual": int(rul)
    }

def run_evaluation_case(engine_id: int, iloc_index: int, expected_status: str) -> dict:
    """Run a single evaluation case and return results."""
    print(f"\n{'='*60}")
    print(f"Testing Engine {engine_id} at iloc[{iloc_index}] — Expected: {expected_status}")
    print(f"{'='*60}")

    snapshot = get_snapshot(engine_id, iloc_index)
    print(f"RUL: {snapshot['rul_actual']} cycles")

    langfuse_handler = CallbackHandler()
    start_time = time.time()

    try:
        result = pipeline.invoke(
            {
                "engine_id": snapshot["engine_id"],
                "cycle": snapshot["cycle"],
                "sensor_readings": snapshot["sensor_readings"],
                "rul_actual": snapshot["rul_actual"],
                "is_valid": False,
                "validation_errors": [],
                "anomalies": [],
                "anomaly_summary": "",
                "debate_transcript": "",
                "diagnosis": "",
                "diagnosis_confidence": "",
                "maintenance_plan": "",
                "risk_level": RiskLevel.UNKNOWN,
                "final_report": "",
                "alert_operator": False
            },
            config={"callbacks": [langfuse_handler]}
        )
        langfuse.flush()

        elapsed = time.time() - start_time
        actual_status = result["risk_level"].value

        evaluation = {
            "engine_id": engine_id,
            "iloc_index": iloc_index,
            "rul_actual": snapshot["rul_actual"],
            "expected_status": expected_status,
            "actual_status": actual_status,
            "correct": actual_status == expected_status,
            "anomaly_count": len(result["anomalies"]),
            "anomaly_severity": result["anomaly_summary"].split("**Overall Severity:**")[1].split("\n")[0].strip() if "**Overall Severity:**" in result["anomaly_summary"] else "N/A",
            "diagnosis_confidence": result["diagnosis_confidence"],
            "alert_operator": result["alert_operator"],
            "elapsed_seconds": round(elapsed, 2)
        }

        status_icon = "✅" if evaluation["correct"] else "❌"
        print(f"{status_icon} Expected: {expected_status}, Got: {actual_status}, Time: {elapsed:.1f}s")

        return evaluation

    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        print(traceback.format_exc())
        return {
            "engine_id": engine_id,
            "iloc_index": iloc_index,
            "rul_actual": snapshot["rul_actual"],
            "expected_status": expected_status,
            "actual_status": "ERROR",
            "correct": False,
            "error": str(e),
            "elapsed_seconds": round(time.time() - start_time, 2)
        }

# ─────────────────────────────────────────────
# Define test cases
# ─────────────────────────────────────────────
test_cases = [
    # Engine 1
    (1,  -20, "CRITICAL"),
    (1,  -50, "WARNING"),
    (1,   10, "HEALTHY"),
    # Engine 5
    (5,  -20, "CRITICAL"),
    (5,  -50, "WARNING"),
    (5,   10, "HEALTHY"),
    # Engine 10
    (10, -20, "CRITICAL"),
    (10, -50, "WARNING"),
    (10,  10, "HEALTHY"),
]


# ─────────────────────────────────────────────
# Run evaluation
# ─────────────────────────────────────────────
print("Starting pipeline evaluation...")
print(f"Total test cases: {len(test_cases)}")
print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")

results = []
for engine_id, iloc_index, expected_status in test_cases:
    result = run_evaluation_case(engine_id, iloc_index, expected_status)
    results.append(result)

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
print(f"\n{'='*60}")
print("EVALUATION SUMMARY")
print(f"{'='*60}")

correct = sum(1 for r in results if r.get("correct", False))
total = len(results)
accuracy = correct / total * 100

print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")
print(f"Average time per case: {sum(r['elapsed_seconds'] for r in results)/total:.1f}s")

print(f"\nResults by status:")
for status in ["CRITICAL", "WARNING", "HEALTHY"]:
    status_results = [r for r in results if r["expected_status"] == status]
    status_correct = sum(1 for r in status_results if r.get("correct", False))
    print(f"  {status}: {status_correct}/{len(status_results)} correct")

print(f"\nDetailed Results:")
for r in results:
    icon = "✅" if r.get("correct") else "❌"
    print(f"  {icon} Engine {r['engine_id']} iloc[{r['iloc_index']}]: "
          f"RUL={r['rul_actual']}, "
          f"Expected={r['expected_status']}, "
          f"Got={r['actual_status']}, "
          f"Time={r['elapsed_seconds']}s")

# Save results
results_path = 'docs/evaluation_results.json'
with open(results_path, 'w') as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results
    }, f, indent=2)

print(f"\nResults saved to {results_path}")
langfuse.flush()