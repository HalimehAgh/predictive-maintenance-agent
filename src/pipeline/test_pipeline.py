import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import pandas as pd
from pipeline import pipeline
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from schemas.pipeline_state import RiskLevel
from dotenv import load_dotenv

load_dotenv(override=True)

langfuse= Langfuse()
langfuse_handler = CallbackHandler()

# Load a real engine snapshot from the dataset
columns = ['unit', 'cycle', 'os1', 'os2', 'os3'] + \
          [f'sensor_{i}' for i in range(1, 22)]
train = pd.read_csv('data/train_FD001.txt', sep=r'\s+', header=None, names=columns)

# Pick engine 1 at its last cycle
engine_1 = train[train['unit'] == 1]
snapshot = engine_1.iloc[-50]  

# Compute RUL from actual data
max_cycle_engine1 = engine_1['cycle'].max()
current_cycle = int(snapshot['cycle'])
rul_actual = max_cycle_engine1 - current_cycle

print(f"Engine 1 — Max cycle: {max_cycle_engine1}, Current cycle: {current_cycle}, RUL: {rul_actual}")

sensor_readings = {f'sensor_{i}': snapshot[f'sensor_{i}'] for i in range(1, 22)}

# Run the pipeline
result = pipeline.invoke({
    "engine_id": 1,
    "cycle": int(snapshot['cycle']),
    "sensor_readings": sensor_readings,
    "rul_actual": rul_actual,
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

config = {"callbacks": [langfuse_handler]}
)
langfuse.flush()
print("\n" + "="*50)
print(result["final_report"])