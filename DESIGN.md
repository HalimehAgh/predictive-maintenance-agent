# System Design Document

## Problem Statement

Industrial turbofan engines degrade gradually over time. Operators managing large 
fleets cannot manually monitor every sensor of every engine every cycle. This system 
provides an agentic AI solution that autonomously monitors engine health, diagnoses 
faults, retrieves maintenance procedures, and alerts operators when action is required.

## Dataset

**NASA CMAPSS FD001** — Turbofan Engine Degradation Simulation Dataset
- 100 engines, single operating condition, single fault mode (HPC Degradation)
- 21 sensors per engine, sampled every operational cycle
- Ground truth RUL available for evaluation

### Key EDA Findings
- 7 sensors are constant and carry no degradation signal (dropped)
- 14 sensors show meaningful variance
- All 14 useful sensors show clear degradation trends (rising or falling)
- Degradation accelerates non-linearly in the last ~20% of engine life
- Data-driven RUL thresholds: Critical < 30 cycles, Warning < 80 cycles

## Architecture
```
Input: Engine sensor snapshot
            ↓
    [LangGraph Pipeline]
    ├── Node 1: Input Validator (guardrail)
    ├── Node 2: Anomaly Detector (OpenAI Agents SDK)
    │           ├── Statistical pre-filter (2σ threshold)
    │           └── LLM reasoning (structured Pydantic output)
    ├── Node 3: Diagnosis Debate (AutoGen)
    │           ├── Mechanical Engineer Agent
    │           ├── Electrical Engineer Agent
    │           └── Judge Agent → structured verdict
    ├── Node 4: Knowledge Crew (CrewAI + MCP)
    │           ├── Maintenance Researcher (Filesystem MCP + Brave Search MCP)
    │           ├── Risk Assessment Specialist
    │           └── Report Writer
    ├── Node 5: Guardrail Check
    │           └── RUL-based risk classification + operator alert forcing
    └── Node 6: Report Generator
                └── Structured Markdown report
            ↓
Output: Maintenance report + operator alert
```

## Framework Selection Rationale

### LangGraph — Orchestration Layer
**Why:** LangGraph provides stateful, conditional pipeline orchestration with 
explicit node and edge definitions. For an industrial system, predictability 
and auditability are critical — LangGraph's graph-based approach makes the 
control flow explicit and inspectable, unlike fully autonomous agent systems.

### AutoGen — Multi-Agent Diagnosis
**Why:** AutoGen's RoundRobinGroupChat enables structured multi-round debates 
between specialized agents. The mechanical vs electrical debate pattern naturally 
maps to how real engineering teams diagnose complex faults — no single expert 
has complete knowledge, and debate surfaces better diagnoses.

### CrewAI — Knowledge & Maintenance Crew
**Why:** CrewAI's role-based crew model with sequential task execution is ideal 
for the research → assess → report workflow. Each crew member has a clear 
responsibility and passes findings to the next, mirroring real maintenance teams.

### OpenAI Agents SDK — Anomaly Detection
**Why:** The SDK's native support for structured outputs (Pydantic models) and 
its lightweight design make it ideal for the anomaly detection node, which 
requires deterministic, schema-enforced outputs to feed into downstream agents.

## MCP Integration

Two MCP servers are used by the Maintenance Researcher agent:

| Server | Purpose | Fallback |
|--------|---------|---------|
| Filesystem MCP | Read local maintenance manuals | — |
| Brave Search MCP | Search web for additional procedures | Used if local manuals insufficient |

The filesystem-first, web-search-fallback pattern ensures:
- Fast responses when local knowledge is sufficient
- Comprehensive coverage for edge cases
- Reduced API costs in production

## Guardrails

| Guardrail | Type | Implementation |
|-----------|------|---------------|
| Input validation | Structural | Check all 14 useful sensors present, no null values |
| RUL threshold enforcement | Safety | Force CRITICAL alert if RUL ≤ 30, override agent recommendations |
| Structured output enforcement | Schema | Pydantic models for anomaly detection and judge verdict |
| Hallucination mitigation | Grounding | Agents receive actual sensor values and baselines, not abstractions |

## Observability

All agent interactions are traced via **Langfuse**:
- Full LangGraph pipeline trace with per-node latency
- AutoGen debate transcript captured as nested spans
- CrewAI crew execution tracked
- OpenAI Agents SDK anomaly detection traced

## Evaluation

Pipeline evaluated on 9 test cases across 3 engines × 3 risk zones:
- **Accuracy: 9/9 (100%)**
- Average pipeline latency: ~99 seconds per run
- Consistent performance across different engines and risk levels
