import json
import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
from langfuse import get_client
from dotenv import load_dotenv

load_dotenv(override=True)

langfuse = get_client()

# Load sensor config
config_path = os.path.join(os.path.dirname(__file__), '../schemas/sensor_config.json')
with open(config_path) as f:
    SENSOR_CONFIG = json.load(f)

manuals_path = os.path.join(os.path.dirname(__file__), '../../data/manuals')

# ─────────────────────────────────────────────
# MCP Server Parameters
# ─────────────────────────────────────────────
filesystem_server_params = StdioServerParameters(
    command="npx",
    args=[
        "-y",
        "@modelcontextprotocol/server-filesystem",
        manuals_path
    ],
    env={**os.environ}
)

brave_server_params = StdioServerParameters(
    command="npx",
    args=[
        "-y",
        "@modelcontextprotocol/server-brave-search"
    ],
    env={
        **os.environ,
        "BRAVE_API_KEY": os.getenv("BRAVE_API_KEY", "")
    }
)


def run_knowledge_crew(
    engine_id: int,
    cycle: int,
    rul: int,
    diagnosis: str,
    risk_level: str,
    anomalies: list
) -> str:
    """Run the CrewAI knowledge crew with MCP tools."""

    with langfuse.start_as_current_observation(
        as_type="span",
        name="knowledge_crew"
    ) as span:
        span.update(input={
            "engine_id": engine_id,
            "cycle": cycle,
            "rul": rul,
            "diagnosis": diagnosis,
            "risk_level": risk_level
        })

        # ─────────────────────────────────────────────
        # Start both MCP servers and collect tools
        # ─────────────────────────────────────────────
        print("Connecting to Filesystem MCP server...")
        fs_adapter = MCPServerAdapter(filesystem_server_params)
        filesystem_tools = fs_adapter.tools
        print(f"  ✅ Filesystem tools: {[t.name for t in filesystem_tools]}")

        print("Connecting to Brave Search MCP server...")
        brave_adapter = MCPServerAdapter(brave_server_params)
        brave_tools = brave_adapter.tools
        print(f"  ✅ Brave tools: {[t.name for t in brave_tools]}")

        all_researcher_tools = filesystem_tools + brave_tools

        try:
            # ─────────────────────────────────────────────
            # Define agents with MCP tools
            # ─────────────────────────────────────────────
            maintenance_researcher = Agent(
                role="Maintenance Researcher",
                goal="Find the most relevant maintenance procedures for the diagnosed engine fault",
                backstory="""You are an expert aviation maintenance researcher with 20 years of experience.
You specialize in turbofan engine maintenance procedures.
Always read local maintenance manuals first using the filesystem tools,
then use web search as fallback if more information is needed.""",
                tools=all_researcher_tools,
                verbose=True,
                allow_delegation=False
            )

            risk_assessor = Agent(
                role="Risk Assessment Specialist",
                goal="Assess the operational risk level based on diagnosis and RUL",
                backstory="""You are a senior aviation safety engineer specializing in risk assessment.
You evaluate engine health data and remaining useful life to determine operational risk.
You are conservative — when in doubt, always recommend the safer option.""",
                tools=[],
                verbose=True,
                allow_delegation=False
            )

            report_writer = Agent(
                role="Maintenance Report Writer",
                goal="Produce a clear structured maintenance report for the operator",
                backstory="""You are a technical writer specializing in aviation maintenance documentation.
You produce clear, actionable Markdown reports with specific action items.""",
                tools=[],
                verbose=True,
                allow_delegation=False
            )

            # ─────────────────────────────────────────────
            # Define tasks
            # ─────────────────────────────────────────────
            thresholds = SENSOR_CONFIG["rul_thresholds"]

            research_task = Task(
                description=f"""Research maintenance procedures for this engine fault:

**Diagnosis:** {diagnosis}
**Risk Level:** {risk_level}
**RUL:** {rul} cycles remaining

Step 1: Use filesystem tools to read all available maintenance manuals.
Step 2: Identify procedures relevant to: {diagnosis}
Step 3: If needed, use Brave Search to find additional turbofan maintenance guidance.

Summarize the most relevant procedures found.""",
                agent=maintenance_researcher,
                expected_output="Summary of relevant maintenance procedures"
            )

            risk_task = Task(
                description=f"""Assess operational risk for this engine:

**Diagnosis:** {diagnosis}
**RUL:** {rul} cycles remaining
**Anomalous Sensors:** {anomalies if anomalies else 'None flagged'}
**Critical threshold:** {thresholds['critical_below']} cycles
**Warning threshold:** {thresholds['warning_above']} cycles

Determine:
1. Risk level (CRITICAL / WARNING / HEALTHY)
2. Urgency of maintenance action
3. Safety implications of continued operation
4. Whether to ground the engine or continue monitoring""",
                agent=risk_assessor,
                expected_output="Risk assessment with risk level and grounding recommendation"
            )

            report_task = Task(
                description=f"""Write a comprehensive maintenance report for Engine {engine_id}.

**Engine ID:** {engine_id}
**Current Cycle:** {cycle}
**RUL:** {rul} cycles remaining

Use findings from the researcher and risk assessor to write a Markdown report with these sections:

## Maintenance Findings
## Risk Assessment
## Recommended Actions
## Timeline
## Safety Notes""",
                agent=report_writer,
                expected_output="Complete Markdown maintenance report"
            )

            # ─────────────────────────────────────────────
            # Run the crew
            # ─────────────────────────────────────────────
            crew = Crew(
                agents=[maintenance_researcher, risk_assessor, report_writer],
                tasks=[research_task, risk_task, report_task],
                process=Process.sequential,
                verbose=True
            )

            print("Running knowledge crew...")
            result = crew.kickoff()
            maintenance_plan = str(result)

            span.update(output={"maintenance_plan": maintenance_plan[:500]})
            langfuse.flush()

        finally:
            fs_adapter.stop()
            brave_adapter.stop()
            print("MCP servers stopped cleanly")

    return maintenance_plan