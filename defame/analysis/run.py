# Configure ezmm to use a separate temp directory for analysis
# This prevents SQLite database conflicts with running experiments
# MUST be done before any ezmm imports (including indirect imports through defame.common)
import sys
from pathlib import Path

# Create a mock ezmm.config module with our custom temp_dir before ezmm loads
# This hacky solution is required because ezmm uses a relative path and initializes the database at import time,
# which doesn't play well with a multi-node setup.
class MockEzmmConfig:
    temp_dir = Path.home() / ".cache" / "ezmm_analysis"
    items_dir = temp_dir / "items"

sys.modules['ezmm.config'] = MockEzmmConfig()
print("Configured ezmm temp dir to", MockEzmmConfig.temp_dir)

from defame.analysis.analyzer_config import AnalyzerConfig
from defame.analysis.llm_helper import AnalyzerLLMHelper
from defame.analysis.analysis_pipeline import AnalysisPipeline

if __name__ == "__main__":
    # Use the Nov 10 experiment - it has proper structured logs
    path = "/mnt/vast/workspaces/PI_Rohrbach/mk79honu/DEFAME/out/averitec/summary/dynamic/llama4_maverick/2025-11-10_08-48 averitec_llama4_maverick"
    path = "/mnt/vast/workspaces/PI_Rohrbach/mk79honu/DEFAME/out/verite/summary/dynamic/llama4_maverick/2025-11-10_16-21 verite_llama4_maverick"
    path = "/mnt/vast/workspaces/PI_Rohrbach/mk79honu/DEFAME/out/mocheg/summary/dynamic/llama4_maverick/2025-11-08_21-53 mocheg_llama4_maverick"

    path = "/mnt/vast/workspaces/PI_Rohrbach/mk79honu/DEFAME/out/mocheg/summary/dynamic/llama4_maverick/2025-11-27_10-49 first_150"

    llm = AnalyzerLLMHelper.from_config_dict({
        # "llm_model": "llama4_maverick",
        # "llm_model": "gpt_4o",
        "llm_model": "gpt_5",
        "llm_temperature": 0.1,
        "llm_max_response_len": 100_000,
    })
    config = AnalyzerConfig(llm=llm, use_multiprocessing=False)  # Disable to avoid SQLite locking

    # pipeline = AnalysisPipeline(
    #     analyze_search_coverage=True,
    #     analyze_query_strategy=True,
    #     analyze_source_utilization=True,
    #     analyze_reasoning=True,
    #     analyze_confidence=True,
    #     analyze_iterations=True
    # )

    pipeline = AnalysisPipeline(
        analyze_search_coverage=False,
        analyze_query_strategy=False,
        analyze_source_utilization=False,
        analyze_reasoning=False,
        analyze_confidence=False,
        analyze_iterations=True
    )

    results = pipeline.analyze(
        experiment_dir=path,
        config=config,
        first_n=150,
        save=True
    )
