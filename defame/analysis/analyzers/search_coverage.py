import multiprocessing
from pydantic import BaseModel
from scipy.stats import pearsonr
from tqdm import tqdm

from defame.analysis.data_models import ClaimData, ExperimentData
from defame.analysis.analyzers.base_analyzer import BaseAnalyzer


class SearchCoverageExtractedData(BaseModel):
    num_sources: int
    num_unique_sources: int
    num_useful_sources: int
    success: bool


class SearchCoverageMetrics(BaseModel):
    avg_num_sources: float
    avg_num_unique_sources: float
    avg_num_useful_sources: float
    corr_num_sources_success: tuple[float, float]
    corr_num_unique_sources_success: tuple[float, float]
    corr_num_useful_sources_success: tuple[float, float]


class SearchCoverageInsights(BaseModel):
    insights_summary: str


class SearchCoverageAnalyzer(BaseAnalyzer[SearchCoverageExtractedData, SearchCoverageMetrics, SearchCoverageInsights]):
    """Analyzer that computes these metrics:
    - Number of unique URLs/sources retrieved per claim (i.e. per log file)
    - Number of sources marked as useful per claim

    Extraction is done algorithmically from parsed iterations.
    - Get all domains and count unique ones
    - Track which sources were marked as useful

    Example:
        from defame.analysis.analyzers.analyzer_config import AnalyzerConfig

        # This analyzer doesn't need LLM, so use base config
        config = AnalyzerConfig(use_llm=False, use_multiprocessing=True)
        analyzer = SourceCountingAnalyzer(config=config)
        result = analyzer.analyze(experiment_log)
    """

    def extract_data(self, experiment_data: ExperimentData) -> list[SearchCoverageExtractedData]:
        results = []
        if self.config.use_multiprocessing:
            with multiprocessing.Pool() as pool:
                for result in pool.imap_unordered(_process_claim_data, experiment_data.claims):
                    if result is not None:
                        results.append(result)
        else:
            for claim_data in tqdm(experiment_data.claims, desc="Search Coverage", unit="claim"):
                result = _process_claim_data(claim_data)
                if result is not None:
                    results.append(result)
        return results

    def compute_metrics(self, extracted_data: list[SearchCoverageExtractedData]) -> SearchCoverageMetrics:
        total_sources = sum(r.num_sources for r in extracted_data)
        total_unique_sources = sum(r.num_unique_sources for r in extracted_data)
        total_useful_sources = sum(r.num_useful_sources for r in extracted_data)
        num_claims = len(extracted_data)

        # Compute correlation metrics between source counts and success
        # pearsonr requires at least 2 data points
        if num_claims < 2:
            corr_sources = (0.0, 1.0)
            corr_unique_sources = (0.0, 1.0)
            corr_useful_sources = (0.0, 1.0)
        else:
            try:
                corr_sources = pearsonr([r.num_sources for r in extracted_data], [r.success for r in extracted_data])
                corr_unique_sources = pearsonr([r.num_unique_sources for r in extracted_data], [r.success for r in extracted_data])
                corr_useful_sources = pearsonr([r.num_useful_sources for r in extracted_data], [r.success for r in extracted_data])
            except Exception:
                corr_sources = (0.0, 1.0)
                corr_unique_sources = (0.0, 1.0)
                corr_useful_sources = (0.0, 1.0)

        return SearchCoverageMetrics(
            avg_num_sources=total_sources / num_claims if num_claims > 0 else 0,
            avg_num_unique_sources=total_unique_sources / num_claims if num_claims > 0 else 0,
            avg_num_useful_sources=total_useful_sources / num_claims if num_claims > 0 else 0,
            corr_num_sources_success=corr_sources,
            corr_num_unique_sources_success=corr_unique_sources,
            corr_num_useful_sources_success=corr_useful_sources
        )
    
    def generate_insights(self, metrics: SearchCoverageMetrics) -> SearchCoverageInsights:
        def interpret_corr(name: str, corr: tuple[float, float]) -> str:
            r, p = corr
            if p > 0.05:
                return f"{name}: no significant correlation (r={r:.2f}, p={p:.3f})"
            strength = "strong" if abs(r) > 0.5 else "moderate" if abs(r) > 0.3 else "weak"
            direction = "positive" if r > 0 else "negative"
            return f"{name}: {strength} {direction} correlation (r={r:.2f}, p={p:.3f})"

        summary = "\n".join([
            interpret_corr("Total sources vs. success", metrics.corr_num_sources_success),
            interpret_corr("Unique sources vs. success", metrics.corr_num_unique_sources_success),
            interpret_corr("Useful sources vs. success", metrics.corr_num_useful_sources_success),
        ])

        return SearchCoverageInsights(insights_summary=summary)

def _process_claim_data(claim_data: ClaimData) -> SearchCoverageExtractedData | None:
    try:
        sources = []
        useful_sources = []

        for iteration in claim_data.log.iterations:
            # Extract all result URLs from evidence retrieval actions
            for action in iteration.evidence_retrieval:
                for result in action.results:
                    sources.append(result.url)

                    # Track useful sources
                    if result.marked_useful:
                        useful_sources.append(result.url)

        return SearchCoverageExtractedData(
            num_sources=len(sources),
            num_unique_sources=len(set(sources)),
            num_useful_sources=len(set(useful_sources)),
            success=claim_data.correct
        )
    except Exception as e:
        print(f"Error processing claim {claim_data.claim_id}: {e}")
        import traceback
        traceback.print_exc()
        return None
