"""Data extraction logic for Source Utilization and Tool Effectiveness Analyzer."""

from __future__ import annotations

import re
from sentence_transformers import SentenceTransformer

from defame.analysis.analyzers.source_utilization.models import (
    CrossReference,
    IterationToolEffectiveness,
    IterationUtilization,
    SourceUtilizationExtractedData,
    ToolExecution,
)
from defame.analysis.analyzers.source_utilization.prompts import (
    get_cross_reference_identification_prompt,
    get_reasoning_citation_prompt,
)
from defame.analysis.data_models import ClaimData
from defame.analysis.llm_helper import AnalyzerLLMHelper


class SourceUtilizationExtractor:
    """Extracts source utilization and tool effectiveness data from claim logs."""

    def __init__(
        self,
        llm_helper: AnalyzerLLMHelper | None = None,
        use_embedding_clustering: bool = True,
    ):
        """
        Initialize the extractor.

        Args:
            llm_helper: LLM helper for reasoning citation and cross-reference analysis
            use_embedding_clustering: Whether to use embedding-based clustering for cross-refs
        """
        self.llm_helper = llm_helper
        self.use_embedding_clustering = use_embedding_clustering
        self.embedding_model = None

        if use_embedding_clustering:
            # Initialize sentence transformer for embedding-based cross-ref clustering
            try:
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                print(f"Warning: Could not load embedding model: {e}")
                self.use_embedding_clustering = False

    def extract_iteration_utilization(
        self, iteration_num: int, claim_data: ClaimData
    ) -> IterationUtilization:
        """
        Extract source utilization data for a single iteration.

        Args:
            iteration_num: Iteration number (0-indexed)
            claim_data: Claim log containing parsed iterations

        Returns:
            IterationUtilization data
        """
        if iteration_num >= len(claim_data.log.iterations):
            return IterationUtilization(
                iteration_num=iteration_num,
                total_sources=0,
                unique_sources=0,
                useful_sources=0
            )

        iteration = claim_data.log.iterations[iteration_num]

        # Collect all sources from evidence retrieval actions
        all_sources = []
        useful_sources_set = set()

        for action in iteration.evidence_retrieval:
            for result in action.results:
                all_sources.append(result.url)
                if result.marked_useful:
                    useful_sources_set.add(result.url)

        total_sources = len(all_sources)
        unique_sources = len(set(all_sources))
        useful_sources = len(useful_sources_set)

        return IterationUtilization(
            iteration_num=iteration_num,
            total_sources=total_sources,
            unique_sources=unique_sources,
            useful_sources=useful_sources,
        )

    def extract_iteration_tool_effectiveness(
        self, iteration_num: int, claim_data: ClaimData
    ) -> IterationToolEffectiveness:
        """
        Extract tool effectiveness data for a single iteration.

        The log structure follows this pattern:
          Searching [platform] with query: [query]
          Got X new source(s): [urls]
          Generated response: [summary for each URL]
          Useful result: [if any sources were useful]
          Generated response: [final summary after all sources]

        So each search produces:
        - 1 result_block with N URLs
        - N+1 evidence_summaries (one per URL + final summary)
        - M useful_results (where M <= N)

        We match search_executions[i] -> result_blocks[i], then check if any
        useful_results came from that search's URLs.

        Args:
            iteration_num: Iteration number (0-indexed)
            claim_data: Claim log containing parsed iterations

        Returns:
            IterationToolEffectiveness data
        """
        if iteration_num >= len(claim_data.log.iterations):
            return IterationToolEffectiveness(
                iteration_num=iteration_num,
                tool_executions=[],
            )

        iteration = claim_data.log.iterations[iteration_num]

        tool_executions = []

        # Process evidence retrieval actions (what was actually executed)
        for action in iteration.evidence_retrieval:
            # Check for errors related to this action
            has_error = len(action.errors) > 0

            # Get result URLs
            result_urls = [result.url for result in action.results]
            is_empty_result = len(result_urls) == 0

            # Check if any useful results came from this action
            num_useful = sum(1 for result in action.results if result.marked_useful)

            # If we got sources but none were marked useful, this indicates
            # NONE responses or low-quality content
            is_none_response = len(result_urls) > 0 and num_useful == 0

            success = not has_error

            tool_executions.append(
                ToolExecution(
                    tool_name=action.tool,
                    query=action.query or "",
                    success=success,
                    is_none_response=is_none_response,
                    is_empty_result=is_empty_result,
                )
            )

        return IterationToolEffectiveness(
            iteration_num=iteration_num,
            tool_executions=tool_executions,
        )

    def extract_cross_references_llm(
        self, claim_text: str, useful_results: list[tuple[str, str]], claim_id: str | None = None
    ) -> list[CrossReference]:
        """
        Extract cross-references using LLM to identify corroborating sources.

        Args:
            claim_text: The claim being fact-checked
            useful_results: List of (url, content) tuples
            claim_id: Optional claim ID for better error messages

        Returns:
            List of CrossReference objects
        """
        if not self.llm_helper or len(useful_results) < 2:
            return []

        prompt = get_cross_reference_identification_prompt(claim_text, useful_results)

        try:
            result = self.llm_helper.extract_json(prompt, temperature=0.3)
            if not result:
                claim_info = f" for claim {claim_id}" if claim_id else ""
                print(f"Warning: LLM cross-reference extraction returned empty result{claim_info}")
                return []

            if not isinstance(result, list):
                claim_info = f" for claim {claim_id}" if claim_id else ""
                print(f"Warning: LLM cross-reference extraction returned non-list result{claim_info}: {type(result)}")
                return []

            # Convert to CrossReference objects
            cross_refs = []
            url_by_idx = {i + 1: url for i, (url, _) in enumerate(useful_results)}

            for item in result:
                if not isinstance(item, dict):
                    continue

                fact = item.get("fact", "")
                source_indices = item.get("supporting_sources", [])

                if not fact or not source_indices or len(source_indices) < 2:
                    continue

                # Map indices to URLs
                supporting_urls = [
                    url_by_idx[idx] for idx in source_indices if idx in url_by_idx
                ]

                if len(supporting_urls) >= 2:
                    cross_refs.append(
                        CrossReference(
                            fact_statement=fact,
                            supporting_sources=supporting_urls,
                        )
                    )

            return cross_refs

        except Exception as e:
            claim_info = f" for claim {claim_id}" if claim_id else ""
            print(f"Warning: LLM cross-reference extraction failed{claim_info}: {e}")
            import traceback
            traceback.print_exc()
            return []

    def extract_cross_references_embedding(
        self, useful_results: list[tuple[str, str]], similarity_threshold: float = 0.7
    ) -> list[CrossReference]:
        """
        Extract cross-references using embedding-based clustering.

        Args:
            useful_results: List of (url, content) tuples
            similarity_threshold: Cosine similarity threshold for clustering

        Returns:
            List of CrossReference objects
        """
        if (
            not self.use_embedding_clustering
            or not self.embedding_model
            or len(useful_results) < 2
        ):
            return []

        try:
            from sklearn.metrics.pairwise import cosine_similarity

            # Extract texts and encode
            texts = [content for _, content in useful_results]
            embeddings = self.embedding_model.encode(texts)

            # Compute similarity matrix
            sim_matrix = cosine_similarity(embeddings)

            # Find clusters of similar evidence
            n = len(texts)
            visited = [False] * n
            clusters = []

            for i in range(n):
                if visited[i]:
                    continue

                cluster = [i]
                visited[i] = True

                for j in range(i + 1, n):
                    if not visited[j] and sim_matrix[i][j] >= similarity_threshold:
                        cluster.append(j)
                        visited[j] = True

                if len(cluster) >= 2:
                    clusters.append(cluster)

            # Convert clusters to CrossReference objects
            cross_refs = []
            for cluster in clusters:
                # Use the longest text as the fact statement
                fact_idx = max(cluster, key=lambda idx: len(texts[idx]))
                fact_statement = texts[fact_idx][:200]  # Truncate for readability

                supporting_urls = [useful_results[idx][0] for idx in cluster]

                cross_refs.append(
                    CrossReference(
                        fact_statement=fact_statement,
                        supporting_sources=supporting_urls,
                    )
                )

            return cross_refs

        except Exception as e:
            print(f"Warning: Embedding cross-reference extraction failed: {e}")
            return []

    def extract_reasoning_citation(
        self, claim_text: str, useful_results: list[tuple[str, str]], reasoning_text: str, claim_id: str | None = None
    ) -> tuple[bool, float]:
        """
        Extract whether reasoning cites evidence and quality score.

        Args:
            claim_text: The claim being fact-checked
            useful_results: List of (url, content) tuples
            reasoning_text: The reasoning/analysis text from elaboration
            claim_id: Optional claim ID for better error messages

        Returns:
            Tuple of (cites_evidence: bool, citation_score: float)
        """
        if not self.llm_helper or not reasoning_text or not useful_results:
            return False, 0.0

        # Extract evidence summaries
        evidence_summaries = [content for _, content in useful_results]

        prompt = get_reasoning_citation_prompt(
            claim_text, evidence_summaries, reasoning_text
        )

        try:
            response = self.llm_helper.generate(prompt, temperature=0.1)

            # Ensure response is a string
            if not isinstance(response, str):
                claim_info = f" for claim {claim_id}" if claim_id else ""
                print(f"Warning: Reasoning citation extraction returned non-string{claim_info}: {type(response)}")
                return False, 0.0

            # Parse response
            cites_evidence = False
            citation_score = 0.0

            # Extract CITES_EVIDENCE
            cites_match = re.search(
                r"CITES_EVIDENCE:\s*([YN])", response, re.IGNORECASE
            )
            if cites_match:
                cites_evidence = cites_match.group(1).upper() == "Y"
            else:
                claim_info = f" for claim {claim_id}" if claim_id else ""
                print(f"Warning: Could not parse CITES_EVIDENCE from response{claim_info}")

            # Extract CITATION_SCORE
            score_match = re.search(r"CITATION_SCORE:\s*([\d.]+)", response)
            if score_match:
                citation_score = float(score_match.group(1))
                citation_score = max(0.0, min(1.0, citation_score))  # Clamp to [0, 1]
            else:
                claim_info = f" for claim {claim_id}" if claim_id else ""
                print(f"Warning: Could not parse CITATION_SCORE from response{claim_info}")

            return cites_evidence, citation_score

        except Exception as e:
            claim_info = f" for claim {claim_id}" if claim_id else ""
            print(f"Warning: Reasoning citation extraction failed{claim_info}: {e}")
            import traceback
            traceback.print_exc()
            return False, 0.0

    def process_claim_data(self, claim_data: ClaimData) -> SourceUtilizationExtractedData:
        """
        Process a single claim and extract all source utilization data.

        Args:
            claim_data: Claim data to process

        Returns:
            SourceUtilizationExtractedData object
        """
        # Extract per-iteration data
        iteration_utilization = []
        iteration_tool_effectiveness = []

        for i in range(len(claim_data.log.iterations)):
            iteration_utilization.append(
                self.extract_iteration_utilization(i, claim_data)
            )
            iteration_tool_effectiveness.append(
                self.extract_iteration_tool_effectiveness(i, claim_data)
            )

        # Compute overall utilization metrics (only non-computable fields)
        # Collect all unique sources and useful sources across iterations
        all_unique_sources = set()
        all_useful_sources = set()

        for iteration in claim_data.log.iterations:
            for action in iteration.evidence_retrieval:
                for result in action.results:
                    all_unique_sources.add(result.url)
                    if result.marked_useful:
                        all_useful_sources.add(result.url)

        overall_unique_sources = len(all_unique_sources)
        overall_useful_sources = len(all_useful_sources)

        # Extract cross-references
        useful_results = []
        for iteration in claim_data.log.iterations:
            for action in iteration.evidence_retrieval:
                for result in action.results:
                    if result.marked_useful and result.summary:
                        useful_results.append((result.url, result.summary))

        # Try both LLM and embedding-based approaches
        cross_refs_llm = self.extract_cross_references_llm(
            claim_data.claim_text, useful_results, claim_id=claim_data.claim_id
        )
        cross_refs_embedding = self.extract_cross_references_embedding(useful_results)

        # Use LLM results if available, otherwise embedding results
        cross_references = cross_refs_llm if cross_refs_llm else cross_refs_embedding

        # Extract reasoning citation
        reasoning_text = ""
        for iteration in claim_data.log.iterations:
            if iteration.elaboration and iteration.elaboration.analysis_text:
                reasoning_text += iteration.elaboration.analysis_text + "\n"

        reasoning_cites_evidence, reasoning_citation_score = (
            self.extract_reasoning_citation(
                claim_data.claim_text, useful_results, reasoning_text, claim_id=claim_data.claim_id
            )
        )

        return SourceUtilizationExtractedData(
            claim_id=claim_data.claim_id,
            claim_text=claim_data.claim_text,
            prediction_correct=claim_data.correct,
            iteration_utilization=iteration_utilization,
            overall_unique_sources=overall_unique_sources,
            overall_useful_sources=overall_useful_sources,
            iteration_tool_effectiveness=iteration_tool_effectiveness,
            cross_references=cross_references,
            reasoning_cites_evidence=reasoning_cites_evidence,
            reasoning_citation_score=reasoning_citation_score,
        )
