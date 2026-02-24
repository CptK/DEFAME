import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from defame.analysis.analyzer_config import AnalyzerConfig
from defame.analysis.data_models import ClaimData
from defame.analysis.analyzers.query_strategy.models import (
    QueryStrategyExtractedData,
    ExtractedIterationData,
    QueryInfo,
    QueryEvolution,
    ToolDiversity,
    ToolChoiceCoherence,
    CounterEvidenceSeeking,
    SearchAngleAnalysis,
)
from defame.analysis.analyzers.query_strategy.prompts import (
    COUNTER_EVIDENCE_SEEKING_PROMPT,
    QUERY_SPECIFICITY_PROMPT,
    GROUP_QUERIES_BY_ANGLE_PROMPT,
    TOOL_COHERENCE_RATING_PROMPT
)


class QueryStrategyDataExtractor:
    def __init__(self, config: AnalyzerConfig):
        self.config = config

    def process_claim_data(self, claim_data: ClaimData) -> QueryStrategyExtractedData | None:
        """
        Process a single claim to extract query strategy data.

        This is where the main extraction logic goes:
        1. Extract queries from each iteration
        2. Compute query-level metrics (specificity, overlap, counter-evidence)
        3. Compute claim-level aggregates (evolution, diversity, coherence)
        """
        try:
            iterations = self._extract_iterations(claim_data)
            return QueryStrategyExtractedData(
                claim_id=int(claim_data.claim_id) if claim_data.claim_id.isdigit() else 0,
                success=claim_data.correct,
                claim_text=claim_data.claim_text,
                iterations=iterations,
                query_evolution=self._extract_query_evolution(iterations),
                tool_diversity=self._extract_tool_diversity(iterations),
                tool_choice_coherence=self._extract_tool_choice_coherence(iterations, claim_data),
                counter_evidence_seeking=self._extract_counter_evidence_seeking(iterations),
                search_angle_analysis=self._extract_search_angle_analysis(
                    [q.query_text for it in iterations for q in it.queries],
                    claim_data.claim_text
                )
            )
        except Exception as e:
            print(f"Error processing claim {claim_data.claim_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _extract_iterations(self, claim_data: ClaimData) -> list[ExtractedIterationData]:
        """
        Extract iteration-level data from the claim.

        Args:
            claim_data: The claim data to process

        Returns:
            List of ExtractedIterationData for each iteration
        """
        iterations = []
        for i, iteration in enumerate(claim_data.log.iterations):
            queries = []
            # Parse actions_planned strings to extract tool and query
            # Format: "tool_name(query)"
            for action_str in iteration.planning.actions_planned:
                match = re.match(r'(\w+)\((.*)\)', action_str)
                if match:
                    tool_name = match.group(1)
                    query = match.group(2)
                    queries.append(QueryInfo(
                        query_text=query,
                        tool_name=tool_name,
                        specificity_score=self._rate_query_specificity(query),
                        keyword_overlap_with_claim=self._rate_keyword_overlap(query, claim_data.claim_text),
                        is_counter_evidence_seeking=self._is_counter_evidence_seeking(query)
                    ))


            iterations.append(
                ExtractedIterationData(
                    iteration_number=i + 1,
                    queries=queries,
                    tool_types_used=[query.tool_name for query in queries]
                )
            )

        return iterations

    def _rate_query_specificity(self, query: str) -> float | None:
        """
        Rate a query's specificity for fact-checking purposes.

        Args:
            query: The search query to rate

        Returns:
            Specificity score from 1.0 to 5.0, or None if undetermined
        """
        if self.config.llm:
            return self.config.llm.get_numeric_rating(QUERY_SPECIFICITY_PROMPT.format(query=query), scale=(1, 5))
        return None
    
    def _rate_keyword_overlap(self, query: str, claim: str) -> float | None:
        """
        Rate the keyword overlap between a query and the claim.

        Args:
            query: The search query
            claim: The claim being fact-checked

        Returns:
            Overlap score from 0.0 to 1.0, or None if undetermined
        """
        try:
            # Use TF-IDF vectorizer to compute cosine similarity between query and claim
            vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([claim, query])

            # Compute cosine similarity between claim (row 0) and query (row 1)
            # Use getrow() for proper sparse matrix access
            claim_vec = tfidf_matrix.getrow(0)
            query_vec = tfidf_matrix.getrow(1)
            similarity = cosine_similarity(claim_vec, query_vec)[0, 0]

            return float(similarity)
        except Exception as e:
            # If there's an error (e.g., empty strings), return None
            return None

    def _is_counter_evidence_seeking(self, query: str) -> bool | None:
        """
        Determine if a query seeks counter-evidence.

        Args:
            query: The search query to analyze

        Returns:
            True if query seeks counter-evidence, False if not, None if undetermined
        """
        if self.config.llm:
            result = self.config.llm.classify(COUNTER_EVIDENCE_SEEKING_PROMPT.format(query=query), options=["yes", "no"])
            return result == "yes"
        return None

    def _calculate_lexical_diversity(self, queries1: list[str], queries2: list[str]) -> float:
        """
        Calculate lexical diversity between two sets of queries.

        Uses Jaccard distance (1 - Jaccard similarity) to measure how different
        the vocabularies are between two query sets.

        Args:
            queries1: First set of queries
            queries2: Second set of queries

        Returns:
            Diversity score from 0.0 (identical) to 1.0 (completely different)
        """
        if not queries1 or not queries2:
            return 0.0

        # Extract words from both query sets (lowercase and tokenize)
        words1 = set()
        for q in queries1:
            words1.update(q.lower().split())

        words2 = set()
        for q in queries2:
            words2.update(q.lower().split())

        # Handle empty sets
        if not words1 or not words2:
            return 0.0

        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        if union == 0:
            return 0.0

        jaccard_similarity = intersection / union

        # Return Jaccard distance (diversity = 1 - similarity)
        # Higher values mean more diverse/different vocabularies
        return 1.0 - jaccard_similarity

    def _extract_query_evolution(self, iterations: list[ExtractedIterationData]) -> QueryEvolution:
        """
        Extract query evolution patterns across iterations.

        Args:
            iterations: List of ExtractedIterationData

        Returns:
            QueryEvolution object capturing trends
        """
        # Calculate lexical diversity between consecutive iterations
        lexical_diversity_scores = []
        for i in range(len(iterations) - 1):
            current_queries = [q.query_text for q in iterations[i].queries]
            next_queries = [q.query_text for q in iterations[i + 1].queries]
            diversity = self._calculate_lexical_diversity(current_queries, next_queries)
            lexical_diversity_scores.append(diversity)

        return QueryEvolution(
            specificity_trend=[it.avg_specificity for it in iterations],
            keyword_overlap_trend=[it.avg_keyword_overlap for it in iterations],
            lexical_diversity_between_iterations=lexical_diversity_scores,
            query_count_per_iteration=[it.num_queries for it in iterations]
        )
    
    def _extract_tool_diversity(self, iterations: list[ExtractedIterationData]) -> ToolDiversity:
        """
        Extract tool diversity and redundancy metrics.

        Args:
            iterations: List of ExtractedIterationData

        Returns:
            ToolDiversity object capturing diversity metrics
        """
        return ToolDiversity(
            unique_tool_types=list(set(tool for it in iterations for tool in it.tool_types_used)),
            tool_type_counts={
                tool: sum(it.tool_types_used.count(tool) for it in iterations) for tool in set(tool for it in iterations for tool in it.tool_types_used)
            },
            num_redundant_queries=self._count_redundant_queries([q.query_text for it in iterations for q in it.queries]),
            total_queries=sum(it.num_queries for it in iterations)
        )
    
    def _count_redundant_queries(self, queries: list[str]) -> int:
        """
        Count the number of redundant (similar/duplicate) queries.

        A query is considered redundant if it has high similarity (>0.8 cosine similarity) with any previous
        query in the list.

        Args:
            queries: List of all queries made

        Returns:
            Number of redundant queries
        """
        if len(queries) <= 1:
            return 0

        try:
            # Use TF-IDF to vectorize all queries
            vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(queries)

            # Compute pairwise cosine similarity matrix
            similarity_matrix = cosine_similarity(tfidf_matrix)

            # Count redundant queries
            # A query is redundant if it has similarity > 0.8 with any earlier query
            redundant_count = 0
            similarity_threshold = 0.8

            for i in range(1, len(queries)):
                # Check similarity with all previous queries
                if np.any(similarity_matrix[i, :i] > similarity_threshold):
                    redundant_count += 1

            return redundant_count

        except Exception as e:
            # If there's an error (e.g., all queries are empty), return 0
            return 0

    def _extract_tool_choice_coherence(
            self,
            iterations: list[ExtractedIterationData],
            claim_data: ClaimData
        ) -> ToolChoiceCoherence:
        """
        Extract tool choice coherence metrics.

        Args:
            iterations: List of ExtractedIterationData
            claim_data: The claim data being processed

        Returns:
            ToolChoiceCoherence object capturing coherence scores
        """
        coherence_scores = []
        for i in range(len(iterations) - 1):
            previous_iteration = iterations[i]
            next_iteration = iterations[i + 1]
            if previous_iteration.queries and next_iteration.queries:
                # get evidence from previous iteration and rate coherence
                iteration_info = claim_data.log.iterations[i]
                previous_evidence = ""
                # Extract evidence from result summaries
                all_evidences = []
                for action in iteration_info.evidence_retrieval:
                    for result in action.results:
                        if result.summary:
                            all_evidences.append(result.summary)
                previous_evidence = " ".join(all_evidences)

                if previous_evidence:
                    # Rate coherence based on previous evidence
                    score = self._rate_tool_choice_coherence(
                        previous_action=previous_iteration.queries[-1].tool_name,
                        previous_evidence=previous_evidence,
                        next_action=next_iteration.queries[0].tool_name
                    )
                else:
                    score = 1.0  # If there is no evidence any action is coherent
                if score is not None:
                    coherence_scores.append(score)

        return ToolChoiceCoherence(coherence_scores=coherence_scores)

    def _rate_tool_choice_coherence(
        self,
        previous_action: str,
        previous_evidence: str,
        next_action: str
    ) -> float | None:
        """
        Rate whether next action follows logically from previous evidence.

        Args:
            previous_action: Previous action description
            previous_evidence: Evidence from previous action
            next_action: Next action description

        Returns:
            Coherence score from 0.0 to 1.0, or None if undetermined
        """
        if not self.config.llm:
            return None

        prompt_text = TOOL_COHERENCE_RATING_PROMPT.format(
            previous_action=previous_action,
            previous_evidence=previous_evidence,
            next_action=next_action
        )

        rating = self.config.llm.get_numeric_rating(prompt_text, scale=(1, 5))
        # Normalize to 0-1 scale
        return (rating - 1) / 4.0
    
    def _extract_counter_evidence_seeking(self, iterations: list[ExtractedIterationData]) -> CounterEvidenceSeeking:
        """
        Extract counter-evidence seeking behavior metrics.

        Args:
            iterations: List of ExtractedIterationData

        Returns:
            CounterEvidenceSeeking object capturing metrics
        """
        queries = [q.query_text for it in iterations for q in it.queries if q.is_counter_evidence_seeking]
        return CounterEvidenceSeeking(
            num_counter_evidence_queries=len(queries),
            total_queries=sum(it.num_queries for it in iterations),
            counter_evidence_queries=queries
        )

    def _extract_search_angle_analysis(self, queries: list[str], claim: str) -> SearchAngleAnalysis:
        """
        Group queries by the search angle they cover.

        Args:
            queries: List of queries to group
            claim: The claim being fact-checked

        Returns:
            SearchAngleAnalysis object capturing grouped queries
        """
        result = {"General": queries}
        if queries and self.config.llm:
            queries_str = "\n".join(f"{i+1}. {q}" for i, q in enumerate(queries))

            grouping = self.config.llm.extract_json(GROUP_QUERIES_BY_ANGLE_PROMPT.format(
                claim=claim,
                queries_str=queries_str
            ))

            # Check that grouping is a dictionary before accessing .items()
            if grouping and isinstance(grouping, dict):
                # Convert indices back to query strings
                result = {}
                for angle, indices in grouping.items():
                    # Ensure indices is a list
                    if isinstance(indices, list):
                        result[angle] = [queries[i-1] for i in indices if isinstance(i, int) and 0 < i <= len(queries)]
                
        return SearchAngleAnalysis(
            num_distinct_angles=len(result),
            angle_groups=list(result.values()),
            angle_labels=list(result.keys())
        )
