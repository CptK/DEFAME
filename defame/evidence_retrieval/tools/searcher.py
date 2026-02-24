import re
from datetime import datetime, timedelta, date
import time
from typing import Any, cast

from ezmm import Image, MultimodalSequence
from jinja2.exceptions import TemplateSyntaxError
from openai import APIError

from config.globals import api_keys
from defame.common import Report, Prompt, logger, Action, Evidence
from defame.common.structured_logger import StructuredLogger
from defame.evidence_retrieval import scraper
from defame.evidence_retrieval.integrations.search import SearchResults, SearchPlatform, PLATFORMS, KnowledgeBase
from defame.evidence_retrieval.integrations.search.common import Query, SearchMode, Source, WebSource
from defame.evidence_retrieval.tools.tool import Tool
from defame.utils.console import gray


class Search(Action):
    """Runs a search on the specified platform to retrieve helpful sources. Useful
    to find new knowledge. Some platforms also support images, e.g.,
    Reverse Image Search (RIS), or
    search modes (like 'news', 'places'), and additional parameters like date limits.
    If a platform does not support some of the parameters, they will be ignored.
    If you run multiple search queries, vary them."""
    name = "search"

    platform: SearchPlatform
    query: Query

    def __init__(
        self,
        query: str | None = None,
        image: str | None = None,
        platform: str = "google",
        mode: str = "search",
        limit: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None
    ) -> None:
        """
        @param query: The textual search query. At least one of `query` or `image` must
            be set.
        @param image: The reference of an image. Use this if you want to perform Reverse
            Image Search (RIS). RIS is helpful to find sources that contain the same or
            similar images. If you also provide `query`, the query will be treated as
            additional context, constraining the search results respectively.
        @param platform: The platform/engine to run the query on. Choose from the
            available platforms below.
        @param mode: The search mode or category. Choose from
            `search` for standard, open search (default),
            `images` for retrieving images for a given text query (useful for verifying
                claims that feature visuals),
            `news` for searching (recent) news articles,
            `places` for searching places.
        @param limit: The maximum number of search results to retrieve.
        @param start_date: Returns search results on or after this date. Use ISO format.
        @param end_date: Returns search results before or on this date. Use ISO format.
        """
        self._save_parameters(locals())

        try:
            self.platform = PLATFORMS[platform]
        except KeyError:
            logger.warning(f"Platform {platform} is not available. Defaulting to Google.")
            self.platform = PLATFORMS["google"]

        img = Image(reference=image) if image else None

        try:
            search_mode = SearchMode(mode) if mode else None
        except ValueError:
            search_mode = None

        try:
            start_date_datetime = date.fromisoformat(start_date) if start_date else None
        except ValueError:
            start_date_datetime = None

        try:
            end_date_datetime = date.fromisoformat(end_date) if end_date else None
        except ValueError:
            end_date_datetime = None

        self.query = Query(
            text=query,
            image=cast(Image | None, img),
            search_mode=search_mode,
            limit=limit,
            start_date=start_date_datetime,
            end_date=end_date_datetime
        )

    def __eq__(self, other):
        return isinstance(other, Search) and self.query == other.query and self.name == other.name

    def __hash__(self):
        return hash((self.name, self.query))


class Searcher(Tool):
    """Searches the specified platform (Google, Wikipedia, ...) for useful sources."""
    # TODO: Rank or annotate the websites according to their credibility, like MUSE
    name = "searcher"
    platforms: list[SearchPlatform]

    n_retrieved_results: int
    n_unique_retrieved_results: int

    def __init__(
        self,
        search_config: dict[str, dict] | None = None,
        limit_per_search: int = 5,
        max_result_len: int | None = None,  # chars
        extract_sentences: bool = False,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.limit_per_search = limit_per_search
        self.max_result_len = max_result_len  # chars
        self.extract_sentences = extract_sentences
        self.restrict_results_before_time: datetime | None = None  # date restriction for all search actions

        self.platforms = self._initialize_platforms(search_config)
        self.known_sources: set[Source] = set()

        self.actions = self._define_actions()

        self.reset()

    def perform(
        self, action: Action, summarize: bool = True, structured_logger: StructuredLogger | None = None, **kwargs
    ) -> Evidence:
        """Override to log search results after summarization is complete."""
        start_time = time.time()

        # Perform the search
        result = self._perform(action, structured_logger=None)  # Don't pass logger to _perform
        execution_time = time.time() - start_time

        # Summarize the results (this is where source.takeaways and usefulness are set)
        summary = self._summarize(result, **kwargs) if summarize else None
        evidence = Evidence(result, action, takeaways=summary)

        # Now log with complete information including summaries and usefulness
        if structured_logger and result and isinstance(action, Search):
            self._log_search_results(structured_logger, action, result, execution_time)

        return evidence

    def _initialize_platforms(self, search_config: dict | None) -> list[SearchPlatform]:
        if search_config is None:
            search_config = self._get_default_search_config()

        platforms = []
        for platform, kwargs in search_config.items():
            if kwargs is None:
                kwargs = {}
            if platform == "averitec_kb":
                kwargs["device"] = self.device
            platform_cls = PLATFORMS[platform]
            platform = platform_cls(max_search_results=self.limit_per_search, **kwargs)
            platforms.append(platform)

        return platforms

    def _get_default_search_config(self):
        if api_keys["serper_api_key"]:
            return {"google": {}}
        else:
            logger.warning("No Serper API key (needed for Google) provided. Falling back to DuckDuckGo.")
            return {"duckduckgo": {}}

    def _define_actions(self) -> list[type[Action]]:
        """Adds a list of the available search platforms to the Search
        action class which will be used in the LLM prompt.."""
        platforms_info = "Available search platforms:"
        for platform in self.platforms:
            platforms_info += f"\n`{platform.name}`: {platform.description}"
        Search.additional_info = platforms_info
        return [Search]

    def _perform(
        self, action: Search, structured_logger: StructuredLogger | None = None
    ) -> SearchResults | None:
        """Validates the search query (by enforcing potential restrictions)
        and runs it."""
        query = action.query

        # Set the strictest specified end date
        if self.restrict_results_before_time is not None:
            max_date = self.restrict_results_before_time.date() - timedelta(days=1)
            if query.end_date is not None:
                query.end_date = min(query.end_date, max_date)
            else:
                query.end_date = max_date

        # Set the strictest search limit
        if self.limit_per_search is not None:
            if query.limit is not None:
                query.limit = min(query.limit, self.limit_per_search)
            else:
                query.limit = self.limit_per_search

        # Ensure the given platform is available
        platform = self.get_platform(action.platform.name)
        if not platform:
            platform = self.platforms[0]
            logger.warning(f"Platform {action.platform.name} is not initialized/allowed. "
                           f"Defaulting to {platform.name}.")

        # Run the query
        results = self._search(platform, query)

        # Note: Logging moved to perform() method to capture summaries and usefulness

        return results

    def _search(self, platform: SearchPlatform, query: Query) -> SearchResults | None:
        """Executes the given search query on the given platform and processes the results.
        Removes known results."""

        # Run search and retrieve sources
        results = platform.search(query)
        sources = results.sources[:self.limit_per_search] if results else []
        self.n_retrieved_results += len(sources)

        # Remove known sources
        sources = self._remove_known_sources(sources)
        self.n_unique_retrieved_results += len(sources)

        # Log search results
        if len(sources) > 0:
            logger.log(f"Got {len(sources)} new source(s):")
            logger.log("\n".join([s.reference for s in sources]))
        else:
            logger.log("No new sources found.")

        # Scrape the pages of the results
        sources_to_scrape = [s for s in sources if isinstance(s, WebSource)]
        scraper.scrape_sources(sources_to_scrape)

        # Modify the raw source text to avoid jinja errors when used in prompt
        self._postprocess_sources(sources, query)
        self._register_sources(sources)

        if len(sources) > 0 and results:
            results.sources = sources
            return results

    def _remove_known_sources(self, sources: list[Source]) -> list[Source]:
        """Removes already known sources from the list `sources`."""
        return [r for r in sources if r not in self.known_sources]

    def _register_sources(self, sources: list[Source]):
        """Adds the provided list of sources to the set of known sources."""
        self.known_sources |= set(sources)

    def reset(self):
        """Removes all known web sources and resets the search platforms."""
        self.known_sources = set()
        self.n_retrieved_results = 0
        self.n_unique_retrieved_results = 0
        for platform in self.platforms:
            platform.reset()

    def _postprocess_sources(self, sources: list[Source], query: Query) -> None:
        for source in sources:
            if source.is_loaded():
                processed = self._postprocess_single_source(str(source.content), query)
                source.content = MultimodalSequence(processed)

    def _postprocess_single_source(self, content: str, query: Query) -> str:
        """Prepares the result contents before LLM processing:
        1. Optionally extracts relevant sentences from the result text using keywords
            from the query.
        2. Removes all double curly braces to avoid conflicts with Jinja.
        3. Optionally truncates the result text to a maximum length."""
        if self.extract_sentences:
            keywords = re.findall(r'\b\w+\b', query.text.lower()) or query.text
            relevant_content = extract_relevant_sentences(content, keywords)[:10]
            relevant_text = ' '.join(relevant_content)
            content = relevant_text or content

        content = re.sub(r"\{\{.*}}", "", content)

        if self.max_result_len is not None:
            content = content[:self.max_result_len]

        return content

    def _summarize(self, results: SearchResults, doc: Report | None = None) -> MultimodalSequence | None:
        assert doc is not None
        if results:
            for source in results.sources:
                self._summarize_single_source(source, doc)
            return self._summarize_summaries(results, doc)
        else:
            return None

    def _truncate_for_context(self, primary: str, context: str, reserve: int = 1500) -> tuple[str, str]:
        """Truncate primary and context strings to fit the LLM's context window.
        Prioritizes keeping the primary content over the context."""
        max_tokens = self.llm.context_window - reserve
        primary_tokens = self.llm.count_tokens(primary)
        context_tokens = self.llm.count_tokens(context)
        total = primary_tokens + context_tokens

        if total <= max_tokens:
            return primary, context

        logger.debug(f"Summarize prompt has ~{total} tokens, exceeding limit of "
                     f"{max_tokens} by ~{total - max_tokens}. Truncating.")

        # Cap context (doc) at 25% of budget, give the rest to primary (source/summaries)
        max_context_tokens = min(context_tokens, max_tokens // 4)
        max_primary_tokens = max_tokens - max_context_tokens

        if primary_tokens > max_primary_tokens:
            ratio = max_primary_tokens / primary_tokens
            primary = primary[:int(len(primary) * ratio)]
        if context_tokens > max_context_tokens:
            ratio = max_context_tokens / context_tokens
            context = context[:int(len(context) * ratio)]

        return primary, context

    def _summarize_single_source(self, source: Source, doc: Report):
        # Skip summarization if source wasn't successfully scraped
        if not source.is_loaded():
            source.takeaways = MultimodalSequence("NONE")
            return

        source_str, doc_str = self._truncate_for_context(str(source), str(doc))
        prompt = Prompt(
            placeholder_targets={"[SOURCE]": source_str, "[DOC]": doc_str},
            name="SummarizeSourcePrompt",
            template_file_path="defame/prompts/summarize_source.md",
        )

        try:
            summary = self.llm.generate(prompt, max_attempts=3)
            if not summary:
                summary = "NONE"
        except APIError as e:
            logger.info(f"APIError: {e} - Skipping the summary for {source}.")
            logger.log(f"Used prompt:\n{str(prompt)}")
            summary = "NONE"
        except TemplateSyntaxError as e:
            logger.info(f"TemplateSyntaxError: {e} - Skipping the summary for {source}.")
            summary = "NONE"
        except ValueError as e:
            logger.warning(f"ValueError: {e} - Skipping the summary for {source}.")
            summary = "NONE"
        except Exception as e:
            logger.log(f"Error while summarizing! {e} - Skipping the summary for {source}.")
            summary = "NONE"

        source.takeaways = MultimodalSequence(summary)

        if source.is_relevant():
            logger.log("Useful source: " + gray(str(source)))

    def _summarize_summaries(self, result: SearchResults, doc: Report | None) -> MultimodalSequence | None:
        """Generates a summary, aggregating all relevant information from the
        identified and relevant sources."""

        summaries = [str(source) for source in result.sources if source.is_relevant()]
        if len(summaries) == 0:  # No relevant sources
            return None
        elif len(summaries) == 1:
            # No further summarization needed as we have only one source
            return MultimodalSequence(summaries[0])

        # Disable summary of summaries:
        # relevant_sources = "\n\n".join([str(s) for s in result.sources if s.is_relevant()])
        # return MultimodalSequence(relevant_sources)

        # Prepare the prompt for the LLM
        summaries_str, doc_str = self._truncate_for_context(str(result), str(doc))
        summarize_prompt = Prompt(
            placeholder_targets={"[SUMMARIES]": summaries_str, "[DOC]": doc_str},
            name="SummarizeSummariesPrompt",
            template_file_path="defame/prompts/summarize_summaries.md",
        )

        return MultimodalSequence(self.llm.generate(summarize_prompt))

    def _log_search_results(
            self,
            structured_logger: StructuredLogger,
            action: Search,
            results: SearchResults,
            execution_time: float
        ):
        """Log search results to structured logger after summarization is complete."""
        # Extract platform and query from action
        platform = action.platform
        query = action.query

        # Prepare parameters
        parameters = {}
        if query.start_date:
            parameters["start_date"] = query.start_date.isoformat()
        if query.end_date:
            parameters["end_date"] = query.end_date.isoformat()
        if query.limit:
            parameters["limit"] = query.limit
        if query.search_mode:
            parameters["search_mode"] = query.search_mode.value

        # Prepare results (now with summaries and usefulness from post-summarization)
        result_dicts = []
        for source in results.sources:
            result_dict = {
                "url": source.reference if hasattr(source, 'reference') else str(source),
                "platform": platform.name,
            }

            if hasattr(source, 'title') and source.title:
                result_dict["title"] = source.title

            if hasattr(source, 'snippet') and source.snippet:
                result_dict["snippet"] = str(source.snippet)

            if hasattr(source, 'content') and source.content:
                result_dict["content"] = str(source.content)[:1000]  # Limit content length

            if hasattr(source, 'takeaways') and source.takeaways:
                result_dict["summary"] = str(source.takeaways)

            # Mark if useful (relevant) - ensure boolean value
            if hasattr(source, 'is_relevant'):
                useful = source.is_relevant()
                result_dict["useful"] = bool(useful) if useful is not None else False
            else:
                result_dict["useful"] = False

            result_dicts.append(result_dict)

        # Log the search action
        structured_logger.log_evidence_retrieval(
            action_type="search",
            tool=self.name,
            query=query.text if query.text else None,
            platform=platform.name,
            parameters=parameters,
            results=result_dicts,
            execution_time=execution_time
        )

    def get_stats(self) -> dict[str, Any]:
        return {
            "Total searches": sum([platform.n_searches for platform in self.platforms]),
            "Platform stats": {platform.name: platform.stats for platform in self.platforms},
        }

    def get_platform(self, name: str) -> SearchPlatform | None:
        for platform in self.platforms:
            if platform.name == name:
                return platform

    def set_time_restriction(self, before: datetime | None):
        self.restrict_results_before_time = before

    def set_claim_id(self, claim_id: str):
        super().set_claim_id(claim_id)
        kb = self.get_platform(KnowledgeBase.name)
        if kb:
            kb.current_claim_id = int(claim_id)


def extract_relevant_sentences(text, keywords):
    sentences = re.split(r'(?<=[.!?]) +', text)
    relevant_sentences = []
    for sentence in sentences:
        score = sum(1 for word in keywords if word in sentence.lower())
        if score > 0:
            relevant_sentences.append((sentence, score))
    relevant_sentences.sort(key=lambda x: x[1], reverse=True)
    return [sentence for sentence, score in relevant_sentences]
