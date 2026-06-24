"""Search client abstraction for ResearcherAgent."""

import logging

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient:
    """Provider-agnostic search client with Tavily + mock fallback."""

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""
        settings = get_settings()
        if settings.tavily_api_key:
            return self._tavily_search(query, max_results, settings.tavily_api_key)
        logger.warning("No TAVILY_API_KEY set — using mock search results")
        return self._mock_search(query, max_results)

    @staticmethod
    def _tavily_search(query: str, max_results: int, api_key: str) -> list[SourceDocument]:
        try:
            from tavily import TavilyClient  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("tavily package not installed — falling back to mock search")
            return SearchClient._mock_search(query, max_results)

        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, max_results=max_results)
        return [
            SourceDocument(
                title=r.get("title", ""),
                url=r.get("url"),
                snippet=r.get("content", ""),
            )
            for r in response.get("results", [])
        ]

    @staticmethod
    def _mock_search(query: str, max_results: int) -> list[SourceDocument]:
        mock_data = [
            SourceDocument(title=f"Source {i+1}: {query[:40]}", url=f"https://example.com/doc{i+1}", snippet=f"This is a mock snippet about '{query}' from source {i+1}. It contains relevant information for research purposes.")
            for i in range(min(max_results, 3))
        ]
        return mock_data
