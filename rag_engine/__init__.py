"""
rag_engine — public surface.

Exports the two retrieval functions used by main.py and CLI utilities.
Dispatch between standard and multi-query search happens in main.py's
_retrieve() helper, not here.
"""
from .search import search_rag
from .multi_query import multi_query_search

__all__ = ["search_rag", "multi_query_search"]
