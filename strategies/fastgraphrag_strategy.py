"""
FastGraphRAG strategy, option F1 (blueprint §B, §H) - Category C: SIMPLIFIED HEURISTIC.

WHAT THE REAL METHOD DOES:
FastGraphRAG (Circlemind, 2024) extracts EVERY chunk with an LLM and uses
personalised PageRank only at retrieval time. It has no pre-extraction
selection step at all.

WHAT THIS HEURISTIC DOES:
We borrow its "central entities matter" idea, using a graph that is cheap to
build BEFORE any LLM call:
1. Find named entities in each chunk - spaCy NER if installed
   (pip install spacy && python -m spacy download en_core_web_sm), otherwise a
   capitalised-phrase rule (src/text_utils.py).
2. Link two entities if they appear in the same chunk (co-occurrence graph).
3. Score each chunk by the SUM of the normalised degree centrality of the
   entities it mentions; rank descending, tie-break by chunk_id.

WHY IT IS NOT KET-RAG TWICE: KET-RAG's graph is over CHUNKS (chunk-chunk text
similarity); this graph is over ENTITIES (entity-entity co-occurrence), and the
score is degree centrality, not PageRank (blueprint §H requires them to differ).

WHAT TO SAY IN THE REPORT: "a centrality-based selection heuristic inspired by
FastGraphRAG", never "FastGraphRAG's selection strategy" - it has none.
"""

from __future__ import annotations

import itertools

import networkx as nx

from src.chunking import Chunk
from src.text_utils import capitalized_phrases, load_spacy, normalize_name
from strategies.base_strategy import SelectionStrategy

# spaCy entity types that name things (dates, numbers, money etc. are excluded).
ENTITY_TYPES = {"PERSON", "ORG", "GPE", "LOC", "NORP", "FAC", "EVENT", "WORK_OF_ART", "PRODUCT", "LAW"}


class FastGraphRAGStrategy(SelectionStrategy):
    name = "fastgraphrag"

    def __init__(self, use_spacy: bool = True, spacy_model: str = "en_core_web_sm"):
        self.nlp = load_spacy(spacy_model, disable=["lemmatizer"]) if use_spacy else None
        self.ner_backend = f"spacy:{spacy_model}" if self.nlp else "regex-capitalised"
        self.last_run_info: dict = {}

    def find_entities(self, text: str) -> set[str]:
        if self.nlp is not None:
            names = [e.text for e in self.nlp(text).ents if e.label_ in ENTITY_TYPES]
        else:
            names = capitalized_phrases(text)
        return {normalize_name(n) for n in names if len(n.strip()) > 1}

    def rank(self, chunks: list[Chunk]) -> list[str]:
        mentions = {c.chunk_id: self.find_entities(c.text) for c in chunks}

        graph = nx.Graph()
        for entities in mentions.values():
            graph.add_nodes_from(entities)
            graph.add_edges_from(itertools.combinations(sorted(entities), 2))
        centrality = nx.degree_centrality(graph) if graph.number_of_nodes() > 1 else {}

        scores = {cid: sum(centrality.get(e, 0.0) for e in ents) for cid, ents in mentions.items()}
        ranked = sorted(scores, key=lambda cid: (-scores[cid], cid))

        self.last_run_info = {"ner_backend": self.ner_backend,
                              "n_entities": graph.number_of_nodes(),
                              "n_entity_edges": graph.number_of_edges()}
        return ranked
