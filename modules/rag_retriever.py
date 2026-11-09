"""
RAG Knowledge Retrieval Engine for Nutrition Guidelines

Loads curated nutritional guidelines from the local knowledge base (WHO, ICMR-NIN, USDA),
ranks relevant documents based on user query tokens and keywords,
and formats retrieved passages with verified source citations.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

KNOWLEDGE_FILE = Path(__file__).resolve().parent.parent / "data" / "nutrition_guidelines.json"


class NutritionKnowledgeRetriever:
    """
    Retrieval engine for authoritative nutrition guidelines.
    """

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or KNOWLEDGE_FILE
        self._guidelines: List[Dict[str, Any]] = []
        self._load_knowledge_base()

    def _load_knowledge_base(self):
        """Loads JSON guidelines from disk."""
        if not self.data_path.exists():
            self._guidelines = []
            return

        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                self._guidelines = json.load(f)
        except Exception as e:
            print(f"Error loading nutrition guidelines: {e}")
            self._guidelines = []

    def get_all_topics(self) -> List[str]:
        """Returns unique list of guideline topics."""
        return list({item.get("topic", "General") for item in self._guidelines})

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves the top-k most relevant guidelines matching the user query.
        Uses token overlap, keyword matches, and title matching for scoring.
        """
        if not self._guidelines or not query.strip():
            return []

        # Tokenize query into lowercase words (filtering short stop-words)
        stop_words = {
            "what", "are", "the", "and", "for", "how", "much", "does", "can", "should", "with", "from", "you",
            "day", "days", "daily", "every", "realistically", "count", "tell", "please", "about", "your", "give",
            "ways", "tips", "like", "good", "well", "best", "need", "want", "would", "could", "some"
        }
        query_tokens = [
            w for w in re.findall(r"\w+", query.lower())
            if len(w) > 2 and w not in stop_words
        ]

        # Check if query has Indian nutrition context
        indian_terms = {"india", "indian", "icmr", "nin", "dal", "roti", "millets", "ragi", "jowar", "chapati", "dahi", "sambar", "khichdi", "curd"}
        has_indian_context = any(t in query.lower() for t in indian_terms)

        scored_results = []
        for item in self._guidelines:
            score = 0
            title = item.get("title", "").lower()
            content = item.get("content", "").lower()
            keywords = [k.lower() for k in item.get("keywords", [])]
            org = item.get("source_organization", "").lower()
            has_kw_or_title = False

            for token in query_tokens:
                # High weight for keyword match
                if any(token in kw for kw in keywords):
                    score += 6
                    has_kw_or_title = True
                # Title match
                if token in title:
                    score += 5
                    has_kw_or_title = True
                # Exact word in content (bounded by word boundaries)
                elif re.search(rf"\b{re.escape(token)}\b", content):
                    score += 1

            # Prioritise ICMR-NIN evidence for Indian nutrition questions
            if has_indian_context and ("icmr" in org or "nin" in org):
                score += 3

            # Only retain guidelines with genuine keyword/title relevance or meaningful score
            if score >= 3 and has_kw_or_title:
                scored_results.append((score, item))

        # Sort descending by score
        scored_results.sort(key=lambda x: x[0], reverse=True)

        if not scored_results:
            return []

        # Filter out irrelevant low-scoring documents if top document is a strong match
        top_score = scored_results[0][0]
        min_threshold = max(3, top_score * 0.35)
        filtered = [item for sc, item in scored_results if sc >= min_threshold]

        return filtered[:top_k]

    def format_context_for_prompt(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved passages into a structured text context block
        for IBM Granite prompt grounding.
        """
        if not retrieved_docs:
            return "No specific nutrition guidelines retrieved."

        context_blocks = []
        for i, doc in enumerate(retrieved_docs, start=1):
            source = doc.get("source", "Unknown Authority")
            title = doc.get("title", "")
            content = doc.get("content", "")
            doc_name = doc.get("document_name", "")
            org = doc.get("source_organization", "")

            # Identify regional jurisdiction clearly
            if "usda" in org.lower():
                scope_str = "Regional Scope: United States Dietary Guidelines (USDA / HHS) — US Comparative Guidance"
            elif "icmr" in org.lower() or "nin" in org.lower():
                scope_str = "Regional Scope: Indian National Dietary Guidelines (ICMR-NIN) — Primary Authority for Indian Diets"
            else:
                scope_str = "Regional Scope: Global Public Health Guidelines (WHO)"

            block_lines = [
                f"[Document {i}]",
                f"Source: {source}",
                f"{scope_str}",
                f"Topic: {title}",
            ]
            if doc_name:
                block_lines.append(f"Official Publication: {doc_name}")
            block_lines.append(f"Content: {content}")

            context_blocks.append("\n".join(block_lines))

        return "\n\n".join(context_blocks)
