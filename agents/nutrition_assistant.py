"""
Feature 1: Nutrition Knowledge Assistant

Combines local RAG knowledge retrieval (WHO, ICMR-NIN, USDA) with
IBM Granite foundation model generation.
Includes mandatory non-clinical safety guardrails and source citations.
"""

from typing import Dict, Any, List, Optional
from modules.rag_retriever import NutritionKnowledgeRetriever
from modules.watsonx_client import WatsonxGraniteClient

# Standard safety disclaimer required across all outputs
SAFETY_DISCLAIMER = (
    "⚠️ Medical Disclaimer: This information is provided for educational purposes only "
    "and does not constitute medical advice, clinical diagnosis, or medical nutrition therapy. "
    "Always consult a registered dietitian, physician, or certified healthcare provider for "
    "individual medical needs or health conditions."
)


class NutritionKnowledgeAssistant:
    """
    RAG-powered Nutrition Knowledge Assistant.
    """

    def __init__(self, client: Optional[WatsonxGraniteClient] = None):
        self.retriever = NutritionKnowledgeRetriever()
        self.client = client or WatsonxGraniteClient()

    @staticmethod
    def ensure_complete_response(text: str) -> str:
        """
        Post-processes model output to ensure response completeness:
        - If generation is truncated mid-sentence, trims back to the last complete sentence.
        - Guarantees that the medical safety disclaimer is included.
        """
        if not text or not text.strip():
            return ""

        cleaned = text.strip()

        # If text ends abruptly without terminal punctuation, trim back to the last complete sentence
        terminal_puncts = {".", "!", "?", "\"", "”", "\n", ")"}
        if cleaned[-1] not in terminal_puncts:
            last_punct = max(cleaned.rfind("."), cleaned.rfind("!"), cleaned.rfind("?"))
            if last_punct > 0:
                cleaned = cleaned[:last_punct + 1].strip()

        # Ensure safety disclaimer is present
        if "disclaimer" not in cleaned.lower():
            cleaned += f"\n\n{SAFETY_DISCLAIMER}"

        return cleaned

    def build_prompt(self, query: str, context: str) -> str:
        """
        Builds a grounded prompt for IBM Granite foundation model with clear
        prioritization of ICMR-NIN guidelines for Indian nutrition contexts.
        """
        prompt = f"""You are the Nutrition Knowledge Assistant, an evidence-based AI assistant for nutrition guidance.

CONTEXT FROM VERIFIED NUTRITION AUTHORITIES:
--------------------------------------------
{context}
--------------------------------------------

USER QUESTION:
{query}

INSTRUCTIONS:
1. Ground your response strictly in the verified nutrition context provided above.
2. Prioritise ICMR-NIN evidence for questions concerning Indian dietary patterns. If citing USDA guidelines, explicitly identify them as United States guidance.
3. If the user asks for a specific count of items (such as 'three simple ways'), provide all requested recommendations in full, using complete, grammatically finished sentences.
4. Clearly cite the authoritative source (e.g., [Source: ICMR-NIN], [Source: WHO], or [Source: USDA]) for every factual recommendation.
5. Be clear, concise, objective, and beginner-friendly. Structure your points cleanly so the full answer fits without truncation.
6. Do not offer clinical diagnoses or prescribe treatments.
7. Conclude your response with the required medical safety disclaimer.

ANSWER:"""
        return prompt

    def answer_question(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Processes a nutrition question:
        1. Retrieves relevant guidelines from the local knowledge base.
        2. Formats the grounding context.
        3. Attempts IBM Granite inference if configured, or returns grounded context.
        """
        docs = self.retriever.search(query, top_k=top_k)
        formatted_context = self.retriever.format_context_for_prompt(docs)
        prompt = self.build_prompt(query, formatted_context)

        sources = list({doc.get("source", "Authoritative Source") for doc in docs})

        result = {
            "success": False,
            "query": query,
            "retrieved_docs": docs,
            "sources": sources,
            "prompt": prompt,
            "answer": None,
            "used_live_model": False,
            "disclaimer": SAFETY_DISCLAIMER,
            "status_message": "",
        }

        # Check if IBM watsonx client is configured
        if self.client.is_configured():
            try:
                # Pass sufficient token budget (400 tokens) to prevent cutoff
                params = {
                    "max_new_tokens": 400,
                    "min_new_tokens": 50,
                    "temperature": 0.2,
                    "decoding_method": "greedy",
                }
                raw_response = self.client.generate_response(prompt=prompt, params=params)
                result["answer"] = self.ensure_complete_response(raw_response)
                result["used_live_model"] = True
                result["success"] = True
                result["status_message"] = "Response generated by IBM Granite via watsonx.ai."
            except Exception as e:
                result["status_message"] = f"watsonx.ai Notice: {str(e)}"
        else:
            result["status_message"] = (
                "IBM watsonx credentials are not yet configured. "
                "Retrieved authoritative guidelines are displayed below."
            )

        return result
