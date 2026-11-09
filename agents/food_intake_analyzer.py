"""
Feature 3: Food Intake Analyzer

Collects user food consumption logs (food name, portion, meal type),
calculates nutrient estimates using verified Indian Food Composition Tables (IFCT 2017)
and USDA FoodData Central data, retrieves ICMR-NIN/WHO guidelines via RAG,
and invokes IBM Granite 4.0 to provide practical, evidence-based nutritional commentary.

Safety Guardrails:
- Strictly verified food composition dataset (zero hallucinated nutrient values).
- Unmatched foods are explicitly flagged as "Nutrition data unavailable".
- All numerical totals computed deterministically in Python, never invented by the LLM.
- Clear non-clinical educational disclaimers.
- Does not diagnose nutritional deficiencies or prescribe therapeutic diets.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from modules.rag_retriever import NutritionKnowledgeRetriever
from modules.watsonx_client import WatsonxGraniteClient

ANALYZER_SAFETY_DISCLAIMER = (
    "⚠️ Educational Disclaimer: This intake analysis provides educational estimates based on standard "
    "food composition tables (ICMR-NIN IFCT 2017 and USDA FoodData Central). It does not constitute "
    "clinical medical nutrition therapy, medical diagnosis of nutritional deficiencies, or a therapeutic prescription. "
    "A single day's food log does not represent an individual's long-term nutritional status. "
    "Always consult a registered clinical dietitian or physician for personal medical or therapeutic guidance."
)

COMMON_ALLERGENS = {
    "peanuts": {
        "display": "Peanuts / Groundnuts",
        "keywords": ["peanut", "peanuts", "groundnut", "groundnuts", "moongphali", "mungfali", "singdana"],
    },
    "dairy": {
        "display": "Dairy / Milk",
        "keywords": ["dairy", "milk", "paneer", "curd", "dahi", "cheese", "ghee", "butter", "whey", "yogurt", "chai"],
    },
    "gluten": {
        "display": "Gluten / Wheat",
        "keywords": ["gluten", "wheat", "atta", "maida", "sooji", "suji", "semolina", "roti", "chapati", "paratha", "bread"],
    },
    "nuts": {
        "display": "Tree Nuts",
        "keywords": ["almond", "almonds", "badam", "cashew", "walnut", "pista", "pistachio"],
    },
}


class FoodIntakeAnalyzer:
    """
    Agent responsible for tracking food consumption logs, computing deterministic
    nutrient metrics against authoritative food tables, retrieving RAG evidence,
    and generating educational AI commentary via IBM Granite.
    """

    VALID_MEAL_TYPES = [
        "Breakfast",
        "Lunch",
        "Evening Snack",
        "Dinner",
    ]

    def validate_entry(self, food_name: str, portion: Any, meal_type: str) -> Tuple[bool, List[str]]:
        """
        Validates individual food log entry parameters.
        Returns:
            Tuple of (is_valid: bool, error_messages: List[str])
        """
        errors = []
        if not food_name or not str(food_name).strip():
            errors.append("Food name cannot be empty.")

        try:
            p = float(portion)
            if p <= 0:
                errors.append("Portion size must be greater than 0.")
            elif p > 50:
                errors.append("Portion size exceeds realistic single-day limit (maximum 50 servings).")
        except (ValueError, TypeError):
            errors.append("Portion size must be a valid number.")

        if meal_type not in self.VALID_MEAL_TYPES:
            errors.append(f"Meal type must be one of: {', '.join(self.VALID_MEAL_TYPES)}.")

        return len(errors) == 0, errors

    @staticmethod
    def identify_allergens(food_name: str, aliases: Optional[List[str]] = None) -> List[str]:
        """
        Identifies common potential allergen categories (peanuts, dairy, gluten, tree nuts)
        present in a food name or its known aliases.
        """
        tokens = [str(food_name).lower()]
        if aliases:
            tokens.extend([str(a).lower() for a in aliases])
        combined = " ".join(tokens)

        detected = []
        for allergen_key, allergen_info in COMMON_ALLERGENS.items():
            for kw in allergen_info["keywords"]:
                pattern = rf"(?:\b|_){re.escape(kw)}(?:\b|_)"
                if re.search(pattern, combined):
                    if allergen_info["display"] not in detected:
                        detected.append(allergen_info["display"])
                    break
        return detected

    @staticmethod
    def ensure_complete_response(text: str) -> str:
        """
        Post-processes model output to ensure response completeness:
        - If generation is truncated mid-sentence, trims back to the last complete sentence.
        - Guarantees that the educational safety disclaimer is appended if missing.
        """
        if not text or not text.strip():
            return ""

        cleaned = text.strip()

        # If text ends abruptly without terminal punctuation, trim back to the last complete sentence
        terminal_puncts = {".", "!", "?", '"', '”', ")", "*"}
        if cleaned[-1] not in terminal_puncts:
            last_punct = max(cleaned.rfind("."), cleaned.rfind("!"), cleaned.rfind("?"))
            if last_punct > 0:
                cleaned = cleaned[:last_punct + 1].strip()

        # Ensure educational safety disclaimer is present
        if "disclaimer" not in cleaned.lower():
            cleaned += f"\n\n{ANALYZER_SAFETY_DISCLAIMER}"

        return cleaned

    def __init__(
        self,
        client: Optional[WatsonxGraniteClient] = None,
        data_path: Optional[Path] = None,
    ):
        self.client = client or WatsonxGraniteClient()
        self.retriever = NutritionKnowledgeRetriever()

        if data_path is None:
            self.data_path = Path(__file__).resolve().parent.parent / "data" / "food_composition.json"
        else:
            self.data_path = data_path

        self.foods: List[Dict[str, Any]] = self._load_composition_data()

    def _load_composition_data(self) -> List[Dict[str, Any]]:
        """
        Loads the verified food composition dataset from data/food_composition.json.
        """
        if not self.data_path.exists():
            return []
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def get_available_foods(self) -> List[Dict[str, Any]]:
        """
        Returns list of all available verified food items for UI selection.
        """
        return self.foods

    def find_food(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Searches the food database for a match by ID, exact name, or alias.
        Returns the matching food dictionary, or None if no verified entry exists.
        """
        if not query or not query.strip():
            return None

        clean_query = query.strip().lower()

        # 1. Exact ID match
        for food in self.foods:
            if food.get("id", "").lower() == clean_query:
                return food

        # 2. Exact or substring name match
        for food in self.foods:
            food_name = food.get("name", "").lower()
            if clean_query == food_name:
                return food

        # 3. Alias exact match
        for food in self.foods:
            aliases = [a.lower() for a in food.get("aliases", [])]
            if clean_query in aliases:
                return food

        # 4. Token substring matching in name or aliases
        for food in self.foods:
            food_name = food.get("name", "").lower()
            aliases = [a.lower() for a in food.get("aliases", [])]
            if clean_query in food_name or any(clean_query in a for a in aliases):
                return food
            if any(a in clean_query for a in aliases):
                return food

        return None

    def calculate_intake(self, food_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Deterministically calculates the nutrient totals from a user's food log.
        Does NOT guess or fabricate nutrients for unknown foods.
        """
        total_calories = 0.0
        total_protein = 0.0
        total_carbs = 0.0
        total_fat = 0.0
        total_fiber = 0.0
        total_sodium = 0.0

        item_breakdowns = []
        unmatched_items = []
        matched_items_count = 0

        for entry in food_log:
            food_name = str(entry.get("food_name", "")).strip()
            if not food_name:
                continue

            try:
                portion = float(entry.get("portion", 1.0))
                if portion <= 0:
                    portion = 1.0
            except (ValueError, TypeError):
                portion = 1.0

            meal_type = entry.get("meal_type", "Lunch")

            matched_food = self.find_food(food_name)

            if matched_food:
                matched_items_count += 1
                cals = round(float(matched_food.get("calories_kcal", 0.0)) * portion, 1)
                protein = round(float(matched_food.get("protein_g", 0.0)) * portion, 1)
                carbs = round(float(matched_food.get("carbohydrates_g", 0.0)) * portion, 1)
                fat = round(float(matched_food.get("fat_g", 0.0)) * portion, 1)
                fiber = round(float(matched_food.get("dietary_fiber_g", 0.0)) * portion, 1)
                sodium = round(float(matched_food.get("sodium_mg", 0.0)) * portion, 1)
                item_allergens = self.identify_allergens(matched_food["name"], matched_food.get("aliases", []))

                total_calories += cals
                total_protein += protein
                total_carbs += carbs
                total_fat += fat
                total_fiber += fiber
                total_sodium += sodium

                item_breakdowns.append({
                    "food_name": matched_food["name"],
                    "logged_name": food_name,
                    "meal_type": meal_type,
                    "portion": portion,
                    "serving_desc": matched_food.get("default_serving_description", ""),
                    "source": matched_food.get("source", "IFCT 2017"),
                    "source_code": matched_food.get("source_code", matched_food.get("source", "IFCT 2017")),
                    "matched": True,
                    "allergens": item_allergens,
                    "calories_kcal": cals,
                    "protein_g": protein,
                    "carbohydrates_g": carbs,
                    "fat_g": fat,
                    "dietary_fiber_g": fiber,
                    "sodium_mg": sodium,
                })
            else:
                unmatched_items.append(food_name)
                unmatched_allergens = self.identify_allergens(food_name)
                item_breakdowns.append({
                    "food_name": food_name,
                    "logged_name": food_name,
                    "meal_type": meal_type,
                    "portion": portion,
                    "serving_desc": "Portion specified: x" + str(portion),
                    "source": "None - Nutrition data unavailable in verified dataset",
                    "source_code": "Unmatched in IFCT 2017 / USDA",
                    "matched": False,
                    "allergens": unmatched_allergens,
                    "calories_kcal": None,
                    "protein_g": None,
                    "carbohydrates_g": None,
                    "fat_g": None,
                    "dietary_fiber_g": None,
                    "sodium_mg": None,
                })

        # Calculate macronutrient calorie ratios if calories > 0
        # Protein = 4 kcal/g, Carbs = 4 kcal/g, Fat = 9 kcal/g
        macro_cals_protein = total_protein * 4.0
        macro_cals_carbs = total_carbs * 4.0
        macro_cals_fat = total_fat * 9.0
        macro_sum = macro_cals_protein + macro_cals_carbs + macro_cals_fat

        if macro_sum > 0:
            protein_pct = round((macro_cals_protein / macro_sum) * 100, 1)
            carbs_pct = round((macro_cals_carbs / macro_sum) * 100, 1)
            fat_pct = round((macro_cals_fat / macro_sum) * 100, 1)
        else:
            protein_pct, carbs_pct, fat_pct = 0.0, 0.0, 0.0

        is_partial = (len(unmatched_items) > 0 and matched_items_count > 0)
        has_no_data = (matched_items_count == 0)

        if is_partial:
            disclaimer_label = (
                f"⚠️ PARTIAL ESTIMATE: Totals reflect {matched_items_count} of {len(item_breakdowns)} logged food items. "
                f"{len(unmatched_items)} item(s) ([{', '.join(unmatched_items)}]) have unavailable nutritional data in verified tables "
                "and are completely EXCLUDED from numerical totals."
            )
        elif has_no_data:
            disclaimer_label = (
                f"⚠️ NUTRITIONAL DATA UNAVAILABLE: None of the {len(item_breakdowns)} logged item(s) could be verified in "
                "the IFCT 2017 / USDA composition tables. Zero numerical totals are fabricated."
            )
        else:
            disclaimer_label = (
                f"✅ COMPLETE VERIFIED ESTIMATE: All {matched_items_count} logged item(s) calculated deterministically from "
                "standard IFCT 2017 & USDA food composition tables."
            )

        return {
            "total_items": len(item_breakdowns),
            "matched_items_count": matched_items_count,
            "unmatched_items_count": len(unmatched_items),
            "unmatched_items": unmatched_items,
            "is_partial_estimate": is_partial,
            "has_no_verified_data": has_no_data,
            "total_calories_kcal": round(total_calories, 1),
            "total_protein_g": round(total_protein, 1),
            "total_carbohydrates_g": round(total_carbs, 1),
            "total_fat_g": round(total_fat, 1),
            "total_dietary_fiber_g": round(total_fiber, 1),
            "total_sodium_mg": round(total_sodium, 1),
            "macronutrient_ratios": {
                "carbohydrates_pct": carbs_pct,
                "carbs_pct": carbs_pct,
                "protein_pct": protein_pct,
                "fat_pct": fat_pct,
            },
            "item_breakdowns": item_breakdowns,
            "disclaimer_label": disclaimer_label,
        }

    def retrieve_guidelines(self) -> List[Dict[str, Any]]:
        """
        Retrieves relevant public health and ICMR-NIN nutrition guidelines
        to ground IBM Granite's commentary.
        """
        query = "ICMR-NIN protein RDA macronutrient balance dietary fiber sodium limits balanced diet"
        return self.retriever.search(query, top_k=3)

    def build_prompt(
        self,
        intake_summary: Dict[str, Any],
        context: str,
    ) -> str:
        """
        Constructs a grounded prompt for IBM Granite 4.0 with deterministic numbers and RAG evidence.
        """
        items_text = []
        for item in intake_summary["item_breakdowns"]:
            if item["matched"]:
                items_text.append(
                    f"- {item['meal_type']}: {item['food_name']} (Portion: x{item['portion']}) -> "
                    f"{item['calories_kcal']} kcal, {item['protein_g']}g protein, "
                    f"{item['carbohydrates_g']}g carbs, {item['fat_g']}g fat, "
                    f"{item['dietary_fiber_g']}g fiber [Source: {item['source']}]"
                )
            else:
                items_text.append(
                    f"- {item['meal_type']}: {item['food_name']} (Portion: x{item['portion']}) -> "
                    f"[Nutrition data unavailable in verified dataset — EXCLUDED from calculated totals]"
                )

        logged_items_str = "\n".join(items_text)
        macros = intake_summary["macronutrient_ratios"]

        estimate_status_header = (
            f"PARTIAL ESTIMATE ({intake_summary['matched_items_count']} of {intake_summary['total_items']} items calculated; "
            f"{intake_summary['unmatched_items_count']} item(s) excluded due to missing composition data)"
            if intake_summary.get("is_partial_estimate")
            else f"COMPLETE ESTIMATE (All {intake_summary['matched_items_count']} items calculated from verified tables)"
        )

        unmatched_note = ""
        if intake_summary["unmatched_items_count"] > 0:
            unmatched_list = ", ".join(intake_summary["unmatched_items"])
            unmatched_note = (
                f"\nCRITICAL INSTRUCTION ON EXCLUDED FOODS: {intake_summary['unmatched_items_count']} item(s) "
                f"([{unmatched_list}]) could not be verified in IFCT 2017 / USDA composition tables and have been "
                "completely EXCLUDED from the numerical totals below. "
                "You must explicitly inform the user that these totals represent a PARTIAL estimate. "
                "DO NOT invent, fabricate, or guess nutritional numbers for these excluded items.\n"
            )

        prompt = f"""You are the Food Intake Analyzer, an evidence-based nutritional analysis assistant.
You provide educational commentary on logged food intake grounded in ICMR-NIN and WHO public health standards.
You are NOT a doctor, clinical dietitian, or medical provider. Do not diagnose diseases or deficiencies.

VERIFIED PUBLIC HEALTH GUIDELINES:
---------------------------------------------------------------------
{context}
---------------------------------------------------------------------

CALCULATED FOOD INTAKE SUMMARY (From IFCT 2017 / USDA Tables):
---------------------------------------------------------------------
Status: {estimate_status_header}

Logged Food Items:
{logged_items_str}

Deterministic Nutrient Totals:
- Estimated Total Calories: {intake_summary['total_calories_kcal']} kcal
- Estimated Total Protein: {intake_summary['total_protein_g']} g ({macros['protein_pct']}% of macro calories)
- Estimated Total Carbohydrates: {intake_summary['total_carbohydrates_g']} g ({macros['carbs_pct']}% of macro calories)
- Estimated Total Fat: {intake_summary['total_fat_g']} g ({macros['fat_pct']}% of macro calories)
- Estimated Total Dietary Fiber: {intake_summary['total_dietary_fiber_g']} g
- Estimated Total Sodium: {intake_summary['total_sodium_mg']} mg
---------------------------------------------------------------------{unmatched_note}

TASK & CONSTRAINTS:
1. Explain the logged food intake in clear, non-judgmental language.
2. If this is a partial estimate, explicitly remind the user which items could not be calculated.
3. Identify general dietary strengths (e.g., presence of whole foods, pulses, fiber, vegetable intake).
4. Identify potential nutritional gaps or areas of improvement compared against general ICMR-NIN benchmarks (e.g., protein sufficiency against ~0.83 g/kg, sodium moderation <2000mg, fiber adequacy).
5. Suggest practical, balanced whole-food choices to complement the logged meals.
6. Emphasize that a single day's food intake does not indicate chronic deficiency or long-term nutritional status.
7. Do NOT invent, fabricate, or alter the numerical values above.
8. Structure each section concisely (2-3 direct sentences or bullet points) so that all observations and the safety disclaimer fit completely without truncation.
9. Conclude with a clear educational safety disclaimer.

NUTRITIONAL ANALYSIS:"""
        return prompt

    def _call_client(self, prompt: str, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Safely invokes the Watsonx client, supporting both generate_with_metadata and legacy generate_response.
        """
        if hasattr(self.client, "generate_with_metadata"):
            try:
                meta = self.client.generate_with_metadata(prompt=prompt, params=params)
                if isinstance(meta, dict) and isinstance(meta.get("text"), str):
                    return meta["text"], meta
            except Exception:
                pass
        raw = self.client.generate_response(prompt=prompt, params=params)
        return raw, {"token_usage": {}, "finish_reason": "unknown", "response_length": len(raw)}

    def analyze_intake(self, food_log: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Complete analysis pipeline:
        1. Validates food log.
        2. Calculates nutrient estimates deterministically.
        3. Retrieves grounded RAG context.
        4. Calls IBM Granite 4.0 via WatsonxGraniteClient.
        5. Returns structured result.
        """
        if not food_log or len(food_log) == 0:
            return {
                "success": False,
                "intake_summary": None,
                "analysis_commentary": None,
                "retrieved_docs": [],
                "prompt": "",
                "status_message": "Food log is empty. Please add at least one food item before analyzing.",
                "disclaimer": ANALYZER_SAFETY_DISCLAIMER,
                "used_live_model": False,
            }

        intake_summary = self.calculate_intake(food_log)
        retrieved_docs = self.retrieve_guidelines()
        formatted_context = self.retriever.format_context_for_prompt(retrieved_docs)
        prompt = self.build_prompt(intake_summary, formatted_context)

        result = {
            "success": False,
            "intake_summary": intake_summary,
            "analysis_commentary": None,
            "generation_metadata": {},
            "retrieved_docs": retrieved_docs,
            "prompt": prompt,
            "status_message": "",
            "disclaimer": ANALYZER_SAFETY_DISCLAIMER,
            "used_live_model": False,
        }

        if self.client.is_configured():
            try:
                params = {
                    "max_new_tokens": 700,
                    "min_new_tokens": 20,
                    "temperature": 0.3,
                }
                raw_response, meta = self._call_client(prompt=prompt, params=params)
                completed_response = self.ensure_complete_response(raw_response)
                result["analysis_commentary"] = completed_response
                result["generation_metadata"] = meta
                result["success"] = True
                result["used_live_model"] = True
                result["status_message"] = "Dietary intake analysis successfully completed with IBM Granite 4.0."
            except Exception as e:
                result["analysis_commentary"] = None
                result["status_message"] = f"IBM watsonx inference notice: {str(e)}"
        else:
            result["analysis_commentary"] = None
            result["status_message"] = (
                "IBM watsonx credentials are not configured. Nutrient totals calculated deterministically, "
                "but AI commentary cannot be generated."
            )

        return result
