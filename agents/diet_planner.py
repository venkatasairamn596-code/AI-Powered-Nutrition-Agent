"""
Feature 2: Personalised Diet Planner

Collects user nutritional preferences, fitness goals, and allergen constraints,
retrieves grounded ICMR-NIN evidence using RAG, and prompts IBM Granite to
generate a culturally relevant, practical Indian meal plan with nutritional reasoning.

Includes comprehensive safety guardrails:
- Allergen synonym expansion
- Post-generation allergen safety scanning & blocking
- Under-18 adolescent growth protections
- Clinical condition advisories
"""

import re
from typing import Dict, Any, List, Tuple, Optional, Set
from modules.rag_retriever import NutritionKnowledgeRetriever
from modules.watsonx_client import WatsonxGraniteClient

PLANNER_SAFETY_DISCLAIMER = (
    "⚠️ Medical Disclaimer: This personalized meal plan is generated for general wellness "
    "and educational purposes only. It does not constitute medical nutrition therapy, clinical "
    "dietetics, or a medical prescription. Individuals with chronic medical conditions "
    "(such as diabetes, chronic kidney disease, severe food allergies, or cardiovascular disease) "
    "must consult a registered dietitian or physician before making significant dietary modifications."
)

# Comprehensive food allergen synonym mapping
ALLERGEN_SYNONYMS: Dict[str, List[str]] = {
    "peanut": ["peanut", "peanuts", "groundnut", "groundnuts", "moongphali", "mungfali", "arachis"],
    "dairy": ["milk", "dairy", "paneer", "cheese", "curd", "dahi", "yogurt", "yoghurt", "butter", "ghee", "whey", "buttermilk", "cream", "malai", "khoya", "mawa", "lactose"],
    "gluten": ["gluten", "wheat", "atta", "maida", "sooji", "suji", "semolina", "rava", "barley", "rye"],
    "egg": ["egg", "eggs", "anda", "albumin", "mayonnaise"],
    "soy": ["soy", "soya", "soybean", "tofu", "edamame"],
    "tree nut": ["tree nut", "tree nuts", "almond", "almonds", "badam", "cashew", "cashews", "kaju", "walnut", "walnuts", "akhrot", "pistachio", "pista"],
    "shellfish": ["shellfish", "prawn", "prawns", "shrimp", "shrimps", "crab", "crabs", "lobster", "lobsters"],
    "sesame": ["sesame", "til"],
    "mustard": ["mustard", "sarson", "rai"],
}


class PersonalisedDietPlanner:
    """
    Agent responsible for input validation, RAG evidence retrieval,
    safety auditing, and IBM Granite prompt assembly for personalized Indian meal planning.
    """

    VALID_DIET_PREFERENCES = [
        "Vegetarian",
        "Eggetarian",
        "Non-Vegetarian",
        "Vegan",
    ]

    VALID_ACTIVITY_LEVELS = [
        "Sedentary (desk job, minimal exercise)",
        "Moderate (active daily routine / 3-4 days exercise)",
        "Heavy / Very Active (physical labor or daily intense athletics)",
    ]

    VALID_GOALS = [
        "Balanced Daily Nutrition",
        "Weight Management / Calorie Consciousness",
        "Muscle Maintenance & Strength",
        "Energy, Stamina & Gut Health",
    ]

    def __init__(self, client: Optional[WatsonxGraniteClient] = None):
        self.retriever = NutritionKnowledgeRetriever()
        self.client = client or WatsonxGraniteClient()

    @staticmethod
    def get_allergen_search_terms(allergies_text: str) -> Set[str]:
        """
        Parses declared food allergies and expands them with common synonyms and Indian regional terms.
        """
        if not allergies_text or allergies_text.strip().lower() in ["none", "no", "n/a", "nil"]:
            return set()

        raw_tokens = [t.strip().lower() for t in re.split(r"[,;/]+", allergies_text) if t.strip()]
        search_terms = set(raw_tokens)

        for token in raw_tokens:
            for root_key, synonyms in ALLERGEN_SYNONYMS.items():
                if token in root_key or any(token in syn for syn in synonyms):
                    search_terms.update(synonyms)

        return search_terms

    def check_post_generation_safety(self, meal_plan: str, allergies_text: str) -> Tuple[bool, List[str]]:
        """
        Post-generation safety audit:
        Scans the generated meal plan text to verify that none of the user's declared allergens
        or their synonyms were inadvertently suggested by the AI.

        Returns:
            Tuple of (is_safe: bool, detected_violating_terms: List[str])
        """
        search_terms = self.get_allergen_search_terms(allergies_text)
        if not search_terms:
            return True, []

        plan_lower = meal_plan.lower()
        detected_violations = []

        for term in sorted(search_terms):
            # Match whole words to avoid false substring matches
            pattern = rf"\b{re.escape(term)}\b"
            if re.search(pattern, plan_lower):
                detected_violations.append(term)

        is_safe = len(detected_violations) == 0
        return is_safe, detected_violations

    def validate_inputs(self, profile: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """
        Validates the user input profile.
        Returns:
            Tuple of (is_valid: bool, errors: List[str], warnings: List[str])
        """
        errors = []
        warnings = []

        # 1. Age validation
        age = profile.get("age")
        if age is None:
            errors.append("Age is required.")
        else:
            try:
                age_val = int(age)
                if age_val < 10 or age_val > 115:
                    errors.append("Please enter a valid age between 10 and 115.")
                elif age_val < 18:
                    warnings.append(
                        "Adolescent Growth Safeguard: For individuals under 18, adult weight-loss "
                        "and calorie-restriction regimens are not clinically advised. The meal plan "
                        "will focus strictly on wholesome nutrient density, energy sufficiency, and natural growth."
                    )
            except (ValueError, TypeError):
                errors.append("Age must be a valid integer.")

        # 2. Dietary preference validation
        diet = profile.get("dietary_preference", "")
        if not diet or diet not in self.VALID_DIET_PREFERENCES:
            errors.append(f"Please select a valid dietary preference ({', '.join(self.VALID_DIET_PREFERENCES)}).")

        # 3. Activity level validation
        activity = profile.get("activity_level", "")
        if not activity:
            errors.append("Activity level is required.")

        # 4. Nutrition goal validation
        goal = profile.get("nutrition_goal", "")
        if not goal:
            errors.append("Nutrition goal is required.")

        # 5. Food allergies & health warnings
        allergies = profile.get("food_allergies", "").strip()
        restrictions = profile.get("dietary_restrictions", "").strip()

        if allergies and allergies.lower() not in ["none", "no", "n/a", "nil"]:
            expanded = self.get_allergen_search_terms(allergies)
            warnings.append(
                f"Declared Allergies: '{allergies}'. The safety filter has expanded this to actively block: "
                f"[{', '.join(sorted(list(expanded)[:8]))}...]. All recommendations will be verified."
            )

        # 6. Clinical conditions detection
        clinical_keywords = {
            "diabetes": "Diabetes / Blood Glucose Disorder",
            "diabetic": "Diabetes / Blood Glucose Disorder",
            "kidney": "Renal / Kidney Disease",
            "renal": "Renal / Kidney Disease",
            "dialysis": "Dialysis Treatment",
            "cardiac": "Cardiovascular / Heart Disease",
            "heart": "Cardiovascular / Heart Disease",
            "hypertension": "Hypertension / High Blood Pressure",
            "celiac": "Celiac Disease / Gluten Enteropathy",
            "coeliac": "Celiac Disease / Gluten Enteropathy",
            "liver": "Hepatic / Liver Disease",
        }
        combined_text = f"{allergies} {restrictions}".lower()
        detected_conditions = set()
        for kw, condition_name in clinical_keywords.items():
            if kw in combined_text:
                detected_conditions.add(condition_name)

        if detected_conditions:
            warnings.append(
                f"Clinical Health Advisory: You reported conditions ({', '.join(detected_conditions)}). "
                "The Nutrition Agent provides general educational guidance only and cannot prescribe therapeutic medical nutrition therapy. "
                "Always seek personalized supervision from your primary healthcare provider or clinical dietitian."
            )

        is_valid = len(errors) == 0
        return is_valid, errors, warnings

    def retrieve_evidence(self, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Retrieves relevant ICMR-NIN and public health guidelines based on user profile.
        """
        query_terms = [
            profile.get("dietary_preference", "balanced diet"),
            profile.get("nutrition_goal", "macronutrients"),
            "ICMR balanced diet millets protein water",
        ]
        query = " ".join(query_terms)
        return self.retriever.search(query, top_k=3)

    def build_prompt(self, profile: Dict[str, Any], context: str) -> str:
        """
        Builds a grounded, safety-constrained prompt for IBM Granite.
        """
        allergies = profile.get("food_allergies", "").strip() or "None reported"
        restrictions = profile.get("dietary_restrictions", "").strip() or "None reported"
        preferred_foods = profile.get("preferred_foods", "").strip() or "Standard healthy Indian regional foods"
        age = profile.get("age", 25)

        expanded_allergens = self.get_allergen_search_terms(allergies)
        allergen_constraint_str = ", ".join(sorted(expanded_allergens)) if expanded_allergens else "None"

        # Special adolescent guidance instruction
        adolescent_instruction = ""
        if isinstance(age, (int, float)) and age < 18:
            adolescent_instruction = (
                "ADOBLESCENT GROWTH SAFEGUARD: The user is under 18 years old. "
                "Do NOT recommend restrictive calorie deficits, fasting, or bodybuilding supplements. "
                "Focus entirely on wholesome nutrient density, energy for academic/sports performance, and balanced growth.\n"
            )

        prompt = f"""You are the Personalised Diet Planner, an evidence-based Indian dietary advisor.
You provide educational nutrition guidance grounded in ICMR-NIN standards. You are NOT a doctor and must not prescribe clinical treatments.

VERIFIED NUTRITIONAL GUIDELINES (ICMR-NIN & PUBLIC HEALTH STANDARDS):
---------------------------------------------------------------------
{context}
---------------------------------------------------------------------

USER PROFILE:
- Age: {age} years
- Dietary Preference: {profile.get('dietary_preference')}
- Physical Activity Level: {profile.get('activity_level')}
- Primary Nutrition Goal: {profile.get('nutrition_goal')}
- Preferred Regional Indian Foods: {preferred_foods}
- CRITICAL Food Allergies to AVOID: {allergies} (Expanded prohibited ingredients: [{allergen_constraint_str}])
- Special Dietary Restrictions: {restrictions}

TASK & CONSTRAINTS:
1. Design a practical, culturally authentic 1-day Indian meal plan with 4 parts:
   - Breakfast
   - Lunch
   - Evening Snack / Mid-day refreshment
   - Dinner
2. STRICT ALLERGEN SAFETY: Do NOT include, mention, or suggest any ingredient from: [{allergen_constraint_str}].
{adolescent_instruction}3. Ground your plan in ICMR-NIN principles:
   - Combine whole grains / millets (ragi, jowar, bajra, brown rice, whole wheat) with pulses/dal in healthy ratios.
   - Include diverse vegetables and colorful produce.
   - Align hydration with activity level.
4. Nutritional Reasoning: Briefly explain why these specific meals were chosen for this user profile.
5. Identify Assumptions: State clearly that quantities are educational estimates, not clinical prescriptions.
6. Safety Disclaimer: Conclude with the required non-medical safety notice.

MEAL PLAN:"""
        return prompt

    def generate_plan(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Full orchestration pipeline with pre-validation and post-generation safety scan:
        1. Validates user input.
        2. Retrieves ICMR-NIN evidence.
        3. Assembles prompt.
        4. Calls IBM Granite via watsonx.ai client.
        5. Runs post-generation allergen safety scan.
        """
        is_valid, errors, warnings = self.validate_inputs(profile)

        if not is_valid:
            return {
                "success": False,
                "meal_plan": None,
                "retrieved_docs": [],
                "prompt": "",
                "validation_errors": errors,
                "safety_warnings": warnings,
                "allergen_blocked": False,
                "disclaimer": PLANNER_SAFETY_DISCLAIMER,
                "status_message": "Input validation failed. Please correct the fields above.",
                "used_live_model": False,
            }

        retrieved_docs = self.retrieve_evidence(profile)
        formatted_context = self.retriever.format_context_for_prompt(retrieved_docs)
        prompt = self.build_prompt(profile, formatted_context)

        result = {
            "success": False,
            "meal_plan": None,
            "retrieved_docs": retrieved_docs,
            "prompt": prompt,
            "validation_errors": [],
            "safety_warnings": warnings,
            "allergen_blocked": False,
            "disclaimer": PLANNER_SAFETY_DISCLAIMER,
            "status_message": "",
            "used_live_model": False,
        }

        if self.client.is_configured():
            try:
                params = {
                    "max_new_tokens": 450,
                    "min_new_tokens": 10,
                    "temperature": 0.25,
                }
                raw_response = self.client.generate_response(prompt=prompt, params=params)
                generated_plan = raw_response.strip()

                # POST-GENERATION SAFETY SCAN
                allergies = profile.get("food_allergies", "")
                is_safe, violating_terms = self.check_post_generation_safety(generated_plan, allergies)

                if not is_safe:
                    # Allergen safety violation detected: BLOCK THE MEAL PLAN
                    result["success"] = False
                    result["meal_plan"] = None
                    result["allergen_blocked"] = True
                    result["status_message"] = (
                        f"🚨 Safety Alert: The generated meal plan contained prohibited ingredient(s) "
                        f"matching your allergen profile ({', '.join(violating_terms)}). For your safety, "
                        "this meal plan has been suppressed. Please consult a registered clinical dietitian."
                    )
                else:
                    result["meal_plan"] = generated_plan
                    result["success"] = True
                    result["used_live_model"] = True
                    result["status_message"] = "Personalised meal plan successfully generated and verified by safety audit."

            except Exception as e:
                result["meal_plan"] = None
                result["status_message"] = f"IBM watsonx inference notice: {str(e)}"
        else:
            result["meal_plan"] = None
            result["status_message"] = (
                "IBM watsonx credentials are not configured. Cannot generate live meal plan."
            )

        return result
