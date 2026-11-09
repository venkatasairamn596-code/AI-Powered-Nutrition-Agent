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
        Handles both individual tokens and compound expressions (e.g. 'peanut allergy', 'dairy intolerance').
        """
        if not allergies_text or allergies_text.strip().lower() in ["none", "no", "n/a", "nil"]:
            return set()

        raw_tokens = [t.strip().lower() for t in re.split(r"[,;/]+", allergies_text) if t.strip()]
        search_terms = set(raw_tokens)

        # Also extract individual words from tokens to catch phrases like 'peanut allergy'
        words_to_check = set(raw_tokens)
        for t in raw_tokens:
            words_to_check.update(re.findall(r"\b[a-zA-Z]{3,}\b", t))

        stop_words = {"allergy", "allergies", "allergic", "severe", "intolerance", "intolerant", "sensitive", "none", "and", "with", "only"}
        for word in words_to_check:
            if word in stop_words:
                continue
            for root_key, synonyms in ALLERGEN_SYNONYMS.items():
                if word == root_key or word in synonyms or root_key in word or any(syn in word for syn in synonyms):
                    search_terms.update(synonyms)

        return search_terms

    @classmethod
    def is_occurrence_safe_avoidance(cls, text: str, start: int, end: int, term: str) -> bool:
        """
        Evaluates whether a specific occurrence of an allergen term (at text[start:end])
        is strictly and unambiguously part of an avoidance warning, safety disclaimer, or allergy declaration,
        and NOT part of an actual food/ingredient recommendation.

        Safety Principle: If ambiguous, returns False (which causes the plan to be blocked).
        """
        left_text = text[:start]
        right_text = text[end:]

        # 1. Find nearest sentence boundaries preceding and following the term
        sent_start_indices = [
            left_text.rfind("."),
            left_text.rfind("!"),
            left_text.rfind("?"),
            left_text.rfind("\n"),
        ]
        sent_start = max(sent_start_indices)
        sent_start = 0 if sent_start == -1 else sent_start + 1

        sent_end_indices = [
            right_text.find("."),
            right_text.find("!"),
            right_text.find("?"),
            right_text.find("\n"),
        ]
        valid_ends = [b for b in sent_end_indices if b != -1]
        sent_end = len(text) if not valid_ends else end + min(valid_ends)

        sentence = text[sent_start:sent_end].strip()
        sent_lower = sentence.lower()

        # 2. Find clause boundaries within the sentence (semicolons, dashes, contrasting conjunctions)
        clause_start_in_left = max(
            left_text.rfind(";"),
            left_text.rfind("—"),
            left_text.rfind("--"),
        )
        contrast_match = None
        for m in re.finditer(r"\b(?:but|however|although|yet|except|whereas)\b", left_text[sent_start:], re.IGNORECASE):
            contrast_match = sent_start + m.end()

        effective_start = max(sent_start, clause_start_in_left + 1 if clause_start_in_left != -1 else 0)
        if contrast_match and contrast_match > effective_start:
            effective_start = contrast_match

        clause_left = text[effective_start:start].lower().strip()

        clause_end_in_right = min([b for b in [
            right_text.find(";"),
            right_text.find("—"),
            right_text.find("--"),
        ] if b != -1] or [len(right_text)])

        m_right = re.search(r"\b(?:but|however|although|yet|except|whereas)\b", right_text[:clause_end_in_right], re.IGNORECASE)
        if m_right:
            clause_end_in_right = m_right.start()

        effective_end = min(sent_end, end + clause_end_in_right)
        clause_right = text[end:effective_end].lower().strip()

        # 3. Check for positive culinary preparation / serving words immediately preceding the allergen
        # (unless governed by an avoidance or negative phrase like 'do not eat')
        positive_prep_pattern = (
            r"\b(?:roasted|handful\s+of|crushed|boiled|fried|garnish(?:ed)?\s+with|"
            r"topped\s+with|serve(?:d)?\s+with|cooked\s+in|eat|snack\s+on|"
            r"cup\s+of|tbsp\s+of|tablespoon\s+of|spoon\s+of|grams?\s+of|g\s+of)\s*$"
        )
        is_negated_prep = bool(re.search(r"\b(?:do\s+not|never|should\s+not|must\s+not|cannot|avoid)\s+\w+\s*$", clause_left))
        if not is_negated_prep and re.search(positive_prep_pattern, clause_left):
            return False

        # 4. Category 1: Explicit Preceding Avoidance Verb / Negation in clause_left
        avoidance_prefix_pattern = (
            r"\b(?:"
            r"avoid(?:s|ed|ing|ance)?|"
            r"strictly\s+avoid(?:s|ed|ing|ance)?|"
            r"must\s+avoid|"
            r"should\s+avoid|"
            r"do\s+not\s+(?:consume|eat|use|include|add|have|take|recommend)|"
            r"never\s+(?:consume|eat|use|include|add|have|take|recommend)|"
            r"cannot\s+(?:consume|eat|use|include|add|have|take)|"
            r"must\s+not\s+(?:consume|eat|use|include|add|have|take)|"
            r"should\s+not\s+(?:consume|eat|use|include|add|have|take)|"
            r"exclude(?:s|d|ing)?|"
            r"exclusion\s+of|"
            r"without|"
            r"with\s+no|"
            r"with\s+zero|"
            r"free\s+(?:from|of)|"
            r"contains?\s+no|"
            r"containing\s+no|"
            r"eliminate(?:s|d|ing)?|"
            r"elimination\s+of|"
            r"omit(?:s|ted|ting)?|"
            r"omission\s+of|"
            r"steer\s+clear\s+of|"
            r"stay\s+away\s+from|"
            r"refrain\s+from|"
            r"prohibit(?:s|ed|ing)?|"
            r"prohibition\s+of|"
            r"no|"
            r"zero"
            r")\b"
        )
        avoidance_match = None
        for m in re.finditer(avoidance_prefix_pattern, clause_left):
            avoidance_match = m

        if avoidance_match:
            between_text = clause_left[avoidance_match.end():].strip()
            conflicting_verbs = re.search(r"\b(?:eat|enjoy|include|add|have|consume|serve|recommend|try|with)\b", between_text)
            if not conflicting_verbs:
                return True

        # 5. Category 2: Explicit Following Safety Predicate in clause_right
        safety_predicate_pattern = (
            r"^(?:\s*(?:and|or|,|:)\s+[\w\s,]+?)?\s*(?:"
            r":?\s*(?:is|are|was|were)?\s*(?:strictly\s+|completely\s+)?(?:unsafe|not\s+safe|prohibited|forbidden|excluded|omitted|avoided|restricted|none|not\s+recommended)|"
            r":?\s*(?:has|have)\s+been\s+(?:strictly\s+|completely\s+)?(?:excluded|omitted|avoided|eliminated|removed)|"
            r":?\s*(?:should|must)\s+not\s+be\s+(?:consumed|eaten|used|included|added)|"
            r":?\s*(?:is|are)\s+to\s+be\s+avoided|"
            r":?\s*(?:is|are)\s+(?:strictly\s+)?off-limits|"
            r":?\s*(?:must\s+be\s+avoided|avoid\s+completely|avoid\s+strictly|avoided\s+completely)"
            r")\b"
        )
        if re.search(safety_predicate_pattern, clause_right):
            return True

        # 6. Category 3: Allergy Profile / Clinical Header Context
        allergy_header_pattern = (
            r"\b(?:"
            r"(?:critical\s+)?(?:food\s+)?allerg(?:y|ies|ic|ens?)(?:\s+to\s+avoid)?(?:\s*[:\-])?|"
            r"prohibited\s+(?:food\s+)?allerg(?:ens?|ies)(?:\s*[:\-])?|"
            r"allergic\s+to(?:\s*[:\-])?|"
            r"allergy\s+to(?:\s*[:\-])?|"
            r"allergies\s+to(?:\s*[:\-])?|"
            r"user[\'’]?s?\s+(?:reported\s+)?allerg(?:y|ies|ens?)(?:\s*[:\-])?|"
            r"declared\s+(?:food\s+)?allerg(?:y|ies)(?:\s*[:\-])?"
            r")\s*$"
        )
        if re.search(allergy_header_pattern, clause_left):
            return True

        if re.search(r"^\s*allerg(?:y|ies|ic)\b", clause_right):
            return True

        if "allerg" in sent_lower:
            if re.search(rf"\b(?:due\s+to|because\s+of|as\s+the\s+user\s+is|since\s+the\s+user\s+is)\b[^.!?;\n]*?\b{re.escape(term)}\b[^.!?;\n]*?\ballerg", sent_lower):
                return True
            if re.search(rf"\b{re.escape(term)}\b\s+allerg(?:y|ies|ic)\b", sent_lower):
                return True

        # 7. Category 4: Hyphenated free suffix
        if clause_right.startswith("-free"):
            return True

        # Ambiguous or positive: unsafe
        return False

    def check_post_generation_safety(self, meal_plan: str, allergies_text: str) -> Tuple[bool, List[str]]:
        """
        Post-generation safety audit:
        Scans the generated meal plan text to verify that none of the user's declared allergens
        or their synonyms were inadvertently suggested or recommended by the AI.

        Safety Guardrail Rules:
        1. If an allergen term is detected as a recommended food or ingredient in any meal,
           the plan is marked UNSAFE and blocked.
        2. If an allergen term appears strictly and unambiguously within a negative warning,
           avoidance advisory, or medical disclaimer (e.g. 'Avoid peanuts', 'Peanuts are unsafe for this user'),
           it does NOT trigger a false positive when no allergen is recommended.
        3. If there is ANY ambiguity about whether an allergen is being recommended, safety takes
           precedence: the plan is marked UNSAFE and blocked.

        Returns:
            Tuple of (is_safe: bool, detected_violating_terms: List[str])
        """
        search_terms = self.get_allergen_search_terms(allergies_text)
        if not search_terms or not meal_plan:
            return True, []

        detected_violations = []

        # Find all occurrences of any allergen term in the meal plan
        for term in sorted(search_terms, key=len, reverse=True):
            pattern = rf"\b{re.escape(term)}\b"
            for match in re.finditer(pattern, meal_plan, re.IGNORECASE):
                start, end = match.span()
                # Verify if this specific occurrence is unambiguously an avoidance warning / disclaimer
                if not self.is_occurrence_safe_avoidance(meal_plan, start, end, term):
                    if term not in detected_violations:
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

    @classmethod
    def check_plan_completeness(cls, meal_plan: str) -> Tuple[bool, List[str]]:
        """
        Validates that the generated meal plan contains all four required meals
        (Breakfast, Lunch, Evening Snack, Dinner) and nutritional reasoning,
        and does not terminate abruptly.

        Note: The medical disclaimer is application-controlled and appended after validation,
        so it is not required in the model's raw token stream.

        Returns:
            Tuple of (is_complete: bool, missing_elements: List[str])
        """
        if not meal_plan or not meal_plan.strip():
            return False, ["Entire meal plan is empty"]

        text = meal_plan.strip()
        text_lower = text.lower()
        missing = []

        # 1. Four meals
        if not re.search(r"\bbreakfast\b", text_lower):
            missing.append("Breakfast")
        if not re.search(r"\blunch\b", text_lower):
            missing.append("Lunch")
        if not re.search(r"\b(?:evening\s+snack|mid-day\s+snack|mid-day\s+refreshment|afternoon\s+snack|snack|refreshment)\b", text_lower):
            missing.append("Evening Snack")
        if not re.search(r"\bdinner\b", text_lower):
            missing.append("Dinner")

        # 2. Nutritional Reasoning (flexible matching to avoid false positives on wording variations)
        reasoning_pattern = (
            r"\b(?:"
            r"nutritional\s+reasoning|reasoning|rationale|nutritional\s+justification|"
            r"why\s+these\s+meals|nutritional\s+benefits?|nutrition\s+benefits?|"
            r"nutritional\s+balance|nutritional\s+focus|nutrition\s+notes?|"
            r"dietary\s+rationale|nutritional\s+analysis|nutritional\s+value|"
            r"nutritional\s+explanation|meal\s+rationale|health\s+benefits?|"
            r"icmr-nin\s+alignment|macronutrient\s+balance"
            r")\b"
        )
        if not re.search(reasoning_pattern, text_lower):
            missing.append("Nutritional Reasoning")

        # 3. Check abrupt truncation:
        # A. Ends in unclosed/dangling punctuation: e.g. 'Curd (Yogurt) ('
        cleaned_tail = text.rstrip(" *_\t\r\n")
        if cleaned_tail and cleaned_tail[-1] in "(,:-[{/\\":
            missing.append("Unfinished sentence / dangling punctuation")
        # B. Unclosed parentheses or brackets:
        elif text.count("(") > text.count(")") or text.count("[") > text.count("]"):
            missing.append("Unclosed parenthesis or bracket")
        # C. Dangling grammatical conjunction/preposition at the very end:
        elif re.search(r"\b(?:and|or|with|of|the|to|in|for|containing|including|plus|such\s+as)\s*[*_]*$", text, re.IGNORECASE):
            missing.append("Dangling trailing conjunction")

        is_complete = len(missing) == 0
        return is_complete, missing

    def build_prompt(self, profile: Dict[str, Any], context: str) -> str:
        """
        Builds a grounded, safety-constrained prompt for IBM Granite.
        Requests concise, structured output to avoid token exhaustion.
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
1. Four Complete Meals (Be concise and practical; 1-2 lines per meal):
   - Breakfast: Wholesome morning meal with regional items.
   - Lunch: Balanced mid-day meal adhering to ICMR-NIN cereal-pulse-vegetable ratios.
   - Evening Snack: Afternoon nourishment or refreshment.
   - Dinner: Light, nourishing evening meal.
2. STRICT ALLERGEN SAFETY: Do NOT include, mention, or suggest any ingredient from: [{allergen_constraint_str}].
{adolescent_instruction}3. Ground your plan in ICMR-NIN principles:
   - Combine whole grains / millets with pulses/dal in healthy ratios.
   - Limit cereals to a maximum of 45% of total energy; prioritize vegetables.
   - Align hydration with activity level.
4. Nutritional Reasoning: Write a brief paragraph (2-3 sentences) explaining why these specific meals were chosen for this user profile.
5. Retrieved Guideline Evidence: Explicitly cite the guideline references used (e.g. [Source: ICMR-NIN Dietary Guidelines for Indians 2024]).
6. Scientific Integrity & No Unsupported Claims: Do NOT make speculative or unverified nutritional claims, including unverified glycemic-index (GI) or disease-reversal claims.
7. Assumptions: State clearly that quantities are educational estimates, not clinical prescriptions.

MEAL PLAN:"""
        return prompt

    def build_retry_prompt(self, profile: Dict[str, Any], context: str) -> str:
        """
        Builds a concise, focused prompt for retry attempts to prevent token exhaustion
        and ensure complete generation of all four meals and nutritional reasoning.
        """
        allergies = profile.get("food_allergies", "").strip() or "None reported"
        restrictions = profile.get("dietary_restrictions", "").strip() or "None reported"
        preferred_foods = profile.get("preferred_foods", "").strip() or "Standard healthy Indian regional foods"
        age = profile.get("age", 25)

        expanded_allergens = self.get_allergen_search_terms(allergies)
        allergen_constraint_str = ", ".join(sorted(expanded_allergens)) if expanded_allergens else "None"

        adolescent_instruction = ""
        if isinstance(age, (int, float)) and age < 18:
            adolescent_instruction = (
                "ADOBLESCENT GROWTH SAFEGUARD: User is under 18. Wholesome nutrient density only; no restrictive calorie cuts.\n"
            )

        concise_context = (
            "ICMR-NIN 2024 Standards: Maximum 45% cereals from whole grains/millets, "
            "14-15% pulses/eggs/flesh foods, 20-30% healthy fats, 0.83g/kg protein RDA, "
            "diverse vegetables, adequate hydration."
        )

        retry_prompt = f"""You are the Personalised Diet Planner. Generate a concise, complete 1-day Indian meal plan grounded in ICMR-NIN guidelines.

CORE GUIDELINES: {concise_context}

USER PROFILE:
- Age: {age} | Preference: {profile.get('dietary_preference')} | Activity: {profile.get('activity_level')}
- Goal: {profile.get('nutrition_goal')} | Regional Cuisine: {preferred_foods}
- Prohibited Allergens: [{allergen_constraint_str}] | Restrictions: {restrictions}
{adolescent_instruction}
FORMAT REQUIREMENT (Be concise, direct, and complete):
Breakfast: [1-2 lines detailing items]
Lunch: [1-2 lines detailing items]
Evening Snack: [1 line detailing items]
Dinner: [1-2 lines detailing items]
Nutritional Reasoning: [2-3 sentences explaining ICMR-NIN alignment and macronutrient balance]
Guideline Sources: [Source: ICMR-NIN Dietary Guidelines for Indians 2024]

STRICT SAFETY: Never recommend any ingredient from: [{allergen_constraint_str}]. No unsupported or glycemic-index claims.

MEAL PLAN:"""
        return retry_prompt

    def _call_client(self, prompt: str, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Executes model inference using watsonx client, capturing diagnostic metadata
        while maintaining full backward compatibility with mocks of generate_response.
        """
        # If client has generate_with_metadata and returns a real metadata dict with str text:
        if hasattr(self.client, "generate_with_metadata"):
            try:
                meta = self.client.generate_with_metadata(prompt=prompt, params=params)
                if isinstance(meta, dict) and isinstance(meta.get("text"), str):
                    return meta["text"], meta
            except Exception:
                pass

        # Primary / fallback path through generate_response (supports all mocks and direct calls)
        resp = self.client.generate_response(prompt=prompt, params=params)
        text = str(resp) if resp is not None else ""
        meta = {
            "text": text,
            "stop_reason": "unknown",
            "generated_token_count": len(text.split()),
            "input_token_count": 0,
            "params_used": params,
        }
        return text, meta

    def generate_plan(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Full orchestration pipeline with pre-validation, token budget allocation,
        diagnostic metadata tracking, completeness verification, focused retry,
        and post-generation safety scan.
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
                "diagnostics": None,
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
            "diagnostics": {},
        }

        if self.client.is_configured():
            try:
                # Token budget configured to 700 tokens (supporting full 4-meal plan + reasoning + citations + disclaimer)
                params = {
                    "max_new_tokens": 700,
                    "min_new_tokens": 150,
                    "temperature": 0.25,
                    "decoding_method": "greedy",
                }

                # Attempt 1: Standard prompt
                generated_plan, attempt_1_meta = self._call_client(prompt=prompt, params=params)
                generated_plan = generated_plan.strip()
                allergies = profile.get("food_allergies", "")

                # 1. Allergen Safety Scan (Highest Priority)
                is_safe, violating_terms = self.check_post_generation_safety(generated_plan, allergies)
                if not is_safe:
                    result["success"] = False
                    result["meal_plan"] = None
                    result["allergen_blocked"] = True
                    result["status_message"] = (
                        f"🚨 Safety Alert: The generated meal plan contained prohibited ingredient(s) "
                        f"matching your allergen profile ({', '.join(violating_terms)}). For your safety, "
                        "this meal plan has been suppressed. Please consult a registered clinical dietitian."
                    )
                    return result

                # 2. Completeness & Truncation Verification on Attempt 1
                is_complete, missing_elements = self.check_plan_completeness(generated_plan)

                # Capture Attempt 1 Diagnostics
                result["diagnostics"]["attempt_1"] = {
                    "max_new_tokens": params.get("max_new_tokens"),
                    "response_length_chars": len(generated_plan),
                    "response_length_words": len(generated_plan.split()),
                    "finish_reason": attempt_1_meta.get("stop_reason", "unknown"),
                    "generated_tokens": attempt_1_meta.get("generated_token_count", 0),
                    "input_tokens": attempt_1_meta.get("input_token_count", 0),
                    "is_complete": is_complete,
                    "missing_elements": missing_elements,
                }

                # Attempt 2: Focused retry prompt if attempt 1 was incomplete or truncated
                retried = False
                if not is_complete:
                    retried = True
                    retry_prompt = self.build_retry_prompt(profile, formatted_context)
                    retry_plan, attempt_2_meta = self._call_client(prompt=retry_prompt, params=params)
                    retry_plan = retry_plan.strip()

                    # Verify allergen safety on retry output
                    is_safe_retry, violating_terms_retry = self.check_post_generation_safety(retry_plan, allergies)
                    if not is_safe_retry:
                        result["success"] = False
                        result["meal_plan"] = None
                        result["allergen_blocked"] = True
                        result["status_message"] = (
                            f"🚨 Safety Alert: The generated meal plan contained prohibited ingredient(s) "
                            f"matching your allergen profile ({', '.join(violating_terms_retry)}). For your safety, "
                            "this meal plan has been suppressed. Please consult a registered clinical dietitian."
                        )
                        return result

                    is_complete_retry, missing_elements_retry = self.check_plan_completeness(retry_plan)

                    result["diagnostics"]["attempt_2"] = {
                        "max_new_tokens": params.get("max_new_tokens"),
                        "response_length_chars": len(retry_plan),
                        "response_length_words": len(retry_plan.split()),
                        "finish_reason": attempt_2_meta.get("stop_reason", "unknown"),
                        "generated_tokens": attempt_2_meta.get("generated_token_count", 0),
                        "input_tokens": attempt_2_meta.get("input_token_count", 0),
                        "is_complete": is_complete_retry,
                        "missing_elements": missing_elements_retry,
                    }
                    result["diagnostics"]["responses_differ"] = (generated_plan != retry_plan)

                    if is_complete_retry:
                        generated_plan = retry_plan
                        is_complete = True
                        missing_elements = []
                    else:
                        is_complete = False
                        missing_elements = missing_elements_retry

                # If still incomplete after retry: DO NOT display as a successfully generated complete plan
                if not is_complete:
                    result["success"] = False
                    result["meal_plan"] = None
                    result["status_message"] = (
                        f"Generation Alert: The generated meal plan was incomplete or cut off before completion "
                        f"(missing: {', '.join(missing_elements)}). For your safety, truncated plans are not displayed. "
                        "Please try generating again."
                    )
                    return result

                # Ensure Grounding Evidence and trusted Medical Disclaimer sections are integrated into text
                plan_lower = generated_plan.lower()
                if not re.search(r"\b(?:grounding\s+evidence|guidelines?\s+used|source:?\s*icmr)\b", plan_lower):
                    doc_titles = [f"- {d.get('title')} [{d.get('source', 'ICMR-NIN')}]" for d in retrieved_docs]
                    evidence_block = "\n\n📚 Grounding Evidence & Guidelines Used:\n" + "\n".join(doc_titles)
                    generated_plan += evidence_block

                if "disclaimer" not in plan_lower:
                    generated_plan += f"\n\n{PLANNER_SAFETY_DISCLAIMER}"

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
