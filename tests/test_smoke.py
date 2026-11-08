"""
Basic Smoke Tests for Nutrition Agent Project Structure and Integration Code
"""

import os
from pathlib import Path
from modules.watsonx_client import WatsonxGraniteClient


def test_watsonx_client_initialization_defaults():
    """
    Ensure the client initializes properly without throwing errors,
    and accurately reflects unconfigured state when no credentials are provided.
    """
    client = WatsonxGraniteClient(api_key="", project_id="")
    status = client.get_configuration_status()

    assert status["is_ready"] is False
    assert status["has_api_key"] is False
    assert status["has_project_id"] is False
    assert status["target_model"] == "ibm/granite-4-h-small"


def test_watsonx_client_test_connection_unconfigured():
    """
    Verify test_connection returns safe failure message when credentials are empty
    without making external API requests.
    """
    client = WatsonxGraniteClient(api_key="", project_id="")
    result = client.test_connection()

    assert result["success"] is False
    assert "not configured" in result["message"]
    assert result["details"]["is_ready"] is False


def test_watsonx_client_placeholder_filtering():
    """
    Ensure placeholder values from template files are not recognized as real credentials.
    """
    client = WatsonxGraniteClient(
        api_key="your_ibm_cloud_api_key_here",
        project_id="your_watsonx_project_id_here"
    )
    assert client.is_configured() is False


def test_project_folders_exist():
    """
    Verify that required project folders exist.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for folder in ["agents", "data", "documentation", "modules", "tests"]:
        folder_path = os.path.join(base_dir, folder)
        assert os.path.isdir(folder_path), f"Folder missing: {folder}"


def test_env_example_has_required_keys():
    """
    Ensure .env.example contains all mandatory IBM watsonx keys.
    """
    base_dir = Path(__file__).resolve().parent.parent
    env_example = base_dir / ".env.example"
    assert env_example.exists(), ".env.example file is missing"

    content = env_example.read_text(encoding="utf-8")
    assert "WATSONX_APIKEY" in content
    assert "WATSONX_PROJECT_ID" in content
    assert "WATSONX_URL" in content


def test_rag_retriever_load_and_search():
    """
    Verify NutritionKnowledgeRetriever loads data and finds relevant topics.
    """
    from modules.rag_retriever import NutritionKnowledgeRetriever
    retriever = NutritionKnowledgeRetriever()
    results = retriever.search("sugar recommendations", top_k=2)
    assert len(results) > 0
    assert any("sugar" in r["title"].lower() or "who" in r["source"].lower() for r in results)


def test_nutrition_assistant_prompt_structure():
    """
    Verify NutritionKnowledgeAssistant builds safe, grounded prompts with citations.
    """
    from agents.nutrition_assistant import NutritionKnowledgeAssistant, SAFETY_DISCLAIMER
    assistant = NutritionKnowledgeAssistant()
    prompt = assistant.build_prompt(
        query="What is the sodium limit?",
        context="[Document 1]\nSource: WHO Guidelines\nTopic: Sodium\nContent: Less than 2000mg sodium daily."
    )
    assert "CONTEXT FROM VERIFIED NUTRITION AUTHORITIES" in prompt
    assert "WHO Guidelines" in prompt
    assert "Disclaimer" in SAFETY_DISCLAIMER


def test_guidelines_metadata_integrity():
    """
    Verify that every entry in nutrition_guidelines.json contains authoritative
    metadata including organization, document title, publication year, and source URL.
    """
    import json
    from pathlib import Path
    data_file = Path(__file__).resolve().parent.parent / "data" / "nutrition_guidelines.json"
    assert data_file.exists(), "nutrition_guidelines.json does not exist"

    with open(data_file, "r", encoding="utf-8") as f:
        guidelines = json.load(f)

    assert len(guidelines) >= 8, f"Expected at least 8 guidelines, found {len(guidelines)}"

    for item in guidelines:
        assert item.get("id"), "Missing id"
        assert item.get("title"), f"Missing title in {item.get('id')}"
        assert item.get("source_organization"), f"Missing source_organization in {item.get('id')}"
        assert isinstance(item.get("publication_year"), int), f"Invalid publication_year in {item.get('id')}"
        assert item.get("publication_year") >= 2015, f"Year too old in {item.get('id')}"
        assert item.get("document_name"), f"Missing document_name in {item.get('id')}"
        assert item.get("source_url", "").startswith("https://"), f"Invalid source_url in {item.get('id')}"
        assert len(item.get("content", "")) > 50, f"Content too short in {item.get('id')}"


def test_numerical_recommendations_presence():
    """
    Verify numerical claims against WHO, ICMR-NIN and USDA standards.
    """
    import json
    from pathlib import Path
    data_file = Path(__file__).resolve().parent.parent / "data" / "nutrition_guidelines.json"
    with open(data_file, "r", encoding="utf-8") as f:
        guidelines = json.load(f)

    all_content = " ".join(item["content"] for item in guidelines)

    # Protein RDA 0.83 g/kg
    assert "0.83" in all_content, "Missing ICMR-NIN 0.83 g/kg protein recommendation"
    # Sodium < 2,000 mg
    assert "2,000 mg" in all_content, "Missing WHO 2,000 mg sodium limit"
    # Free sugars < 10%
    assert "10%" in all_content, "Missing WHO 10% free sugars recommendation"
    # Fiber 14g per 1,000 kcal
    assert "14 grams" in all_content and "1,000" in all_content, "Missing USDA 14g/1,000 kcal fiber benchmark"


def test_usda_and_icmr_source_distinctions():
    """
    Verify USDA 2025-2030 URL and population vs individual distinction notes.
    """
    import json
    from pathlib import Path
    data_file = Path(__file__).resolve().parent.parent / "data" / "nutrition_guidelines.json"
    with open(data_file, "r", encoding="utf-8") as f:
        guidelines = json.load(f)

    # Verify USDA DGA 2025-2030 entry
    usda_item = next(item for item in guidelines if item["id"] == "usda-dietary-fiber")
    assert "2025–2030" in usda_item["document_name"]
    assert usda_item["source_url"] == "https://www.fns.usda.gov/cnpp/dietary-guidelines-americans"
    assert usda_item["publication_year"] == 2025

    # Verify population recommendation notes vs individual prescriptions
    for item in guidelines:
        content = item["content"].lower()
        assert "individual" in content or "population" in content, (
            f"Guideline {item['id']} must clearly distinguish population benchmarks from individual requirements"
        )


def test_diet_planner_input_validation():
    """
    Test input validation for PersonalisedDietPlanner.
    """
    from agents.diet_planner import PersonalisedDietPlanner

    planner = PersonalisedDietPlanner()

    # Valid profile
    valid_profile = {
        "age": 25,
        "dietary_preference": "Vegetarian",
        "activity_level": "Moderate (active daily routine / 3-4 days exercise)",
        "nutrition_goal": "Balanced Daily Nutrition",
        "food_allergies": "None",
        "dietary_restrictions": "None",
        "preferred_foods": "Roti, Dal, Sabzi",
    }
    is_valid, errors, warnings = planner.validate_inputs(valid_profile)
    assert is_valid is True
    assert len(errors) == 0

    # Invalid age
    invalid_profile = valid_profile.copy()
    invalid_profile["age"] = -5
    is_valid, errors, _ = planner.validate_inputs(invalid_profile)
    assert is_valid is False
    assert any("age" in err.lower() for err in errors)

    # Invalid diet preference
    invalid_profile2 = valid_profile.copy()
    invalid_profile2["dietary_preference"] = "KetoOnly"
    is_valid, errors, _ = planner.validate_inputs(invalid_profile2)
    assert is_valid is False
    assert any("dietary preference" in err.lower() for err in errors)


def test_diet_planner_allergen_and_clinical_safety_checks():
    """
    Verify allergen warnings and clinical condition safety guardrails in Diet Planner.
    """
    from agents.diet_planner import PersonalisedDietPlanner

    planner = PersonalisedDietPlanner()

    # Allergen warning
    allergy_profile = {
        "age": 30,
        "dietary_preference": "Vegetarian",
        "activity_level": "Moderate (active daily routine / 3-4 days exercise)",
        "nutrition_goal": "Balanced Daily Nutrition",
        "food_allergies": "Peanuts, Dairy",
        "dietary_restrictions": "None",
    }
    is_valid, errors, warnings = planner.validate_inputs(allergy_profile)
    assert is_valid is True
    assert any("peanuts" in w.lower() for w in warnings)

    # Clinical condition warning
    clinical_profile = allergy_profile.copy()
    clinical_profile["dietary_restrictions"] = "Chronic Kidney Disease stage 3"
    _, _, warnings = planner.validate_inputs(clinical_profile)
    assert any("clinical" in w.lower() or "medical" in w.lower() for w in warnings)


def test_diet_planner_prompt_structure():
    """
    Verify that Diet Planner constructs prompt with allergen safety, ICMR-NIN principles and disclaimers.
    """
    from agents.diet_planner import PersonalisedDietPlanner, PLANNER_SAFETY_DISCLAIMER

    planner = PersonalisedDietPlanner()
    profile = {
        "age": 28,
        "dietary_preference": "Vegetarian",
        "activity_level": "Moderate (active daily routine / 3-4 days exercise)",
        "nutrition_goal": "Balanced Daily Nutrition",
        "food_allergies": "Peanuts",
        "dietary_restrictions": "None",
        "preferred_foods": "Millets, Dal, Sabzi",
    }
    context = "[Document 1]\nSource: ICMR-NIN\nTopic: Balanced Diet\nContent: 50-56% carbohydrates, 10-15% protein."
    prompt = planner.build_prompt(profile, context)

    assert "VERIFIED NUTRITIONAL GUIDELINES" in prompt
    assert "STRICT ALLERGEN SAFETY" in prompt
    assert "Peanuts" in prompt
    assert "ICMR-NIN" in prompt
    assert "Disclaimer" in PLANNER_SAFETY_DISCLAIMER


def test_diet_planner_allergen_synonym_expansion():
    """
    Verify that reported allergens expand to common Indian regional and ingredient synonyms.
    """
    from agents.diet_planner import PersonalisedDietPlanner

    planner = PersonalisedDietPlanner()

    # Peanuts -> groundnut, moongphali
    terms = planner.get_allergen_search_terms("peanuts")
    assert "peanut" in terms or "peanuts" in terms
    assert "groundnut" in terms
    assert "moongphali" in terms

    # Dairy -> paneer, curd, ghee, dahi
    dairy_terms = planner.get_allergen_search_terms("dairy")
    assert "paneer" in dairy_terms
    assert "curd" in dairy_terms
    assert "ghee" in dairy_terms
    assert "dahi" in dairy_terms

    # Gluten -> wheat, atta, maida, sooji
    gluten_terms = planner.get_allergen_search_terms("gluten")
    assert "wheat" in gluten_terms
    assert "atta" in gluten_terms
    assert "maida" in gluten_terms
    assert "sooji" in gluten_terms

    # Tree nuts -> badam, kaju, almond
    nut_terms = planner.get_allergen_search_terms("tree nuts")
    assert "badam" in nut_terms
    assert "kaju" in nut_terms
    assert "almond" in nut_terms

    # None / empty -> empty set
    assert planner.get_allergen_search_terms("None") == set()
    assert planner.get_allergen_search_terms("") == set()


def test_diet_planner_post_generation_blocking():
    """
    Verify post-generation safety check blocks plans containing declared allergens or synonyms.
    """
    from agents.diet_planner import PersonalisedDietPlanner

    planner = PersonalisedDietPlanner()

    # Case 1: Direct allergen violation
    unsafe_plan_1 = (
        "Breakfast: Poha with curry leaves and roasted peanuts.\n"
        "Lunch: Brown rice, dal, and vegetable curry.\n"
        "Dinner: Chapati with paneer bhurji."
    )
    is_safe, detected = planner.check_post_generation_safety(unsafe_plan_1, "peanuts")
    assert is_safe is False
    assert "peanuts" in detected

    # Case 2: Regional synonym violation (user reported dairy; plan suggests paneer)
    unsafe_plan_2 = (
        "Breakfast: Oats with almond milk.\n"
        "Lunch: Roti with yellow dal and salad.\n"
        "Dinner: Spinach and paneer curry with millets."
    )
    is_safe, detected = planner.check_post_generation_safety(unsafe_plan_2, "dairy")
    assert is_safe is False
    assert "paneer" in detected

    # Case 3: Completely safe plan
    safe_plan = (
        "Breakfast: Sprouted moong chilla with mint chutney.\n"
        "Lunch: Brown rice, yellow lentil sambar, and stir-fried bhindi.\n"
        "Dinner: Ragi roti with mixed vegetable curry."
    )
    is_safe, detected = planner.check_post_generation_safety(safe_plan, "dairy, peanuts")
    assert is_safe is True
    assert len(detected) == 0


def test_diet_planner_blocks_plan_in_pipeline():
    """
    Verify generate_plan suppresses the meal plan and flags allergen_blocked=True
    when mock model inadvertently emits an allergen.
    """
    from unittest.mock import MagicMock
    from agents.diet_planner import PersonalisedDietPlanner

    mock_client = MagicMock()
    mock_client.is_configured.return_value = True
    # The model generated a plan that contains groundnuts despite user having a peanut allergy
    mock_client.generate_response.return_value = (
        "Breakfast: Poha topped with crispy groundnuts.\n"
        "Lunch: Dal tadka and brown rice.\n"
        "Dinner: Khichdi."
    )

    planner = PersonalisedDietPlanner(client=mock_client)
    profile = {
        "age": 28,
        "dietary_preference": "Vegetarian",
        "activity_level": "Moderate (active daily routine / 3-4 days exercise)",
        "nutrition_goal": "Balanced Daily Nutrition",
        "food_allergies": "Peanuts",
        "dietary_restrictions": "None",
        "preferred_foods": "Poha, Dal, Khichdi",
    }

    result = planner.generate_plan(profile)

    # Must be blocked: no unsafe meal plan presented to the user
    assert result["success"] is False
    assert result["allergen_blocked"] is True
    assert result["meal_plan"] is None
    assert "prohibited ingredient" in result["status_message"].lower()
    assert "groundnut" in result["status_message"].lower()


def test_diet_planner_under_18_safeguards():
    """
    Verify adolescent growth safeguards are enforced for users under 18.
    """
    from agents.diet_planner import PersonalisedDietPlanner

    planner = PersonalisedDietPlanner()

    profile_minor = {
        "age": 15,
        "dietary_preference": "Vegetarian",
        "activity_level": "Moderate (active daily routine / 3-4 days exercise)",
        "nutrition_goal": "Weight Management / Calorie Consciousness",
        "food_allergies": "None",
        "dietary_restrictions": "None",
        "preferred_foods": "Idli, Sambar",
    }

    is_valid, errors, warnings = planner.validate_inputs(profile_minor)
    assert is_valid is True
    assert any("adolescent growth safeguard" in w.lower() for w in warnings)

    # Check prompt contains adolescent guidance
    prompt = planner.build_prompt(profile_minor, context="[Context]")
    assert "ADOBLESCENT GROWTH SAFEGUARD" in prompt
    assert "calorie deficits" in prompt.lower()


def test_diet_planner_medical_condition_warnings():
    """
    Verify clinical condition detection and professional consultation advisories.
    """
    from agents.diet_planner import PersonalisedDietPlanner

    planner = PersonalisedDietPlanner()

    conditions_to_test = [
        ("Type 2 Diabetes", "diabetes"),
        ("Chronic Kidney Disease", "kidney"),
        ("Cardiovascular Heart Disease", "heart"),
        ("Celiac Disease", "celiac"),
    ]

    for condition_str, expected_kw in conditions_to_test:
        profile = {
            "age": 45,
            "dietary_preference": "Vegetarian",
            "activity_level": "Sedentary (desk job, minimal exercise)",
            "nutrition_goal": "Balanced Daily Nutrition",
            "food_allergies": "None",
            "dietary_restrictions": f"Diagnosed with {condition_str}",
            "preferred_foods": "Roti, Dal",
        }
        _, _, warnings = planner.validate_inputs(profile)
        assert any("clinical health advisory" in w.lower() for w in warnings), (
            f"Failed to produce clinical warning for condition: {condition_str}"
        )


def test_diet_planner_failed_inference_handling():
    """
    Ensure errors from IBM watsonx / network failure do NOT fabricate fake meal plans.
    """
    from unittest.mock import MagicMock
    from agents.diet_planner import PersonalisedDietPlanner

    # Test 1: Exception during generation
    mock_failing_client = MagicMock()
    mock_failing_client.is_configured.return_value = True
    mock_failing_client.generate_response.side_effect = RuntimeError("IBM watsonx connection timeout")

    planner = PersonalisedDietPlanner(client=mock_failing_client)
    profile = {
        "age": 30,
        "dietary_preference": "Vegetarian",
        "activity_level": "Moderate (active daily routine / 3-4 days exercise)",
        "nutrition_goal": "Balanced Daily Nutrition",
        "food_allergies": "None",
        "dietary_restrictions": "None",
        "preferred_foods": "Dal, Roti",
    }

    result = planner.generate_plan(profile)
    assert result["success"] is False
    assert result["meal_plan"] is None
    assert "connection timeout" in result["status_message"]

    # Test 2: Unconfigured client
    unconfigured_client = WatsonxGraniteClient(api_key="", project_id="")
    planner_unconf = PersonalisedDietPlanner(client=unconfigured_client)
    result_unconf = planner_unconf.generate_plan(profile)
    assert result_unconf["success"] is False
    assert result_unconf["meal_plan"] is None
    assert "not configured" in result_unconf["status_message"]


def test_food_composition_data_integrity():
    """
    Verify that food_composition.json exists, is valid JSON, and adheres to
    ICMR-NIN IFCT 2017 / USDA FoodData Central authoritative schema.
    """
    import json
    from pathlib import Path
    data_file = Path(__file__).resolve().parent.parent / "data" / "food_composition.json"
    assert data_file.exists(), "food_composition.json is missing"

    with open(data_file, "r", encoding="utf-8") as f:
        foods = json.load(f)

    assert len(foods) >= 25, f"Expected at least 25 foods, found {len(foods)}"

    required_keys = [
        "id", "name", "category", "source", "default_serving_description",
        "serving_weight_g", "calories_kcal", "protein_g", "carbohydrates_g",
        "fat_g", "dietary_fiber_g", "sodium_mg"
    ]

    for item in foods:
        for k in required_keys:
            assert k in item, f"Missing key {k} in item {item.get('id')}"
        assert any("ifct" in item["source"].lower() or "usda" in item["source"].lower() for item in [item]), (
            f"Source must cite IFCT or USDA in {item.get('id')}"
        )
        assert item["calories_kcal"] >= 0
        assert item["protein_g"] >= 0
        assert item["carbohydrates_g"] >= 0
        assert item["fat_g"] >= 0


def test_food_intake_analyzer_matching():
    """
    Test food name, id, and alias resolution.
    """
    from agents.food_intake_analyzer import FoodIntakeAnalyzer

    analyzer = FoodIntakeAnalyzer()

    # Exact ID match
    f1 = analyzer.find_food("roti_whole_wheat")
    assert f1 is not None
    assert "Roti" in f1["name"]

    # Alias match (chapati, phulka)
    f2 = analyzer.find_food("chapati")
    assert f2 is not None
    assert f2["id"] == "roti_whole_wheat"

    f3 = analyzer.find_food("moongphali")
    assert f3 is not None
    assert "peanuts" in f3["id"]

    # Unknown food must return None (no guessing)
    f_unknown = analyzer.find_food("super_space_gelatin_unobtainium")
    assert f_unknown is None


def test_food_intake_analyzer_deterministic_calculation():
    """
    Test mathematical accuracy of nutrient totals and portion scaling.
    """
    from agents.food_intake_analyzer import FoodIntakeAnalyzer

    analyzer = FoodIntakeAnalyzer()

    log = [
        {"food_name": "Whole Wheat Roti / Chapati", "portion": 2.0, "meal_type": "Lunch"},
        {"food_name": "Cooked Toor / Arhar Dal (with tadka)", "portion": 1.0, "meal_type": "Lunch"},
    ]

    summary = analyzer.calculate_intake(log)

    # Roti: 120 kcal * 2 = 240 kcal, Dal: 150 kcal * 1 = 150 kcal -> Total 390 kcal
    assert summary["total_calories_kcal"] == 390.0
    # Protein: 3.6 * 2 + 7.6 * 1 = 14.8g
    assert summary["total_protein_g"] == 14.8
    assert summary["matched_items_count"] == 2
    assert summary["unmatched_items_count"] == 0
    assert summary["macronutrient_ratios"]["protein_pct"] > 0
    assert summary["macronutrient_ratios"]["carbs_pct"] > 0


def test_food_intake_analyzer_unmatched_food_handling():
    """
    Test that unverified foods are marked as 'Nutrition data unavailable',
    have None for nutrient fields rather than a misleading zero,
    are excluded from totals, and flag partial daily estimates.
    """
    from agents.food_intake_analyzer import FoodIntakeAnalyzer

    analyzer = FoodIntakeAnalyzer()

    log = [
        {"food_name": "Whole Wheat Roti / Chapati", "portion": 1.0, "meal_type": "Lunch"},
        {"food_name": "Unknown Mystery Dish", "portion": 1.0, "meal_type": "Lunch"},
    ]

    summary = analyzer.calculate_intake(log)

    assert summary["total_items"] == 2
    assert summary["matched_items_count"] == 1
    assert summary["unmatched_items_count"] == 1
    assert "Unknown Mystery Dish" in summary["unmatched_items"]
    assert summary["is_partial_estimate"] is True
    assert summary["has_no_verified_data"] is False
    assert "PARTIAL ESTIMATE" in summary["disclaimer_label"]
    assert "EXCLUDED" in summary["disclaimer_label"]

    # Totals only reflect the matched roti (120 kcal, 3.6g protein)
    assert summary["total_calories_kcal"] == 120.0
    assert summary["total_protein_g"] == 3.6

    unmatched_entry = next(i for i in summary["item_breakdowns"] if not i["matched"])
    # Crucial safety requirement: Never treat unavailable nutritional data as an actual zero!
    assert unmatched_entry["calories_kcal"] is None
    assert unmatched_entry["protein_g"] is None
    assert unmatched_entry["carbohydrates_g"] is None
    assert unmatched_entry["fat_g"] is None
    assert unmatched_entry["dietary_fiber_g"] is None
    assert unmatched_entry["sodium_mg"] is None
    assert "unavailable" in unmatched_entry["source"].lower()


def test_food_intake_analyzer_all_unsupported_foods():
    """
    Test when all logged items lack verified nutritional data.
    """
    from agents.food_intake_analyzer import FoodIntakeAnalyzer

    analyzer = FoodIntakeAnalyzer()
    log = [
        {"food_name": "Alien Food 1", "portion": 1.0, "meal_type": "Lunch"},
        {"food_name": "Alien Food 2", "portion": 2.0, "meal_type": "Dinner"},
    ]
    summary = analyzer.calculate_intake(log)

    assert summary["matched_items_count"] == 0
    assert summary["unmatched_items_count"] == 2
    assert summary["has_no_verified_data"] is True
    assert summary["is_partial_estimate"] is False
    assert "NUTRITIONAL DATA UNAVAILABLE" in summary["disclaimer_label"]
    assert summary["total_calories_kcal"] == 0.0
    assert summary["total_protein_g"] == 0.0


def test_food_intake_analyzer_partial_estimate_prompt():
    """
    Verify prompt explicitly instructs IBM Granite about partial estimates and excluded items.
    """
    from agents.food_intake_analyzer import FoodIntakeAnalyzer

    analyzer = FoodIntakeAnalyzer()
    log = [
        {"food_name": "Whole Wheat Roti / Chapati", "portion": 2.0, "meal_type": "Lunch"},
        {"food_name": "Unverified Street Snack", "portion": 1.0, "meal_type": "Evening Snack"},
    ]
    summary = analyzer.calculate_intake(log)
    prompt = analyzer.build_prompt(summary, context="[ICMR-NIN Guidelines]")

    assert "PARTIAL ESTIMATE" in prompt
    assert "EXCLUDED from calculated totals" in prompt
    assert "CRITICAL INSTRUCTION ON EXCLUDED FOODS" in prompt
    assert "Unverified Street Snack" in prompt
    assert "DO NOT invent, fabricate, or guess nutritional numbers" in prompt



def test_food_intake_analyzer_empty_log_handling():
    """
    Verify empty food log is safely rejected before calling IBM Granite.
    """
    from agents.food_intake_analyzer import FoodIntakeAnalyzer

    analyzer = FoodIntakeAnalyzer()
    res = analyzer.analyze_intake([])

    assert res["success"] is False
    assert "empty" in res["status_message"].lower()
    assert res["analysis_commentary"] is None


def test_food_intake_analyzer_prompt_structure():
    """
    Verify FoodIntakeAnalyzer prompt includes deterministic numbers, RAG evidence, and constraints.
    """
    from agents.food_intake_analyzer import FoodIntakeAnalyzer, ANALYZER_SAFETY_DISCLAIMER

    analyzer = FoodIntakeAnalyzer()
    log = [
        {"food_name": "Steamed Idli", "portion": 2.0, "meal_type": "Breakfast"},
    ]
    summary = analyzer.calculate_intake(log)
    prompt = analyzer.build_prompt(summary, context="[ICMR-NIN Protein Benchmark: 0.83 g/kg]")

    assert "CALCULATED FOOD INTAKE SUMMARY" in prompt
    assert "Steamed Idli" in prompt
    assert "130.0 kcal" in prompt or "130" in prompt
    assert "ICMR-NIN" in prompt
    assert "Disclaimer" in ANALYZER_SAFETY_DISCLAIMER


def test_food_intake_analyzer_mock_watsonx_inference():
    """
    Verify full intake analysis pipeline with mock IBM Granite client.
    """
    from unittest.mock import MagicMock
    from agents.food_intake_analyzer import FoodIntakeAnalyzer

    mock_client = MagicMock()
    mock_client.is_configured.return_value = True
    mock_client.generate_response.return_value = (
        "Dietary Review: Your intake of Idli and Sambar provides wholesome complex carbohydrates and legumes. "
        "Strengths: Fermented foods support gut health. Areas for improvement: Add fresh fruit for dietary fiber."
    )

    analyzer = FoodIntakeAnalyzer(client=mock_client)
    sample_log = [
        {"food_name": "Steamed Idli", "portion": 2.0, "meal_type": "Breakfast"},
        {"food_name": "Vegetable Sambar", "portion": 1.0, "meal_type": "Breakfast"},
    ]

    result = analyzer.analyze_intake(sample_log)

    assert result["success"] is True
    assert result["used_live_model"] is True
    assert "Dietary Review" in result["analysis_commentary"]
    assert len(result["retrieved_docs"]) > 0
    assert result["intake_summary"]["total_calories_kcal"] > 0


def test_food_intake_analyzer_entry_and_portion_validation():
    """
    Verify individual food log entry and portion size validation logic.
    """
    from agents.food_intake_analyzer import FoodIntakeAnalyzer

    analyzer = FoodIntakeAnalyzer()

    # 1. Valid entry
    is_valid, errors = analyzer.validate_entry("Whole Wheat Roti / Chapati", 2.0, "Lunch")
    assert is_valid is True
    assert len(errors) == 0

    # 2. Empty food name
    is_valid, errors = analyzer.validate_entry("", 1.0, "Lunch")
    assert is_valid is False
    assert any("empty" in e.lower() for e in errors)

    # 3. Zero portion
    is_valid, errors = analyzer.validate_entry("Steamed Idli", 0.0, "Breakfast")
    assert is_valid is False
    assert any("greater than 0" in e.lower() for e in errors)

    # 4. Negative portion
    is_valid, errors = analyzer.validate_entry("Steamed Idli", -1.5, "Breakfast")
    assert is_valid is False
    assert any("greater than 0" in e.lower() for e in errors)

    # 5. Non-numeric portion
    is_valid, errors = analyzer.validate_entry("Steamed Idli", "invalid_portion", "Breakfast")
    assert is_valid is False
    assert any("valid number" in e.lower() for e in errors)

    # 6. Unrealistic portion (> 50 servings)
    is_valid, errors = analyzer.validate_entry("Steamed Idli", 100.0, "Breakfast")
    assert is_valid is False
    assert any("limit" in e.lower() for e in errors)

    # 7. Invalid meal type
    is_valid, errors = analyzer.validate_entry("Steamed Idli", 2.0, "MidnightFeast")
    assert is_valid is False
    assert any("meal type" in e.lower() for e in errors)


def test_food_intake_analyzer_multi_meal_daily_totals():
    """
    Verify complete multi-meal daily food log aggregation across Breakfast, Lunch, Snack, Dinner.
    """
    from agents.food_intake_analyzer import FoodIntakeAnalyzer

    analyzer = FoodIntakeAnalyzer()

    full_day_log = [
        # Breakfast
        {"food_name": "Steamed Idli", "portion": 2.0, "meal_type": "Breakfast"},
        {"food_name": "Vegetable Sambar", "portion": 1.0, "meal_type": "Breakfast"},
        # Lunch
        {"food_name": "Whole Wheat Roti / Chapati", "portion": 2.0, "meal_type": "Lunch"},
        {"food_name": "Cooked Toor / Arhar Dal (with tadka)", "portion": 1.0, "meal_type": "Lunch"},
        {"food_name": "Curd / Plain Dahi (Whole Cow Milk)", "portion": 1.0, "meal_type": "Lunch"},
        # Evening Snack
        {"food_name": "Roasted Peanuts / Groundnuts", "portion": 1.0, "meal_type": "Evening Snack"},
        {"food_name": "Indian Masala Chai (with Milk and 1 tsp Sugar)", "portion": 1.0, "meal_type": "Evening Snack"},
        # Dinner
        {"food_name": "Moong Dal Khichdi", "portion": 1.0, "meal_type": "Dinner"},
        {"food_name": "Fresh Green Salad (Cucumber, Tomato, Carrot)", "portion": 1.0, "meal_type": "Dinner"},
    ]

    summary = analyzer.calculate_intake(full_day_log)

    # Totals verification
    assert summary["total_items"] == 9
    assert summary["matched_items_count"] == 9
    assert summary["unmatched_items_count"] == 0

    # Calories: (65*2 + 95) + (120*2 + 150 + 92) + (172 + 78) + (220 + 24) = 225 + 482 + 250 + 244 = 1201.0
    assert summary["total_calories_kcal"] == 1201.0

    # Protein: (2.1*2 + 3.6) + (3.6*2 + 7.6 + 4.8) + (7.6 + 2.1) + (6.8 + 1.0) = 7.8 + 19.6 + 9.7 + 7.8 = 44.9
    assert summary["total_protein_g"] == 44.9

    # Fiber and sodium presence
    assert summary["total_dietary_fiber_g"] > 0
    assert summary["total_sodium_mg"] > 0

    # Macronutrient ratios sum to ~100%
    ratios = summary["macronutrient_ratios"]
    macro_sum = ratios["carbs_pct"] + ratios["protein_pct"] + ratios["fat_pct"]
    assert 99.0 <= macro_sum <= 101.0


def test_food_intake_analyzer_failed_inference_handling():
    """
    Ensure watsonx connection errors do NOT fabricate fake nutrition data or crash pipeline,
    and deterministic calculations remain intact.
    """
    from unittest.mock import MagicMock
    from agents.food_intake_analyzer import FoodIntakeAnalyzer

    # Test 1: Exception during IBM Granite call
    mock_failing_client = MagicMock()
    mock_failing_client.is_configured.return_value = True
    mock_failing_client.generate_response.side_effect = RuntimeError("IBM watsonx network timeout")

    analyzer = FoodIntakeAnalyzer(client=mock_failing_client)
    sample_log = [
        {"food_name": "Whole Wheat Roti / Chapati", "portion": 2.0, "meal_type": "Lunch"},
        {"food_name": "Cooked Toor / Arhar Dal (with tadka)", "portion": 1.0, "meal_type": "Lunch"},
    ]

    result = analyzer.analyze_intake(sample_log)
    assert result["success"] is False
    assert result["analysis_commentary"] is None
    assert "network timeout" in result["status_message"]
    # Deterministic calculations must still be preserved accurately!
    assert result["intake_summary"] is not None
    assert result["intake_summary"]["total_calories_kcal"] == 390.0
    assert result["intake_summary"]["total_protein_g"] == 14.8

    # Test 2: Unconfigured client
    unconfigured_client = WatsonxGraniteClient(api_key="", project_id="")
    analyzer_unconf = FoodIntakeAnalyzer(client=unconfigured_client)
    res_unconf = analyzer_unconf.analyze_intake(sample_log)
    assert res_unconf["success"] is False
    assert res_unconf["analysis_commentary"] is None
    assert "not configured" in res_unconf["status_message"]
    assert res_unconf["intake_summary"]["total_calories_kcal"] == 390.0


def test_food_intake_analyzer_safety_disclaimers():
    """
    Verify non-clinical safety disclaimers, deficiency non-diagnosis,
    and single-day limitation statements.
    """
    from agents.food_intake_analyzer import ANALYZER_SAFETY_DISCLAIMER, FoodIntakeAnalyzer

    # Core disclaimer content checks
    assert "Educational Disclaimer" in ANALYZER_SAFETY_DISCLAIMER
    assert "medical nutrition therapy" in ANALYZER_SAFETY_DISCLAIMER
    assert "deficiencies" in ANALYZER_SAFETY_DISCLAIMER
    assert "long-term" in ANALYZER_SAFETY_DISCLAIMER
    assert "dietitian" in ANALYZER_SAFETY_DISCLAIMER

    # Calculation disclaimer label
    analyzer = FoodIntakeAnalyzer()
    summary = analyzer.calculate_intake([{"food_name": "Fresh Banana", "portion": 1.0, "meal_type": "Breakfast"}])
    assert "estimate" in summary["disclaimer_label"].lower()
    assert "IFCT 2017" in summary["disclaimer_label"]


def test_general_health_advisor_normal_question_and_rag():
    """
    Test standard everyday nutrition question with RAG evidence and mock Granite response.
    """
    from unittest.mock import MagicMock
    from agents.general_health_advisor import GeneralHealthAdvisor, ADVISOR_SAFETY_DISCLAIMER

    mock_client = MagicMock()
    mock_client.is_configured.return_value = True
    mock_client.generate_response.return_value = (
        "Hydration Advice: Drinking approximately 2 to 2.5 liters of fluids daily supports metabolic function. "
        "Unsweetened tea counts toward hydration, though pure water should remain your core source."
    )

    advisor = GeneralHealthAdvisor(client=mock_client)
    res = advisor.advise(
        query="How much water should I drink every day and does tea count?",
        age_group="Young Adult / College (18–25)",
        user_context="Desk worker sitting 7 hours a day",
    )

    assert res["success"] is True
    assert res["used_live_model"] is True
    assert "Hydration Advice" in res["answer"]
    assert len(res["retrieved_docs"]) > 0
    assert any("water" in d["content"].lower() or "hydration" in d["content"].lower() for d in res["retrieved_docs"])
    assert "Educational Disclaimer" in res["disclaimer"]


def test_general_health_advisor_source_attribution():
    """
    Verify that retrieved guidelines contain official title, organization, year, and URL.
    """
    from agents.general_health_advisor import GeneralHealthAdvisor

    advisor = GeneralHealthAdvisor()
    docs = advisor.retrieve_evidence("balanced diet millets protein")

    assert len(docs) > 0
    for d in docs:
        assert d.get("title")
        assert d.get("source_organization")
        assert d.get("publication_year") >= 2015
        assert d.get("source_url", "").startswith("https://")


def test_general_health_advisor_missing_evidence_handling():
    """
    Verify handling when a query does not match any specific guideline.
    """
    from unittest.mock import MagicMock
    from agents.general_health_advisor import GeneralHealthAdvisor

    mock_client = MagicMock()
    mock_client.is_configured.return_value = True
    mock_client.generate_response.return_value = "General advice based on public health principles."

    advisor = GeneralHealthAdvisor(client=mock_client)
    # Query completely unrelated to nutrition guidelines
    prompt = advisor.build_prompt(
        query="What is quantum computing?",
        age_group="Prefer not to say",
        user_context="",
        evidence_context="",
        has_evidence=False,
    )

    assert "No direct guideline match found" in prompt
    assert "Rely strictly on general public health principles" in prompt


def test_general_health_advisor_inference_failure_handling():
    """
    Ensure IBM Granite network or quota failures do NOT fabricate fake AI responses.
    """
    from unittest.mock import MagicMock
    from agents.general_health_advisor import GeneralHealthAdvisor

    # Exception during generation
    mock_failing_client = MagicMock()
    mock_failing_client.is_configured.return_value = True
    mock_failing_client.generate_response.side_effect = RuntimeError("watsonx connection timeout")

    advisor = GeneralHealthAdvisor(client=mock_failing_client)
    res = advisor.advise(query="How to avoid afternoon energy crashes?")

    assert res["success"] is False
    assert res["answer"] is None
    assert "connection timeout" in res["status_message"]

    # Unconfigured client
    unconf_client = WatsonxGraniteClient(api_key="", project_id="")
    advisor_unconf = GeneralHealthAdvisor(client=unconf_client)
    res_unconf = advisor_unconf.advise(query="How to balance meals?")
    assert res_unconf["success"] is False
    assert res_unconf["answer"] is None
    assert "not configured" in res_unconf["status_message"]


def test_general_health_advisor_urgent_symptoms_escalation():
    """
    Verify acute medical emergencies immediately bypass dietary advice and trigger emergency guidance.
    """
    from agents.general_health_advisor import GeneralHealthAdvisor

    advisor = GeneralHealthAdvisor()
    urgent_queries = [
        "I am having severe chest pain and shortness of breath, what should I eat?",
        "Sudden numbness in my arm and blood in vomit",
        "My friend fainted and has swelling of lips from severe allergic reaction",
    ]

    for q in urgent_queries:
        res = advisor.advise(query=q)
        assert res["success"] is True
        assert res["used_live_model"] is False  # Must NOT make model calls for emergencies!
        assert res["risk_assessment"]["risk_level"] == "urgent"
        assert "IMMEDIATE MEDICAL ATTENTION REQUIRED" in res["answer"]
        assert "emergency" in res["answer"].lower()


def test_general_health_advisor_clinical_escalation():
    """
    Verify pregnancy, chronic diseases, and eating disorders trigger clinical consultation advisories.
    """
    from agents.general_health_advisor import GeneralHealthAdvisor

    advisor = GeneralHealthAdvisor()

    cases = [
        ("I am pregnant in my second trimester, should I fast?", "pregnancy"),
        ("I have chronic kidney disease on dialysis, what fruit to eat?", "chronic_disease"),
        ("I have been starving myself and purging, how to lose more weight?", "eating_disorder"),
    ]

    for q, expected_cat in cases:
        risk = advisor.detect_safety_risks(q)
        assert risk["risk_level"] == "clinical"
        assert risk["category"] == expected_cat
        assert risk["advisory"] is not None


def test_general_health_advisor_prohibited_prescription_requests():
    """
    Verify medication and supplement dosage prescription requests are rejected.
    """
    from agents.general_health_advisor import GeneralHealthAdvisor

    advisor = GeneralHealthAdvisor()
    prohibited_queries = [
        "What medicine should I take for hypertension?",
        "What dosage of vitamin D drug should I take to cure my fatigue?",
        "Diagnose me: do I have scurvy?",
    ]

    for q in prohibited_queries:
        risk = advisor.detect_safety_risks(q)
        assert risk["risk_level"] == "prohibited_prescription"
        assert "Professional Scope Notice" in risk["advisory"]


def test_general_health_advisor_empty_input():
    """
    Verify empty or whitespace query is rejected cleanly before any model call.
    """
    from agents.general_health_advisor import GeneralHealthAdvisor

    advisor = GeneralHealthAdvisor()
    res = advisor.advise(query="   ")
    assert res["success"] is False
    assert "valid question" in res["status_message"].lower()
    assert res["answer"] is None


def test_general_health_advisor_severe_allergies_and_special_populations():
    """
    Verify severe allergies, young children, and older adults with dysphagia trigger clinical advisories.
    """
    from agents.general_health_advisor import GeneralHealthAdvisor

    advisor = GeneralHealthAdvisor()

    # 1. Severe food allergy
    risk_allergy = advisor.detect_safety_risks("I have a severe food allergy to peanuts and carry an epipen, can I eat bakery sweets?")
    assert risk_allergy["risk_level"] == "clinical"
    assert risk_allergy["category"] == "severe_allergy"
    assert "Severe Food Allergy" in risk_allergy["advisory"]

    # 2. Young children
    risk_child = advisor.detect_safety_risks("What toddler diet plan supports pediatric growth?")
    assert risk_child["risk_level"] == "clinical"
    assert risk_child["category"] == "infant_pediatric"
    assert "Pediatric" in risk_child["advisory"]

    # 3. Geriatric complex needs (dysphagia / swallowing difficulties)
    risk_geriatric = advisor.detect_safety_risks("My elderly grandmother has severe dysphagia and swallowing difficulty, how to hydrate her?")
    assert risk_geriatric["risk_level"] == "clinical"
    assert risk_geriatric["category"] == "geriatric_complex"
    assert "Geriatric" in risk_geriatric["advisory"]


def test_general_health_advisor_ordinary_questions_not_flagged():
    """
    Ensure everyday wellness and nutrition questions are NOT false-positively flagged as emergencies.
    """
    from agents.general_health_advisor import GeneralHealthAdvisor

    advisor = GeneralHealthAdvisor()

    ordinary_questions = [
        "How much water should I drink after a 30 minute morning jog?",
        "Is eating spinach good for natural iron intake?",
        "What are simple, balanced breakfast ideas for a college student?",
        "How do I read nutrition labels on packaged cereals to check for added sugars?",
        "Are all cooking fats unhealthy, or how should I balance cooking oils?",
        "What foods help prevent afternoon tiredness at a desk job?",
    ]

    for q in ordinary_questions:
        risk = advisor.detect_safety_risks(q)
        assert risk["risk_level"] == "none", f"Ordinary query incorrectly flagged: {q} -> {risk}"
        assert risk["advisory"] is None





