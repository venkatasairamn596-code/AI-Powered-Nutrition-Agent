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

