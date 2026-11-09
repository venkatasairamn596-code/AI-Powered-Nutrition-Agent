"""
Nutrition Agent - Streamlit Application
IBM SkillsBuild Agentic AI Internship
"""

import streamlit as st
import os
from modules.watsonx_client import WatsonxGraniteClient
from agents import (
    NutritionKnowledgeAssistant,
    SAFETY_DISCLAIMER,
    PersonalisedDietPlanner,
    PLANNER_SAFETY_DISCLAIMER,
    FoodIntakeAnalyzer,
    ANALYZER_SAFETY_DISCLAIMER,
    GeneralHealthAdvisor,
    ADVISOR_SAFETY_DISCLAIMER,
)

# Page configuration
st.set_page_config(
    page_title="Nutrition Agent - IBM SkillsBuild",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    # Sidebar navigation
    st.sidebar.title("🥗 Nutrition Agent")
    st.sidebar.markdown("**IBM SkillsBuild Agentic AI Internship**")
    st.sidebar.divider()

    menu_selection = st.sidebar.radio(
        "Navigation",
        [
            "🏠 Welcome & Overview",
            "1️⃣ Nutrition Knowledge Assistant",
            "2️⃣ Personalised Diet Planner",
            "3️⃣ Food Intake Analyzer",
            "4️⃣ General Health Advisor",
            "⚙️ Integration Status",
        ],
    )

    client = WatsonxGraniteClient()
    config_status = client.get_configuration_status()

    # Route navigation
    if menu_selection == "🏠 Welcome & Overview":
        render_welcome_page(config_status)
    elif menu_selection == "1️⃣ Nutrition Knowledge Assistant":
        render_knowledge_assistant(client)
    elif menu_selection == "2️⃣ Personalised Diet Planner":
        render_diet_planner(client)
    elif menu_selection == "3️⃣ Food Intake Analyzer":
        render_food_intake_analyzer(client)
    elif menu_selection == "4️⃣ General Health Advisor":
        render_general_health_advisor(client)
    elif menu_selection == "⚙️ Integration Status":
        render_status_page(client, config_status)
    else:
        render_feature_placeholder(menu_selection)


def render_welcome_page(config_status: dict):
    st.title("🥗 Welcome to Nutrition Agent")
    st.caption("AI-Powered Dietary & Wellness Companion | Powered by IBM Granite & watsonx.ai")

    st.markdown(
        """
    Welcome to the **Nutrition Agent** project workspace! This application is designed to provide 
    evidence-based dietary planning, nutrition question answering, and food intake evaluation 
    using **IBM Granite** foundation models and **Retrieval-Augmented Generation (RAG)**.
    """
    )

    st.divider()

    # Planned Features Grid
    st.subheader("🎯 Core Application Capabilities")

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("### 1. 📖 Nutrition Knowledge Assistant")
            st.markdown(
                """
                - Answers evidence-based questions about nutrients, vitamins, and minerals.
                - Powered by **RAG** over curated nutritional guidelines (WHO, ICMR, USDA).
                - *Status: Ready for agent pipeline integration.*
                """
            )

        with st.container(border=True):
            st.markdown("### 2. 📋 Personalised Diet Planner")
            st.markdown(
                """
                - Generates tailored daily/weekly meal schedules.
                - Accommodates dietary restrictions, health goals, and personal preferences.
                - *Status: Ready for agent pipeline integration.*
                """
            )

    with col2:
        with st.container(border=True):
            st.markdown("### 3. 🔍 Food Intake Analyzer")
            st.markdown(
                """
                - Breaks down logged meals into macronutrients and calorie estimations.
                - Identifies potential nutritional deficiencies or surpluses.
                - *Status: Ready for agent pipeline integration.*
                """
            )

        with st.container(border=True):
            st.markdown("### 4. 💡 General Health & Nutrition Advisor")
            st.markdown(
                """
                - Offers lifestyle guidance, hydration tracking, and general wellness habits.
                - Non-diagnostic wellness support grounded in verified knowledge.
                - *Status: Ready for agent pipeline integration.*
                """
            )

    st.divider()

    # Technology Stack & Setup Overview
    st.subheader("🛠️ Technology Architecture")
    tech_col1, tech_col2, tech_col3 = st.columns(3)

    with tech_col1:
        st.markdown("**LLM Foundation**")
        st.write("• IBM Granite Models")
        st.write("• IBM watsonx.ai SDK")

    with tech_col2:
        st.markdown("**Orchestration & RAG**")
        st.write("• LangFlow Workflows")
        st.write("• Grounded Nutrition Knowledge Base")

    with tech_col3:
        st.markdown("**User Experience**")
        st.write("• Streamlit Interactive Web App")
        st.write("• Python 3 Modular Architecture")


def render_knowledge_assistant(client: WatsonxGraniteClient):
    st.title("📖 Nutrition Knowledge Assistant")
    st.caption("Evidence-Based Nutrition Guidance Grounded in WHO, ICMR-NIN & USDA Standards | Powered by IBM Granite & RAG")

    st.warning(SAFETY_DISCLAIMER)

    sample_questions = [
        "Select a sample question or type below...",
        "What is the recommended daily intake of free sugars according to WHO?",
        "How much protein should an adult consume daily according to ICMR-NIN guidelines?",
        "What are the WHO recommendations for daily sodium and salt intake?",
        "How much water should an adult drink daily?",
        "What are the dietary fiber recommendations according to dietary guidelines?",
        "What are the essential micronutrients vulnerable to widespread deficiency?",
    ]

    selected_sample = st.selectbox("💡 Quick Sample Questions:", sample_questions)

    default_text = ""
    if selected_sample != sample_questions[0]:
        default_text = selected_sample

    user_query = st.text_area(
        "Enter your nutrition question:",
        value=default_text,
        placeholder="e.g. What are the healthy daily limits for salt and sugar?",
        height=100,
    )

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        search_clicked = st.button("🔍 Ask Nutrition Assistant", type="primary", use_container_width=True)

    if search_clicked:
        if not user_query.strip():
            st.error("Please enter a nutrition question to search.")
            return

        assistant = NutritionKnowledgeAssistant(client=client)

        with st.spinner("Retrieving verified guidelines and generating grounded response..."):
            result = assistant.answer_question(user_query.strip())

        # Display Section: Retrieved RAG Evidence
        st.subheader("📚 Verified Dietary Evidence (RAG Context)")
        retrieved_docs = result.get("retrieved_docs", [])

        if retrieved_docs:
            for doc in retrieved_docs:
                with st.expander(f"📌 {doc.get('title')} ({doc.get('source')})", expanded=True):
                    st.write(doc.get("content"))
                    caption_text = f"**Authority:** {doc.get('source')} | **Topic:** {doc.get('topic')}"
                    if doc.get("source_url"):
                        caption_text += f" | 🔗 [Official Document Link]({doc.get('source_url')})"
                    st.caption(caption_text)
        else:
            st.info("No specific guidelines retrieved for this topic.")

        # Display Section: Model Answer / Status
        st.divider()
        st.subheader("🤖 Nutrition Assistant Guidance")

        if result.get("used_live_model") and result.get("answer"):
            st.success("✅ Grounded response generated by IBM Granite:")
            st.markdown(result["answer"])
        else:
            st.info(f"ℹ️ {result.get('status_message', 'Awaiting live inference.')}")
            if not result.get("used_live_model"):
                st.markdown(
                    """
                    > **Note for watsonx.ai Live Generation**: 
                    > The RAG context and grounded prompt pipeline are ready. When you associate your Watson Machine Learning (WML) service instance with the project in watsonx.ai, live Granite responses will appear directly here.
                    """
                )

        # Citations
        if result.get("sources"):
            st.markdown("---")
            st.markdown("**Cited Public Health Authorities:**")
            for s in result["sources"]:
                st.markdown(f"- 🏛️ `{s}`")

        # Mandatory Safety Footer
        st.caption(SAFETY_DISCLAIMER)


def render_diet_planner(client: WatsonxGraniteClient):
    st.title("📋 Personalised Diet Planner")
    st.caption("Evidence-Grounded Indian Meal Planning Based on ICMR-NIN Guidelines | Powered by IBM Granite")

    st.warning(PLANNER_SAFETY_DISCLAIMER)

    planner = PersonalisedDietPlanner(client=client)

    st.subheader("👤 User Profile & Dietary Requirements")

    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age (years):", min_value=10, max_value=115, value=25, step=1)
            dietary_preference = st.selectbox(
                "Dietary Preference:",
                planner.VALID_DIET_PREFERENCES,
                index=0,
            )
            activity_level = st.selectbox(
                "Physical Activity Level:",
                planner.VALID_ACTIVITY_LEVELS,
                index=1,
            )

        with col2:
            nutrition_goal = st.selectbox(
                "Primary Nutrition Goal:",
                planner.VALID_GOALS,
                index=0,
            )
            food_allergies = st.text_input(
                "Food Allergies to Exclude:",
                placeholder="e.g. Peanuts, Dairy, Shellfish, Gluten, None",
                help="List any allergens that must be strictly excluded from the meal plan.",
            )
            dietary_restrictions = st.text_input(
                "Special Dietary Restrictions / Notes:",
                placeholder="e.g. Low sodium, Diabetic-friendly, Lactose-free, None",
            )

        preferred_foods = st.text_input(
            "Preferred Regional Indian Foods / Staples:",
            placeholder="e.g. North Indian (Roti, Dal, Paneer), South Indian (Idli, Sambar, Rice), Millets (Ragi, Jowar)",
            help="Tell the assistant what regional foods you prefer or normally have access to.",
        )

        generate_btn = st.button("🥗 Generate Personalised Meal Plan", type="primary", use_container_width=True)

    if generate_btn:
        profile = {
            "age": int(age),
            "dietary_preference": dietary_preference,
            "activity_level": activity_level,
            "nutrition_goal": nutrition_goal,
            "food_allergies": food_allergies,
            "dietary_restrictions": dietary_restrictions,
            "preferred_foods": preferred_foods,
        }

        # Validate inputs first
        is_valid, errors, warnings = planner.validate_inputs(profile)

        if not is_valid:
            for err in errors:
                st.error(f"❌ {err}")
            return

        for warn in warnings:
            st.warning(f"⚠️ {warn}")

        with st.spinner("Retrieving ICMR-NIN evidence and generating personalized Indian meal plan with IBM Granite..."):
            result = planner.generate_plan(profile)

        if result.get("success") and result.get("meal_plan"):
            st.success("✅ Personalised Indian Meal Plan Generated Successfully:")

            with st.container(border=True):
                st.markdown(result["meal_plan"])

            # Retrieved evidence expander
            if result.get("retrieved_docs"):
                with st.expander("📚 Grounding Evidence (ICMR-NIN Guidelines Used)", expanded=False):
                    for d in result["retrieved_docs"]:
                        st.markdown(f"**{d.get('title')}** ({d.get('source')})")
                        st.caption(d.get("content"))
                        if d.get("source_url"):
                            st.caption(f"🔗 [Official Publication]({d.get('source_url')})")
                        st.divider()

        else:
            if result.get("allergen_blocked"):
                st.error(result.get("status_message"))
            else:
                st.warning(f"Generation Notice: {result.get('status_message')}")
            if result.get("retrieved_docs"):
                with st.expander("📚 Retrieved Guidelines (RAG Context)", expanded=True):
                    for d in result["retrieved_docs"]:
                        st.write(f"**{d.get('title')}**: {d.get('content')}")

        st.caption(PLANNER_SAFETY_DISCLAIMER)


def render_food_intake_analyzer(client: WatsonxGraniteClient):
    st.title("3️⃣ Food Intake Analyzer")
    st.caption("Deterministic Nutrition Tracking (IFCT 2017 & USDA) + IBM Granite 4.0 Dietary Commentary")

    analyzer = FoodIntakeAnalyzer(client=client)

    if "food_log" not in st.session_state:
        st.session_state.food_log = []

    if "intake_analysis_result" not in st.session_state:
        st.session_state.intake_analysis_result = None

    st.markdown(
        """
        Log your meals throughout the day to receive **deterministic nutrient estimates** 
        calculated from the authoritative **Indian Food Composition Tables (IFCT 2017)** and **USDA FoodData Central**, 
        accompanied by **IBM Granite 4.0** dietary analysis grounded in **ICMR-NIN 2024** guidelines.
        """
    )

    # Educational Disclaimer Banner
    st.info(ANALYZER_SAFETY_DISCLAIMER)

    # Log Input Form
    with st.container(border=True):
        st.subheader("📝 Add Food Item to Daily Log")
        available_foods = analyzer.get_available_foods()
        food_options = ["-- Select from verified Indian & global foods --"] + [
            f"{f['name']} ({f.get('default_serving_description', '1 serving')})" for f in available_foods
        ] + ["✏️ Enter custom food name / search..."]

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            selected_option = st.selectbox(
                "Food Item",
                options=food_options,
                index=0,
                help="Select from verified food composition entries or choose custom food.",
            )

            custom_food_name = ""
            if selected_option == "✏️ Enter custom food name / search...":
                custom_food_name = st.text_input(
                    "Custom Food Name",
                    placeholder="e.g. Masala Dosa, Sprouted Moong Salad, Pizza...",
                )

        with col2:
            meal_type = st.selectbox(
                "Meal Type",
                options=analyzer.VALID_MEAL_TYPES,
                index=1,
            )

        with col3:
            portion = st.number_input(
                "Portion / Servings",
                min_value=0.25,
                max_value=10.0,
                value=1.0,
                step=0.25,
                format="%.2f",
                help="Multiplier relative to standard serving (e.g., 2.0 = two servings/rotis/cups).",
            )

        add_col, sample_col = st.columns([1, 1])
        with add_col:
            if st.button("➕ Add Item to Daily Log", use_container_width=True):
                target_name = ""
                if selected_option == "✏️ Enter custom food name / search...":
                    target_name = custom_food_name.strip()
                elif selected_option != "-- Select from verified Indian & global foods --":
                    idx = food_options.index(selected_option) - 1
                    target_name = available_foods[idx]["name"]

                is_valid, val_errors = analyzer.validate_entry(target_name, portion, meal_type)
                if is_valid:
                    st.session_state.food_log.append({
                        "food_name": target_name,
                        "portion": float(portion),
                        "meal_type": meal_type,
                    })
                    st.session_state.intake_analysis_result = None
                    st.rerun()
                else:
                    for err in val_errors:
                        st.warning(err)

        with sample_col:
            if st.button("📋 Load Balanced Sample Day (3 Meals + Snack)", use_container_width=True):
                st.session_state.food_log = [
                    {"food_name": "Steamed Idli", "portion": 2.0, "meal_type": "Breakfast"},
                    {"food_name": "Vegetable Sambar", "portion": 1.0, "meal_type": "Breakfast"},
                    {"food_name": "Whole Wheat Roti / Chapati", "portion": 2.0, "meal_type": "Lunch"},
                    {"food_name": "Cooked Toor / Arhar Dal (with tadka)", "portion": 1.0, "meal_type": "Lunch"},
                    {"food_name": "Mixed Vegetable Sabzi", "portion": 1.0, "meal_type": "Lunch"},
                    {"food_name": "Curd / Plain Dahi (Whole Cow Milk)", "portion": 1.0, "meal_type": "Lunch"},
                    {"food_name": "Roasted Peanuts / Groundnuts", "portion": 1.0, "meal_type": "Evening Snack"},
                    {"food_name": "Indian Masala Chai (with Milk and 1 tsp Sugar)", "portion": 1.0, "meal_type": "Evening Snack"},
                    {"food_name": "Moong Dal Khichdi", "portion": 1.0, "meal_type": "Dinner"},
                    {"food_name": "Fresh Green Salad (Cucumber, Tomato, Carrot)", "portion": 1.0, "meal_type": "Dinner"},
                ]
                st.session_state.intake_analysis_result = None
                st.rerun()

    # Current Daily Food Log
    st.divider()
    st.subheader(f"📋 Daily Food Log ({len(st.session_state.food_log)} items)")

    if len(st.session_state.food_log) == 0:
        st.info("Your daily food log is currently empty. Add foods above or click 'Load Balanced Sample Day' to test.")
    else:
        for i, entry in enumerate(st.session_state.food_log):
            c1, c2, c3, c4, c5 = st.columns([1.5, 2.5, 1, 2, 0.8])
            matched = analyzer.find_food(entry["food_name"])
            with c1:
                st.write(f"**{entry['meal_type']}**")
            with c2:
                st.write(entry["food_name"])
                item_allergens = analyzer.identify_allergens(
                    entry["food_name"],
                    matched.get("aliases", []) if matched else None
                )
                if item_allergens:
                    st.caption(f"⚠️ Allergen: {', '.join(item_allergens)}")
            with c3:
                st.write(f"x{entry['portion']:.2f}")
            with c4:
                if matched:
                    st.caption(f"✅ {matched.get('source', 'IFCT 2017')}")
                else:
                    st.caption("⚠️ Nutrition data unavailable")
            with c5:
                if st.button("🗑️", key=f"del_{i}", help="Remove this item"):
                    st.session_state.food_log.pop(i)
                    st.session_state.intake_analysis_result = None
                    st.rerun()

        col_clear, col_analyze = st.columns([1, 2])
        with col_clear:
            if st.button("🧹 Clear Entire Log", use_container_width=True):
                st.session_state.food_log = []
                st.session_state.intake_analysis_result = None
                st.rerun()

        with col_analyze:
            analyze_clicked = st.button("🔍 Analyze Daily Intake with IBM Granite", type="primary", use_container_width=True)

        if analyze_clicked:
            with st.spinner("Calculating deterministic nutrients & consulting IBM Granite 4.0..."):
                st.session_state.intake_analysis_result = analyzer.analyze_intake(st.session_state.food_log)

    # Render Analysis Results
    if st.session_state.intake_analysis_result:
        res = st.session_state.intake_analysis_result
        intake = res["intake_summary"]

        st.divider()
        if intake.get("is_partial_estimate"):
            st.subheader("📊 Calculated Nutrient Intake Summary (⚠️ Partial Daily Estimate)")
        elif intake.get("has_no_verified_data"):
            st.subheader("📊 Calculated Nutrient Intake Summary (Data Unavailable)")
        else:
            st.subheader("📊 Calculated Nutrient Intake Summary (Verified Daily Estimate)")

        st.caption(intake["disclaimer_label"])

        if intake.get("is_partial_estimate"):
            st.warning(
                f"⚠️ **Partial Daily Estimate**: Totals reflect **{intake['matched_items_count']} of {intake['total_items']}** logged items. "
                f"**{intake['unmatched_items_count']} item(s)** ([{', '.join(intake['unmatched_items'])}) have **unavailable nutritional data** in the verified IFCT 2017 / USDA tables. "
                "Per safety rules, they are completely excluded from numerical totals rather than guessed or treated as zero."
            )
        elif intake.get("has_no_verified_data"):
            st.error(
                f"⚠️ **Nutritional Data Unavailable**: None of the {intake['total_items']} logged item(s) could be verified in "
                "the IFCT 2017 / USDA composition tables. Zero numerical totals are fabricated."
            )

        # High-level metric cards
        if not intake.get("has_no_verified_data"):
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            partial_tag = " (Partial)" if intake.get("is_partial_estimate") else ""
            m1.metric("Calories" + partial_tag, f"{intake['total_calories_kcal']} kcal")
            m2.metric("Protein", f"{intake['total_protein_g']} g", f"{intake['macronutrient_ratios']['protein_pct']}% cals")
            m3.metric("Carbohydrates", f"{intake['total_carbohydrates_g']} g", f"{intake['macronutrient_ratios']['carbs_pct']}% cals")
            m4.metric("Total Fat", f"{intake['total_fat_g']} g", f"{intake['macronutrient_ratios']['fat_pct']}% cals")
            m5.metric("Dietary Fiber", f"{intake['total_dietary_fiber_g']} g")
            m6.metric("Sodium", f"{intake['total_sodium_mg']} mg")

        # Detailed item breakdown table
        with st.expander("📋 Detailed Item-by-Item Breakdown & Traceable Sources", expanded=False):
            breakdown_data = []
            for b in intake["item_breakdowns"]:
                allergen_str = ", ".join(b.get("allergens", [])) if b.get("allergens") else "None detected"
                breakdown_data.append({
                    "Meal": b["meal_type"],
                    "Food Item": b["food_name"],
                    "Portion": f"x{b['portion']}",
                    "Allergens": allergen_str,
                    "Calories (kcal)": f"{b['calories_kcal']} kcal" if b["matched"] else "Data unavailable (Excluded)",
                    "Protein (g)": f"{b['protein_g']} g" if b["matched"] else "Data unavailable (Excluded)",
                    "Carbs (g)": f"{b['carbohydrates_g']} g" if b["matched"] else "Data unavailable (Excluded)",
                    "Fat (g)": f"{b['fat_g']} g" if b["matched"] else "Data unavailable (Excluded)",
                    "Fiber (g)": f"{b['dietary_fiber_g']} g" if b["matched"] else "Data unavailable (Excluded)",
                    "Sodium (mg)": f"{b['sodium_mg']} mg" if b["matched"] else "Data unavailable (Excluded)",
                    "Table Reference": b.get("source_code", b["source"]),
                })
            st.dataframe(breakdown_data, use_container_width=True)

        # IBM Granite AI Analysis
        st.subheader("🤖 IBM Granite 4.0 Nutritional Commentary")
        if res.get("analysis_commentary"):
            st.markdown(res["analysis_commentary"])
            st.caption(f"Status: {res.get('status_message')}")
        else:
            st.warning(f"AI Commentary Notice: {res.get('status_message')}")

        # Grounded RAG Evidence
        if res.get("retrieved_docs"):
            st.divider()
            st.subheader("📚 Grounded Public Health Guidelines (ICMR-NIN & WHO)")
            for d in res["retrieved_docs"]:
                with st.container(border=True):
                    st.markdown(f"**{d.get('title')}** — *{d.get('source_organization')} ({d.get('publication_year')})*")
                    st.caption(f"Document: {d.get('document_name')}")
                    st.write(d.get("content"))
                    if d.get("source_url"):
                        st.markdown(f"🔗 [Official Reference Document]({d.get('source_url')})")

        st.caption(res["disclaimer"])


def render_general_health_advisor(client: WatsonxGraniteClient):
    st.title("4️⃣ General Health & Nutrition Advisor")
    st.caption("Everyday Wellness Coaching, Practical Habit Guidance & Clinical Safety Guardrails")

    advisor = GeneralHealthAdvisor(client=client)

    st.markdown(
        """
        Ask practical questions about **hydration, balanced daily routines, reading nutrition labels, 
        student eating habits, food safety, and debunking common dietary myths**. 
        All advice is grounded in **ICMR-NIN 2024** and **WHO** public health evidence.
        """
    )

    # Visible non-medical safety banner
    st.info(ADVISOR_SAFETY_DISCLAIMER)

    # Example questions selector
    selected_example = st.selectbox(
        "💡 Select an example question (or write your own below):",
        options=["-- Choose an example question or enter custom inquiry below --"] + advisor.EXAMPLE_QUESTIONS,
        index=0,
    )

    default_q = ""
    if selected_example != "-- Choose an example question or enter custom inquiry below --":
        default_q = selected_example

    with st.form("health_advisor_form"):
        user_query = st.text_area(
            "Your Health & Nutrition Question:",
            value=default_q,
            placeholder="e.g. How can I stay properly hydrated throughout the workday? Or: How do I read food labels in India to spot hidden sugars?",
            help="Enter any general nutrition, wellness, or dietary habit question.",
        )

        col1, col2 = st.columns(2)
        with col1:
            age_group = st.selectbox(
                "Age Group (Optional):",
                options=advisor.AGE_GROUPS,
                index=0,
                help="Helps the advisor provide age-appropriate habit suggestions (e.g. adolescent growth vs older adult digestive comfort).",
            )
        with col2:
            user_context = st.text_input(
                "Daily Routine / Lifestyle Context (Optional):",
                placeholder="e.g. Desk job with little movement, night-shift student, vegetarian runner...",
                help="Optional lifestyle context without sharing sensitive personal health information.",
            )

        submit_advisor = st.form_submit_button("💡 Get General Wellness Advice", type="primary", use_container_width=True)

    if submit_advisor:
        if not user_query.strip():
            st.warning("Please enter a question or choose an example from above.")
        else:
            with st.spinner("Analyzing inquiry & retrieving evidence for IBM Granite 4.0..."):
                result = advisor.advise(
                    query=user_query,
                    age_group=age_group,
                    user_context=user_context,
                )

            # Check for Urgent Emergency Warning
            if result["risk_assessment"]["risk_level"] == "urgent":
                st.error(result["answer"])
            else:
                # If clinical advisory exists, display prominent warning
                if result.get("safety_advisory"):
                    st.warning(result["safety_advisory"])

                # Display Granite Answer
                if result.get("success") and result.get("answer"):
                    st.subheader("🤖 Everyday Wellness Guidance (IBM Granite 4.0)")
                    st.markdown(result["answer"])
                    st.caption(f"Status: {result.get('status_message')}")
                else:
                    st.error(f"Generation Notice: {result.get('status_message')}")

                # Display Retrieved RAG Evidence
                st.divider()
                st.subheader("📚 Verified Public Health Guidelines (ICMR-NIN & WHO)")
                if result.get("retrieved_docs") and len(result["retrieved_docs"]) > 0:
                    for d in result["retrieved_docs"]:
                        with st.container(border=True):
                            st.markdown(f"**{d.get('title')}** — *{d.get('source_organization')} ({d.get('publication_year')})*")
                            st.caption(f"Document: {d.get('document_name')}")
                            st.write(d.get("content"))
                            if d.get("source_url"):
                                st.markdown(f"🔗 [Official Reference Document]({d.get('source_url')})")
                else:
                    st.info(
                        "ℹ️ Note: No direct local guideline match was retrieved from the local dataset for this specific topic. "
                        "The advice above is grounded strictly in foundational public health principles."
                    )

            st.caption(result["disclaimer"])


def render_feature_placeholder(feature_name: str):
    st.title(feature_name)
    st.info(
        f"🚧 **{feature_name}** is currently in initial setup mode. "
        "The agent logic and IBM Granite pipeline will be connected in upcoming stages."
    )
    st.markdown(
        """
        ### Next Steps for this Module:
        1. Define domain prompts and LangFlow agent flow.
        2. Ingest domain-specific nutrition guidelines into the `data/` directory for RAG.
        3. Connect the IBM Granite model client from `modules/watsonx_client.py`.
        """
    )


def render_status_page(client: WatsonxGraniteClient, config_status: dict):
    st.title("⚙️ System & Integration Diagnostics")
    st.markdown("Check project configuration and IBM watsonx.ai connectivity prerequisites.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("IBM watsonx.ai Configuration")
        if config_status["is_ready"]:
            st.success("✅ watsonx credentials are fully detected in environment!")
        else:
            st.warning("⚠️ watsonx credentials are not yet configured.")
            st.caption(
                "Copy `.env.example` to `.env` and fill in your `WATSONX_APIKEY` "
                "and `WATSONX_PROJECT_ID`."
            )

        st.json(
            {
                "API Key Configured": config_status["has_api_key"],
                "Project ID Configured": config_status["has_project_id"],
                "Masked Project ID": config_status.get("project_id_masked", ""),
                "Endpoint URL": config_status["service_url"],
                "Target Granite Model": config_status["target_model"],
            }
        )

        st.markdown("---")
        st.markdown("#### Test Connectivity")
        st.caption("Verifies authentication with IBM watsonx.ai without running paid inference.")

        if st.button("🔌 Test watsonx Connection", use_container_width=True):
            with st.spinner("Checking IBM watsonx.ai connectivity..."):
                test_result = client.test_connection()
                if test_result["success"]:
                    st.success(f"✅ {test_result['message']}")
                else:
                    st.warning(f"⚠️ {test_result['message']}")

    with col2:
        st.subheader("Workspace Directory Health")
        folders = ["agents", "data", "documentation", "modules", "tests"]
        folder_health = {}
        for f in folders:
            folder_health[f] = "✅ Present" if os.path.isdir(f) else "❌ Missing"
        st.json(folder_health)


if __name__ == "__main__":
    main()
