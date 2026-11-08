# Nutrition Agent - Architecture & Design

## 1. Project Overview
The **Nutrition Agent** is an agentic AI solution developed for the **IBM SkillsBuild Agentic AI Internship**. The application leverages **IBM Granite** foundation models via **IBM watsonx.ai**, combined with **RAG (Retrieval-Augmented Generation)**, **LangFlow**, and a **Streamlit** user interface.

---

## 2. Core Planned Features
1. **Nutrition Knowledge Assistant**: Responds to factual nutrition questions with grounded citations retrieved from trusted dietary guides (RAG).
2. **Personalised Diet Planner**: Formulates tailored daily or weekly meal plans matching personal health metrics (age, weight, height, activity level, dietary restrictions).
3. **Food Intake Analyzer**: Evaluates meals logged by the user and breaks down estimated macronutrients, calories, and micronutrient gaps.
4. **General Health and Nutrition Advisor**: Offers proactive lifestyle advice, hydration tracking reminders, and wellness guidance.

---

## 3. Technology Stack
- **Frontend / UI**: [Streamlit](https://streamlit.io/)
- **Foundation LLM**: IBM Granite (via `ibm-watsonx-ai` SDK)
- **Workflow & Agent Orchestration**: LangFlow / LangChain
- **Knowledge Retrieval (RAG)**: Document vector store with verified nutrition data
- **Language**: Python 3.10+

---

## 4. Directory Structure
```text
Nutrition-Agent/
├── .env.example              # Template for environment variables (IBM watsonx credentials)
├── .gitignore                # Git exclusions (credentials, virtual envs)
├── README.md                 # Project landing page and quickstart instructions
├── requirements.txt          # Python dependencies
├── app.py                    # Streamlit web application entry point
├── agents/                   # Agent logic and prompt pipelines
├── data/                     # RAG reference documents and guidelines
├── documentation/            # Architectural and design documentation
├── modules/                  # Reusable utilities (e.g., IBM watsonx client)
└── tests/                    # Unit and integration test cases
```

---

## 5. Security & Credentials
- All sensitive tokens (`WATSONX_APIKEY`, `WATSONX_PROJECT_ID`) are loaded strictly through environment variables (`.env`).
- Never commit actual API keys to version control.
