# 🥗 Nutrition Agent

**AI-Powered Nutrition Agent for IBM SkillsBuild Agentic AI Internship**

---

## 📌 Project Overview
The **Nutrition Agent** is an agentic AI application designed to offer evidence-based nutrition intelligence, personalized meal planning, dietary analysis, and health recommendations.

The project is built around **IBM Granite** foundation models running on **IBM watsonx.ai**, integrated with **RAG (Retrieval-Augmented Generation)**, **LangFlow**, and a **Streamlit** user interface.

---

## 🚀 Key Features Roadmap
1. **Nutrition Knowledge Assistant**: Evidence-grounded answers to nutrition, dietary science, and vitamin queries using verified nutritional guidelines.
2. **Personalised Diet Planner**: Custom daily and weekly meal plans tailored to user demographics, fitness goals, and allergies.
3. **Food Intake Analyzer**: Quantitative breakdown of logged meals into calories, macronutrients (proteins, carbs, fats), and micronutrients.
4. **General Health and Nutrition Advisor**: Proactive hydration, balanced eating habits, and lifestyle tips.

---

## 🛠️ Mandatory Technology Stack
- **Foundation LLM**: IBM Granite via [IBM watsonx.ai](https://www.ibm.com/watsonx)
- **Programming Language**: Python 3.10+
- **User Interface**: [Streamlit](https://streamlit.io/)
- **Workflow Orchestration**: [LangFlow](https://www.langflow.org/)
- **Information Grounding**: RAG (Retrieval-Augmented Generation)

---

## 📂 Project Directory Structure

```text
Nutrition-Agent/
│
├── .env.example              # Template for IBM watsonx credentials (safe to commit)
├── .gitignore                # Prevents secrets (.env) and virtual environments from being tracked
├── README.md                 # Project documentation and guide
├── requirements.txt          # Python library dependencies
├── app.py                    # Main Streamlit web application entry point
│
├── agents/                   # Agent logic and prompt pipelines for the 4 core features
│   └── __init__.py
│
├── data/                     # Nutrition knowledge base documents and guidelines for RAG
│   └── README.md
│
├── documentation/            # Architectural diagrams and technical documentation
│   └── architecture.md
│
├── modules/                  # Reusable modules and API connectors
│   ├── __init__.py
│   └── watsonx_client.py     # Clean IBM Granite / watsonx.ai client wrapper
│
└── tests/                    # Unit and smoke tests
    └── test_smoke.py
```

---

## 🔐 Credentials & Security
- **Never expose or commit your API keys.**
- Copy `.env.example` to create a local `.env` file:
  ```bash
  copy .env.example .env
  ```
- Fill in your IBM Cloud credentials inside `.env`:
  ```ini
  WATSONX_APIKEY=your_actual_api_key_here
  WATSONX_PROJECT_ID=your_watsonx_project_id_here
  WATSONX_URL=https://us-south.ml.cloud.ibm.com
  GRANITE_MODEL_ID=ibm/granite-3-8b-instruct
  ```
- `.gitignore` is already configured to keep `.env` private.

---

## ⚙️ Getting Started (When Ready to Install)

> [!NOTE]
> Per project guidelines, no packages have been installed automatically. When you are ready, follow the steps below:

### 1. Create a Python Virtual Environment (Recommended)
```bash
python -m venv venv
```
Activate it:
- **Windows (Command Prompt / PowerShell)**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **macOS / Linux**:
  ```bash
  source venv/bin/activate
  ```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application
```bash
streamlit run app.py
```

### 4. Run Smoke Tests
```bash
pytest tests/
```

---

## 🧩 Architectural Principles
- **No Fake AI Responses**: Modules strictly interface with real IBM Granite foundation model endpoints when credentials are provided. Unconfigured states provide informative diagnostics rather than simulated answers.
- **Modular Design**: UI (`app.py`), agents (`agents/`), data (`data/`), and model connectors (`modules/`) are strictly decoupled.
