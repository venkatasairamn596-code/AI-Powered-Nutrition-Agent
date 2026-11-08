"""
Feature 4: General Health & Nutrition Advisor

An interactive, evidence-grounded advisor for everyday nutrition, healthy eating habits,
hydration, daily routines, food safety, and debunking common dietary myths.

Distinction from Feature 1:
- Feature 1 provides factual guideline Q&A retrieval.
- Feature 4 delivers contextual everyday wellness guidance, evaluating safety boundaries,
  analyzing user context and age group, providing actionable lifestyle advice, and
  actively routing users to healthcare professionals when clinical red flags appear.

Safety Guardrails:
- Does NOT diagnose illnesses, medical conditions, or nutrient deficiencies.
- Does NOT prescribe medicines, supplements, clinical diets, or medication changes.
- Escalates urgent symptoms, pregnancy, eating disorders, and chronic conditions to professional care.
- Clear educational non-medical disclaimer.
"""

import re
from typing import Dict, Any, List, Optional, Tuple

from modules.rag_retriever import NutritionKnowledgeRetriever
from modules.watsonx_client import WatsonxGraniteClient

ADVISOR_SAFETY_DISCLAIMER = (
    "⚠️ Educational Disclaimer: The General Health & Nutrition Advisor provides general educational "
    "information on nutrition, everyday wellness habits, and healthy eating patterns. It does not "
    "constitute medical advice, diagnosis, treatment, or prescription of drugs/supplements. "
    "If you are experiencing acute symptoms, managing a chronic condition, pregnant, or suspect a "
    "deficiency, consult a registered dietitian, physician, or qualified healthcare professional."
)

URGENT_KEYWORDS = [
    "chest pain", "shortness of breath", "difficulty breathing", "blood in vomit",
    "severe dizziness", "fainted", "fainting", "sudden numbness", "anaphylaxis",
    "swelling of lips", "severe allergic reaction", "poisoning", "coughing blood",
]

CLINICAL_KEYWORDS = {
    "prescription_request": ["what medicine", "what drug", "what dosage", "what supplement should i take", "prescribe", "cure my", "do i have a deficiency", "diagnose me"],
    "pregnancy": ["pregnant", "pregnancy", "trimester", "expecting a baby", "prenatal"],
    "eating_disorder": ["anorexia", "bulimia", "binge eating", "purge", "purging", "starving myself"],
    "chronic_disease": ["diabetes", "diabetic", "chronic kidney", "renal failure", "dialysis", "heart disease", "hypertension", "liver disease", "cancer", "chemotherapy"],
    "severe_allergy": ["severe allergy", "severe allergies", "severe food allergy", "food allergy diagnosis", "allergy testing", "epipen", "epinephrine"],
    "geriatric_complex": ["dysphagia", "swallowing difficulty", "swallowing problem", "tube feeding", "elderly weight loss", "dementia nutrition", "frail senior", "older adult malnutrition"],
    "infant_pediatric": ["infant", "newborn", "baby food under 6 months", "formula milk", "toddler", "child nutrition under 5", "pediatric growth"],
}


class GeneralHealthAdvisor:
    """
    Agent responsible for contextual everyday wellness and nutrition advice,
    safety escalation detection, RAG guideline retrieval, and IBM Granite 4.0 orchestration.
    """

    AGE_GROUPS = [
        "Prefer not to say",
        "Adolescent / Student (13–17)",
        "Young Adult / College (18–25)",
        "Adult (26–59)",
        "Older Adult (60+)",
    ]

    EXAMPLE_QUESTIONS = [
        "How much water should I realistically drink every day, and does tea count?",
        "What are simple, budget-friendly ways for a college student to eat balanced meals?",
        "How can I understand nutrition labels when buying packaged foods in India?",
        "Is skipping breakfast bad for energy and metabolism?",
        "Are all fats unhealthy, or how should I balance cooking oils in Indian cooking?",
        "What are practical tips for overcoming afternoon energy crashes without excess caffeine?",
    ]

    def __init__(self, client: Optional[WatsonxGraniteClient] = None):
        self.client = client or WatsonxGraniteClient()
        self.retriever = NutritionKnowledgeRetriever()

    def detect_safety_risks(self, query: str, context_text: str = "") -> Dict[str, Any]:
        """
        Scans question and user context for acute emergency symptoms, clinical conditions,
        eating disorder cues, or prohibited prescription/diagnosis requests.
        """
        combined = f"{query} {context_text}".lower()

        # 1. Urgent Emergency Symptoms Check
        for kw in URGENT_KEYWORDS:
            if re.search(rf"\b{re.escape(kw)}\b", combined):
                return {
                    "risk_level": "urgent",
                    "category": "emergency_symptoms",
                    "keyword_matched": kw,
                    "advisory": (
                        "🚨 IMMEDIATE MEDICAL ATTENTION REQUIRED: Your question mentions potential acute symptoms "
                        f"('{kw}'). The Nutrition Agent is an educational tool and CANNOT assist with medical emergencies. "
                        "Please contact emergency services (e.g., dial 112 in India / 911 / your local emergency number) "
                        "or proceed immediately to the nearest emergency department."
                    ),
                }

        # 2. Clinical Conditions & Prescription Requests Check
        for category, terms in CLINICAL_KEYWORDS.items():
            for kw in terms:
                if re.search(rf"\b{re.escape(kw)}\b", combined):
                    if category == "prescription_request":
                        return {
                            "risk_level": "prohibited_prescription",
                            "category": category,
                            "keyword_matched": kw,
                            "advisory": (
                                "⚕️ Professional Scope Notice: You asked about medical diagnosis, medication, or supplement prescription. "
                                "AI systems cannot evaluate clinical blood tests, diagnose deficiencies, or prescribe pharmacological doses. "
                                "Please consult a licensed physician or registered dietitian for personalized medical advice."
                            ),
                        }
                    elif category == "pregnancy":
                        return {
                            "risk_level": "clinical",
                            "category": category,
                            "keyword_matched": kw,
                            "advisory": (
                                "🤰 Maternal Health Notice: Pregnancy and lactation involve unique, critical nutritional demands "
                                "(such as folate, iron, and iodine requirements). General AI wellness advice cannot replace specialized "
                                "obstetric care. Please discuss your dietary routine directly with your obstetrician or clinical dietitian."
                            ),
                        }
                    elif category == "eating_disorder":
                        return {
                            "risk_level": "clinical",
                            "category": category,
                            "keyword_matched": kw,
                            "advisory": (
                                "💛 Health & Wellbeing Support: Inquiries involving eating habits that cause distress, severe restriction, "
                                "or purging benefit greatly from compassionate, specialized medical support. Please consider speaking "
                                "with a licensed mental health professional or physician specializing in eating disorders."
                            ),
                        }
                    elif category == "chronic_disease":
                        return {
                            "risk_level": "clinical",
                            "category": category,
                            "keyword_matched": kw,
                            "advisory": (
                                "🩺 Clinical Condition Advisory: You mentioned managing a chronic medical condition. "
                                "Nutritional advice for conditions like diabetes, kidney disorders, or cardiovascular disease "
                                "requires individualized medical nutrition therapy (MNT) supervised by your healthcare provider."
                            ),
                        }
                    elif category == "severe_allergy":
                        return {
                            "risk_level": "clinical",
                            "category": category,
                            "keyword_matched": kw,
                            "advisory": (
                                "🥜 Severe Food Allergy Notice: Severe food allergies present potential anaphylaxis risks. "
                                "General AI systems cannot verify the safety of foods, cross-contamination risks, or restaurant menus. "
                                "Consult an allergist or registered clinical dietitian and maintain prescribed emergency auto-injectors."
                            ),
                        }
                    elif category == "geriatric_complex":
                        return {
                            "risk_level": "clinical",
                            "category": category,
                            "keyword_matched": kw,
                            "advisory": (
                                "👵 Geriatric & Complex Care Advisory: Older adults with swallowing difficulties (dysphagia), "
                                "rapid unintentional weight loss, or complex health needs require in-person clinical and speech-pathology/dietetic "
                                "evaluation to ensure safe hydration and prevent aspiration or nutritional decline."
                            ),
                        }
                    elif category == "infant_pediatric":
                        return {
                            "risk_level": "clinical",
                            "category": category,
                            "keyword_matched": kw,
                            "advisory": (
                                "👶 Pediatric Nutrition Notice: Infants, toddlers, and young children have sensitive developmental "
                                "nutritional requirements. Please consult a pediatrician or pediatric dietitian for child growth nutrition."
                            ),
                        }

        return {
            "risk_level": "none",
            "category": "general_wellness",
            "keyword_matched": None,
            "advisory": None,
        }

    def retrieve_evidence(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieves grounded evidence from the curated WHO, ICMR-NIN, and USDA guidelines.
        """
        results = self.retriever.search(query, top_k=3)
        return results

    def build_prompt(
        self,
        query: str,
        age_group: str,
        user_context: str,
        evidence_context: str,
        has_evidence: bool,
        safety_notice: Optional[str] = None,
    ) -> str:
        """
        Builds a safe, grounded prompt for IBM Granite 4.0.
        """
        age_str = f"User Age Group: {age_group}" if age_group and age_group != "Prefer not to say" else "Age Group: Unspecified"
        context_str = f"User Lifestyle / Routine Context: {user_context}" if user_context.strip() else "User Lifestyle / Routine Context: None provided"

        evidence_section = evidence_context if has_evidence else "[No direct guideline match found in verified dataset. Rely strictly on general public health principles and state clearly that specific local guidelines were not retrieved.]"

        safety_instruction = ""
        if safety_notice:
            safety_instruction = (
                f"\nCLINICAL SAFETY ESCALATION:\n{safety_notice}\n"
                "Emphasize this safety advisory in your response and advise professional consultation.\n"
            )

        prompt = f"""You are the General Health & Nutrition Advisor, an evidence-based educator on everyday nutrition and healthy living.
You provide practical, culturally relevant guidance grounded in ICMR-NIN 2024 and WHO public health standards.
You are NOT a doctor, clinical dietitian, or medical provider. You must NEVER diagnose illnesses, prescribe medications, or recommend clinical diets.

VERIFIED EVIDENCE (PUBLIC HEALTH GUIDELINES):
---------------------------------------------------------------------
{evidence_section}
---------------------------------------------------------------------

USER INQUIRY:
Question: {query}
{age_str}
{context_str}
{safety_instruction}
TASK & CONSTRAINTS:
1. Provide a clear, educational, and empathetic answer explaining everyday wellness implications.
2. Ground your advice in the retrieved guidelines whenever relevant; if direct evidence is unavailable, state this transparently.
3. Offer practical, culturally authentic food and routine choices (e.g., traditional Indian foods, balanced meal combinations, hydration tips).
4. Strictly avoid diagnosing illnesses, prescribing supplements or medications, or claiming that the user has a deficiency.
5. If the question touches on clinical conditions, advise consulting a qualified physician or registered dietitian.
6. Do NOT invent citations, statistics, or guideline recommendations.
7. Conclude with a clear educational safety disclaimer.

EDUCATIONAL ADVICE:"""
        return prompt

    def advise(
        self,
        query: str,
        age_group: str = "Prefer not to say",
        user_context: str = "",
    ) -> Dict[str, Any]:
        """
        Full orchestration pipeline for Feature 4:
        1. Validates user inquiry.
        2. Scans for acute or clinical safety risks.
        3. Retrieves evidence via RAG.
        4. Constructs grounded IBM Granite prompt.
        5. Generates AI response or returns immediate safety guidance.
        """
        clean_query = str(query).strip() if query else ""
        if not clean_query:
            return {
                "success": False,
                "answer": None,
                "retrieved_docs": [],
                "risk_assessment": {"risk_level": "none", "category": "empty_input"},
                "safety_advisory": None,
                "status_message": "Please enter a valid question or topic before requesting advice.",
                "disclaimer": ADVISOR_SAFETY_DISCLAIMER,
                "used_live_model": False,
            }

        risk_assessment = self.detect_safety_risks(clean_query, user_context)

        # Immediate escalation for emergency medical symptoms
        if risk_assessment["risk_level"] == "urgent":
            return {
                "success": True,
                "answer": (
                    f"### 🚨 Emergency Medical Warning\n\n"
                    f"{risk_assessment['advisory']}\n\n"
                    "**Why AI cannot respond to this query:** Dietary or wellness advice is not appropriate when acute "
                    "symptoms are present. Please seek immediate, professional clinical care."
                ),
                "retrieved_docs": [],
                "risk_assessment": risk_assessment,
                "safety_advisory": risk_assessment["advisory"],
                "status_message": "Urgent emergency symptom detected. Guidance redirected to emergency medical care.",
                "disclaimer": ADVISOR_SAFETY_DISCLAIMER,
                "used_live_model": False,
            }

        # Evidence retrieval via RAG
        retrieved_docs = self.retrieve_evidence(clean_query)
        has_evidence = len(retrieved_docs) > 0
        formatted_context = self.retriever.format_context_for_prompt(retrieved_docs) if has_evidence else ""

        safety_advisory = risk_assessment.get("advisory")
        prompt = self.build_prompt(
            query=clean_query,
            age_group=age_group,
            user_context=user_context,
            evidence_context=formatted_context,
            has_evidence=has_evidence,
            safety_notice=safety_advisory,
        )

        result = {
            "success": False,
            "answer": None,
            "retrieved_docs": retrieved_docs,
            "risk_assessment": risk_assessment,
            "safety_advisory": safety_advisory,
            "status_message": "",
            "disclaimer": ADVISOR_SAFETY_DISCLAIMER,
            "used_live_model": False,
        }

        if self.client.is_configured():
            try:
                params = {
                    "max_new_tokens": 500,
                    "min_new_tokens": 20,
                    "temperature": 0.25,
                }
                raw_response = self.client.generate_response(prompt=prompt, params=params)
                answer_text = raw_response.strip()

                # If clinical or prescription notice exists, prepend a clear advisory box
                if safety_advisory:
                    answer_text = f"> **Clinical Advisory**: {safety_advisory}\n\n" + answer_text

                result["answer"] = answer_text
                result["success"] = True
                result["used_live_model"] = True
                result["status_message"] = "General health and nutrition advice generated with IBM Granite 4.0."
            except Exception as e:
                result["answer"] = None
                result["status_message"] = f"IBM watsonx inference notice: {str(e)}"
        else:
            result["answer"] = None
            result["status_message"] = (
                "IBM watsonx credentials are not configured. Cannot generate live AI wellness advice."
            )

        return result
