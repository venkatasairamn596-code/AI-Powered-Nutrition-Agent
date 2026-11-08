"""
Agents Package

Specialized Agent Implementations for Nutrition Agent:
1. Nutrition Knowledge Assistant (Feature 1)
2. Personalised Diet Planner (Feature 2)
3. Food Intake Analyzer (Feature 3)
4. General Health & Nutrition Advisor (Feature 4)
"""

from agents.nutrition_assistant import NutritionKnowledgeAssistant, SAFETY_DISCLAIMER
from agents.diet_planner import PersonalisedDietPlanner, PLANNER_SAFETY_DISCLAIMER
from agents.food_intake_analyzer import FoodIntakeAnalyzer, ANALYZER_SAFETY_DISCLAIMER
from agents.general_health_advisor import GeneralHealthAdvisor, ADVISOR_SAFETY_DISCLAIMER

__all__ = [
    "NutritionKnowledgeAssistant",
    "SAFETY_DISCLAIMER",
    "PersonalisedDietPlanner",
    "PLANNER_SAFETY_DISCLAIMER",
    "FoodIntakeAnalyzer",
    "ANALYZER_SAFETY_DISCLAIMER",
    "GeneralHealthAdvisor",
    "ADVISOR_SAFETY_DISCLAIMER",
]
