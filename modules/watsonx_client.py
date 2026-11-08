"""
IBM watsonx.ai Granite Model Client

This module handles connection and inference requests to IBM Granite foundation models
hosted on IBM watsonx.ai.

Credentials and configuration are loaded securely from environment variables (.env file).
No credentials or API keys should ever be hard-coded here.
"""

import os
from typing import Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Locate and safely load the project root .env file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()


class WatsonxGraniteClient:
    """
    Client wrapper for IBM watsonx.ai Granite foundation models.
    Provides configuration checks, connection testing, and model initialization.
    """

    DEFAULT_URL = "https://us-south.ml.cloud.ibm.com"
    DEFAULT_MODEL_ID = "ibm/granite-4-h-small"

    def __init__(
        self,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        url: Optional[str] = None,
        model_id: Optional[str] = None,
        instance_id: Optional[str] = None,
    ):
        self.api_key = (api_key if api_key is not None else os.getenv("WATSONX_APIKEY", "")).strip()
        self.project_id = (project_id if project_id is not None else os.getenv("WATSONX_PROJECT_ID", "")).strip()
        self.url = (url if url is not None else os.getenv("WATSONX_URL", self.DEFAULT_URL)).strip()
        self.model_id = (model_id if model_id is not None else os.getenv("GRANITE_MODEL_ID", self.DEFAULT_MODEL_ID)).strip()
        self.instance_id = (instance_id if instance_id is not None else os.getenv("WATSONX_INSTANCE_ID", "")).strip()
        self._model = None
        self._api_client = None

    def is_configured(self) -> bool:
        """
        Check whether the required IBM watsonx.ai credentials are set.
        Returns True only if both API key and Project ID are non-empty and
        not placeholder template values.
        """
        placeholders = {
            "your_ibm_cloud_api_key_here",
            "your_watsonx_project_id_here",
            "your_actual_api_key_here",
            "your_actual_project_id_here",
            "your_watsonx_instance_id_here",
            "",
        }
        has_real_key = bool(self.api_key and self.api_key not in placeholders)
        has_real_project = bool(self.project_id and self.project_id not in placeholders)
        return has_real_key and has_real_project

    def get_configuration_status(self) -> Dict[str, Any]:
        """
        Returns safe diagnostic information about the configuration.
        Masks all sensitive data to ensure credentials are never exposed in logs or UI.
        """
        masked_project = ""
        if self.project_id:
            if len(self.project_id) > 8:
                masked_project = f"{self.project_id[:4]}...{self.project_id[-4:]}"
            else:
                masked_project = "***"

        masked_instance = ""
        if self.instance_id:
            if len(self.instance_id) > 8:
                masked_instance = f"{self.instance_id[:4]}...{self.instance_id[-4:]}"
            else:
                masked_instance = "***"

        return {
            "has_api_key": bool(self.api_key and "your_" not in self.api_key),
            "has_project_id": bool(self.project_id and "your_" not in self.project_id),
            "has_instance_id": bool(self.instance_id and "your_" not in self.instance_id),
            "project_id_masked": masked_project,
            "instance_id_masked": masked_instance,
            "service_url": self.url,
            "target_model": self.model_id,
            "is_ready": self.is_configured(),
        }

    def test_connection(self) -> Dict[str, Any]:
        """
        Validates IBM watsonx.ai connectivity and authentication when credentials are provided.
        Does NOT make paid inference calls or generate fake responses.
        
        Returns:
            Dict containing:
                - success (bool): True if authentication succeeded
                - message (str): Human-friendly description of status
                - details (dict): Diagnostic info (no secrets)
        """
        if not self.is_configured():
            return {
                "success": False,
                "message": (
                    "IBM watsonx credentials are not configured. Please create a '.env' file "
                    "from '.env.example' and provide your WATSONX_APIKEY and WATSONX_PROJECT_ID."
                ),
                "details": self.get_configuration_status(),
            }

        try:
            from ibm_watsonx_ai import APIClient, Credentials

            creds = Credentials(
                url=self.url,
                api_key=self.api_key,
                instance_id=self.instance_id if self.instance_id else None,
            )
            # Instantiating APIClient authenticates against IBM Cloud IAM
            self._api_client = APIClient(credentials=creds, project_id=self.project_id)

            return {
                "success": True,
                "message": "Successfully authenticated with IBM watsonx.ai!",
                "details": self.get_configuration_status(),
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Connection check failed: {str(e)}",
                "details": self.get_configuration_status(),
            }

    def initialize_model(self):
        """
        Initializes the ModelInference instance from the ibm-watsonx-ai SDK.
        This is called lazily only when actual inference is requested.
        """
        if not self.is_configured():
            raise ValueError(
                "IBM watsonx credentials missing. Please set WATSONX_APIKEY and "
                "WATSONX_PROJECT_ID in your .env file."
            )

        try:
            from ibm_watsonx_ai.foundation_models import ModelInference
            from ibm_watsonx_ai import Credentials

            credentials = Credentials(
                url=self.url,
                api_key=self.api_key,
                instance_id=self.instance_id if self.instance_id else None,
            )

            self._model = ModelInference(
                model_id=self.model_id,
                credentials=credentials,
                project_id=self.project_id,
            )
            return self._model
        except ImportError as e:
            raise ImportError(
                "The 'ibm-watsonx-ai' package is required. "
                "Install it using 'pip install ibm-watsonx-ai'."
            ) from e

    def generate_response(self, prompt: str, params: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate response from IBM Granite foundation model.
        NOTE: No fake AI responses are used. If credentials or dependencies are missing,
        an explicit exception is raised.
        """
        if self._model is None:
            self.initialize_model()

        # Default bounded parameters to prevent unexpected token consumption
        default_params = {
            "decoding_method": "greedy",
            "max_new_tokens": 250,
            "min_new_tokens": 1,
            "temperature": 0.2,
        }
        if params:
            default_params.update(params)

        # Real inference call against IBM Granite
        result = self._model.generate_text(prompt=prompt, params=default_params)
        return result
