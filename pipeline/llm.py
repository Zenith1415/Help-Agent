"""
pipeline/llm.py
Unified LLM client interface for Google Gemini API.
Handles API initialization, prompting, JSON mode, retries, and errors.
"""

import os
import time
import json
import warnings
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# Filter out verbose Google SDK AFC hints
warnings.filterwarnings("ignore", message=".*automatic function calling.*")

DEFAULT_MODEL = "gemini-flash-lite-latest"
FALLBACK_MODEL = "gemini-3.5-flash-lite"


class GeminiClient:
    """Wrapper around Gemini API with graceful error handling and JSON parsing."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = DEFAULT_MODEL):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self._client = None
        self._genai = None

        if not self.api_key:
            print("[!] Warning: GEMINI_API_KEY is not set in environment or .env file.")

    def _init_client(self):
        """Lazy initialization of Google GenAI SDK."""
        if self._client is not None or self._genai is not None:
            return

        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        except (ImportError, Exception):
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._genai = genai.GenerativeModel(self.model_name)

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.1,
        max_retries: int = 3,
        json_output: bool = False
    ) -> str:
        """Generate text completion with retry mechanism."""
        self._init_client()

        for attempt in range(max_retries):
            try:
                if self._client is not None:
                    from google.genai import types
                    config_args = {"temperature": temperature}
                    if system_instruction:
                        config_args["system_instruction"] = system_instruction
                    if json_output:
                        config_args["response_mime_type"] = "application/json"

                    config = types.GenerateContentConfig(**config_args)
                    response = self._client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=config
                    )
                    return response.text.strip()

                elif self._genai is not None:
                    import google.generativeai as genai_module
                    generation_config = {"temperature": temperature}
                    if json_output:
                        generation_config["response_mime_type"] = "application/json"
                    
                    model = genai_module.GenerativeModel(
                        model_name=self.model_name,
                        system_instruction=system_instruction,
                        generation_config=generation_config
                    )
                    response = model.generate_content(prompt)
                    return response.text.strip()
                else:
                    raise RuntimeError("No Gemini SDK could be initialized.")

            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "QUOTA" in err_msg.upper() or "RESOURCE_EXHAUSTED" in err_msg:
                    # Switch to fallback model
                    if self.model_name != FALLBACK_MODEL:
                        self.model_name = FALLBACK_MODEL
                        continue
                if attempt < max_retries - 1:
                    sleep_time = (attempt + 1) * 1.5
                    time.sleep(sleep_time)
                else:
                    raise RuntimeError(f"Gemini API request failed after {max_retries} attempts: {e}")

    def generate_json(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """Generate and parse JSON output safely."""
        raw_text = self.generate(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            json_output=True
        )
        
        # Clean markdown codeblocks if present
        clean_text = raw_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        clean_text = clean_text.strip()

        try:
            return json.loads(clean_text)
        except json.JSONDecodeError as err:
            raise ValueError(f"Failed to parse model output as JSON: {raw_text}") from err


_default_client = None

def get_llm_client() -> GeminiClient:
    global _default_client
    if _default_client is None:
        _default_client = GeminiClient()
    return _default_client
