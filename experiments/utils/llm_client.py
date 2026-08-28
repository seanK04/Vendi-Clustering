import os
from openai import OpenAI
import time

class VLLMClient:
    """
    A client wrapper for vLLM running in OpenAI-compatible mode.
    Designed for use with LLMReducer in Protocol 2.
    """
    def __init__(self, model_name: str, base_url: str = "http://localhost:8000/v1"):
        # vLLM doesn't require a real API key by default, but the library needs a string
        self.client = OpenAI(base_url=base_url, api_key="vllm-on-oscar")
        self.model_name = model_name
    
    def generate(self, system_prompt: str, main_prompt: str) -> str:
        """
        Sends a prompt to the vLLM server and returns the text response.
        Matches the interface expected by LLMReducer.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": main_prompt}
                ],
                temperature=0.0,
                max_tokens=50,
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error connecting to vLLM server: {e}")
            return ""

    def wait_for_server(self, timeout: int = 600):
        """
        Heuristic to wait until the vLLM server is finished loading the model.
        Useful at the start of Protocol 2.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                self.client.models.list()
                print(f"✓ vLLM Server is ready ({self.model_name})")
                return True
            except:
                print("Waiting for vLLM server to load Gemma...")
                time.sleep(10)
        raise TimeoutError("vLLM server failed to start in time.")


class OpenAIClient:
    """
    A client wrapper for OpenAI's hosted Chat Completions API.
    Mirrors the interface of VLLMClient so it can be used as a drop-in
    replacement for LLMReducer in Protocol 2.
    """
    def __init__(self, model_name: str, api_key: str = None, max_retries: int = 3):
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "No OpenAI API key provided. Pass api_key explicitly or set "
                "the OPENAI_API_KEY environment variable."
            )
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.max_retries = max_retries

    def generate(self, system_prompt: str, main_prompt: str) -> str:
        """
        Sends a prompt to the OpenAI Chat Completions API and returns the text response.
        Retries with exponential backoff on failure. Returns "" on terminal failure,
        matching VLLMClient's contract so LLMReducer.get_merge_pair halts gracefully.
        """
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": main_prompt}
                    ],
                    temperature=0.0,
                    max_tokens=50,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    backoff = 2 ** attempt
                    print(
                        f"OpenAI request failed (attempt {attempt + 1}/{self.max_retries}): "
                        f"{e}. Retrying in {backoff}s..."
                    )
                    time.sleep(backoff)

        print(f"OpenAI request failed after {self.max_retries} attempts. Last error: {last_error}")
        return ""

    def wait_for_server(self, timeout: int = 600) -> None:
        """
        No-op for OpenAI (the hosted endpoint is always available).
        Provided so the runner can call it unconditionally without provider-specific branching.
        """
        pass