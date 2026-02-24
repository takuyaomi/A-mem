"""Test utilities for the memory system."""
import json
from typing import List, Optional
from agentic_memory.llm_controller import BaseLLMController


class MockLLMController(BaseLLMController):
    """Mock LLM controller for testing.

    Supports setting separate responses for:
    - analyze_content (Ps1)
    - link generation (Ps2)
    - evolution (Ps3)

    By default, returns a generic empty JSON response.
    """

    def __init__(self):
        self.mock_response = "{}"
        # Ps1: content analysis
        self.analyze_response: Optional[str] = None
        # Ps2: link generation
        self.link_response: Optional[str] = None
        # Ps3: evolution per neighbor (can be a list for sequential calls)
        self.evolution_responses: list[str] = []
        self._evolution_call_idx = 0
        # Track all calls for assertion
        self.call_history: list[dict] = []

    def get_completion(self, prompt: str, response_format: dict = None, temperature: float = 0.7) -> str:
        """Route to the appropriate mock response based on prompt content."""
        self.call_history.append({
            "prompt": prompt,
            "response_format": response_format,
        })

        # Detect prompt type by content
        if "memory linking agent" in prompt:
            if self.link_response is not None:
                return self.link_response
        elif "memory evolution agent" in prompt:
            if self.evolution_responses:
                idx = min(self._evolution_call_idx, len(self.evolution_responses) - 1)
                self._evolution_call_idx += 1
                return self.evolution_responses[idx]
        elif "structured analysis" in prompt:
            if self.analyze_response is not None:
                return self.analyze_response

        return self.mock_response

    def reset_call_tracking(self):
        self.call_history.clear()
        self._evolution_call_idx = 0

    def get_embedding(self, text: str) -> List[float]:
        """Mock embedding that returns a zero vector"""
        return [0.0] * 384
