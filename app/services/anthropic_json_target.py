# app/services/anthropic_json_target.py

import json
import re
import uuid
from typing import List, Optional

import httpx

from pyrit.prompt_target.common.prompt_chat_target import PromptChatTarget
from pyrit.models import Message, MessagePiece


class AnthropicJsonTarget(PromptChatTarget):
    """
    A hardened Chat Target for PyRIT Self-Ask scorers, using the Anthropic API.

    Enforces:
      - Strict JSON-only system guardrail
      - temperature=0 for deterministic scoring
      - Normalizes response to a single valid JSON object (no prose/markdown)

    This is the Anthropic equivalent of JsonEnforcedChatTarget (which is Azure-only).
    """

    def __init__(
        self,
        *,
        endpoint: str,
        model_name: str,
        api_key: str,
    ):
        super().__init__(endpoint=endpoint, model_name=model_name)
        self.endpoint = endpoint
        self.model_name = model_name
        self.api_key = api_key

        self._system_prompts: dict[str, str] = {}

    # ------------------------------------------------------------------
    # PyRIT interface
    # ------------------------------------------------------------------
    def is_json_response_supported(self) -> bool:
        return True

    def set_system_prompt(
        self,
        *,
        system_prompt: str,
        conversation_id: str,
        attack_identifier: Optional[dict] = None,
        labels: Optional[dict] = None,
    ) -> None:
        self._system_prompts[conversation_id] = system_prompt

    def _validate_request(self, message: Message) -> None:
        if not isinstance(message, Message):
            raise TypeError("Expected a pyrit.models.Message")
        try:
            first_val = message.get_value()
        except Exception:
            first_val = None
        if first_val is None or (isinstance(first_val, str) and first_val.strip() == ""):
            raise ValueError("Empty message value is not allowed")

    # ------------------------------------------------------------------
    # JSON normalization helpers (copied from json_enforced_chat_target.py)
    # ------------------------------------------------------------------
    def _is_valid_json(self, s: str) -> bool:
        try:
            json.loads(s)
            return True
        except Exception:
            return False

    def _extract_balanced_json_object(self, text: str) -> Optional[str]:
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        i = start
        while i < len(text):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1].strip()
                    if self._is_valid_json(candidate):
                        return candidate
                    remainder = text[i + 1:]
                    return self._extract_balanced_json_object(remainder)
            i += 1
        return None

    def _to_strict_json(self, text: str) -> str:
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.IGNORECASE | re.DOTALL)
        if fence:
            candidate = fence.group(1).strip()
            if self._is_valid_json(candidate):
                return candidate

        candidate = self._extract_balanced_json_object(text)
        if candidate:
            return candidate

        rationale = text.strip()
        if len(rationale) > 500:
            rationale = rationale[:500] + "..."
        return json.dumps({"score_value": "false", "rationale": rationale})

    # ------------------------------------------------------------------
    # Main: send prompt with JSON enforcement
    # ------------------------------------------------------------------
    async def send_prompt_async(
        self,
        *,
        message: Message,
        conversation_id: Optional[str] = None,
        custom_json_response_format: Optional[dict] = None,
    ) -> List[Message]:
        self._validate_request(message)

        if conversation_id is None:
            conversation_id = str(uuid.uuid4())

        # JSON-only guardrail system prompt
        json_guardrail = (
            "You are a JSON generator. Return ONLY a single JSON object, no prose, no markdown.\n"
            "Required keys:\n"
            '  - "score_value": string, must be "true" or "false" (lowercase)\n'
            '  - "rationale": string, brief explanation\n'
            'Example: {"score_value":"true","rationale":"..."}\n'
            "Do not include any other fields. Do not include any preamble or commentary."
        )

        # Convert message pieces to Anthropic format
        anthropic_msgs: list[dict] = []
        for piece in message.message_pieces:
            value = piece.converted_value if piece.converted_value is not None else piece.original_value
            role = piece.role
            if role == "assistant":
                anthropic_msgs.append({"role": "assistant", "content": value})
            elif role != "system":
                anthropic_msgs.append({"role": "user", "content": value})

        if not anthropic_msgs:
            anthropic_msgs.append({"role": "user", "content": "(empty)"})

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        body = {
            "model": self.model_name,
            "system": json_guardrail,
            "messages": anthropic_msgs,
            "temperature": 0,
            "max_tokens": 256,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(self.endpoint, headers=headers, json=body)

        if resp.status_code >= 400:
            raise RuntimeError(
                f"Anthropic Messages API error {resp.status_code}: {resp.text}"
            )

        data = resp.json()
        raw = data["content"][0]["text"]
        json_only = self._to_strict_json(raw)

        assistant_piece = MessagePiece(
            role="assistant",
            original_value=json_only,
            conversation_id=conversation_id,
        )
        return [Message([assistant_piece])]
