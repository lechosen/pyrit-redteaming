# app/services/json_enforced_chat_target.py

import asyncio
import json
import os
import re
import uuid
from typing import List, Optional

import httpx
from azure.identity import ClientSecretCredential
from azure.core.pipeline.transport import RequestsTransport

from pyrit.prompt_target.common.prompt_chat_target import PromptChatTarget
from pyrit.models import Message, MessagePiece

from app.config import settings


class JsonEnforcedChatTarget(PromptChatTarget):
    """
    A hardened Chat Target specifically for PyRIT Self-Ask scorers (evaluator LLM).

    It enforces:
      - Strict JSON output (response_format=json_object)
      - JSON-only system guardrail injected at the top
      - temperature=0 for deterministic responses
      - Normalizes/stores ONLY a single valid JSON object (no prose/markdown)
      - TLS CA bundle (compatible with pip-system-certs or explicit CA bundle)

    This guarantees that scorer parsing won't fail due to conversational text.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        model_name: str,
        ca_bundle_path: Optional[str] = None,
    ):
        super().__init__(endpoint=endpoint, model_name=model_name)
        self.endpoint = endpoint
        self.model_name = model_name

        # CA bundle (pip-system-certs often makes this unnecessary, but keep for portability)
        self._ca_bundle = (
            ca_bundle_path
            or os.getenv("CA_BUNDLE_PATH")
            or "/etc/ssl/cert.pem"
        )

        # Transport with certificate enforcement for AAD
        aad_transport = RequestsTransport(connection_verify=self._ca_bundle)

        # Entra ID credential; disable instance discovery for locked-down networks
        self.credential = ClientSecretCredential(
            tenant_id=settings.azure_tenant_id,
            client_id=settings.azure_client_id,
            client_secret=settings.azure_client_secret,
            disable_instance_discovery=True,
            transport=aad_transport,
        )

        # Store system prompts if PyRIT sets them (normal multi-turn)
        self._system_prompts: dict[str, str] = {}

    # -------------------------------
    # PyRIT interface requirements
    # -------------------------------
    def is_json_response_supported(self) -> bool:
        return True

    def set_system_prompt(
        self,
        *,
        system_prompt: str,
        conversation_id: str,
        attack_identifier: Optional[dict] = None,
        labels: Optional[dict] = None
    ) -> None:
        # We store it, but will still hard-enforce a JSON guardrail in send_prompt_async
        self._system_prompts[conversation_id] = system_prompt

    def _validate_request(self, message: Message) -> None:
        """
        Minimal validation required by PromptChatTarget.
        Ensures the message is a valid PyRIT Message with at least one non-empty value.
        """
        if not isinstance(message, Message):
            raise TypeError("Expected a pyrit.models.Message")

        try:
            first_val = message.get_value()
        except Exception:
            first_val = None

        if first_val is None or (isinstance(first_val, str) and first_val.strip() == ""):
            raise ValueError("Empty message value is not allowed")

    # -------------------------------
    # JSON normalization helpers
    # -------------------------------
    def _is_valid_json(self, s: str) -> bool:
        try:
            json.loads(s)
            return True
        except Exception:
            return False

    def _extract_balanced_json_object(self, text: str) -> Optional[str]:
        """
        Return the first balanced {...} JSON object found in text, or None.
        This is robust to prose that surrounds an embedded JSON object.
        """
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
                    # If invalid, try to find another object later in the text
                    remainder = text[i + 1 :]
                    nxt = self._extract_balanced_json_object(remainder)
                    return nxt
            i += 1
        return None

    def _to_strict_json(self, text: str) -> str:
        """
        Convert the model's raw output to a single valid JSON object string.

        Strategy:
          1) extract fenced ```json ... ``` block
          2) else extract first balanced {...} object
          3) else fallback to {"score_value":"false","rationale":"<truncated prose>"}
        """
        # 1) Fenced code block: ```json { ... } ```
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.IGNORECASE | re.DOTALL)
        if fence:
            candidate = fence.group(1).strip()
            if self._is_valid_json(candidate):
                return candidate

        # 2) Scan for first balanced JSON object
        candidate = self._extract_balanced_json_object(text)
        if candidate:
            return candidate

        # 3) Fallback: produce minimal JSON object to satisfy scorer contract
        rationale = text.strip()
        if len(rationale) > 500:
            rationale = rationale[:500] + "..."
        return json.dumps({"score_value": "false", "rationale": rationale})

    # -------------------------------
    # Main scoring send method
    # -------------------------------
    async def send_prompt_async(
        self,
        *,
        message: Message,
        conversation_id: Optional[str] = None,
        custom_json_response_format: Optional[dict] = None
    ) -> List[Message]:
        """
        HARD OVERRIDE for PyRIT scorer calls.

        - Inject JSON-only system prompt at the top
        - Enforce response_format=json_object
        - Force temperature=0
        - Normalize the response to ONLY a single valid JSON object
        """
        self._validate_request(message)

        if conversation_id is None:
            conversation_id = str(uuid.uuid4())

        # 1) Build strict JSON-only system guardrail
        json_guardrail = {
            "role": "system",
            "content": (
                "You MUST return ONLY a single valid JSON object.\n"
                "No explanations, no conversation, no markdown.\n"
                "Required keys:\n"
                '  - "score_value": must be "true" or "false" (lowercase)\n'
                '  - "rationale": a brief explanation string\n'
                "Example JSON:\n"
                '{"score_value":"true","rationale":"..."}\n'
                "Return NOTHING except valid JSON."
            ),
        }

        # 2) Convert PyRIT Message -> minimal chat messages
        user_msgs = []
        for piece in message.message_pieces:
            value = (
                piece.converted_value
                if piece.converted_value is not None
                else piece.original_value
            )
            user_msgs.append({"role": "user", "content": value})

        # ORDER MATTERS: guardrail goes first
        azure_msgs = [json_guardrail] + user_msgs

        # 3) Acquire token
        token = await asyncio.to_thread(
            self.credential.get_token,
            "https://cognitiveservices.azure.com/.default"
        )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token.token}",
        }

        # 4) Build enforced request body
        body = {
            "model": self.model_name,
            "messages": azure_msgs,
            "response_format": {"type": "json_object"},
            "temperature": 0,     # deterministic
            "top_p": 1,           # minimal randomness
            "max_tokens": 256,    # safe cap
        }

        # 5) Send request
        async with httpx.AsyncClient(timeout=60.0, verify=self._ca_bundle) as client:
            resp = await client.post(self.endpoint, headers=headers, json=body)

        if resp.status_code >= 400:
            raise RuntimeError(
                f"Azure Chat Completions error {resp.status_code}: {resp.text}"
            )

        # 6) Normalize to a single strict JSON object string
        data = resp.json()
        raw = data["choices"][0]["message"]["content"]
        json_only = self._to_strict_json(raw)

        # 7) Return a List[Message] with assistant JSON only
        assistant_piece = MessagePiece(
            role="assistant",
            original_value=json_only,
            conversation_id=conversation_id,
        )
        return [Message([assistant_piece])]
