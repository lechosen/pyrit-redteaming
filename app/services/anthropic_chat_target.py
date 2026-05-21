# app/services/anthropic_chat_target.py

import uuid
from typing import List, Optional

import httpx

from pyrit.prompt_target.common.prompt_chat_target import PromptChatTarget
from pyrit.models import Message, MessagePiece


class AnthropicChatTarget(PromptChatTarget):
    """
    PromptChatTarget for the Anthropic Messages API.

    Used as both the objective target (victim) and adversarial target (attacker)
    in multi-turn red-teaming attacks.

    Auth: x-api-key header (no OAuth/Entra ID).
    API:  POST https://api.anthropic.com/v1/messages
    """

    def __init__(
        self,
        *,
        endpoint: str,
        model_name: str,
        api_key: str,
        temperature: float = 0.8,
        max_tokens: int = 4096,
    ):
        super().__init__(endpoint=endpoint, model_name=model_name)
        self.endpoint = endpoint
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Per-conversation system prompts (set by PyRIT multi-turn strategies)
        self._system_prompts: dict[str, str] = {}

    # ------------------------------------------------------------------
    # PromptTarget: validate incoming request
    # ------------------------------------------------------------------
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
    # PromptChatTarget: set system prompt (used by multi-turn strategies)
    # ------------------------------------------------------------------
    def set_system_prompt(
        self,
        *,
        system_prompt: str,
        conversation_id: str,
        attack_identifier: Optional[dict] = None,
        labels: Optional[dict] = None,
    ) -> None:
        self._system_prompts[conversation_id] = system_prompt

    def is_json_response_supported(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Main: send a single chat turn to the Anthropic Messages API
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

        # --- Build the system prompt (top-level field in Anthropic API) ---
        system_parts: list[str] = []
        sys_prompt = self._system_prompts.get(conversation_id)
        if sys_prompt:
            system_parts.append(sys_prompt)

        # --- Convert message pieces to Anthropic messages format ---
        anthropic_msgs: list[dict] = []
        for piece in message.message_pieces:
            role = piece.role
            value = piece.converted_value if piece.converted_value is not None else piece.original_value

            if role == "system":
                # Anthropic does not allow "system" in the messages array
                system_parts.append(value)
            elif role == "assistant":
                anthropic_msgs.append({"role": "assistant", "content": value})
            else:
                anthropic_msgs.append({"role": "user", "content": value})

        # Anthropic requires at least one user message
        if not anthropic_msgs:
            anthropic_msgs.append({"role": "user", "content": "(empty)"})

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        body: dict = {
            "model": self.model_name,
            "messages": anthropic_msgs,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        # Merge all system parts into the top-level "system" field
        if system_parts:
            body["system"] = "\n\n".join(system_parts)

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(self.endpoint, headers=headers, json=body)

        if resp.status_code >= 400:
            raise RuntimeError(
                f"Anthropic Messages API error {resp.status_code}: {resp.text}"
            )

        data = resp.json()
        assistant_text = data["content"][0]["text"]

        assistant_piece = MessagePiece(
            role="assistant",
            original_value=assistant_text,
            conversation_id=conversation_id,
        )
        return [Message([assistant_piece])]
