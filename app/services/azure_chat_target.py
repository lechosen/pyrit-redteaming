
import asyncio
import json
import os
import uuid
from typing import List, Optional

import httpx
from azure.identity import ClientSecretCredential
from azure.core.pipeline.transport import RequestsTransport

from pyrit.prompt_target.common.prompt_chat_target import PromptChatTarget
from pyrit.models import Message, MessagePiece

from app.config import settings


class AzureChatTarget(PromptChatTarget):
    """
    PromptChatTarget for Azure OpenAI Chat Completions with AAD token auth
    and explicit CA bundle enforcement for both token acquisition and HTTPS calls.

    This class satisfies multi-turn strategies (RedTeamingAttack, CrescendoAttack),
    because it implements PromptChatTarget methods like set_system_prompt(...).

    Args:
        endpoint: Full Azure OpenAI Chat Completions URL, e.g.:
                  https://<resource>.cognitiveservices.azure.com/openai/deployments/<deployment>/chat/completions?api-version=2024-02-15-preview
        model_name: Deployment/model name
        temperature: Sampling temperature
        top_p: nucleus sampling
        max_tokens: Optional cap on tokens
        ca_bundle_path: Optional explicit CA bundle path; defaults to env CA_BUNDLE_PATH or /etc/ssl/cert.pem
        disable_instance_discovery: Optional override to skip instance discovery in AAD auth
    """

    def __init__(
        self,
        *,
        endpoint: str,
        model_name: str,
        temperature: float = 0.8,
        top_p: float = 1.0,
        max_tokens: Optional[int] = None,
        ca_bundle_path: Optional[str] = None,
        disable_instance_discovery: Optional[bool] = None,
    ):
        super().__init__(endpoint=endpoint, model_name=model_name)
        self.endpoint = endpoint
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        # ---- CA bundle selection (system trust or corp bundle) ----
        self._ca_bundle = (
            ca_bundle_path
            or os.getenv("CA_BUNDLE_PATH")
            or "/etc/ssl/cert.pem"  # macOS system trust (also valid on many Linux distros if populated)
        )

       
        # ---- AAD token acquisition transport with pinned CA bundle ----
        disable_flag = (
            disable_instance_discovery
            if disable_instance_discovery is not None
            else os.getenv("AZURE_DISABLE_INSTANCE_DISCOVERY", "false").lower() in ("1", "true", "yes")
        )

        aad_transport = RequestsTransport(connection_verify=self._ca_bundle)

        self.credential = ClientSecretCredential(
            tenant_id=settings.azure_tenant_id,
            client_id=settings.azure_client_id,
            client_secret=settings.azure_client_secret,
            disable_instance_discovery=disable_flag,  # pass directly to the credential
            transport=aad_transport,
        )

        # Store per-conversation system prompts
        self._system_prompts: dict[str, str] = {}

    # ----------------------------------------------------------------------
    # PromptTarget abstract: validate incoming request
    # ----------------------------------------------------------------------
    def _validate_request(self, message: Message) -> None:
        if not isinstance(message, Message):
            raise TypeError("Expected a pyrit.models.Message")
        try:
            # Ensure at least first piece has a value
            first_val = message.get_value()
        except Exception:
            first_val = None
        if first_val is None or (isinstance(first_val, str) and first_val.strip() == ""):
            raise ValueError("Empty message value is not allowed")

    # ----------------------------------------------------------------------
    # PromptChatTarget: set system prompt (PyRIT uses this in multi-turn)
    # ----------------------------------------------------------------------
    def set_system_prompt(
        self,
        *,
        system_prompt: str,
        conversation_id: str,
        attack_identifier: Optional[dict] = None,
        labels: Optional[dict] = None
    ) -> None:
        self._system_prompts[conversation_id] = system_prompt

    # PromptChatTarget: declare JSON response mode support
    def is_json_response_supported(self) -> bool:
        return True

    # ----------------------------------------------------------------------
    # Main: send a single chat turn to Azure OpenAI Chat Completions endpoint
    # ----------------------------------------------------------------------
    async def send_prompt_async(
        self,
        *,
        message: Message,
        conversation_id: Optional[str] = None,
        custom_json_response_format: Optional[dict] = None
    ) -> Message:
        self._validate_request(message)

        # Ensure a conversation ID exists
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())

        # Build the full chat history as OpenAI-style messages
        azure_msgs: List[dict] = []

        # Prepend system prompt if we have one
        sys_prompt = self._system_prompts.get(conversation_id)
        if sys_prompt:
            azure_msgs.append({"role": "system", "content": sys_prompt})

        # Convert Message -> list of chat dicts
        for piece in message.message_pieces:
            role = piece.role  # 'system' | 'user' | 'assistant' | ...
            value = piece.converted_value if piece.converted_value is not None else piece.original_value

            if role == "system":
                r = "system"
            elif role == "assistant":
                r = "assistant"
            else:
                r = "user"

            azure_msgs.append({"role": r, "content": value})

        # Acquire AAD bearer token for Cognitive Services
        scope = "https://cognitiveservices.azure.com/.default"
        token = await asyncio.to_thread(self.credential.get_token, scope)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token.token}",
        }

        body = {
            "model": self.model_name,
            "messages": azure_msgs,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.max_tokens:
            body["max_tokens"] = self.max_tokens


        # If the scorer requested a structured/JSON format, enforce JSON mode
        if custom_json_response_format is not None:
            # Enforce Azure JSON mode (widely supported compared to json_schema)
            body["response_format"] = {"type": "json_object"}

            # Strong, explicit, JSON-only guardrail for Self-Ask scorers
            azure_msgs.insert(0, {
                "role": "system",
                "content": (
                    "You are a JSON generator. Return ONLY a single JSON object, no prose, no markdown.\n"
                    "Required keys:\n"
                    '  - "score_value": string, must be "true" or "false" (lowercase)\n'
                    '  - "rationale": string, brief explanation\n'
                    "Example: {\"score_value\":\"true\",\"rationale\":\"...\"}\n"
                    "Do not include any other fields. Do not include any preamble or commentary."
                ),
            })

            # Reduce drift during scoring
            body["temperature"] = 0


        # Proceed with the enforced messages/body
        async with httpx.AsyncClient(timeout=60.0, verify=self._ca_bundle) as client:
            resp = await client.post(self.endpoint, headers=headers, json=body)

        # Use the same CA bundle for the HTTPS request to Azure OpenAI
        async with httpx.AsyncClient(timeout=60.0, verify=self._ca_bundle) as client:
            resp = await client.post(self.endpoint, headers=headers, json=body)

        if resp.status_code >= 400:
            raise RuntimeError(
                f"Azure Chat Completions error {resp.status_code}: {resp.text}"
            )

        data = resp.json()
        assistant_text = data["choices"][0]["message"]["content"]
        
        assistant_piece = MessagePiece(
            role="assistant",
            original_value=assistant_text,
            conversation_id=conversation_id,
        )
        return [Message([assistant_piece])]


