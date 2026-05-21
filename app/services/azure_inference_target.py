
import asyncio
import os
from typing import List

from azure.identity import ClientSecretCredential
from azure.core.pipeline.transport import RequestsTransport

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage

from pyrit.prompt_target import PromptTarget
from pyrit.models import Message, MessagePiece

from app.config import settings


class AzureInferenceTarget(PromptTarget):
    """
    PromptTarget that sends prompts to Azure AI Inference Chat Completions,
    using AAD token auth and enforcing a CA bundle for both token acquisition
    and the client HTTP pipeline.

    This is intended as the *objective target* (system under test). It does not
    implement PromptChatTarget (no set_system_prompt), which is fine for many
    multi-turn strategies where only the adversarial model needs that capability.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        model_name: str,
        temperature: float = 0.3,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        ca_bundle_path: str | None = None,
        disable_instance_discovery: bool | None = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

        # ---- CA bundle selection ----
        self._ca_bundle = (
            ca_bundle_path
            or os.getenv("CA_BUNDLE_PATH")
            or "/etc/ssl/cert.pem"
        )

        # ---- AAD token acquisition transport ----

        disable_flag = (
            disable_instance_discovery
            if disable_instance_discovery is not None
            else os.getenv("AZURE_DISABLE_INSTANCE_DISCOVERY", "false").lower() in ("1", "true", "yes")
        )

        aad_transport = RequestsTransport(connection_verify=self._ca_bundle)

        credential = ClientSecretCredential(
            tenant_id=settings.azure_tenant_id,
            client_id=settings.azure_client_id,
            client_secret=settings.azure_client_secret,
            disable_instance_discovery=disable_flag,  # pass directly
            transport=aad_transport,
        )

        # ---- Azure AI Inference client with pinned CA bundle ----
        svc_transport = RequestsTransport(connection_verify=self._ca_bundle)

        self.client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=credential,
            credential_scopes=["https://cognitiveservices.azure.com/.default"],
            transport=svc_transport,
        )

    # Required by PromptTarget
    def _validate_request(self, message: Message) -> None:
        if not isinstance(message, Message):
            raise TypeError("Expected pyrit.models.Message")
        try:
            first_val = message.get_value()
        except Exception:
            first_val = None
        if first_val is None or (isinstance(first_val, str) and first_val.strip() == ""):
            raise ValueError("Empty message value is not allowed")

    # Required by PromptTarget
    async def send_prompt_async(self, *, message: Message) -> Message:
        self._validate_request(message)

        # Build Azure Inference chat messages (SystemMessage/UserMessage)
        msgs: List[object] = []
        for piece in message.message_pieces:
            role = piece.role
            val = piece.converted_value if piece.converted_value is not None else piece.original_value
            if role == "system":
                msgs.append(SystemMessage(content=val))
            else:
                msgs.append(UserMessage(content=val))

        # Call the (sync) client on a worker thread
        result = await asyncio.to_thread(
            self.client.complete,
            messages=msgs,
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            model=self.model_name,
        )

        assistant_reply = result.choices[0].message.content
        
        assistant_piece = MessagePiece(
            role="assistant",
            original_value=assistant_reply,
            conversation_id=message.conversation_id,
        )
        return [Message([assistant_piece])]


