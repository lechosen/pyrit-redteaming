# app/services/target_factory.py

from app.services.azure_chat_target import AzureChatTarget
from app.services.azure_inference_target import AzureInferenceTarget
from app.services.json_enforced_chat_target import JsonEnforcedChatTarget
from app.services.anthropic_chat_target import AnthropicChatTarget
from app.services.anthropic_json_target import AnthropicJsonTarget
from app.models import ModelConfig


def _is_anthropic(cfg: ModelConfig) -> bool:
    return "anthropic.com" in cfg.endpoint


# Use the same chat API version everywhere for attacker (adversarial) calls
_CHAT_API_VERSION = "2025-01-01-preview"

def build_target(cfg: ModelConfig, *, adversarial: bool = False):
    """
    Build targets used by the attack orchestrator.

    - Anthropic endpoints  -> AnthropicChatTarget (both roles)
    - objective_target     -> AzureInferenceTarget (PromptTarget)
    - adversarial_target   -> AzureChatTarget     (PromptChatTarget; conversational attacker)
    """
    if _is_anthropic(cfg):
        return AnthropicChatTarget(
            endpoint=cfg.endpoint,
            model_name=cfg.model_name,
            api_key=cfg.api_key,
        )

    if adversarial:
        # Conversational attacker (NOT JSON-enforced).
        # cfg.endpoint should stop at /openai/deployments/<deployment>
        return AzureChatTarget(
            endpoint=f"{cfg.endpoint}/chat/completions?api-version={_CHAT_API_VERSION}",
            model_name=cfg.model_name,
        )

    # Default: objective target (Azure AI Inference SDK target)
    return AzureInferenceTarget(
        endpoint=cfg.endpoint,
        model_name=cfg.model_name,
    )


def build_evaluator_target(cfg: ModelConfig):
    """
    Build the JSON-enforced evaluator model for scorers (PromptChatTarget).

    - Anthropic endpoints -> AnthropicJsonTarget
    - Azure endpoints     -> JsonEnforcedChatTarget (hardcoded gpt-4o-mini)
    """
    if _is_anthropic(cfg):
        return AnthropicJsonTarget(
            endpoint=cfg.endpoint,
            model_name=cfg.model_name,
            api_key=cfg.api_key,
        )

    evaluator_endpoint = (
        "https://corpmkt-mrm-aifoundry-01.cognitiveservices.azure.com/"
        "openai/deployments/gpt-4o-mini/chat/completions"
        "?api-version=2025-01-01-preview"
    )

    return JsonEnforcedChatTarget(
        endpoint=evaluator_endpoint,
        model_name="gpt-4o-mini",
    )