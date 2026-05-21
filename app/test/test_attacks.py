
from app.services.target_factory import build_target
from app.models import ModelConfig

cfg = ModelConfig(
    endpoint="https://corpmkt-mrm-aifoundry-01.cognitiveservices.azure.com/openai/deployments/gpt-4.1",
    model_name="gpt-4.1"
)

adv = build_target(cfg, adversarial=True)
obj = build_target(cfg, adversarial=False)

print("Adversarial target:", adv.__class__)
print("Objective target:", obj.__class__)
