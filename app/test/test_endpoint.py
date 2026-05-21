
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.identity import ClientSecretCredential

endpoint = ""
model_name = "gpt-4.1"

credential = ClientSecretCredential(
    tenant_id="",
    client_id="",
    client_secret=""
)

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=credential,
    credential_scopes=["https://cognitiveservices.azure.com/.default"]
)

response = client.complete(
    messages=[
        SystemMessage(content="You are a concise mathematician."),
        UserMessage(content="What is sin(1.3932143222*pi)?"),
    ],
    temperature=0.1,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    model=model_name
)

print(response.choices[0].message.content)




 