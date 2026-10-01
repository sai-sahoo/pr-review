"""Step 1: talk to an LLM once and look at what comes back.

Run:  cd backend && uv run python hello_llm.py
"""

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import get_model

model = get_model("fast")

# A chat is a list of messages. The system message sets behaviour;
# the human message is the actual request.
messages = [
    SystemMessage("You are a senior code reviewer. Be concise."),
    HumanMessage(
        "Review this Python line and name the single biggest problem:\n\n"
        'rows = cursor.execute(f"SELECT * FROM users WHERE id = {request.args[\'id\']}")'
    ),
]

response = model.invoke(messages)  # one blocking request/response, like requests.post()

print("=== Reply ===")
print(response.content)

print("\n=== Metadata ===")
print("type:       ", type(response).__name__)
print("model:      ", response.response_metadata.get("model_name"))
print("token usage:", response.usage_metadata)
