"""Step 2: get typed Finding objects back instead of free text.

Run:  cd backend && uv run python structured_review.py
"""

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import get_model
from app.schemas import Review

# A hardcoded diff with a few planted bugs. Step 3 replaces this with a real PR.
DIFF = '''\
diff --git a/app/users.py b/app/users.py
--- a/app/users.py
+++ b/app/users.py
@@ -1,6 +1,18 @@
 import sqlite3
+import pickle
+
+API_KEY = "sk-live-9f8a7b6c5d4e3f2a1b0c"


 def get_user(conn, user_id):
-    return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
+    return conn.execute(f"SELECT * FROM users WHERE id = {user_id}").fetchone()
+
+
+def load_session(raw_bytes):
+    return pickle.loads(raw_bytes)
+
+
+def average_age(users):
+    return sum(u["age"] for u in users) / len(users)
'''

model = get_model("smart")

# Wraps the model: same .invoke(), but it now returns a Review instance.
# Under the hood the Review JSON schema is sent to the API, and the reply
# is parsed and validated by Pydantic.
reviewer = model.with_structured_output(Review)

messages = [
    SystemMessage(
        "You are a senior Python code reviewer. Report only real problems "
        "introduced by the added (+) lines. Do not invent issues."
    ),
    HumanMessage(f"Review this diff:\n\n{DIFF}"),
]

review = reviewer.invoke(messages)

print("type:", type(review).__name__, "| findings:", len(review.findings))
for f in review.findings:
    print(f"\n[{f.severity.upper()}] {f.category} - {f.file}:{f.line}")
    print(f"  {f.title}")
    print(f"  why: {f.explanation}")
    print(f"  fix: {f.suggestion}")

# Because these are real objects, ordinary Python works on them:
critical = [f for f in review.findings if f.severity == "critical"]
print(f"\ncritical count: {len(critical)}")

# And they serialize cleanly, ready for an API response or a database row.
print("\n=== JSON of the first finding ===")
if review.findings:
    print(review.findings[0].model_dump_json(indent=2))
