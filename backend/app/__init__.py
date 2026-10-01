"""The pr-review backend package.

Python runs this file the first time anything under `app` is imported, so
loading .env here means every module (llm, github_client, ...) sees it.
"""

from pathlib import Path

from dotenv import load_dotenv

# Load the project-root .env (two levels above this file) into os.environ.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")
