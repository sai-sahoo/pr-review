"""Tools the specialists may call.

@tool turns a function into something a model can request:
  name        -> the function name ("read_file")
  description -> the docstring, which is the model's only guide to when to use it
  args schema -> built from the type hints
"""

from langchain_core.tools import BaseTool, tool

from app.github_client import GitHubError, get_file_text
from app.schemas import PullRequest

MAX_LINES_PER_READ = 300  # protects the context window from "read the whole repo"


def make_read_file(pr: PullRequest) -> BaseTool:
    """Build a read_file tool bound to this PR's repo and commit.

    The model should only choose *what* to read. Which repo and which commit
    are facts we fix here, via the closure, so the model can't get them wrong.
    """

    @tool
    def read_file(path: str, start_line: int = 1, end_line: int = 150) -> str:
        """Read numbered lines from a file in the repository, as of this PR's commit.

        Use it when the diff alone is not enough to be sure about a problem:
        to see a whole function around a change, a helper it calls, or a
        definition imported from another file. At most 300 lines per call.
        """
        try:
            lines = get_file_text(pr.owner, pr.repo, pr.head_sha, path).splitlines()
        except GitHubError as e:
            # Return errors as text, don't raise: the model reads this
            # observation and can try another path instead of the run crashing.
            return f"error: {e}"

        start = max(start_line, 1)
        if start > len(lines):
            return f"error: {path} has only {len(lines)} lines"
        end = min(end_line, start + MAX_LINES_PER_READ - 1, len(lines))
        body = "\n".join(f"{n:>4} {lines[n - 1]}" for n in range(start, end + 1))
        return f"{path} (lines {start}-{end} of {len(lines)})\n{body}"

    return read_file
