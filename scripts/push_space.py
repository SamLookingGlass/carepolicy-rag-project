"""Upload this repo to a free Hugging Face ZeroGPU Gradio Space.

The live index goes up with the code. .env does not. Secrets are copied from
the local environment into Space secrets, not into the uploaded files.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

ROOT = Path(__file__).resolve().parent.parent

# GitHub stays without the index. The Space needs it, so this script uploads
# data/processed and data/index and skips secrets and the raw scrape.
IGNORE = [
    ".env",
    ".git/**",
    ".venv/**",
    "venv/**",
    ".cursor/**",
    "archive/**",
    "**/__pycache__/**",
    "**/*.py[cod]",
    "*.egg-info/**",
    ".pytest_cache/**",
    ".ruff_cache/**",
    "data/raw/**",
    "README copy.md",
    "README copy 2.md",
    "**/*.log",
    ".DS_Store",
    "**/.lock",
]


def _load_dotenv(path: Path) -> None:
    """Fill os.environ from .env without printing values. Existing vars win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"Missing {name}. Put it in .env or the environment. It is not uploaded.")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Push CarePolicy to a Hugging Face ZeroGPU Space")
    parser.add_argument("--repo", required=True, help="username/carepolicy-rag")
    parser.add_argument(
        "--token-env",
        default="CAREPOLICY_WRITE_HF_TOKEN",
        help="Env var holding a Hugging Face write token (default: CAREPOLICY_WRITE_HF_TOKEN)",
    )
    args = parser.parse_args()

    _load_dotenv(ROOT / ".env")
    token = _require(args.token_env)
    api = HfApi(token=token)

    try:
        api.create_repo(
            repo_id=args.repo,
            repo_type="space",
            space_sdk="gradio",
            space_hardware="zero-a10g",
            exist_ok=True,
            private=False,
        )
    except HfHubHTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status in {402, 403}:
            raise SystemExit(
                "Hugging Face refused to create the ZeroGPU Space "
                f"({status}). A free personal account can host one only if the "
                "email is verified and the account is older than 30 days. "
                "Nothing was uploaded, and .env was not sent."
            ) from exc
        raise
    api.request_space_hardware(args.repo, hardware="zero-a10g")
    api.add_space_secret(args.repo, "OPENAI_API_KEY", _require("OPENAI_API_KEY"))
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if hf_token:
        api.add_space_secret(args.repo, "HF_TOKEN", hf_token)
    api.add_space_variable(args.repo, "QDRANT_MODE", "local")
    api.add_space_variable(args.repo, "GRADIO_SSR_MODE", "false")

    commit = api.upload_folder(
        repo_id=args.repo,
        repo_type="space",
        folder_path=str(ROOT),
        ignore_patterns=IGNORE,
        commit_message="Deploy CarePolicy demo with the live index",
    )
    print(f"Uploaded {args.repo}")
    print(commit.commit_url if hasattr(commit, "commit_url") else commit)


if __name__ == "__main__":
    main()
