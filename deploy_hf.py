"""
Deploy to Hugging Face Spaces
==============================
Pushes the entire project to a Hugging Face Space as a Docker app.

Prerequisites:
  1. pip install huggingface_hub
  2. Set HF_TOKEN environment variable or run `huggingface-cli login`
  3. Create a Space: https://huggingface.co/new-space
     - SDK: Docker
     - Name: tirumala-darshan-prediction

Usage:
  python deploy_hf.py --repo <username/space-name>

Example:
  python deploy_hf.py --repo madhav456789123/tirumala-darshan-prediction
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Deploy to Hugging Face Spaces")
    parser.add_argument(
        "--repo",
        type=str,
        default="madhav456789123/tirumala-darshan-prediction",
        help="HF Space repo ID (e.g., username/space-name)",
    )
    parser.add_argument("--token", type=str, default=None, help="HF API token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        print("ERROR: No HF_TOKEN found. Set HF_TOKEN env variable or pass --token")
        print("Get a token at: https://huggingface.co/settings/tokens")
        sys.exit(1)

    repo_id = args.repo
    print(f"Deploying to Hugging Face Space: {repo_id}")
    print("=" * 50)

    try:
        from huggingface_hub import HfApi, upload_folder
    except ImportError:
        print("Installing huggingface_hub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import HfApi, upload_folder

    api = HfApi(token=token)

    # Create space if it doesn't exist
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True,
        )
        print(f"Space '{repo_id}' ready.")
    except Exception as e:
        print(f"Note: {e}")

    # Upload the entire project
    print("Uploading files...")
    upload_folder(
        folder_path=".",
        repo_id=repo_id,
        repo_type="space",
        token=token,
        ignore_patterns=[
            ".env",
            ".git/*",
            ".venv_dl/*",
            "__pycache__/*",
            "*.pyc",
            "node_modules/*",
            "client/node_modules/*",
            "client/build/*",
            "eda_outputs/*",
            "*.log",
            "flask_out.log",
            "flask_err.log",
            "app_gradio.py",
            "test_*.py",
            "evaluate_pretrained_ts.py",
            "deploy_hf.py",
        ],
    )

    url = f"https://huggingface.co/spaces/{repo_id}"
    print()
    print("=" * 50)
    print(f"Deployed successfully!")
    print(f"Live at: {url}")
    print("=" * 50)
    print()
    print("IMPORTANT: Set secrets in Space Settings:")
    print(f"  {url}/settings")
    print("  - GROQ_API_KEY: Your Groq API key for LLM chatbot")
    print("  - HF_TOKEN_CHAT: HuggingFace token for fallback LLM")
    print()
    print("The Space will build the Docker image and start automatically.")
    print("First build takes ~5-10 minutes.")


if __name__ == "__main__":
    main()
