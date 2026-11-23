"""deploy_hf.py - create (if needed) and upload the Hugging Face Space.

One-time setup:
    pip install huggingface_hub
    hf auth login            # paste a token with WRITE access (huggingface.co/settings/tokens)
                             # (older versions of the tool: huggingface-cli login)
Deploy / update:
    python deploy_hf.py                     -> Space "<your-username>/scentmate-api"
    python deploy_hf.py myname/other-name   -> custom Space id
Large files (the 19 MB model) are uploaded via Git LFS automatically by huggingface_hub.
"""
import sys
from pathlib import Path

from huggingface_hub import HfApi

FOLDER = Path(__file__).parent / "hf_space"

def main():
    api = HfApi()
    user = api.whoami()["name"]
    repo_id = sys.argv[1] if len(sys.argv) > 1 else f"{user}/scentmate-api"

    required = ["app.py", "Dockerfile", "requirements.txt", "README.md",
                "model/recommender.onnx", "model/items.json", "model/config.json"]
    missing = [f for f in required if not (FOLDER / f).exists()]
    if missing:
        sys.exit(f"Missing {missing} — run train.py and build_model.py first.")

    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=FOLDER,
                      ignore_patterns=["data/*", "__pycache__/*", "*.pyc"],
                      commit_message="Deploy Scentmate API")

    owner, name = repo_id.split("/")
    url = f"https://{owner}-{name}".lower().replace("_", "-").replace(".", "-") + ".hf.space"
    print(f"Space page : https://huggingface.co/spaces/{repo_id}")
    print(f"API URL    : {url}      <- put this in site/index.html (API_URL)")
    print("The first build takes ~2-4 minutes; watch the 'Logs' tab on the Space page.")

if __name__ == "__main__":
    main()
