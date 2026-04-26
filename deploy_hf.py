"""Deploy TriageAI to Hugging Face Spaces."""

from huggingface_hub import HfApi, create_repo

SPACE_ID = "hinex-07/triage-ai-env"

api = HfApi()

# Create Space (or skip if exists)
try:
    create_repo(repo_id=SPACE_ID, repo_type="space", space_sdk="docker", exist_ok=True)
    print(f"Space {SPACE_ID} ready.")
except Exception as e:
    print(f"Space note: {e}")

# Upload all files from finale/ folder
import os
finale_dir = os.path.dirname(os.path.abspath(__file__))

print(f"Uploading from {finale_dir}...")
api.upload_folder(
    folder_path=finale_dir,
    repo_id=SPACE_ID,
    repo_type="space",
    ignore_patterns=[
        "__pycache__/*",
        "*.pyc",
        "*.pyo",
        ".git/*",
        "deploy_hf.py",
        "test_env.py",
        "training/*",
    ]
)
print(f"\nDone! Space: https://huggingface.co/spaces/{SPACE_ID}")
print(f"API:   https://hinex-07-triage-ai-env.hf.space")
