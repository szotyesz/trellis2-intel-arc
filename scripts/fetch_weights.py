"""Download pinned weights into the cache volume. Needs network and a token."""
import pathlib
import sys

from huggingface_hub import constants, snapshot_download

REPOS = {
    "microsoft/TRELLIS.2-4B": "af44b45f2e35a493886929c6d786e563ec68364d",
    "microsoft/TRELLIS-image-large": "25e0d31ffbebe4b5a97464dd851910efc3002d96",
    "facebook/dinov3-vitl16-pretrain-lvd1689m": "ea8dc2863c51be0a264bab82070e3e8836b02d51",
}
PATTERNS = {"microsoft/TRELLIS-image-large": ["ckpts/ss_dec_conv3d_16l8_fp16*"]}

failed = False
for repo, commit in REPOS.items():
    try:
        path = snapshot_download(repo, allow_patterns=PATTERNS.get(repo))
    except Exception as e:  # keep going so one gated repo does not hide the others
        print(f"FAILED {repo}: {type(e).__name__}: {e}")
        failed = True
        continue
    got = pathlib.Path(path).name
    print(f"{'OK' if got == commit else 'CHANGED'} {repo} {got}")
    failed |= got != commit

if list(pathlib.Path(constants.HF_HUB_CACHE).glob("models--briaai*")):
    print("FAILED: RMBG weights in cache")
    failed = True
sys.exit(1 if failed else 0)
