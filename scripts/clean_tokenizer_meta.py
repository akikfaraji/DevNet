"""
Clean the xorzen_agi_tokenizer_65k.meta.json file:
  - Fix the `name` typo: zarx_agi_tokenizer_65k -> xorzen_agi_tokenizer_65k
  - Strip machine-specific Windows training_files paths
"""
import json
from pathlib import Path

meta_path = Path("/home/z/my-project/xorzen_dev/xorzen/tokenizer/pretrained/xorzen_agi_tokenizer_65k.meta.json")

data = json.loads(meta_path.read_text(encoding="utf-8"))

# Fix the name typo
if data.get("name") == "zarx_agi_tokenizer_65k":
    data["name"] = "xorzen_agi_tokenizer_65k"

# Strip machine-specific training_files paths
if "training_files" in data:
    n_files = len(data["training_files"])
    # Replace with a portable summary instead of machine-specific paths
    data["training_files_summary"] = {
        "count": n_files,
        "source": "Project Gutenberg corpus (English literature)",
        "note": "Original absolute filesystem paths stripped for portability. Re-train the tokenizer to regenerate the file list."
    }
    del data["training_files"]

# Also patch description if it still has machine-specific marker
# (None needed - description is generic)

meta_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Cleaned {meta_path}")
print(f"  Name now: {data['name']}")
print(f"  training_files removed, summary added: {data.get('training_files_summary', {}).get('count', 0)} files originally")
