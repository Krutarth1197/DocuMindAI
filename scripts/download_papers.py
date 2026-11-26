import argparse
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument("--root",type=Path,default=Path("."))
ROOT=parser.parse_args().root
import requests
from pathlib import Path

RAW_DIR = ROOT / "data/raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

papers = {
    "attention_is_all_you_need": "1706.03762",
    "bert": "1810.04805",
    "rag": "2005.11401",
    "sentence_bert": "1908.10084",
    "dense_passage_retrieval": "2004.04906"
}

for name, arxiv_id in papers.items():

    url = f"https://arxiv.org/pdf/{arxiv_id}"
    response = requests.get(url, timeout=60)

    if response.status_code == 200 and response.content.startswith(b"%PDF"):
        path = RAW_DIR / f"{name}.pdf"
        path.write_bytes(response.content)
        print(f"Downloaded: {name}")
    else:
        print(f"Failed: {name}")