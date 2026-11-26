import argparse, time, unicodedata, re
from pathlib import Path
import numpy as np
import pandas as pd
import pymupdf4llm
import fitz
from sentence_transformers import SentenceTransformer
parser=argparse.ArgumentParser()
parser.add_argument("--root",type=Path,default=Path("."))
ROOT=parser.parse_args().root
RAW_DIR=ROOT/"data/raw"
for folder in ["data/processed","vector_store"]: (ROOT/folder).mkdir(parents=True,exist_ok=True)
assert len(list(RAW_DIR.glob("*.pdf"))) == 5
embedding_model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",device="cpu")
new_page_records = []

pdf_files = sorted(RAW_DIR.glob("*.pdf"))

for pdf_path in pdf_files:

    print("\nProcessing:", pdf_path.name)

    extracted_pages = pymupdf4llm.to_markdown(
        str(pdf_path),
        page_chunks=True
    )

    with fitz.open(str(pdf_path)) as pdf:
        expected_pages = len(pdf)

    assert len(extracted_pages) == expected_pages

    for page_number, page_data in enumerate(
        extracted_pages,
        start=1
    ):

        new_page_records.append({
            "document": pdf_path.stem,
            "page": page_number,
            "text": page_data["text"].strip()
        })

    print("Pages extracted:", len(extracted_pages))

new_pages_df = pd.DataFrame(new_page_records)

print("\nEXTRACTION COMPLETED")
print("=" * 60)

print(
    "Documents:",
    new_pages_df["document"].nunique()
)

print("Total pages:", len(new_pages_df))

print(
    "Total characters:",
    new_pages_df["text"].str.len().sum()
)


import unicodedata
import re

def clean_layout_text(text):

    text = unicodedata.normalize("NFKC", text)

    # Remove soft hyphens
    text = text.replace("\u00ad", "")

    # Join words split across lines
    text = re.sub(
        r"(?<=\w)-[ \t]*\n[ \t]*(?=\w)",
        "",
        text
    )

    # Preserve link labels, remove URL markup
    text = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        text
    )

    # Remove HTML formatting tags
    text = re.sub(
        r"</?(?:sup|sub|span|br)\b[^>]*>",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # Remove Markdown heading markers
    text = re.sub(
        r"(?m)^\s*#{1,6}\s+",
        "",
        text
    )

    # Remove bold formatting
    text = text.replace("**", "")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


new_pages_df["cleaned_text"] = (
    new_pages_df["text"].apply(clean_layout_text)
)

print("Text normalization completed!")

print(
    "Total cleaned characters:",
    new_pages_df["cleaned_text"].str.len().sum()
)

print(
    "Empty pages:",
    (new_pages_df["cleaned_text"].str.len() == 0).sum()
)
tokenizer = embedding_model.tokenizer

TOKEN_CHUNK_SIZE = 220
TOKEN_OVERLAP = 44

new_chunk_records = []

for _, row in new_pages_df.iterrows():

    text = row["cleaned_text"]

    if not text.strip():
        continue

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        verbose=False
    )

    offsets = encoded["offset_mapping"]

    stride = TOKEN_CHUNK_SIZE - TOKEN_OVERLAP

    chunk_number = 1

    for start in range(0, len(offsets), stride):

        end = min(
            start + TOKEN_CHUNK_SIZE,
            len(offsets)
        )

        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]

        chunk_text = text[
            char_start:char_end
        ].strip()

        if chunk_text:

            token_count = len(
                tokenizer.encode(
                    chunk_text,
                    add_special_tokens=True,
                    truncation=False,
                    verbose=False
                )
            )

            new_chunk_records.append({
                "chunk_id": (
                    f"{row['document']}_"
                    f"p{row['page']}_"
                    f"c{chunk_number}"
                ),
                "document": row["document"],
                "page": int(row["page"]),
                "chunk_number": chunk_number,
                "text": chunk_text,
                "word_count": len(chunk_text.split()),
                "token_count": token_count
            })

            chunk_number += 1

        if end == len(offsets):
            break

new_chunks_df = pd.DataFrame(
    new_chunk_records
).reset_index(drop=True)


assert new_chunks_df.chunk_id.is_unique
assert new_chunks_df.token_count.max() <= embedding_model.max_seq_length
start = time.perf_counter()

new_embeddings = embedding_model.encode(
    new_chunks_df["text"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

new_embeddings = np.ascontiguousarray(
    new_embeddings,
    dtype=np.float32
)

print("\nEmbedding generation completed!")

print("Shape:", new_embeddings.shape)

print(
    "Time taken:",
    round(time.perf_counter() - start, 2),
    "seconds"
)
new_pages_df.to_json(ROOT/"data/processed/extracted_pages.json",orient="records",force_ascii=False,indent=2)
new_chunks_df.to_json(ROOT/"vector_store/chunk_metadata.json",orient="records",force_ascii=False,indent=2)
np.save(ROOT/"vector_store/document_embeddings.npy",new_embeddings)
print("Built",len(new_chunks_df),"chunks")
