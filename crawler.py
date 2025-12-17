import argparse
import hashlib
import os
import re
from urllib.parse import urlparse

from langchain_community.document_loaders import RecursiveUrlLoader
from bs4 import BeautifulSoup


def sanitize_filename(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.netloc + parsed.path
    path = re.sub(r"[^A-Za-z0-9._-]+", "_", path).strip("_")
    if not path or path == "_":
        path = "index"
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"{path}_{digest}.txt"


def html_to_text(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join([l for l in lines if l])
    return text.strip()


def crawl_and_save(base_url: str, output_dir: str, max_depth: int = 3) -> int:
    os.makedirs(output_dir, exist_ok=True)

    loader = RecursiveUrlLoader(
        url=base_url,
        max_depth=max_depth,
        extractor=None,
        timeout=15,
        prevent_outside=True,
    )
    docs = loader.load()
    count = 0
    for doc in docs:
        src = doc.metadata.get("source", base_url)
        if src.lower().endswith(
            (
                ".css",
                ".js",
                ".png",
                ".jpg",
                ".jpeg",
                ".gif",
                ".svg",
                ".webp",
                ".ico",
                ".pdf",
                ".zip",
                ".rar",
                ".7z",
                ".mp4",
                ".mp3",
                ".woff",
                ".woff2",
                ".ttf",
                ".eot",
            )
        ):
            continue
        fname = sanitize_filename(src)
        fpath = os.path.join(output_dir, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(html_to_text(doc.page_content or ""))
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", default="data")
    parser.add_argument("--depth", type=int, default=3)
    args = parser.parse_args()
    total = crawl_and_save(args.url, args.out, args.depth)
    print(f"Salvos {total} arquivos em '{args.out}'")


if __name__ == "__main__":
    main()
