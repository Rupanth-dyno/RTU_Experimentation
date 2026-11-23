import argparse
import csv
import hashlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def download_one(row: dict, output_dir: Path, timeout: float, retries: int) -> dict:
    filename = Path(row["PDF filename"]).name
    url = row["Direct PDF download"].strip()
    destination = output_dir / filename

    if destination.exists() and destination.stat().st_size > 1000:
        data = destination.read_bytes()
        return {
            "filename": filename,
            "status": "cached",
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    last_error = ""
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=timeout, headers={"User-Agent": "RTU-Thesis-Dataset/1.0"}) as response:
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "")
                data = response.content
            if len(data) < 1000 or b"%PDF" not in data[:1024]:
                raise ValueError(f"not a valid PDF response (type={content_type}, bytes={len(data)})")
            destination.write_bytes(data)
            return {
                "filename": filename,
                "status": "downloaded",
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        except Exception as exc:
            last_error = str(exc)
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
    return {"filename": filename, "status": "failed", "error": last_error}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download all PDFs listed in the TSV link manifest.")
    parser.add_argument("--links", default="data/pdf links.txt")
    parser.add_argument("--output-dir", default="data/pdfs")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Path(args.links).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    if not rows:
        parser.error("PDF link manifest contains no rows")
    print(f"Preparing {len(rows)} PDFs -> {output_dir}")
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_one, row, output_dir, args.timeout, args.retries): row
            for row in rows
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if len(results) % 100 == 0:
                print(f"Processed {len(results)}/{len(rows)}")

    downloaded = sum(r["status"] == "downloaded" for r in results)
    cached = sum(r["status"] == "cached" for r in results)
    failed = [r for r in results if r["status"] == "failed"]
    report = {
        "manifest_links": len(rows),
        "downloaded": downloaded,
        "cached": cached,
        "failed": len(failed),
        "available": downloaded + cached,
        "failures": failed,
        "results": results,
    }
    report_path = output_dir / "download_report.json"
    report_path.write_text(__import__("json").dumps(report, indent=2), encoding="utf-8")
    print(
        f"PDF download complete: available={report['available']} "
        f"(downloaded={downloaded}, cached={cached}, failed={len(failed)})"
    )
    print(f"Report: {report_path}")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
