import requests
import csv
import os
import re
from bs4 import BeautifulSoup

# ===== Config =====
URL = "https://www.lehman.edu/student-affairs/office-campus-life/new-student-orientation/"
OUTPUT_DIR = "data_demo/lehman/new_student_orientation"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "lehman_new_student_orientation.csv")
OUTPUT_TXT = os.path.join(OUTPUT_DIR, "lehman_new_student_orientation.txt")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/121.0 Safari/537.36"
}

# Common content wrappers (same pattern as your other scraper)
CONTENT_SELECTORS = "main, article, .content, #content, .entry-content, .page-content"

def scrape(url: str):
    print(f"Fetching: {url} ...")
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    main = soup.select_one(CONTENT_SELECTORS) or soup.body or soup
    # Drop non-content
    for s in main.select("nav, footer, header, script, style, form, aside"):
        s.decompose()

    rows = []
    parts_for_txt = []
    current_section = ""

    # Headings become sections; paragraphs/list items become rows
    for el in main.descendants:
        name = getattr(el, "name", None)
        if name in {"h1", "h2", "h3"}:
            current_section = el.get_text(" ", strip=True)
            if current_section:
                parts_for_txt.append(f"\n{current_section}\n" + "-" * len(current_section))
        elif name in {"p", "li"}:
            t = el.get_text(" ", strip=True)
            if t:
                t = re.sub(r"[ \t]+", " ", t)
                rows.append({"section": current_section or "General", "text": t, "source_url": url})
                parts_for_txt.append(t)

    full_text = "\n".join(parts_for_txt).strip()
    return full_text, rows

def save_outputs(full_text: str, rows: list[dict]):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["section", "text", "source_url"])
        w.writeheader()
        w.writerows(rows)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"Saved CSV → {OUTPUT_CSV}")
    print(f"Saved TXT → {OUTPUT_TXT}")

if __name__ == "__main__":
    full_text, rows = scrape(URL)
    if not rows or len(full_text) < 200:
        raise SystemExit("No/insufficient content extracted — selectors may need tweaking or the page is JS-rendered.")
    save_outputs(full_text, rows)

    print("\nPreview:")
    for r in rows[:5]:
        print(f"- [{r['section']}] {r['text'][:120]}{'...' if len(r['text'])>120 else ''}")
