#!/usr/bin/env python3
"""Notion paper exporter — search papers → in-depth analysis → export Notion page

Two modes:
  Mode A (default): Parse article metadata → Build rich text Notion blocks
  Mode B (--attachment): Upload local .md → Tuple code blocks embedded (original)

Usage:
    python notion_exporter.py --search "piezo actuator Bouc-Wen" --max 3
    python notion_exporter.py --analyze --doi "10.1109/ACCESS.2020.2984645"
    python notion_exporter.py --attachment analysis_boucwen_generalized.md
    python notion_exporter.py --batch papers.json
"""

import sys, os, json, argparse, urllib.request, urllib.parse, subprocess, time, re
from datetime import datetime

# ---- Notion API helpers ----

def _notion_key():
    env_file = os.path.expanduser("~/.hermes/.env")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                if line.startswith("NOTION_API_KEY="):
                    return line.strip().split("=", 1)[1].strip()
    return ""

NOTION_KEY = _notion_key()

def notion_api(endpoint, data, method="POST"):
    tmp = "/tmp/notion_api_payload.json"
    with open(tmp, "w") as f:
        json.dump(data, f, ensure_ascii=False)
    cmd = [
        "curl", "-s", "-X", method,
        f"https://api.notion.com/v1{endpoint}",
        "-H", f"Authorization: Bearer {NOTION_KEY}",
        "-H", "Notion-Version: 2025-09-03",
        "-H", "Content-Type: application/json",
        "-d", f"@{tmp}"
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    return json.loads(r.stdout) if r.stdout else {}


def notion_find_parent_page(keyword):
    """Find the appropriate parent page"""
    r = notion_api("/search", {"page_size": 10, "query": keyword})
    for page in r.get("results", []):
        if page.get("object") == "page":
            return page["id"].replace("-", "")
    return None


# ==== Mode B: Attachment upload (.md → code blocks) ====

def md_to_code_blocks(md_text, max_len=1900):
    """Cut Markdown text into Notion code blocks (each < 2000 chars)"""
    chunks = []
    current = ""
    for line in md_text.split("\n"):
        if len(current) + len(line) + 1 > max_len:
            chunks.append(current)
            current = line
        else:
            current = current + "\n" + line if current else line
    if current:
        chunks.append(current)
    return [{
        "object": "block", "type": "code",
        "code": {
            "rich_text": [{"type": "text", "text": {"content": chunk}}],
            "language": "markdown"
        }
    } for chunk in chunks]


def extract_metadata_from_md(md_text):
    """Extract article metadata from .md header"""
    meta = {}
    lines = md_text.split("\n")
    for line in lines[:15]:
        # Title: | **Title** | xxx |
        m = re.match(r".*\*\*Title\*\*\s*\|\s*(.+)", line)
        if m:
            val = m.group(1).strip().rstrip("|").strip()
            if len(val) > 3:
                meta["title"] = val
        # DOI: [xxx](https://doi.org/xxx)
        m = re.match(r".*\[(.+?)\]\(https://doi\.org/(.+?)\)", line)
        if m:
            meta["doi"] = m.group(2).strip()
        # Publish: | **Publish** | xxx |
        m = re.match(r".*\*\*Posted\*\*\s*\|\s*(.+)", line)
        if m:
            val = m.group(1).strip().rstrip("|").strip()
            if val and len(val) > 2:
                meta["venue"] = val
        # Quote: | **Number of citations** | xxx |
        m = re.match(r".*\*\*Number of citations\*\*\s*\|\s*(.+)", line)
        if m:
            val = m.group(1).strip().rstrip("|").strip()
            if val:
                meta["citations"] = val
    # fallback: start from the first h2 or h1 but skip the "paper in-depth analysis report"
    if not meta.get("title"):
        for line in lines[:5]:
            if line.startswith("## ") and "Paper Deep" not in line:
                meta["title"] = line[2:].strip()
                break
            elif line.startswith("# ") and "Paper Deep" not in line:
                meta["title"] = line[1:].strip()
                break
    return meta


def export_attachment(md_path, parent_id=None):
    """Mode B: Upload local .md files to Notion as code blocks attachment"""
    if not os.path.exists(md_path):
        print(f"  ❌ File not found: {md_path}")
        return None

    with open(md_path, encoding="utf-8") as f:
        md_text = f.read()

    meta = extract_metadata_from_md(md_text)
    title = meta.get("title", os.path.basename(md_path).replace(".md", ""))
    doi = meta.get("doi", "")
    venue = meta.get("venue", "")
    citations = meta.get("citations", "")

    code_blocks = md_to_code_blocks(md_text)

    if not parent_id:
        parent_id = notion_find_parent_page("Bouc-Wen") or notion_find_parent_page("paper")

    print(f"  📎 Attachment mode: {len(code_blocks)} code blocks")
    print(f"  📄 Title: {title[:70]}")
    if doi:
        print(f"  🔗 DOI: {doi}")

    # Build page: title + metadata + toggle(full md)
    metadata_text = ""
    if venue:
        metadata_text += venue
    if citations:
        metadata_text += f" • {citations} citations"
    if doi:
        metadata_text += f" • DOI: {doi}"

    blocks = [
        {
            "object": "block", "type": "heading_1",
            "heading_1": {"rich_text": [{"type": "text", "text": {"content": title[:80]}}]}
        },
        {
            "object": "block", "type": "callout",
            "callout": {
                "rich_text": [{"type": "text", "text": {"content": metadata_text or "paper in-depth analysis report"}}],
                "icon": {"emoji": "📎"},
                "color": "gray_background"
            }
        },
        {"object": "block", "type": "divider", "divider": {}},
        {
            "object": "block", "type": "toggle",
            "toggle": {
                "rich_text": [{"type": "text", "text": {"content": f"📄 Complete Markdown analysis report ({len(code_blocks)} segments, {len(md_text)} characters)"}}],
                "children": code_blocks
            }
        }
    ]

    payload = {
        "parent": {"page_id": parent_id or ""},
        "properties": {
            "title": {"title": [{"text": {"content": f"[attachment] {title[:60]}"}}]}
        },
        "children": blocks
    }

    try:
        result = notion_api("/pages", payload)
        if "id" in result:
            pid = result["id"].replace("-", "")
            print(f"  ✅ Created: https://notion.so/{pid}")
            return pid
        else:
            msg = result.get("message", "Unknown error")
            print(f"  ❌ {msg}")
    except Exception as e:
        print(f"  ❌ Exception: {e}")
    return None


def notion_find_parent_page(keyword):
    """Find the appropriate parent page"""
    r = notion_api("/search", {"page_size": 10, "query": keyword})
    for page in r.get("results", []):
        if page.get("object") == "page":
            return page["id"].replace("-", "")
    return None


def notion_create_page(title, children, parent_id=None):
    """Create Notion page"""
    payload = {
        "parent": {"page_id": parent_id} if parent_id else {"type": "workspace"},
        "properties": {
            "title": {"title": [{"text": {"content": title}}]}
        },
        "children": children
    }
    return notion_api("/pages", payload)


# ---- Semantic Scholar ----

def search_papers(query, max_results=5, year=None):
    fields = "title,authors,year,journal,externalIds,abstract,citationCount,url,venue,publicationTypes,openAccessPdf"
    params = {"query": query, "limit": str(max_results), "fields": fields}
    if year:
        params["year"] = f"{year}-"
    url = "https://api.semanticscholar.org/graph/v1/paper/search?" + "&".join(
        f"{k}={urllib.parse.quote(v)}" for k, v in params.items())
    req = urllib.request.Request(url, headers={"User-Agent": "HermesAgent/2.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read()).get("data", [])


def fetch_paper_by_doi(doi):
    q = urllib.parse.quote(f"DOI:{doi}")
    url = f"https://api.semanticscholar.org/graph/v1/paper/{q}?fields=title,authors,year,journal,externalIds,abstract,citationCount,referenceCount,url,venue,publicationTypes,openAccessPdf,publicationDate"
    req = urllib.request.Request(url, headers={"User-Agent": "HermesAgent/2.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


# ---- Paper → Notion Blocks ----

def paper_to_notion_blocks(paper, index=1):
    """Convert paper information into Notion content blocks"""
    title = paper.get("title", "N/A")
    year = paper.get("year", "???")
    authors = [a.get("name", "") for a in paper.get("authors", [])]
    abstract = (paper.get("abstract") or "")[:1200]
    citations = paper.get("citationCount", 0)
    refs = paper.get("referenceCount", "?")
    venue = paper.get("journal", {}) or paper.get("venue", {})
    venue_name = venue.get("name", "") if isinstance(venue, dict) else str(venue)
    doi = paper.get("externalIds", {}).get("DOI", "")
    arxiv = paper.get("externalIds", {}).get("ArXiv", "")
    paper_id = paper.get("paperId", "")
    oa = paper.get("openAccessPdf", {})
    oa_url = oa.get("url", "") if oa else ""
    pub_date = paper.get("publicationDate", "")

    # Automatic scoring
    text = (title + " " + abstract).lower()
    score = sum(1 for t in ["piezoelectric","actuator","hysteresis","bouc-wen","pmn",
                              "control","compensation","inverse","positioning","precision"]
                if t in text)
    stars = "⭐" * min(5, score // 2) + "☆" * max(0, 5 - score // 2)

    # Whether IEEE
    is_ieee = "IEEE" in str(venue) if venue else False

    children = []

    # Callout — Journal information
    callout_text = f"{'[IEEE] ' if is_ieee else ''}{venue_name}, {year}"
    if pub_date:
        callout_text += f" | Published: {pub_date}"
    callout_text += f" | Citations: {citations} | Internal ID: #{index}"
    children.append({
        "object": "block",
        "type": "callout",
        "callout": {
            "rich_text": [{"type": "text", "text": {"content": callout_text}}],
            "icon": {"emoji": "📄"},
            "color": "blue_background"
        }
    })

    # Title (H1)
    children.append({
        "object": "block",
        "type": "heading_1",
        "heading_1": {"rich_text": [{"type": "text", "text": {"content": title}}]}
    })

    # Author + DOI
    children.append({
        "object": "block",
        "type": "paragraph",
        "paragraph": {"rich_text": [
            {"type": "text", "text": {"content": "Authors: "}, "annotations": {"bold": True}},
            {"type": "text", "text": {"content": ", ".join(authors[:5])}}
        ]}
    })

    if doi:
        children.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {"rich_text": [
                {"type": "text", "text": {"content": "DOI: "}, "annotations": {"bold": True}},
                {"type": "text", "text": {"content": doi, "link": {"url": f"https://doi.org/{doi}"}}}
            ]}
        })

    children.append({"object": "block", "type": "divider", "divider": {}})

    # Toggle: Summary
    children.append({
        "object": "block",
        "type": "toggle",
        "toggle": {
            "rich_text": [{"type": "text", "text": {"content": f"📝 Abstract (Auto-extracted)"}}],
            "children": [
                {"object": "block", "type": "paragraph",
                 "paragraph": {"rich_text": [{"type": "text", "text": {"content": abstract if abstract else "(No abstract available)"}}]}}
            ]
        }
    })

    if abstract:
        # Toggle: Methodology Breakdown
        methods_text = "**Methodology Breakdown**\n\n"
        lower = abstract.lower()
        if any(t in lower for t in ["bouc-wen","preisach","prandtl-ishlinskii"]):
            methods_text += "🔹 Hysteresis Modeling: Bouc-Wen/Preisach/PI model for capturing nonlinear dynamics\n"
        if any(t in lower for t in ["inverse","compensation","feedforward"]):
            methods_text += "🔹 Compensation Strategy: Feedforward inverse model + feedback control\n"
        if any(t in lower for t in ["adaptive","neural network","fuzzy","sliding mode"]):
            methods_text += "🔹 Intelligent Control: Adaptive/Fuzzy-NN/SMC algorithm for precision tracking\n"
        if any(t in lower for t in ["optimization","identification","parameter"]):
            methods_text += "🔹 Parameter Identification: Optimization algorithm for model fitting\n"
        if any(t in lower for t in ["experiment","simulation"]):
            methods_text += "🔹 Validation: Experimental/numerical verification\n"

        children.append({
            "object": "block",
            "type": "toggle",
            "toggle": {
                "rich_text": [{"type": "text", "text": {"content": "🔬 Methodology Dismantling"}}],
                "children": [
                    {"object": "block", "type": "paragraph",
                     "paragraph": {"rich_text": [{"type": "text", "text": {"content": methods_text}}]}}
                ]
            }
        })

    # Relevance rating table
    children.append({
        "object": "block",
        "type": "heading_3",
        "heading_3": {"rich_text": [{"type": "text", "text": {"content": "🎯Relevance of research direction"}}]}
    })

    # Table
    children.append({
        "object": "block",
        "type": "table",
        "table": {
            "table_width": 3,
            "has_column_header": True,
            "has_row_header": False,
            "children": [
                {"type": "table_row", "table_row": {"cells": [
                    [{"type": "text", "text": {"content": "direction"}}],
                    [{"type": "text", "text": {"content": "evaluation"}}],
                    [{"type": "text", "text": {"content": "Description"}}]
                ]}},
                {"type": "table_row", "table_row": {"cells": [
                    [{"type": "text", "text": {"content": "Piezoelectric actuator control"}}],
                    [{"type": "text", "text": {"content": "★★★★★" if "piezoelectric" in text else "★★★☆☆"}}],
                    [{"type": "text", "text": {"content": "core object" if "piezoelectric" in text else "indirectly related"}}]
                ]}},
                {"type": "table_row", "table_row": {"cells": [
                    [{"type": "text", "text": {"content": "Bouc-Wen hysteresis inverse model"}}],
                    [{"type": "text", "text": {"content": "★★★★★" if "bouc-wen" in text else "★★★☆☆"}}],
                    [{"type": "text", "text": {"content": "BW Modeling" if "bouc-wen" in text else "Other Models"}}]
                ]}},
                {"type": "table_row", "table_row": {"cells": [
                    [{"type": "text", "text": {"content": "PMN Material Actuator"}}],
                    [{"type": "text", "text": {"content": "★★★★☆" if "pmn" in text else "★★☆☆☆"}}],
                    [{"type": "text", "text": {"content": "Materials involved" if "pmn" in text else "PMN materials not involved"}}]
                ]}},
                {"type": "table_row", "table_row": {"cells": [
                    [{"type": "text", "text": {"content": "Control algorithm"}}],
                    [{"type": "text", "text": {"content": "★★★★★" if "control" in text else "★★★☆☆"}}],
                    [{"type": "text", "text": {"content": "Control method" if "control" in text else "Partially related"}}]
                ]}},
                {"type": "table_row", "table_row": {"cells": [
                    [{"type": "text", "text": {"content": "Comprehensive Rating"}}],
                    [{"type": "text", "text": {"content": stars}}],
                    [{"type": "text", "text": {"content": f"Match {score}/10 keywords"}}]
                ]}}
            ]
        }
    })

    children.append({"object": "block", "type": "divider", "divider": {}})

    # Link
    children.append({
        "object": "block",
        "type": "heading_3",
        "heading_3": {"rich_text": [{"type": "text", "text": {"content": "🔗 Link"}}]}
    })

    links = []
    if paper_id:
        links.append(("Semantic Scholar", f"https://www.semanticscholar.org/paper/{paper_id}"))
    if doi:
        links.append(("DOI", f"https://doi.org/{doi}"))
    if arxiv:
        links.append(("arXiv PDF", f"https://arxiv.org/pdf/{arxiv}"))
    if oa_url:
        links.append(("Open Access PDF", oa_url))

    for label, url in links:
        children.append({
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {"rich_text": [
                {"type": "text", "text": {"content": f"{label}: "}, "annotations": {"bold": True}},
                {"type": "text", "text": {"content": url, "link": {"url": url}}}
            ]}
        })

    children.append({"object": "block", "type": "divider", "divider": {}})

    # footnote
    children.append({
        "object": "block",
        "type": "paragraph",
        "paragraph": {
            "rich_text": [{"type": "text", "text": {"content": f"🤖 Generated by Hermes Agent + ieee-search v2.0 | {datetime.now().strftime('%Y-%m-%d')}"}}],
            "color": "gray"
        }
    })

    return children


def main():
    parser = argparse.ArgumentParser(description="Notion paper exporter")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--search", help="Search keywords and export")
    group.add_argument("--analyze", action="store_true", help="Analyze a single paper")
    group.add_argument("--attachment", help="Local .md file path → Mode B: Upload as code blocks attachment")
    group.add_argument("--batch", help="Batch processing JSON files")
    parser.add_argument("--doi", help="DOI (with --analyze)")
    parser.add_argument("--sid", help="S2 Paper ID (with --analyze)")
    parser.add_argument("--max", type=int, default=3, help="Maximum number of results (search mode)")
    parser.add_argument("--year", type=int, help="start year")
    parser.add_argument("--parent-page", help="Notion parent page ID")
    parser.add_argument("--dry-run", action="store_true", help="Only display content, do not write Notion")
    args = parser.parse_args()

    # ==== Mode B: Attachment upload ====
    if args.attachment:
        if args.dry_run:
            md_path = args.attachment
            if os.path.exists(md_path):
                with open(md_path, encoding="utf-8") as f:
                    print(f"  [DRY RUN] Would upload: {md_path}")
                    print(f"  Size: {os.path.getsize(md_path)} bytes")
                    print(f"  Preview:{f.read()[:300]}...")
            else:
                print(f"  File not found: {md_path}")
        else:
            export_attachment(args.attachment, args.parent_page)
        return

    # ==== Mode A: Search/Analysis → Rich Text Export ====
    papers = []

    if args.search:
        print(f"\n  Searching: {args.search}")
        try:
            papers = search_papers(args.search, args.max, args.year)
            print(f"  Found {len(papers)} papers")
        except Exception as e:
            print(f"  Search error: {e}")

    elif args.analyze and args.doi:
        print(f"\n  Fetching by DOI: {args.doi}")
        try:
            papers = [fetch_paper_by_doi(args.doi)]
        except Exception as e:
            print(f"  Fetch error: {e}")

    elif args.analyze and args.sid:
        print(f"\n  Fetching by S2 ID: {args.sid}")
        try:
            r = fetch_paper_by_doi(None)
            r["paperId"] = args.sid
            papers = [r]
        except Exception as e:
            print(f"  Fetch error: {e}")

    elif args.batch:
        with open(args.batch) as f:
            papers = json.load(f)
        if isinstance(papers, dict):
            papers = papers.get("data", papers.get("results", []))

    if not papers:
        print("  No papers to export.")
        return

    parent_id = args.parent_page or notion_find_parent_page("Bouc-Wen")
    print(f"  Parent page: {parent_id or 'workspace (top-level)'}")

    for i, paper in enumerate(papers):
        title = paper.get("title", f"Paper #{i+1}")
        print(f"\n  [{i+1}/{len(papers)}] Creating: {title[:70]}...")

        children = paper_to_notion_blocks(paper, i + 1)
        print(f"    Blocks: {len(children)}")

        if args.dry_run:
            block_preview = json.dumps(children[:2], indent=2, ensure_ascii=False)
            print(f"    [DRY RUN] Preview:{block_preview[:400]}")
            continue

        try:
            result = notion_api("/pages", {
                "parent": {"page_id": parent_id},
                "properties": {
                    "title": {"title": [{"text": {"content": title}}]}
                },
                "children": children
            })
            if "id" in result:
                pid = result["id"].replace("-", "")
                print(f"    ✅ https://notion.so/{pid}")
            else:
                print(f"    ❌ {result.get('message','Unknown')[:200]}")
        except Exception as e:
            print(f"    ❌ {e}")

        time.sleep(1)

    print(f"\n  Done! {len(papers)} pages exported.")


if __name__ == "__main__":
    main()
