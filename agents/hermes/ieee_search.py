#!/usr/bin/env python3
"""IEEE/学术论文多源搜索 — Semantic Scholar API (覆盖 IEEE + 全部出版商)
用法:
    python ieee_search.py "piezoelectric actuator control" --max 10 --year 2020
    python ieee_search.py "Bouc-Wen hysteresis model" --field Engineering --min-citations 5
    python ieee_search.py "PMN-PT actuator" --venue-filter IEEE --output results.json
    python ieee_search.py "precision positioning" --sort citations
    python ieee_search.py --batch keywords.txt --max 5  # Batch search
"""

import sys, os, json, argparse, urllib.request, urllib.parse, time
from datetime import datetime


def search_papers(query, max_results=5, year=None, field=None,
                  min_citations=0, sort="relevance", venue_filter=None):
    """Search papers via Semantic Scholar or OpenAlex"""
    # Try Semantic Scholar
    try:
        s2_data = _search_semantic_scholar(query, max_results, year, field,
                                          min_citations, sort, venue_filter)
        if s2_data.get("results"):
            return s2_data
    except Exception as e:
        print(f"  Semantic Scholar: {type(e).__name__} - {e}")
    
    # If S2 fails, use OpenAlex
    print(f"  Falling back to OpenAlex for: {query[:70]}")
    return _search_openalex(query, max_results, year, field,
                           min_citations, sort, venue_filter)


def _search_semantic_scholar(query, max_results=5, year=None, field=None,
                           min_citations=0, sort="relevance", venue_filter=None):
    """Search papers via the Semantic Scholar API"""
    fields = "title,authors,year,journal,externalIds,abstract,citationCount,referenceCount,url,venue,publicationTypes,publicationDate,openAccessPdf,tldr,fieldsOfStudy"
    params = {"query": query, "limit": str(max_results), "fields": fields}

    # Year filter
    if year:
        params["year"] = f"{year}-"
    if field:
        params["fieldsOfStudy"] = field

    # Sort mapping
    sort_map = {"relevance": "relevance", "citations": "citationCount:desc",
                "date": "publicationDate:desc", "recency": "publicationDate:desc"}
    sort_val = sort_map.get(sort, "relevance")
    if sort_val != "relevance":
        params["sort"] = sort_val

    full_url = "https://api.semanticscholar.org/graph/v1/paper/search?" + \
               "&".join(f"{k}={urllib.parse.quote(v)}" for k, v in params.items())

    req = urllib.request.Request(full_url, headers={"User-Agent": "HermesAgent/2.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())

    papers = data.get("data", [])

    # Post-filter
    if venue_filter:
        papers = [p for p in papers if venue_filter.lower() in
                  str(p.get("journal", "") or p.get("venue", "")).lower()]
    if min_citations > 0:
        papers = [p for p in papers if p.get("citationCount", 0) >= min_citations]

    return {"total": data.get("total", 0), "results": papers, "query": query,
            "timestamp": datetime.now().isoformat(), "source": "semantic-scholar"}


def _search_openalex(query, max_results=5, year=None, field=None,
                    min_citations=0, sort="relevance", venue_filter=None):
    """Search papers via the OpenAlex API"""
    # Build API parameters
    params = {"search": query, "per-page": str(max_results), "mailto": "hermes-agent@nousresearch.com"}
    
    # sorting map
    sort_map = {"relevance": "relevance_score", "citations": "cited_by_count:desc",
                "date": "publication_year:desc", "recency": "publication_date:desc"}
    sort_val = sort_map.get(sort, "relevance_score")
    params["sort"] = sort_val
    
    # Domain filtering (OpenAlex uses concept ID)
    if field:
        field_map = {
            "Engineering": "C154945302",
            "Physics": "C121332964",
            "Materials Science": "C144024400",
            "Computer Science": "C41008148",
            "Mathematics": "C33939047"
        }
        concept_id = field_map.get(field.capitalize())
        if concept_id:
            params["filter"] = f"concepts.id:{concept_id}"
    
    # year filter
    if year:
        year_filter = f"from_publication_date:{year}-01-01"
        if "filter" in params:
            params["filter"] += f",{year_filter}"
        else:
            params["filter"] = year_filter
    
    # Build URL
    url = "https://api.openalex.org/works?" + "&".join(
        f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
    
    # Send request
    req = urllib.request.Request(url, headers={"User-Agent": "HermesAgent/2.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    
    # Processing results
    works = data.get("results", [])
    papers = []
    for work in works:
        # Convert to Semantic Scholar-like format
        paper = {
            "title": work.get("title", ""),
            "year": work.get("publication_year", ""),
            "citationCount": work.get("cited_by_count", 0),
            "doi": work["doi"].replace("https://doi.org/", "") if work.get("doi") else "",
            "url": work.get("id", ""),
            "abstract": (work.get("abstract_inverted_index") or {}).get("text", ""),
            "journal": {"name": work.get("primary_location", {}).get("source", {}).get("display_name", "")},
            "authors": [{"name": au["author"].get("display_name", "")} for au in work.get("authorships", [])],
        }
        papers.append(paper)
    
    # Journal filter
    if venue_filter:
        papers = [p for p in papers if venue_filter.lower() in 
                  str(p.get("journal", {}).get("name", "")).lower()]
    
    # Reference filtering
    if min_citations > 0:
        papers = [p for p in papers if p.get("citationCount", 0) >= min_citations]
    
    return {"total": data.get("meta", {}).get("count", 0), "results": papers, "query": query,
            "timestamp": datetime.now().isoformat(), "source": "openalex"}


def format_paper(paper, index=1, detailed=True):
    """格式化单篇论文 (rich output)"""
    title = paper.get("title", "N/A")
    year = paper.get("year", "???")
    citations = paper.get("citationCount", 0)
    refs = paper.get("referenceCount", "?")
    journal = paper.get("journal", "") or ""
    venue = paper.get("venue", "") or ""
    venue_text = journal.get("name", "") if isinstance(journal, dict) else str(journal)
    if not venue_text and isinstance(venue, dict):
        venue_text = venue.get("name", "")
    elif not venue_text:
        venue_text = str(venue)

    authors = [a.get("name", "") for a in paper.get("authors", [])]
    author_str = ", ".join(authors[:3])
    if len(authors) > 3:
        author_str += f"Waiting for {len(authors)} people"

    ext_ids = paper.get("externalIds", {})
    doi = ext_ids.get("DOI", "")
    arxiv_id = ext_ids.get("ArXiv", "")
    paper_url = paper.get("url", "")
    paper_id = paper.get("paperId", "")
    pub_date = paper.get("publicationDate", "")
    pub_types = paper.get("publicationTypes", [])
    pt_str = ",".join(pub_types) if pub_types else "unknown"
    tldr = paper.get("tldr", {})
    oa = paper.get("openAccessPdf", {})
    oa_status = oa.get("status", "CLOSED") if oa else "CLOSED"
    fields = paper.get("fieldsOfStudy", [])

    # Relevance scoring
    text = (title + " " + (paper.get("abstract") or "")).lower()
    score = sum(1 for t in ["piezoelectric","actuator","hysteresis","bouc","pmn",
                              "control","compensation","inverse","model","position"]
                if t in text)
    stars = "★" * min(5, score // 2) + "☆" * max(0, 5 - score // 2)

    abstract = (paper.get("abstract") or "")[:300]

    # Check IEEE
    is_ieee = "IEEE" in str(venue_text) or "ieee" in str(venue_text)
    badge = "[IEEE] " if is_ieee else ""

    lines = []
    lines.append(f"{'─'*65}")
    lines.append(f" #{index}  {stars}  {badge}{title}")
    lines.append(f"{'─'*65}")
    lines.append(f"     Year: {year}  |  Citations: {citations}  |  Ref: {refs}  |  Type: {pt_str}")
    if venue_text:
        lines.append(f"     Venue: {venue_text}")
    lines.append(f"     Authors: {author_str}")
    if fields:
        lines.append(f"     Fields: {', '.join(fields[:4])}")
    if doi:
        lines.append(f"     DOI: https://doi.org/{doi}")
    if arxiv_id:
        lines.append(f"     arXiv: {arxiv_id}  →  https://arxiv.org/pdf/{arxiv_id}")
    if paper_id:
        lines.append(f"     S2: https://www.semanticscholar.org/paper/{paper_id}")
    if oa_status != "CLOSED":
        lines.append(f"     Open Access: {oa_status}")
    if tldr and tldr.get("text"):
        lines.append(f"     TL;DR: {tldr['text'][:120]}...")

    if detailed and abstract:
        lines.append(f"     Abstract: {abstract}...")
    lines.append("")

    return "\n".join(lines), score, is_ieee


def main():
    parser = argparse.ArgumentParser(description="IEEE/academic paper multi-source search v2.0",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="""
Examples:
  python ieee_search.py "piezoelectric actuator hysteresis" --max 10
  python ieee_search.py "Bouc-Wen model" --field Engineering --venue-filter IEEE
  python ieee_search.py "PMN-PT actuator" --sort citations --output results.json
  python ieee_search.py --batch keywords.txt --year 2020 --max 5
""")
    parser.add_argument("query", nargs="?", help="搜索关键词")
    parser.add_argument("--max", type=int, default=8, help="最大结果数 (默认8)")
    parser.add_argument("--year", type=int, help="起始年份")
    parser.add_argument("--field", help="研究领域 (如 Engineering, Materials Science)")
    parser.add_argument("--min-citations", type=int, default=0, help="最低引用数")
    parser.add_argument("--sort", choices=["relevance","citations","date","recency"],
                       default="relevance", help="排序方式")
    parser.add_argument("--venue-filter", help="期刊/出版商过滤 (如 IEEE, Elsevier)")
    parser.add_argument("--output", help="JSON 输出文件")
    parser.add_argument("--json", action="store_true", help="控制台输出 JSON")
    parser.add_argument("--batch", help="批量搜索关键词文件(每行一个)")
    parser.add_argument("--notion-export", action="store_true", help="导出到 Notion")
    parser.add_argument("--quiet", action="store_true", help="简洁输出")
    args = parser.parse_args()

    queries = []
    if args.batch:
        with open(args.batch) as f:
            queries = [line.strip() for line in f if line.strip()]
    elif args.query:
        queries = [args.query]
    else:
        parser.print_help()
        sys.exit(1)

    all_results = []

    for q in queries:
        if not args.quiet:
            print(f"\n  🔍 Searching: \"{q}\"", file=sys.stderr)
            if args.year: print(f"     Year >= {args.year}", file=sys.stderr)
            if args.field: print(f"     Field: {args.field}", file=sys.stderr)
            if args.venue_filter: print(f"     Venue filter: {args.venue_filter}", file=sys.stderr)
            print(f"     Max results: {args.max}\n", file=sys.stderr)

        try:
            data = search_papers(q, args.max, args.year, args.field,
                                args.min_citations, args.sort, args.venue_filter)
        except Exception as e:
            print(f"  ❌ Search error: {e}", file=sys.stderr)
            continue

        papers = data.get("results", [])
        total = data.get("total", 0)

        if not args.quiet:
            print(f"  Total results: {total} | Showing: {len(papers)}\n", file=sys.stderr)

        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            ieee_count = 0
            for i, p in enumerate(papers):
                txt, score, is_ieee = format_paper(p, i + 1, detailed=not args.quiet)
                if is_ieee:
                    ieee_count += 1
                print(txt)

            if not args.quiet:
                print(f"  --- Summary: {len(papers)} papers | {ieee_count} from IEEE ---\n")

        all_results.append(data)

    # Output
    if args.output:
        out = all_results[0] if len(all_results) == 1 else all_results
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"  📁 Saved to: {args.output}", file=sys.stderr)

    # Notion export stub
    if args.notion_export:
        print(f"  📤 Notion export: Run `python scripts/notion_exporter.py --batch {args.output or 'results.json'}`",
              file=sys.stderr)


if __name__ == "__main__":
    main()
