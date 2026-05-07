#!/usr/bin/env python3
"""引用网络分析 — 引用图、相关论文、领域桥梁
用法:
    python citation_network.py --doi "10.1109/ACCESS.2020.2984645"
    python citation_network.py --sid f90836176af876cc622b7b1587641cce26e4564f
    python citation_network.py --search "piezoelectric hysteresis control" --top-cited 20
"""

import sys, json, argparse, urllib.request, urllib.parse, time
from datetime import datetime


def api_get(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": "HermesAgent/2.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def fetch_paper(identifier, id_type="doi"):
    """id_type: doi, sid, arxiv"""
    if id_type == "doi":
        q = urllib.parse.quote(f"DOI:{identifier}")
    elif id_type == "arxiv":
        q = urllib.parse.quote(f"ArXiv:{identifier}")
    else:
        q = identifier
    fields = "title,authors,year,journal,externalIds,citationCount,referenceCount,url,venue"
    url = f"https://api.semanticscholar.org/graph/v1/paper/{q}?fields={fields}"
    return api_get(url)


def fetch_citations(paper_id, limit=20):
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations?fields=title,authors,year,citationCount,abstract,journal,url&limit={limit}"
    return api_get(url)


def fetch_references(paper_id, limit=20):
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references?fields=title,authors,year,citationCount,abstract,journal,url&limit={limit}"
    return api_get(url)


def search_top_cited(query, limit=20):
    q = urllib.parse.quote(query)
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={q}&limit={limit}&sort=citationCount:desc&fields=title,authors,year,citationCount,journal,url"
    return api_get(url)


def format_citation_network(paper, lang="zh"):
    cn_title = "引用网络分析" if lang == "zh" else "Citation Network Analysis"
    title = paper.get("title", "N/A")
    pid = paper.get("paperId", "")
    cites = paper.get("citationCount", 0)
    refs = paper.get("referenceCount", 0)

    output = f"""# {cn_title}

## 中心论文
- **{title}**
- Citations: {cites} | References: {refs}
- [S2 Link](https://www.semanticscholar.org/paper/{pid})

"""

    # Fetch citations and references
    try:
        cit_data = fetch_citations(pid, 10)
        output += "## 📌 被以下论文引用 (Recent Citations)\n\n"
        for c in cit_data.get("data", []):
            cp = c.get("citingPaper", {})
            output += f"- [{cp.get('title','?')[:80]}]({cp.get('url','')}) — {cp.get('year','?')} ({cp.get('citationCount',0)} cites)\n"
        output += "\n"
        time.sleep(1)
    except:
        output += "*(citation data unavailable)*\n\n"

    try:
        ref_data = fetch_references(pid, 10)
        output += "## 📚 引用以下论文 (References)\n\n"
        for r in ref_data.get("data", []):
            rp = r.get("citedPaper", {})
            output += f"- [{rp.get('title','?')[:80]}]({rp.get('url','')}) — {rp.get('year','?')} ({rp.get('citationCount',0)} cites)\n"
        output += "\n"
    except:
        output += "*(reference data unavailable)*\n\n"

    output += f"---\n*{datetime.now().strftime('%Y-%m-%d %H:%M')} | ieee-search v2.0*\n"
    return output


def main():
    parser = argparse.ArgumentParser(description="引用网络分析 v2.0")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--doi", help="DOI")
    group.add_argument("--sid", help="Semantic Scholar Paper ID")
    group.add_argument("--arxiv", help="arXiv ID")
    group.add_argument("--search", help="搜索高被引论文")
    parser.add_argument("--top-cited", type=int, default=20, help="最高被引数")
    parser.add_argument("--output", help="输出文件")
    parser.add_argument("--lang", choices=["zh","en"], default="zh")
    args = parser.parse_args()

    print(f"\n  Citation Network Analysis")
    print(f"  {'='*50}")

    if args.search:
        print(f"  Searching top-cited: {args.search}")
        data = search_top_cited(args.search, args.top_cited)
        papers = data.get("data", [])
        print(f"  Found {len(papers)} papers\n")
        for i, p in enumerate(papers):
            v = p.get("journal", {})
            vn = v.get("name","") if isinstance(v,dict) else str(v)
            print(f"  {i+1:2d}. [Cites:{p.get('citationCount',0):4d}] {p.get('title','?')[:70]}")
            if vn: print(f"          {vn} ({p.get('year','?')})")
        return

    # Single paper analysis
    id_type = "sid"
    identifier = args.sid
    if args.doi:
        id_type = "doi"
        identifier = args.doi
    elif args.arxiv:
        id_type = "arxiv"
        identifier = args.arxiv

    print(f"  Fetching: {identifier} ({id_type})")
    paper = fetch_paper(identifier, id_type)
    print(f"  Title: {paper.get('title','?')[:80]}...")
    print(f"  Citations: {paper.get('citationCount',0)}, References: {paper.get('referenceCount',0)}")

    report = format_citation_network(paper, args.lang)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"  Saved: {args.output}")

    print(f"\n{report[:2000]}...")


if __name__ == "__main__":
    main()
