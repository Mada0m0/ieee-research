#!/usr/bin/env python3
"""Dissertation Deep Analyzer — via Semantic Scholar / DOI / arXiv ID
usage:
    python paper_analyzer.py --doi "10.1109/ACCESS.2020.2984645"
    python paper_analyzer.py --sid f90836176af876cc622b7b1587641cce26e4564f
    python paper_analyzer.py --arxiv "2402.03300"
    python paper_analyzer.py --doi "DOI" --lang zh --output analysis.md
"""

import sys, os, json, argparse, urllib.request, urllib.parse, time
from datetime import datetime


def api_get(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": "HermesAgent/2.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def fetch_paper_by_doi(doi):
    q = urllib.parse.quote(f"DOI:{doi}")
    url = f"https://api.semanticscholar.org/graph/v1/paper/{q}?fields=title,authors,year,journal,externalIds,abstract,citationCount,referenceCount,url,venue,publicationTypes,openAccessPdf,tldr,fieldsOfStudy,publicationDate"
    return api_get(url)


def fetch_paper_by_sid(sid):
    url = f"https://api.semanticscholar.org/graph/v1/paper/{sid}?fields=title,authors,year,journal,externalIds,abstract,citationCount,referenceCount,url,venue,publicationTypes,openAccessPdf,tldr,fieldsOfStudy,publicationDate"
    return api_get(url)


def fetch_paper_by_arxiv(arxiv_id):
    q = urllib.parse.quote(f"ArXiv:{arxiv_id}")
    url = f"https://api.semanticscholar.org/graph/v1/paper/{q}?fields=title,authors,year,journal,externalIds,abstract,citationCount,referenceCount,url,venue,publicationTypes,openAccessPdf,tldr,fieldsOfStudy,publicationDate"
    return api_get(url)


def fetch_citations(paper_id, limit=10):
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations?fields=title,authors,year,citationCount,abstract,journal,url&limit={limit}"
    return api_get(url)


def fetch_references(paper_id, limit=10):
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references?fields=title,authors,year,citationCount,abstract,journal,url&limit={limit}"
    return api_get(url)


def fetch_recommendations(pos_ids, neg_ids=None, limit=10):
    payload = {"positivePaperIds": pos_ids, "negativePaperIds": neg_ids or []}
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "User-Agent": "HermesAgent/2.0",
        "Content-Type": "application/json"
    }
    req = urllib.request.Request(
        f"https://api.semanticscholar.org/recommendations/v1/papers/?limit={limit}",
        data=data, headers=headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def extract_keywords(text):
    """Automatically extract key terms from abstracts"""
    tech_terms = {
        "piezoelectric": "piezoelectric", "actuator": "actuator", "hysteresis": "hysteresis",
        "Bouc-Wen": "Bouc-Wen model", "PMN-PT": "PMN-PT material",
        "nonlinear": "nonlinear", "compensation": "compensation", "inverse": "inverse model",
        "adaptive control": "adaptive control", "sliding mode": "sliding mode control",
        "fuzzy": "fuzzy", "neural network": "neural network",
        "PID": "PID control", "robust": "robust", "optimization": "optimization",
        "positioning": "positioning", "tracking": "tracking", "vibration": "vibration",
        "precision": "precision", "rate-dependent": "rate-dependent",
        "ferroelectric": "ferroelectric", "relaxor": "relaxation", "shear": "shear",
        "Preisach": "Preisach model", "Prandtl-Ishlinskii": "PI model",
        "feedforward": "feedforward", "feedback": "feedback", "observer": "observer",
        "lead magnesium niobate": "lead magnesium niobate", "control algorithm": "control algorithm",
    }
    found = set()
    text_lower = text.lower()
    for term, cn in tech_terms.items():
        if term.lower() in text_lower:
            found.add((term, cn))
    return sorted(found, key=lambda x: -len(x[0]))


def classify_methodology(abstract):
    """Inferring methodology from abstract"""
    lower = abstract.lower()
    methods = []
    if any(t in lower for t in ["bouc-wen", "preisach", "prandtl-ishlinskii"]):
        methods.append("Hysteresis Modeling")
    if any(t in lower for t in ["inverse", "compensation", "feedforward"]):
        methods.append("Hysteresis compensation/feedforward (Compensation/Feedforward)")
    if any(t in lower for t in ["adaptive", "neural network", "fuzzy", "sliding mode", "pid"]):
        methods.append("Intelligent Control (Adaptive/NN/Fuzzy/SMC)")
    if any(t in lower for t in ["optimization", "identification", "parameter"]):
        methods.append("Parameter Identification/Optimization (Identification/Optimization)")
    if any(t in lower for t in ["experiment", "numerical", "simulation"]):
        methods.append("Experimental Validation")
    if any(t in lower for t in ["rate-dependent", "frequency"]):
        methods.append("Rate/frequency dependence analysis")
    return methods if methods else ["General analysis methods"]


def extract_innovation(abstract):
    """Extract innovative points from the abstract"""
    lower = abstract.lower()
    innovations = []
    markers = ["propose", "novel", "new", "introduce", "develop", "first", "improved",
               "proposed", "introduced", "developed", "design"]
    sentences = abstract.replace(".\n", ". ").split(". ")
    for sent in sentences:
        if any(m in sent.lower() for m in markers[:3]):
            if len(sent) > 30:
                innovations.append(sent.strip()[:150])
    return innovations[:4] if innovations else [abstract[:150]]


def format_analysis(paper, lang="zh"):
    """Generate in-depth analysis markdown"""
    title = paper.get("title", "N/A")
    year = paper.get("year", "???")
    authors = [a.get("name", "") for a in paper.get("authors", [])]
    abstract = paper.get("abstract", "") or ""
    citations = paper.get("citationCount", 0)
    refs = paper.get("referenceCount", "?")
    venue = paper.get("journal", {}) or paper.get("venue", {})
    venue_name = venue.get("name", "") if isinstance(venue, dict) else str(venue)
    doi = paper.get("externalIds", {}).get("DOI", "")
    arxiv = paper.get("externalIds", {}).get("ArXiv", "")
    paper_id = paper.get("paperId", "")
    tldr = paper.get("tldr", {})
    fields = paper.get("fieldsOfStudy", [])
    pub_types = paper.get("publicationTypes", [])
    oa = paper.get("openAccessPdf", {})
    oa_status = oa.get("status", "CLOSED") if oa else "CLOSED"
    oa_url = oa.get("url", "") if oa else ""
    pub_date = paper.get("publicationDate", "")

    keywords = extract_keywords(abstract) if abstract else []
    methods = classify_methodology(abstract) if abstract else []
    innovations = extract_innovation(abstract) if abstract else []

    relevance_score = 0
    rel_terms = ["piezoelectric", "actuator", "hysteresis", "bouc-wen", "pmn", "control",
                 "compensation", "inverse", "positioning", "precision", "rate-dependent"]
    text = (title + " " + abstract).lower()
    relevance_score = sum(1 for t in rel_terms if t in text)
    relevance = "★★★★★" if relevance_score >= 8 else ("★★★★☆" if relevance_score >= 6 else ("★★★☆☆" if relevance_score >= 4 else "★★☆☆☆"))

    cn_title = "Paper Deep Analysis Report" if lang == "zh" else "Paper Deep Analysis Report"

    output = f"""# {cn_title}

## 📄 Paper information

| Properties | Content |
|------|------|
| **Title** | {title} |
| **Authors** | {', '.join(authors[:4])}{'... +' + str(len(authors)-4) + ' more' if len(authors) > 4 else ''} |
| **Posted** | {venue_name} ({year}) |
| **DOI** | [{doi}](https://doi.org/{doi}) |
| **Number of citations** | {citations} (references: {refs}) |
| **Type** | {', '.join(pub_types) if pub_types else 'N/A'} |
| **Research Field** | {', '.join(fields[:4]) if fields else 'N/A'} |
| **Open Access** | {oa_status}{': ' + oa_url if oa_url else ''} |
| **Publish Date** | {pub_date} |

"""
    if arxiv:
        output += f"- **arXiv**: [{arxiv}](https://arxiv.org/abs/{arxiv})\n\n"

    if tldr and tldr.get("text"):
        output += f"""## 🤖 TL;DR (AI summary)

> {tldr['text']}

"""

    output += f"""## 📝 Original summary

{abstract if abstract else '(No abstract provided)'}

"""

    if keywords:
        kw_cn = ", ".join(f"{e}({c})" for e, c in keywords[:15])
        output += f"""## 🏷️ Automatic extraction of keywords

{kw_cn}

"""

    output += f"""## 🔬 Methodological analysis

"""
    for m in methods:
        output += f"- ✅ {m}\n"
    output += "\n"
    for i, inn in enumerate(innovations, 1):
        output += f"### Innovation {i}\n{inn}\n\n"

    output += f"""## 🎯 Relevance to research direction

| Direction | Relevance | Description |
|------|--------|------|
| Piezoelectric actuator control | {"★★★★★" if any(t in text for t in ["piezoelectric actuator", "positioning", "tracking"]) else "★★★☆☆"} | {"Directly related" if "piezoelectric" in text.lower() else "Partially related"} |
| Bouc-Wen hysteresis inverse model | {"★★★★★" if "bouc-wen" in text.lower() else "★★★☆☆"} | {"Bouc-Wen modeling" if "bouc-wen" in text.lower() else "Other hysteresis models"} |
| PMN material | {"★★★★☆" if "pmn" in text.lower() else "★★☆☆☆"} | {"PMN involved" if "pmn" in text.lower() else "PMN not involved"} |
| Control algorithm | {"★★★★★" if any(t in text for t in ["control", "adaptive", "fuzzy", "neural", "sliding"]) else "★★★☆☆"} | {"Control algorithm" if "control" in text.lower() else "Not involved"} |
| Comprehensive score | **{relevance}** | Match {relevance_score}/{len(rel_terms)} core keywords |

## 🔑 Lessons to learn from

"""
    if "bouc-wen" in text.lower():
        output += "- ✅ The parameter identification method and verification process of the Bouc-Wen model can be referred to\n"
    if "inverse" in text.lower() or "compensation" in text.lower():
        output += "- ✅ The feedforward + feedback control architecture of hysteresis inverse model compensation is worth learning\n"
    if "adaptive" in text.lower() or "neural" in text.lower() or "fuzzy" in text.lower():
        output += "- ✅ For the implementation method of intelligent control algorithm (Adaptive/Fuzzy-NN/SMC) on the piezoelectric platform, please refer to\n"
    if "experiment" in text.lower():
        output += "- ✅ Experimental verification scheme and hardware platform can be used as reference\n"
    if "rate-dependent" in text.lower():
        output += "- ✅ The handling of rate-dependent hysteresis deserves attention\n"
    output += "- 📌 The framework of this paper in terms of hysteresis modeling and control compensation has general reference value\n\n"

    output += f"""## 🔗 All links

| Platform | Link |
|------|------|
| Semantic Scholar | https://www.semanticscholar.org/paper/{paper_id} |
| DOI | https://doi.org/{doi} |
"""
    if arxiv:
        output += f"| arXiv | https://arxiv.org/abs/{arxiv} |\n"
        output += f"| arXiv PDF | https://arxiv.org/pdf/{arxiv} |\n"
    if oa_url:
        output += f"| Open Access PDF | {oa_url} |\n"

    output += f"""\n---
*Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Tool: Hermes Agent ieee-search v2.0*\n"""

    return output


def main():
    parser = argparse.ArgumentParser(description="Paper depth analyzer")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--doi", help="DOI")
    group.add_argument("--sid", help="Semantic Scholar Paper ID")
    group.add_argument("--arxiv", help="arXiv ID")
    parser.add_argument("--lang", choices=["zh", "en"], default="zh", help="Output Language")
    parser.add_argument("--output", help="output file")
    parser.add_argument("--with-citations", action="store_true", help="Get citation network at the same time")
    args = parser.parse_args()

    print(f"\n  Analyzing paper via ieee-search v2.0...")
    print(f"  {'='*55}")

    try:
        if args.doi:
            print(f"  Fetching by DOI: {args.doi}")
            paper = fetch_paper_by_doi(args.doi)
        elif args.sid:
            print(f"  Fetching by S2 ID: {args.sid}")
            paper = fetch_paper_by_sid(args.sid)
        else:
            print(f"  Fetching by arXiv: {args.arxiv}")
            paper = fetch_paper_by_arxiv(args.arxiv)
    except Exception as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    title = paper.get("title", "Unknown")
    print(f"  Title: {title[:80]}...")
    print(f"  Generating analysis...")

    report = format_analysis(paper, args.lang)

    if args.with_citations:
        pid = paper.get("paperId", "")
        print(f"  Fetching citation network...")
        try:
            cites = fetch_citations(pid)
            refs = fetch_references(pid)
            report += "\n\n## 📊 Reference network\n\n"
            report += "### is cited by the following papers (Top 5)\n"
            for c in cites.get("data", [])[:5]:
                cp = c.get("citingPaper", {})
                report += f"- [{cp.get('title','?')}]({cp.get('url','')}) ({cp.get('year','?')})\n"
            report += "\n### Quote the following papers (Top 5)\n"
            for r in refs.get("data", [])[:5]:
                rp = r.get("citedPaper", {})
                report += f"- [{rp.get('title','?')}]({rp.get('url','')}) ({rp.get('year','?')})\n"
        except:
            pass

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"  Report saved to: {args.output}")

    print(f"\n{report}")


if __name__ == "__main__":
    main()
