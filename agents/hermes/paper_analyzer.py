#!/usr/bin/env python3
"""论文深度分析器 — 通过 Semantic Scholar / DOI / arXiv ID
用法:
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
    """从摘要中自动提取关键术语"""
    tech_terms = {
        "piezoelectric": "压电", "actuator": "执行器", "hysteresis": "迟滞",
        "Bouc-Wen": "Bouc-Wen模型", "PMN-PT": "PMN-PT材料",
        "nonlinear": "非线性", "compensation": "补偿", "inverse": "逆模型",
        "adaptive control": "自适应控制", "sliding mode": "滑模控制",
        "fuzzy": "模糊", "neural network": "神经网络",
        "PID": "PID控制", "robust": "鲁棒", "optimization": "优化",
        "positioning": "定位", "tracking": "跟踪", "vibration": "振动",
        "precision": "精密", "rate-dependent": "速率依赖",
        "ferroelectric": "铁电", "relaxor": "弛豫", "shear": "剪切",
        "Preisach": "Preisach模型", "Prandtl-Ishlinskii": "PI模型",
        "feedforward": "前馈", "feedback": "反馈", "observer": "观测器",
        "lead magnesium niobate": "铌镁酸铅", "control algorithm": "控制算法",
    }
    found = set()
    text_lower = text.lower()
    for term, cn in tech_terms.items():
        if term.lower() in text_lower:
            found.add((term, cn))
    return sorted(found, key=lambda x: -len(x[0]))


def classify_methodology(abstract):
    """从摘要推断方法论"""
    lower = abstract.lower()
    methods = []
    if any(t in lower for t in ["bouc-wen", "preisach", "prandtl-ishlinskii"]):
        methods.append("迟滞建模 (Hysteresis Modeling)")
    if any(t in lower for t in ["inverse", "compensation", "feedforward"]):
        methods.append("迟滞补偿/前馈 (Compensation/Feedforward)")
    if any(t in lower for t in ["adaptive", "neural network", "fuzzy", "sliding mode", "pid"]):
        methods.append("智能控制 (Adaptive/NN/Fuzzy/SMC)")
    if any(t in lower for t in ["optimization", "identification", "parameter"]):
        methods.append("参数辨识/优化 (Identification/Optimization)")
    if any(t in lower for t in ["experiment", "numerical", "simulation"]):
        methods.append("实验验证 (Experimental Validation)")
    if any(t in lower for t in ["rate-dependent", "frequency"]):
        methods.append("速率/频率依赖分析")
    return methods if methods else ["通用分析方法"]


def extract_innovation(abstract):
    """从摘要中提取创新点"""
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
    """生成深度分析 markdown"""
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

    cn_title = "论文深度分析报告" if lang == "zh" else "Paper Deep Analysis Report"

    output = f"""# {cn_title}

## 📄 论文信息

| 属性 | 内容 |
|------|------|
| **标题** | {title} |
| **作者** | {', '.join(authors[:4])}{'... +' + str(len(authors)-4) + ' more' if len(authors) > 4 else ''} |
| **发表** | {venue_name} ({year}) |
| **DOI** | [{doi}](https://doi.org/{doi}) |
| **引用数** | {citations} (references: {refs}) |
| **类型** | {', '.join(pub_types) if pub_types else 'N/A'} |
| **研究领域** | {', '.join(fields[:4]) if fields else 'N/A'} |
| **Open Access** | {oa_status}{': ' + oa_url if oa_url else ''} |
| **发布日期** | {pub_date} |

"""
    if arxiv:
        output += f"- **arXiv**: [{arxiv}](https://arxiv.org/abs/{arxiv})\n\n"

    if tldr and tldr.get("text"):
        output += f"""## 🤖 TL;DR (AI 总结)

> {tldr['text']}

"""

    output += f"""## 📝 摘要原文

{abstract if abstract else '(未提供摘要)'}

"""

    if keywords:
        kw_cn = ", ".join(f"{e}({c})" for e, c in keywords[:15])
        output += f"""## 🏷️ 关键词自动提取

{kw_cn}

"""

    output += f"""## 🔬 方法论分析

"""
    for m in methods:
        output += f"- ✅ {m}\n"
    output += "\n"
    for i, inn in enumerate(innovations, 1):
        output += f"### 创新点 {i}\n{inn}\n\n"

    output += f"""## 🎯 与研究方向相关度

| 方向 | 相关度 | 说明 |
|------|--------|------|
| 压电执行器控制 | {"★★★★★" if any(t in text for t in ["piezoelectric actuator", "positioning", "tracking"]) else "★★★☆☆"} | {"直接相关" if "piezoelectric" in text.lower() else "部分相关"} |
| Bouc-Wen迟滞逆模型 | {"★★★★★" if "bouc-wen" in text.lower() else "★★★☆☆"} | {"Bouc-Wen 建模" if "bouc-wen" in text.lower() else "其他迟滞模型"} |
| PMN材料 | {"★★★★☆" if "pmn" in text.lower() else "★★☆☆☆"} | {"涉及PMN" if "pmn" in text.lower() else "未涉及PMN"} |
| 控制算法 | {"★★★★★" if any(t in text for t in ["control", "adaptive", "fuzzy", "neural", "sliding"]) else "★★★☆☆"} | {"控制算法" if "control" in text.lower() else "未涉及"} |
| 综合评分 | **{relevance}** | 匹配 {relevance_score}/{len(rel_terms)} 个核心关键词 |

## 🔑 可借鉴之处

"""
    if "bouc-wen" in text.lower():
        output += "- ✅ Bouc-Wen 模型的参数辨识方法和验证流程可参考\n"
    if "inverse" in text.lower() or "compensation" in text.lower():
        output += "- ✅ 迟滞逆模型补偿的前馈+反馈控制架构值得借鉴\n"
    if "adaptive" in text.lower() or "neural" in text.lower() or "fuzzy" in text.lower():
        output += "- ✅ 智能控制算法 (Adaptive/Fuzzy-NN/SMC) 在压电平台上的实现方法可参考\n"
    if "experiment" in text.lower():
        output += "- ✅ 实验验证方案和硬件平台可做参考\n"
    if "rate-dependent" in text.lower():
        output += "- ✅ 速率依赖迟滞的处理方法值得关注\n"
    output += "- 📌 该论文在迟滞建模与控制补偿方面的框架具有通用参考价值\n\n"

    output += f"""## 🔗 所有链接

| 平台 | 链接 |
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
*分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 工具: Hermes Agent ieee-search v2.0*\n"""

    return output


def main():
    parser = argparse.ArgumentParser(description="论文深度分析器")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--doi", help="DOI")
    group.add_argument("--sid", help="Semantic Scholar Paper ID")
    group.add_argument("--arxiv", help="arXiv ID")
    parser.add_argument("--lang", choices=["zh", "en"], default="zh", help="输出语言")
    parser.add_argument("--output", help="输出文件")
    parser.add_argument("--with-citations", action="store_true", help="同时获取引用网络")
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
            report += "\n\n## 📊 引用网络\n\n"
            report += "### 被以下论文引用 (Top 5)\n"
            for c in cites.get("data", [])[:5]:
                cp = c.get("citingPaper", {})
                report += f"- [{cp.get('title','?')}]({cp.get('url','')}) ({cp.get('year','?')})\n"
            report += "\n### 引用以下论文 (Top 5)\n"
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
