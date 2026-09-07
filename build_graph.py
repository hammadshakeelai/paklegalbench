import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).parent
CORPUS_PATH = Path(os.environ.get("PLB_CORPUS_PATH", "chunks.json" if Path("chunks.json").exists() else "seed_corpus.json"))
OUTPUT_PATH = ROOT / "statute_graph.json"

LIST_PAT = re.compile(
    r"\b(?:sections?|articles?|secs?\.?|arts?\.?)\s+([0-9]+[A-Za-z]?(?:\s*(?:,|to|and|-|–)\s*[0-9]+[A-Za-z]?)*)",
    re.I,
)


def parse_citations(text: str) -> list[str]:
    """Parse single citations, comma-separated lists, and bounded ranges from statutory text."""
    out = []
    for m in LIST_PAT.finditer(text):
        raw = m.group(1).strip()
        # Check bounded range, e.g. "300 to 304" or "496-498"
        range_m = re.match(r"^(\d+)\s*(?:to|-|–)\s*(\d+)$", raw, re.I)
        if range_m:
            start, end = int(range_m.group(1)), int(range_m.group(2))
            if 0 < end - start <= 25:
                for s in range(start, end + 1):
                    out.append(str(s))
                continue
        # Split by comma or 'and'
        items = re.split(r"[,;]|\band\b", raw, flags=re.I)
        for it in items:
            it = it.strip()
            num_m = re.match(r"^([0-9]+(?:[-–]?[A-Za-z])?)$", it)
            if num_m:
                out.append(num_m.group(1))
    return out


def build_statutory_graph():
    raw = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    chunks = raw["chunks"] if isinstance(raw, dict) else raw

    ref_to_id = {}
    id_to_citation = {}
    valid_citations = set()
    for c in chunks:
        act = c.get("act_short")
        sec = c.get("section", "").replace("-", "").replace(" ", "").upper()
        cid = c.get("id")
        if act and sec and cid:
            ref_to_id[(act, sec)] = cid
            cit_label = f"{act} Section {c.get('section')}" if act != "Constitution" else f"Constitution, Article {c.get('section')}"
            id_to_citation[cid] = cit_label
            valid_citations.add((act, sec))

    graph: dict[str, list[str]] = {}
    reverse_graph: dict[str, list[str]] = {}
    citation_graph: dict[str, list[str]] = {}
    edges = []

    for c in chunks:
        cid = c["id"]
        act = c.get("act_short")
        sec_norm = c.get("section", "").replace("-", "").replace(" ", "").upper()
        text = c.get("text", "")
        found = parse_citations(text)
        targets = set()
        cit_targets = set()

        for raw_s in found:
            target_sec = raw_s.replace("-", "").replace(" ", "").upper()
            target_id = ref_to_id.get((act, target_sec))
            if target_id and target_id != cid:
                targets.add(target_id)
                if target_id in id_to_citation:
                    cit_targets.add(id_to_citation[target_id])

        if targets:
            graph[cid] = sorted(list(targets))
            for t in targets:
                reverse_graph.setdefault(t, []).append(cid)
                edges.append({"source": cid, "target": t, "type": "REFERENCES"})

        if cit_targets:
            citation_graph[f"{act}:{sec_norm}"] = sorted(list(cit_targets))

    # Add top incoming citations to citation_adjacency if not already present
    for target_id, sources in reverse_graph.items():
        # Find act and sec for target_id
        target_chunk = next((c for c in chunks if c["id"] == target_id), None)
        if not target_chunk:
            continue
        act = target_chunk.get("act_short")
        sec_norm = target_chunk.get("section", "").replace("-", "").replace(" ", "").upper()
        key = f"{act}:{sec_norm}"
        current = set(citation_graph.get(key, []))
        for s_id in sources[:5]:  # limit to top 5 incoming references
            if s_id in id_to_citation:
                src_cit = id_to_citation[s_id]
                current.add(src_cit)
        if current:
            citation_graph[key] = sorted(list(current))

    result = {
        "metadata": {
            "source_corpus": str(CORPUS_PATH),
            "total_nodes": len(chunks),
            "nodes_with_out_edges": len(graph),
            "total_directed_edges": len(edges),
        },
        "adjacency": graph,
        "reverse_adjacency": {k: sorted(v) for k, v in reverse_graph.items()},
        "citation_adjacency": citation_graph,
        "edges": edges,
    }

    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Graph built successfully:")
    print(f"  Nodes: {len(chunks)}")
    print(f"  Nodes with references: {len(graph)}")
    print(f"  Directed edges: {len(edges)}")
    print(f"  Provisions with citation adjacency: {len(citation_graph)}")
    print(f"  Saved to: {OUTPUT_PATH}")
    return result


if __name__ == "__main__":
    build_statutory_graph()

