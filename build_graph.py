import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).parent
CORPUS_PATH = Path(os.environ.get("PLB_CORPUS_PATH", "chunks.json" if Path("chunks.json").exists() else "seed_corpus.json"))
OUTPUT_PATH = ROOT / "statute_graph.json"

SEC_PAT = re.compile(r"\b(?:section|article|sec\.?|art\.?)\s*([0-9]+(?:[-–]?[A-Za-z])?)\b", re.I)

def build_statutory_graph():
    raw = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    chunks = raw["chunks"] if isinstance(raw, dict) else raw

    ref_to_id = {}
    valid_citations = set()
    for c in chunks:
        act = c.get("act_short")
        sec = c.get("section", "").replace("-", "").replace(" ", "").upper()
        cid = c.get("id")
        if act and sec and cid:
            ref_to_id[(act, sec)] = cid
            valid_citations.add((act, sec))

    graph = {}
    reverse_graph = {}
    citation_graph = {}
    edges = []

    for c in chunks:
        cid = c["id"]
        act = c.get("act_short")
        sec_norm = c.get("section", "").replace("-", "").replace(" ", "").upper()
        text = c.get("text", "")
        matches = SEC_PAT.findall(text)
        targets = set()
        cit_targets = set()
        for m in matches:
            target_sec = m.replace("-", "").replace(" ", "").upper()
            target_id = ref_to_id.get((act, target_sec))
            if target_id and target_id != cid:
                targets.add(target_id)
            if (act, target_sec) in valid_citations and target_sec != sec_norm:
                cit_targets.add(f"{act} Section {m}" if act != "Constitution" else f"Constitution, Article {m}")

        if targets:
            graph[cid] = sorted(list(targets))
            for t in targets:
                reverse_graph.setdefault(t, []).append(cid)
                edges.append({"source": cid, "target": t, "type": "REFERENCES"})

        if cit_targets:
            citation_graph[f"{act}:{sec_norm}"] = sorted(list(cit_targets))

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
    print(f"  Saved to: {OUTPUT_PATH}")
    return result

if __name__ == "__main__":
    build_statutory_graph()
