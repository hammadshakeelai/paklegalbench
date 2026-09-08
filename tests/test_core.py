"""Tests for the pure functions. These are what let you refactor retrieval
without re-running the whole app to find out you broke it."""
import sys, pathlib, re
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from index import extract_references, rrf, tokenize


def test_tokenize_keeps_hyphenated_sections():
    assert "489-f" in tokenize("Section 489-F PPC")
    assert "10a" in tokenize("Article 10A")


def test_reference_article():
    assert extract_references("Article 199") == [("199", "Constitution")]


def test_reference_bare_act_suffix():
    assert ("302", "PPC") in extract_references("302 PPC")


def test_reference_abbreviated_section():
    assert ("497", "CrPC") in extract_references("u/s 497 CrPC")


def test_reference_hyphenated():
    assert ("489-F", "PPC") in extract_references("Section 489-F PPC")


def test_no_reference_in_plain_question():
    assert extract_references("can I get bail") == []


def test_rrf_rewards_agreement():
    """A doc ranked mid by both channels should beat one ranked top by only one."""
    a, b = ["x", "y"], ["z", "y"]
    scores = dict(rrf([a, b], k=1))
    assert scores["y"] > scores["x"]


def test_rrf_weights_applied():
    s = dict(rrf([["a"], ["b"]], k=60, weights=[2.0, 1.0]))
    assert s["a"] == 2 * s["b"]


def test_rrf_empty():
    assert rrf([]) == []


def test_reference_variants():
    assert ("489F", "PPC") in extract_references("489F PPC")
    assert ("10-A", "Constitution") in extract_references("Article 10-A")
    assert ("302", "PPC") in extract_references("Section 302 P.P.C.")
    assert ("497", "CrPC") in extract_references("u/s 497 Cr.P.C.")


def test_exact_channel_normalization():
    from index import Retriever, load_chunks
    r = Retriever(load_chunks(), load_dense=False)
    assert "ppc-1860-s489f" in r._exact_channel("489F PPC")
    assert "ppc-1860-s489f" in r._exact_channel("Section 489-F PPC")
    assert "const-1973-a10a" in r._exact_channel("Article 10-A")
    assert "const-1973-a10a" in r._exact_channel("Article 10A")
    assert "crpc-1898-s265k" in r._exact_channel("265K CrPC")


# --- Edge cases: empty, whitespace, and special symbols ---


def test_tokenize_empty_and_whitespace():
    assert tokenize("") == []
    assert tokenize("   ") == []
    assert tokenize(" \t \n \r ") == []


def test_tokenize_special_symbols():
    assert tokenize("!@#$%^&*()_+=-`~[]\\{}|;':\",./<>?") == []
    assert tokenize("Section @#$ 302 !!! PPC ???") == ["section", "302", "ppc"]
    assert tokenize("Section 489 / F PPC") == ["section", "489", "f", "ppc"]
    assert tokenize("Section 489—F PPC") == ["section", "489-f", "ppc"]


def test_extract_references_empty_and_whitespace():
    assert extract_references("") == []
    assert extract_references("   ") == []
    assert extract_references(" \t \n ") == []


def test_extract_references_special_symbols():
    assert extract_references("!@#$%^&*()_+") == []
    assert extract_references("??? ... !!!") == []
    assert extract_references("*** 302 PPC ***") == [("302", "PPC")]
    assert extract_references("(u/s 497 Cr.P.C.)") == [("497", "CrPC")]
    assert extract_references("[Article 199]") == [("199", "Constitution")]
    assert extract_references("Section 302, PPC.") == [("302", "PPC")]


# --- Edge cases: multiple citations in one query ---


def test_extract_references_multiple_citations_ppc_and_crpc():
    refs = extract_references("302 PPC and 497 CrPC")
    assert ("302", "PPC") in refs
    assert ("497", "CrPC") in refs
    assert len(refs) == 2

    refs_prefixed = extract_references("Section 302 PPC and Section 497 CrPC")
    assert ("302", "PPC") in refs_prefixed
    assert ("497", "CrPC") in refs_prefixed


def test_extract_references_multiple_citations_same_act():
    refs = extract_references("Article 199 and Article 10A")
    assert ("199", "Constitution") in refs
    assert ("10A", "Constitution") in refs
    assert len(refs) == 2


def test_extract_references_multiple_citations_cross_act():
    refs = extract_references("Article 199 of Constitution and 302 PPC")
    assert ("199", "Constitution") in refs
    assert ("302", "PPC") in refs
    assert len(refs) == 2

    refs2 = extract_references("Section 489-F PPC and u/s 497 CrPC")
    assert ("489-F", "PPC") in refs2
    assert ("497", "CrPC") in refs2
    assert len(refs2) == 2


def test_exact_channel_multiple_citations():
    from index import Retriever, load_chunks
    r = Retriever(load_chunks(), load_dense=False)
    exact = r._exact_channel("302 PPC and 497 CrPC")
    assert "ppc-1860-s302" in exact
    assert "crpc-1898-s497" in exact

    exact_cross = r._exact_channel("Article 199 and 302 PPC")
    assert "const-1973-a199" in exact_cross
    assert "ppc-1860-s302" in exact_cross


def test_retrieval_and_refusal_empty_and_whitespace():
    from index import Retriever, load_chunks
    r = Retriever(load_chunks(), load_dense=False)
    for q in ["", "   ", " \t\n\r "]:
        hits = r.search(q)
        assert hits == []
        assert r.refusal_reason(q, hits) == "nothing_retrieved"
        assert r.should_refuse(hits, query=q) is True


def test_retrieval_and_refusal_special_symbols():
    from index import Retriever, load_chunks
    r = Retriever(load_chunks(), load_dense=False)
    for q in ["!@#$%^&*()_+=-", "??? ... !!!"]:
        hits = r.search(q)
        assert hits == []
        assert r.refusal_reason(q, hits) == "nothing_retrieved"
        assert r.should_refuse(hits, query=q) is True


def test_retrieval_and_refusal_multiple_citations():
    from index import Retriever, load_chunks
    r = Retriever(load_chunks(), load_dense=False)
    q = "302 PPC and 497 CrPC"
    hits = r.search(q)
    assert len(hits) >= 2
    top_ids = [h.chunk.id for h in hits[:2]]
    assert "ppc-1860-s302" in top_ids
    assert "crpc-1898-s497" in top_ids
    # Multi-citation query should NOT be falsely refused
    assert r.should_refuse(hits, query=q) is False
    assert r.refusal_reason(q, hits) is None


# --- Pure function boundary & edge cases ---


def test_rrf_boundary_conditions():
    assert rrf([[], []]) == []
    res = rrf([["x", "y"], []], k=60)
    assert len(res) == 2
    assert res[0][0] == "x"
    assert res[1][0] == "y"


def test_bm25_empty_and_oov():
    from index import BM25
    bm = BM25([["punishment", "murder", "ppc"], ["cheating", "property", "ppc"]])
    assert bm.scores([]) == [0.0, 0.0]
    assert bm.scores(["unknownterm123xyz"]) == [0.0, 0.0]


# --- Vernacular statutory synonyms & filler words ---


def test_vernacular_references():
    assert ("154", "CrPC") in extract_references("how is an FIR registered")
    assert ("154", "CrPC") in extract_references("First Information Report")
    assert ("173", "CrPC") in extract_references("what is a challan")
    assert ("498", "CrPC") in extract_references("how do I get pre-arrest bail")
    assert ("498", "CrPC") in extract_references("anticipatory bail procedure")
    assert ("497", "CrPC") in extract_references("can bail be granted in a non-bailable offence")
    assert ("496", "CrPC") in extract_references("grant of bail in bailable offence")
    assert ("199", "Constitution") in extract_references("which court has writ jurisdiction in Pakistan")
    assert ("199", "Constitution") in extract_references("filing a writ petition")
    assert ("302", "PPC") in extract_references("qatl-i-amd penalty")
    assert ("302", "PPC") in extract_references("punishment for intentional murder")
    assert ("489-F", "PPC") in extract_references("what happens if my cheque bounces")
    assert ("489-F", "PPC") in extract_references("punishment for bounced cheque")
    assert ("489-F", "PPC") in extract_references("cheque dishonour case")


def test_filler_words_in_stopwords():
    from index import STOPWORDS
    assert "please" in STOPWORDS
    assert "tell" in STOPWORDS
    assert "me" in STOPWORDS
    assert "kya" in STOPWORDS
    assert "hai" in STOPWORDS
    assert "hain" in STOPWORDS


def test_stemming_inflections():
    from index import stem
    assert stem("detained") == "detain"
    assert stem("detention") == "deten"
    assert stem("producing") == "produc"
    assert stem("produced") == "produc"
    assert stem("bounces") == "bounc"


def test_exact_channel_vernacular_resolution():
    from index import Retriever, load_chunks
    r = Retriever(load_chunks(), load_dense=False)
    assert "ppc-1860-s489f" in r._exact_channel("what happens if my cheque bounces")
    assert "crpc-1898-s173" in r._exact_channel("what is a challan")
    assert "crpc-1898-s154" in r._exact_channel("how is an FIR registered")
    assert "const-1973-a199" in r._exact_channel("which court has writ jurisdiction in Pakistan")


def test_urdu_script_references():
    assert ("302", "PPC") in extract_references("دفعہ ۳۰۲ تعزیرات پاکستان")
    assert ("199", "Constitution") in extract_references("آرٹیکل ۱۹۹ آئین پاکستان")
    assert ("302", "PPC") in extract_references("قتل عمد کی سزا کیا ہے")
    assert ("497", "CrPC") in extract_references("ضمانت کے بنیادی اصول")
    assert ("154", "CrPC") in extract_references("ایف آئی آر درج کروانا")
    assert ("173", "CrPC") in extract_references("پولیس چالان")
    assert ("489-F", "PPC") in extract_references("چیک باؤنس کا مقدمہ")
    assert ("199", "Constitution") in extract_references("ہائی کورٹ میں رٹ پٹیشن")


def test_bounded_section_ranges():
    refs_const = extract_references("Articles 8 to 10 of Constitution")
    sections = [s for s, a in refs_const]
    assert "8" in sections and "9" in sections and "10" in sections

    refs_ppc = extract_references("Sections 300 to 302 PPC")
    ppc_secs = [s for s, a in refs_ppc]
    assert "300" in ppc_secs and "301" in ppc_secs and "302" in ppc_secs


def test_graph_context_expansion():
    from index import Retriever, load_chunks, RetrievalConfig
    r = Retriever(load_chunks(), load_dense=False)
    # Search with graph expansion enabled
    cfg = RetrievalConfig(use_graph_context=True, top_k=2)
    hits = r.search("302 PPC", cfg)
    assert len(hits) >= 1
    assert hits[0].chunk.citation() == "PPC Section 302"
    # Ensure graph adjacency helper executes gracefully
    adj = r._get_graph_adjacency()
    assert isinstance(adj, dict)


def test_law_gat_benchmark():
    import law_gat_eval
    questions = law_gat_eval.load_dataset()
    assert len(questions) == 30
    from index import Retriever, load_chunks, RetrievalConfig
    r = Retriever(load_chunks(), load_dense=False)
    stats = law_gat_eval.evaluate_retrieval(r, questions, RetrievalConfig())
    assert stats["overall"]["recall@1"] >= 0.90
    assert stats["overall"]["recall@5"] >= 0.95
    assert stats["overall"]["mrr"] >= 0.95
    assert stats["overall"]["false_refusal_rate"] == 0.0


def test_clean_chunks_corpus():
    import json
    from pathlib import Path
    chunks_path = Path("chunks.json")
    assert chunks_path.exists(), "chunks.json missing"
    chunks = json.load(open(chunks_path, encoding="utf-8"))["chunks"]
    assert len(chunks) >= 1200, f"Expected >= 1200 chunks, got {len(chunks)}"

    # Ensure no TOC dotted lines exist in any chunk
    dotted = [c for c in chunks if re.search(r"\.{4,}", c["marginal_note"]) or re.search(r"\.{4,}", c["text"][:100])]
    assert len(dotted) == 0, f"Found {len(dotted)} dotted TOC artifacts in chunks.json"

    # Ensure all IDs are uniquely defined
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk IDs found in chunks.json"

    # Ensure essential statutory provisions are present and populated with substantive text
    by_id = {c["id"]: c for c in chunks}
    required = [
        "const-1973-a8", "const-1973-a9", "const-1973-a10", "const-1973-a10a",
        "const-1973-a184", "const-1973-a189", "const-1973-a199", "const-1973-a228",
        "ppc-1860-s300", "ppc-1860-s302", "ppc-1860-s420", "ppc-1860-s489f",
        "crpc-1898-s154", "crpc-1898-s173", "crpc-1898-s496", "crpc-1898-s497", "crpc-1898-s498"
    ]
    for req in required:
        assert req in by_id, f"Missing required provision {req}"
        assert len(by_id[req]["text"]) > 50, f"Provision {req} has insufficient substantive text"


def test_legal_uqa_benchmark_mapping():
    from eval import load_legal_uqa
    cases = load_legal_uqa("validation")
    assert len(cases) == 124, f"Expected 124 validation cases, got {len(cases)}"
    for c in cases:
        assert c.get("gold_ids"), f"Missing gold_ids for case {c.get('id')}"
        assert c["gold_ids"][0].startswith("const-1973-a"), f"Invalid gold_id format: {c['gold_ids'][0]}"


def test_expand_multilingual_query():
    from index import expand_multilingual_query

    # English query should remain unmodified
    eng_q = "What is the penalty for murder under 302 PPC?"
    assert expand_multilingual_query(eng_q) == eng_q

    # Urdu queries should append corresponding English statutory terminology
    urdu_q1 = "کیا قومی اسمبلی میں خواتین کے لیے مخصوص نشستیں ہیں؟"
    expanded1 = expand_multilingual_query(urdu_q1)
    assert "national assembly" in expanded1
    assert "reserved seats" in expanded1

    urdu_q2 = "مقامی حکومتوں کے اختیارات اور ذمہ داریاں"
    expanded2 = expand_multilingual_query(urdu_q2)
    assert "local government" in expanded2
    assert "devolve" in expanded2


def test_urdu_conceptual_retrieval():
    from index import Retriever, load_chunks, RetrievalConfig
    r = Retriever(load_chunks(), load_dense=False)
    cfg = RetrievalConfig()

    # Query in pure Urdu script asking about fair trial
    hits_trial = r.search("منصفانہ ٹرائل کا حق اور قانونی تقاضے", cfg)
    assert len(hits_trial) > 0
    assert hits_trial[0].chunk.citation() == "Constitution, Article 10A"

    # Query in pure Urdu script asking about right to information
    hits_info = r.search("معلومات تک رسائی کا بنیادی حق", cfg)
    assert len(hits_info) > 0
    assert hits_info[0].chunk.citation() == "Constitution, Article 19A"

    # If full chunks.json is present, also test local government
    from pathlib import Path
    if Path("chunks.json").exists():
        r_full = Retriever(load_chunks("chunks.json"), load_dense=False)
        hits_local = r_full.search("مقامی حکومتوں کے اختیارات اور ذمہ داریاں", cfg)
        assert len(hits_local) > 0
        assert "140A" in hits_local[0].chunk.citation() or "Local Government" in hits_local[0].chunk.marginal_note


def test_stanford_hallucination_benchmark():
    import stanford_eval
    from index import Retriever, load_chunks
    r = Retriever(load_chunks(), load_dense=False)
    report = stanford_eval.run_evaluation(r, mode="mock")

    assert report["citation_metrics"]["citation_grounding_rate"] == 1.0
    assert report["citation_metrics"]["citation_hallucination_rate"] == 0.0
    assert report["refusal_metrics"]["premise_verification_rate"] == 1.0
    assert report["safety_metrics"]["disclaimer_compliance_rate"] == 1.0


def test_formal_citation():
    from index import Chunk
    c_ppc = Chunk(id="ppc-1860-s302", act="Pakistan Penal Code, 1860", act_short="PPC", year=1860,
                  chapter="", section="302", section_label="Section 302", marginal_note="Punishment of qatl-i-amd",
                  text="Whoever commits qatl-i-amd shall...", jurisdiction="federal", status="in_force", source_url="")
    assert c_ppc.formal_citation() == "Section 302, Pakistan Penal Code, 1860 (Act XLV of 1860)"

    c_crpc = Chunk(id="crpc-1898-s497", act="Code of Criminal Procedure, 1898", act_short="CrPC", year=1898,
                   chapter="", section="497", section_label="Section 497", marginal_note="When bail may be taken",
                   text="When any person accused of any non-bailable offence...", jurisdiction="federal", status="in_force", source_url="")
    assert c_crpc.formal_citation() == "Section 497, Code of Criminal Procedure, 1898 (Act V of 1898)"

    c_const = Chunk(id="const-1973-a199", act="Constitution of the Islamic Republic of Pakistan, 1973", act_short="Constitution", year=1973,
                    chapter="", section="199", section_label="Article 199", marginal_note="Jurisdiction of High Court",
                    text="Subject to the Constitution, a High Court may...", jurisdiction="federal", status="in_force", source_url="")
    assert c_const.formal_citation() == "Article 199, Constitution of the Islamic Republic of Pakistan, 1973"


def test_statutory_definitions():
    from index import extract_references
    assert ("4", "CrPC") in extract_references("what is a cognizable offence")
    assert ("4", "CrPC") in extract_references("definition of non-cognizable offence")
    assert ("299", "PPC") in extract_references("what is culpable homicide")
    assert ("4", "CrPC") in extract_references("قابل دست اندازی جرم کی تعریف")








