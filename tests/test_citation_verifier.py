"""Tests for deterministic citation verification and post-processing in llm.py."""
import pytest
from index import Hit, Chunk
from llm import verify_and_clean_citations


@pytest.fixture
def mock_hits():
    chunk199 = Chunk(
        id="const-1973-a199",
        act="Constitution of the Islamic Republic of Pakistan, 1973",
        act_short="Constitution",
        year=1973,
        chapter="Judicature",
        section="199",
        section_label="Article 199",
        marginal_note="Jurisdiction of High Court",
        text="Subject to the Constitution, a High Court may make an order...",
        jurisdiction="PK",
        status="in_force",
        source_url="http://pakistancode.gov.pk",
        verified=True,
    )
    chunk302 = Chunk(
        id="ppc-1860-s302",
        act="Pakistan Penal Code",
        act_short="PPC",
        year=1860,
        chapter="Offences Affecting Life",
        section="302",
        section_label="Section 302",
        marginal_note="Punishment of qatl-i-amd",
        text="Whoever commits qatl-i-amd shall be punished with death as qisas...",
        jurisdiction="PK",
        status="in_force",
        source_url="http://pakistancode.gov.pk",
        verified=True,
    )
    return [
        Hit(chunk=chunk199, score=1.0, channels=["exact"]),
        Hit(chunk=chunk302, score=0.9, channels=["exact"]),
    ]


def test_verifier_strips_malicious_urls(mock_hits):
    raw = "Murder is punishable by death [PPC Section 302](https://evil.com/phish)."
    cleaned, log = verify_and_clean_citations(raw, mock_hits)
    assert "https://evil.com" not in cleaned
    assert "[PPC Section 302]" in cleaned


def test_verifier_flags_unretrieved_valid_statute(mock_hits):
    single_hit = [mock_hits[0]]
    raw = "Writs are under [Constitution, Article 199], and bail is under [CrPC Section 498]."
    cleaned, log = verify_and_clean_citations(raw, single_hit)
    assert "[Constitution, Article 199]" in cleaned
    assert "[Unverified: CrPC 498]" in cleaned
    assert any(item["status"] == "UNGROUNDED_OR_FABRICATED" for item in log)


def test_verifier_flags_out_of_bounds_fabricated_section(mock_hits):
    raw = "Penalty is prescribed under [PPC Section 999]."
    cleaned, log = verify_and_clean_citations(raw, mock_hits)
    assert "[Unverified: PPC 999]" in cleaned


def test_verifier_supports_subsections(mock_hits):
    raw = "Relief is granted under [Constitution, Article 199(1)(a)(i)]."
    cleaned, log = verify_and_clean_citations(raw, mock_hits)
    assert "[Constitution, Article 199]" in cleaned
    assert "Unverified" not in cleaned


def test_verifier_homoglyph_normalization(mock_hits):
    # Cyrillic 'е', 'о', 'Р', 'С'
    raw = "Punishment under [Sеctiоn 302 РРС]."
    cleaned, log = verify_and_clean_citations(raw, mock_hits)
    assert "[PPC Section 302]" in cleaned
    assert "Unverified" not in cleaned


def test_verifier_preserves_non_statutory_brackets(mock_hits):
    raw = "According to legal precedent [1], the petition succeeds [Constitution, Article 199]."
    cleaned, log = verify_and_clean_citations(raw, mock_hits)
    assert "[1]" in cleaned
    assert "[Constitution, Article 199]" in cleaned
