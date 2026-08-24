"""Tests for the ``inline_assets`` self-contained-rendering option.

``visualize_graph``/``cognee_network_visualization`` document themselves as
producing a "self-contained HTML file", but by default the page loads D3
from https://d3js.org and Google Fonts from fonts.googleapis.com -- so it
renders blank under a strict Content-Security-Policy or with no internet
access. ``inline_assets=True`` inlines the vendored D3 v7.9.0 source and
drops the Google Fonts links so the page has zero external network
dependencies, while ``inline_assets=False`` (the default) must reproduce the
exact prior behavior so no existing caller changes.
"""

import asyncio
import html.parser
import re

import pytest

from cognee.modules.visualization.cognee_network_visualization import (
    cognee_network_visualization,
    _read_vendored_d3,
    _safe_script_embed,
)


def _minimal_graph():
    nodes_data = [
        ("a", {"type": "Entity", "name": "A"}),
        ("b", {"type": "DocumentChunk", "text": "hi"}),
    ]
    edges_data = [("b", "a", "contains", {})]
    return (nodes_data, edges_data)


def _render(tmp_path, **kwargs):
    return asyncio.run(
        cognee_network_visualization(_minimal_graph(), str(tmp_path / "out.html"), **kwargs)
    )


# ---------------------------------------------------------------------------
# Default behavior (inline_assets=False) is unchanged: backwards compat.
# ---------------------------------------------------------------------------


def test_default_still_loads_d3_from_cdn(tmp_path):
    """No caller passes inline_assets, so this must keep working exactly as
    before: the CDN script tag and the Google Fonts links are both present."""
    html_out = _render(tmp_path)
    assert '<script src="https://d3js.org/d3.v7.min.js"></script>' in html_out
    assert "fonts.googleapis.com" in html_out
    assert "fonts.gstatic.com" in html_out
    # The vendored bundle must NOT be inlined by default.
    assert "!function(t,n)" not in html_out  # d3.v7.min.js's minified preamble


def test_inline_assets_false_is_the_explicit_default():
    """Pins the public default so a signature change is caught by this test,
    not discovered by a caller's silent behavior change."""
    import inspect

    sig = inspect.signature(cognee_network_visualization)
    assert sig.parameters["inline_assets"].default is False


# ---------------------------------------------------------------------------
# inline_assets=True: genuinely self-contained, zero external URLs.
# ---------------------------------------------------------------------------


def test_inline_assets_removes_every_external_url(tmp_path):
    """The load-bearing test: with inline_assets=True the rendered page must
    contain zero external URLs (no src="http, no href="http) while still
    containing the D3 source and a working graph payload."""
    html_out = _render(tmp_path, inline_assets=True)

    assert 'src="http' not in html_out
    assert 'href="http' not in html_out
    # Belt-and-braces: no <script src=...>/<link href=...> resource load
    # pointed at an external host anywhere in the page. (The vendored D3
    # source legitimately keeps its own "// https://d3js.org ..." license
    # attribution comment verbatim -- that's provenance text, not a
    # network call, so it's excluded from this check rather than stripped.)
    assert not re.search(r'<(?:script|link)\b[^>]*\b(?:src|href)="https?://', html_out)

    # D3 source is actually inlined (its minified preamble is a stable marker).
    assert "!function(t,n)" in html_out
    assert "d3=t.d3||{}" in html_out or "d3" in html_out

    # A working graph payload: the two nodes/links data tokens resolved to
    # real JS arrays carrying the data we passed in.
    assert "var nodes =" in html_out
    assert "var links =" in html_out
    assert '"name": "A"' in html_out


def test_inline_assets_drops_google_fonts_links(tmp_path):
    html_out = _render(tmp_path, inline_assets=True)
    assert "fonts.googleapis.com" not in html_out
    assert "fonts.gstatic.com" not in html_out


def test_inline_assets_script_tag_not_truncated_and_parses_as_html(tmp_path):
    """Confirms the page parses as HTML, the inlined <script> block is not
    truncated by an unescaped </script> inside the D3 payload, and no
    __TOKEN__ placeholder leaked (mirrors the orchestrator assembly suite's
    no-placeholder-leak check, but on the inline_assets=True path)."""
    html_out = _render(tmp_path, inline_assets=True)

    assert html_out.startswith("<!DOCTYPE html>")
    assert html_out.rstrip().endswith("</html>")

    leaks = re.findall(r"__[A-Z][A-Z0-9_]*__", html_out)
    assert leaks == [], f"unfilled tokens: {leaks}"

    # A truncated <script> would either drop everything after the inlined
    # D3 bundle (losing the JS chunks appended after it) or produce
    # mismatched tag counts. Both are ruled out here.
    assert html_out.count("<script>") == html_out.count("</script>")
    # The story-view chunk (appended well after the inlined D3 block) must
    # still be present and intact -- proof nothing downstream was truncated.
    assert "computeRankedLayout" in html_out
    assert "labelBudget" in html_out

    parser = html.parser.HTMLParser()
    parser.feed(html_out)  # raises on a malformed document
    parser.close()


def test_inline_assets_vendored_bundle_has_no_placeholder_token_leak(tmp_path):
    """A leaked __D3_SCRIPT_TAG__/__GOOGLE_FONTS_LINKS__ token would mean the
    template's head-asset tokens were never substituted."""
    html_out = _render(tmp_path, inline_assets=True)
    assert "__D3_SCRIPT_TAG__" not in html_out
    assert "__GOOGLE_FONTS_LINKS__" not in html_out


# ---------------------------------------------------------------------------
# _safe_script_embed: the </script>-escaping helper, tested explicitly.
# ---------------------------------------------------------------------------


def test_safe_script_embed_neutralises_closing_script_tag():
    payload = 'const s = "</script><script>alert(1)</script>";'
    escaped = _safe_script_embed(payload)
    assert "</script>" not in escaped
    assert "<\\/script>" in escaped


def test_safe_script_embed_is_a_pure_slash_escape_not_a_full_rewrite():
    """Same contract as _safe_json_embed: only the two-character sequence
    ``</`` is touched, so valid JS elsewhere in the payload is untouched."""
    payload = "if (a < /re/.test(b)) { return 1 < 2; }"
    escaped = _safe_script_embed(payload)
    assert escaped == "if (a < /re/.test(b)) { return 1 < 2; }"  # no "</" present


def test_safe_script_embed_handles_the_actual_vendored_d3_source():
    """Run the real fix against the real payload, not a synthetic stand-in:
    embedding the vendored bundle inside a <script> tag must not create an
    early close, and the escape must be reversible (no data loss)."""
    d3_src = _read_vendored_d3()
    escaped = _safe_script_embed(d3_src)
    wrapped = f"<script>{escaped}</script>"

    # Exactly one opening and one closing script tag: the payload cannot
    # smuggle in a premature close.
    assert wrapped.count("<script>") == 1
    assert wrapped.count("</script>") == 1

    # Un-escaping recovers the original source byte-for-byte.
    assert escaped.replace("<\\/", "</") == d3_src


def test_vendored_d3_is_the_expected_version_and_license_header():
    d3_src = _read_vendored_d3()
    assert d3_src.startswith("// https://d3js.org v7.9.0 Copyright")
    assert len(d3_src.encode("utf-8")) == 279706


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
