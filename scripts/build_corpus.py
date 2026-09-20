#!/usr/bin/env python3
"""Assemble the tokenizer differential corpus (PLAN.md task 4.5.3).

Task 4.5.3 asks for "~100k lines of real multilingual corpus" run through both the
Python oracle and the Go tokenizer, with mismatches attributed per stage. The plan
names no source, so this script defines one -- and records it, because 4.5.4 has to
say what was run and a reviewer has to be able to re-run it.

Four streams, none of which is redundant:

  tatoeba  ~100k sentences across ~429 languages, sampled per language from the
           Tatoeba per-language exports. This is the "real corpus" half: natural
           word boundaries, contractions, realistic whitespace and punctuation in
           every script the multilingual checkpoint claims to cover.

  probes   Generated. Natural prose cannot reach a codepoint that was assigned
           after the text was written, and that is precisely the defect class R1
           names: Go 1.26's `unicode` tables are Unicode 15.0.0 while the UCD on a
           current distribution is 15.1.0 and Oniguruma (behind HF tokenizers)
           pins its own. So: every block boundary, every general-category
           transition, the whitespace and format characters the ByteLevel
           predicates classify, combining-mark sequences, and -- where the
           distribution ships it -- the columns of UCD NormalizationTest.txt,
           which is a purpose-built NFC conformance corpus.

  vocab    The multilingual checkpoint's own 256k vocabulary, decoded back to
           text. Every one of these is a string the model was trained to emit, so
           a tokenizer that cannot round-trip them is wrong about its own dictionary.

  locale   gettext catalogues from /usr/share/locale. A different register from
           prose: dense in format specifiers, markup, mnemonics and punctuation
           runs. Read with the stdlib `gettext` parser, so no subprocess.

Output is JSONL -- `{"src": ..., "text": ...}` -- and not plain text, because the
probe stream deliberately contains newlines, tabs and lone surrogated bytes, and a
line-oriented format would mangle exactly the inputs worth testing.

Everything lands under build/ (gitignored). Nothing here is checked in: the corpus
is a one-off run's input, and testdata/ stays the reviewed 103-case corpus.

Usage:
    .venv-ref/bin/python scripts/build_corpus.py --out build/corpus
"""

from __future__ import annotations

import argparse
import bz2
import gettext
import hashlib
import json
import random
import re
import sys
import unicodedata
import urllib.error
import urllib.request
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

# The same seed scripts/dump_python_parity.py:49 uses. Every sample here is drawn
# from a seeded Random, so two runs of this script select the same lines.
SEED = 0

TATOEBA_BASE = "https://downloads.tatoeba.org/exports/per_language"
# Tatoeba serves a plain Apache index; the language directories are the only
# three-letter hrefs on it.
TATOEBA_INDEX_RE = re.compile(r'href="([a-z]{3})/"')

USER_AGENT = "go-laya-parity-harness/0.1 (PLAN.md task 4.5.3)"

UCD_DIR = Path("/usr/share/unicode")
LOCALE_DIR = Path("/usr/share/locale")

# Per-stream line budgets. Tatoeba carries the "~100k lines" the task asks for;
# the other three are additive coverage, not a substitute for it.
DEFAULT_BUDGETS = {
    "tatoeba": 100_000,
    "probes": 20_000,
    "vocab": 20_000,
    "locale": 20_000,
    "added": 20_000,
}


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# Stream A: Tatoeba
# --------------------------------------------------------------------------- #


def fetch(url: str, timeout: int = 120) -> tuple[bytes, dict[str, str]]:
    """GET url, returning the body and the response headers we record."""
    # Fixed https:// base, no user-controlled scheme: TATOEBA_BASE is a constant
    # and only the three-letter language code varies.
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
        headers = {
            k.lower(): resp.headers.get(k, "") for k in ("Last-Modified", "Content-Length", "ETag")
        }
    return body, headers


def tatoeba_languages() -> list[str]:
    body, _ = fetch(f"{TATOEBA_BASE}/")
    langs = sorted(set(TATOEBA_INDEX_RE.findall(body.decode("utf-8", "replace"))))
    if not langs:
        raise RuntimeError(f"no language directories parsed from {TATOEBA_BASE}/")
    return langs


CACHE_PER_LANG = 2000


def tatoeba_one(lang: str, cache: Path) -> tuple[list[str], dict[str, Any]]:
    """Sample up to CACHE_PER_LANG sentences for one language, caching the sample.

    The archive itself is never kept. The per-language exports run from a few
    hundred bytes to tens of megabytes and we want a few hundred lines from each,
    so caching the sample rather than the source keeps a full corpus build to a
    few megabytes on disk.

    The cap is deliberately far above any single language's share of the budget:
    the allocation in stream_tatoeba is water-filling, so a language's share
    depends on how many *other* languages ran dry, and a cache sized to one
    particular budget would force a re-download every time the budget moved.
    """
    sample_path = cache / f"{lang}.txt"
    meta_path = cache / f"{lang}.json"
    if sample_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text("utf-8"))
        # A cache written under a smaller cap is only reusable when it already
        # holds everything that language has.
        if meta.get("cache_per_lang", 0) >= CACHE_PER_LANG or meta.get("lines", 0) >= meta.get(
            "available", 0
        ):
            lines = sample_path.read_text("utf-8").split("\n")
            return [x for x in lines if x], meta

    url = f"{TATOEBA_BASE}/{lang}/{lang}_sentences.tsv.bz2"
    try:
        body, headers = fetch(url)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as err:
        return [], {"lang": lang, "url": url, "error": str(err), "lines": 0}

    try:
        text = bz2.decompress(body).decode("utf-8", "replace")
    except (OSError, EOFError) as err:
        return [], {"lang": lang, "url": url, "error": f"bz2: {err}", "lines": 0}

    # id \t iso3 \t sentence
    sentences = []
    for row in text.split("\n"):
        parts = row.split("\t")
        if len(parts) >= 3 and parts[2]:
            sentences.append(parts[2])

    rng = random.Random(f"{SEED}:{lang}")
    picked = (
        sentences if len(sentences) <= CACHE_PER_LANG else rng.sample(sentences, CACHE_PER_LANG)
    )
    picked.sort()

    meta = {
        "lang": lang,
        "url": url,
        "available": len(sentences),
        "lines": len(picked),
        "cache_per_lang": CACHE_PER_LANG,
        "source_bytes": len(body),
        "source_sha256": hashlib.sha256(body).hexdigest(),
        "last_modified": headers.get("last-modified", ""),
    }
    cache.mkdir(parents=True, exist_ok=True)
    sample_path.write_text("\n".join(picked), "utf-8")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), "utf-8")
    return picked, meta


def allocate(pools: dict[str, int], budget: int) -> dict[str, int]:
    """Water-fill `budget` across per-language pool sizes.

    An equal quota wastes the head of the distribution: most of Tatoeba's 429
    languages hold a handful of sentences, so a flat share leaves the budget
    unmet while eng/deu/rus sit on thousands unused. Handing each round's
    remainder back to the languages that still have material spends the whole
    budget while keeping the long tail fully represented -- every language
    contributes everything it has before any language contributes a second share.
    """
    alloc = dict.fromkeys(pools, 0)
    remaining = budget
    while remaining > 0:
        active = [lg for lg, n in pools.items() if alloc[lg] < n]
        if not active:
            break
        share = max(1, remaining // len(active))
        progress = False
        for lang in sorted(active):
            if remaining <= 0:
                break
            take = min(share, pools[lang] - alloc[lang], remaining)
            if take > 0:
                alloc[lang] += take
                remaining -= take
                progress = True
        if not progress:
            break
    return alloc


def stream_tatoeba(budget: int, cache: Path, workers: int) -> tuple[list[tuple[str, str]], dict]:
    langs = tatoeba_languages()
    log(f"tatoeba: {len(langs)} languages, cache cap {CACHE_PER_LANG}/language")

    pools: dict[str, list[str]] = {}
    errors: dict[str, str] = {}
    shas: dict[str, str] = {}
    available: dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(lambda lg: tatoeba_one(lg, cache), langs)
        for lang, (lines, meta) in zip(langs, results, strict=True):
            if meta.get("error"):
                errors[lang] = meta["error"]
                continue
            pools[lang] = lines
            available[lang] = meta.get("available", len(lines))
            if meta.get("source_sha256"):
                shas[lang] = meta["source_sha256"]

    alloc = allocate({lg: len(v) for lg, v in pools.items()}, budget)

    out: list[tuple[str, str]] = []
    per_lang: dict[str, int] = {}
    for lang in sorted(pools):
        n = alloc[lang]
        if n:
            per_lang[lang] = n
            out.extend((f"tatoeba:{lang}", s) for s in pools[lang][:n])

    info = {
        "source": "Tatoeba per-language sentence exports",
        "url": f"{TATOEBA_BASE}/",
        "licence": "CC-BY 2.0 FR (not redistributed; build/ is gitignored)",
        "languages": len(per_lang),
        "cache_per_language": CACHE_PER_LANG,
        "allocation": "water-filling across per-language pools",
        "lines": len(out),
        "errors": errors,
        "per_language": dict(sorted(per_lang.items())),
        "available_per_language": dict(sorted(available.items())),
        "source_sha256": dict(sorted(shas.items())),
    }
    return out, info


# --------------------------------------------------------------------------- #
# Stream B: generated Unicode probes
# --------------------------------------------------------------------------- #

MAX_CP = 0x110000


def interesting_codepoints() -> list[int]:
    """Codepoints where a table disagreement would show: every boundary.

    Walking all 1.1M codepoints and keeping the ones whose category, or
    whose is-space/is-alnum classification, differs from their predecessor
    yields every edge of every range Go's `unicode` tables and Oniguruma's
    could disagree about -- which is a superset of the block boundaries and
    far smaller than the full space.
    """
    picked: list[int] = []
    prev = None
    for cp in range(MAX_CP):
        if 0xD800 <= cp <= 0xDFFF:  # surrogates are not representable in UTF-8
            continue
        ch = chr(cp)
        cat = unicodedata.category(ch)
        key = (cat, ch.isspace(), ch.isprintable())
        if key != prev:
            picked.append(cp)
            if cp > 0:
                picked.append(cp - 1)
            prev = key
    return sorted(set(c for c in picked if not 0xD800 <= c <= 0xDFFF))


def normalization_test_lines() -> Iterator[tuple[str, str]]:
    """The UCD NFC conformance corpus, if the distribution ships it."""
    for name in ("NormalizationTest.txt", "NormalizationTest.txt.bz2"):
        path = UCD_DIR / name
        if not path.exists():
            continue
        raw = bz2.decompress(path.read_bytes()) if path.suffix == ".bz2" else path.read_bytes()
        for row in raw.decode("utf-8", "replace").split("\n"):
            row = row.split("#", 1)[0].strip()
            if not row or row.startswith("@"):
                continue
            for col in row.rstrip(";").split(";"):
                cps = [int(x, 16) for x in col.split() if x]
                if cps and not any(0xD800 <= c <= 0xDFFF for c in cps):
                    yield f"probes:nfc:{path.name}", "".join(chr(c) for c in cps)
        return


def stream_probes(budget: int) -> tuple[list[tuple[str, str]], dict]:
    out: list[tuple[str, str]] = []

    cps = interesting_codepoints()
    for cp in cps:
        ch = chr(cp)
        # Bare, and in the three positions the ByteLevel give-back rule
        # (bytelevel.go:126) and Metaspace's prepend treat differently.
        out.append(("probes:bare", ch))
        out.append(("probes:inword", f"a{ch}b"))
        out.append(("probes:lead", f"{ch}ab"))
        out.append(("probes:run", f"a{ch}{ch}b"))

    # Combining sequences: a base plus each combining mark, which is where an NFC
    # table difference turns into a different token rather than a different string.
    bases = "aeiouAEIOUकあ"
    marks = [cp for cp in cps if unicodedata.category(chr(cp)).startswith("M")]
    for base in bases:
        for cp in marks:
            out.append(("probes:combining", base + chr(cp)))

    out.extend(normalization_test_lines())

    rng = random.Random(SEED)
    seen = {t for _, t in out}
    info_total = len(out)
    if len(out) > budget:
        out = rng.sample(out, budget)

    info = {
        "source": "generated Unicode probes + UCD NormalizationTest",
        "ucd_dir": str(UCD_DIR),
        "category_boundaries": len(cps),
        "combining_marks": len(marks),
        "generated": info_total,
        "distinct_texts": len(seen),
        "lines": len(out),
    }
    return out, info


# --------------------------------------------------------------------------- #
# Stream C: the multilingual vocabulary
# --------------------------------------------------------------------------- #


def stream_vocab(budget: int, models_root: Path) -> tuple[list[tuple[str, str]], dict]:
    path = models_root / "laya" / "multilingual" / "tokenizer" / "tokenizer.json"
    if not path.exists():
        return [], {
            "source": "multilingual vocabulary",
            "path": str(path),
            "lines": 0,
            "error": "not found",
        }

    doc = json.loads(path.read_text("utf-8"))
    vocab = list(doc["model"]["vocab"])

    texts = []
    for tokstr in vocab:
        # Metaspace: the checkpoint spells a leading space U+2581. Feeding the
        # raw token back in would test the wrong string, so undo the substitution.
        text = tokstr.replace("▁", " ")
        if text and not (text.startswith("<") and text.endswith(">")):
            texts.append(text)

    rng = random.Random(SEED)
    picked = texts if len(texts) <= budget else rng.sample(texts, budget)
    picked.sort()

    info = {
        "source": "multilingual checkpoint vocabulary, U+2581 decoded back to space",
        "path": str(path),
        "vocab_size": len(vocab),
        "eligible": len(texts),
        "lines": len(picked),
    }
    return [("vocab:multilingual", t) for t in picked], info


# --------------------------------------------------------------------------- #
# Stream D: gettext catalogues
# --------------------------------------------------------------------------- #


def stream_locale(budget: int) -> tuple[list[tuple[str, str]], dict]:
    if not LOCALE_DIR.is_dir():
        return [], {
            "source": "gettext catalogues",
            "path": str(LOCALE_DIR),
            "lines": 0,
            "error": "not found",
        }

    catalogues = sorted(LOCALE_DIR.glob("*/LC_MESSAGES/*.mo"))
    rng = random.Random(SEED)

    by_locale: dict[str, set[str]] = {}
    for mo in catalogues:
        loc = mo.parts[-3]
        try:
            with mo.open("rb") as fh:
                # _catalog is private but it is the only way to enumerate a
                # catalogue; the public surface needs a msgid to look one up.
                # getattr so a Python that renames it degrades to zero lines
                # rather than taking the whole corpus build down.
                cat = getattr(gettext.GNUTranslations(fh), "_catalog", {})
        except (OSError, ValueError, UnicodeDecodeError, IndexError, KeyError):
            # A handful of shipped catalogues carry a malformed Plural-Forms
            # header, which the stdlib parser trips over with an IndexError.
            # One unreadable catalogue is not a reason to lose the stream.
            continue
        bucket = by_locale.setdefault(loc, set())
        for value in cat.values():
            if isinstance(value, str) and value:
                bucket.add(value)

    locales = sorted(by_locale)
    quota = max(1, budget // max(1, len(locales)))
    out: list[tuple[str, str]] = []
    per_locale: dict[str, int] = {}
    for loc in locales:
        pool = sorted(by_locale[loc])
        picked = pool if len(pool) <= quota else rng.sample(pool, quota)
        picked.sort()
        per_locale[loc] = len(picked)
        out.extend((f"locale:{loc}", s) for s in picked)

    info = {
        "source": "gettext catalogues under /usr/share/locale, read with stdlib gettext",
        "path": str(LOCALE_DIR),
        "catalogues": len(catalogues),
        "locales": len(locales),
        "quota_per_locale": quota,
        "lines": len(out),
        "per_locale": per_locale,
    }
    return out, info


# --------------------------------------------------------------------------- #
# Stream E: added tokens in context
# --------------------------------------------------------------------------- #

# Natural corpus text does not contain "[MASK]" or "<unused0>", so a corpus built
# only from prose exercises the added-token stage exactly zero times -- verified
# by sabotage: disabling lstrip in tokenizer/added.go left a 2000-line prose
# differential entirely green. Since that stage is one of the five 4.5.3 names,
# it gets a stream of its own.
#
# Both checkpoints' tokens go into both runs on purpose. A content that is an
# added token on English and ordinary text on multilingual (and the reverse) is
# the asymmetry PLAN.md task 4.4.14 found, and it is only reachable by feeding
# each checkpoint the other's vocabulary too.
ADDED_CONTEXTS = (
    "{t}",
    "a{t}b",
    " {t}",
    "{t} ",
    "  {t}",
    "{t}  ",
    "   {t}   ",
    "x {t} y",
    "x\t\t{t}",
    "x\n{t}\n y",
    "{t}{t}",
    "{t} {t}",
    "\u00a0{t}\u00a0",
    "\u200b{t}\u200b",
    "x{t}",
    "{t}x",
)


def added_contents(models_root: Path) -> dict[str, list[str]]:
    """Every added token of every checkpoint, by checkpoint name."""
    out: dict[str, list[str]] = {}
    for name, sub in (("english", ""), ("multilingual", "multilingual")):
        path = models_root / "laya" / sub / "tokenizer" / "tokenizer.json"
        if not path.exists():
            continue
        doc = json.loads(path.read_text("utf-8"))
        out[name] = sorted({t["content"] for t in doc.get("added_tokens", [])})
    return out


def stream_added(budget: int, models_root: Path) -> tuple[list[tuple[str, str]], dict]:
    by_ckpt = added_contents(models_root)
    if not by_ckpt:
        return [], {"source": "added tokens in context", "lines": 0, "error": "no checkpoints"}

    contents = sorted({c for v in by_ckpt.values() for c in v})
    rows: list[tuple[str, str]] = []
    for content in contents:
        for ctx in ADDED_CONTEXTS:
            rows.append(("added:ctx", ctx.format(t=content)))

    rng = random.Random(SEED)
    if len(rows) > budget:
        rows = rng.sample(rows, budget)

    info = {
        "source": "every added token of both checkpoints, in 16 whitespace contexts",
        "contents": len(contents),
        "per_checkpoint": {k: len(v) for k, v in by_ckpt.items()},
        "contexts": len(ADDED_CONTEXTS),
        "lines": len(rows),
    }
    return rows, info


# --------------------------------------------------------------------------- #


def relpath(path: Path) -> str:
    """Repo-relative when it can be, absolute otherwise."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_corpus(path: Path, rows: list[tuple[str, str]]) -> str:
    """Write the deduplicated corpus and return its sha256.

    Deduplication is global and keeps the first source that offered a text, so a
    sentence that Tatoeba and a catalogue both contain is tokenised once and
    attributed to the stream that is meant to cover it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    seen: set[str] = set()
    kept = 0
    with path.open("w", encoding="utf-8") as fh:
        for src, text in rows:
            if not text or text in seen:
                continue
            seen.add(text)
            line = json.dumps({"src": src, "text": text}, ensure_ascii=False) + "\n"
            fh.write(line)
            digest.update(line.encode("utf-8"))
            kept += 1
    log(f"corpus: {kept} unique lines of {len(rows)} offered -> {path}")
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "build" / "corpus")
    ap.add_argument("--models-root", type=Path, default=REPO_ROOT / "models")
    ap.add_argument(
        "--stream",
        action="append",
        choices=sorted(DEFAULT_BUDGETS),
        help="restrict to one stream; repeatable (default: all four)",
    )
    ap.add_argument("--tatoeba-budget", type=int, default=DEFAULT_BUDGETS["tatoeba"])
    ap.add_argument("--probes-budget", type=int, default=DEFAULT_BUDGETS["probes"])
    ap.add_argument("--vocab-budget", type=int, default=DEFAULT_BUDGETS["vocab"])
    ap.add_argument("--locale-budget", type=int, default=DEFAULT_BUDGETS["locale"])
    ap.add_argument("--added-budget", type=int, default=DEFAULT_BUDGETS["added"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    wanted = set(args.stream) if args.stream else set(DEFAULT_BUDGETS)
    out_dir: Path = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, str]] = []
    streams: dict[str, Any] = {}

    if "tatoeba" in wanted:
        got, info = stream_tatoeba(args.tatoeba_budget, out_dir / "tatoeba", args.workers)
        rows += got
        streams["tatoeba"] = info
        log(f"tatoeba: {info['lines']} lines from {info['languages']} languages")
    if "probes" in wanted:
        got, info = stream_probes(args.probes_budget)
        rows += got
        streams["probes"] = info
        log(f"probes: {info['lines']} lines ({info['category_boundaries']} boundaries)")
    if "vocab" in wanted:
        got, info = stream_vocab(args.vocab_budget, args.models_root)
        rows += got
        streams["vocab"] = info
        log(f"vocab: {info['lines']} lines")
    if "locale" in wanted:
        got, info = stream_locale(args.locale_budget)
        rows += got
        streams["locale"] = info
        log(f"locale: {info['lines']} lines from {info.get('locales', 0)} locales")
    if "added" in wanted:
        got, info = stream_added(args.added_budget, args.models_root)
        rows += got
        streams["added"] = info
        log(f"added: {info['lines']} lines from {info.get('contents', 0)} tokens")

    corpus_path = out_dir / "corpus.jsonl"
    sha = write_corpus(corpus_path, rows)

    manifest = {
        "generator": "scripts/build_corpus.py",
        "plan_task": "4.5.3",
        "built": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": SEED,
        "corpus": {
            "path": relpath(corpus_path),
            "sha256": sha,
            "lines": sum(1 for _ in corpus_path.open(encoding="utf-8")),
        },
        "environment": {
            "python": sys.version.split()[0],
            "unicodedata": unicodedata.unidata_version,
        },
        "streams": streams,
    }
    (out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    log(f"manifest: {out_dir / 'MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
