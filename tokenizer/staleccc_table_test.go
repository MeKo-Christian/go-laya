package tokenizer

import (
	"slices"
	"strings"
)

// staleCCC lists the codepoints whose canonical combining class the Rust
// normaliser in tokenizers 0.23.2 lacks: combining marks assigned in Unicode
// 10.0 through 15.0, which it treats as starters, so it never reorders them.
// golang.org/x/text and CPython's unicodedata both do. PLAN.md B.2 keeps
// go-laya Unicode-conformant, so an NFC mismatch on a line carrying one of
// these is a documented deviation, and TestDifferentialCorpus accepts it
// (B.11). Sorted, for hasStaleCCC's binary search.
//
// Provenance: the 108 "codepoints" of build/ccc_disagreement.json, written by
//
//	.venv-ref/bin/python -I scripts/probe_ccc.py --models-root models --out build
//
// at tokenizers 0.23.2 / transformers 5.17.0 (scripts/requirements-ref.txt),
// on 2026-10-11. The run matched the probe's first output (2026-09-20)
// codepoint for codepoint. It excludes the probe's five method artefacts,
// U+0334 and U+0340, U+0341, U+0343, U+0344. Regenerate on every tokenizers
// pin bump; TestStaleCCCTable pins the size.
var staleCCC = [...]rune{
	0x07FD, 0x0898, 0x0899, 0x089A, 0x089B, 0x089C, 0x089D, 0x089E,
	0x089F, 0x08CA, 0x08CB, 0x08CC, 0x08CD, 0x08CE, 0x08CF, 0x08D0,
	0x08D1, 0x08D2, 0x08D3, 0x09FE, 0x0C3C, 0x0D3B, 0x0D3C, 0x0EBA,
	0x1715, 0x1ABF, 0x1AC0, 0x1AC1, 0x1AC2, 0x1AC3, 0x1AC4, 0x1AC5,
	0x1AC6, 0x1AC7, 0x1AC8, 0x1AC9, 0x1ACA, 0x1ACB, 0x1ACC, 0x1ACD,
	0x1ACE, 0x1DF6, 0x1DF7, 0x1DF8, 0x1DF9, 0x1DFA, 0xA82C, 0x10D24,
	0x10D25, 0x10D26, 0x10D27, 0x10EAB, 0x10EAC, 0x10EFD, 0x10EFE, 0x10EFF,
	0x10F46, 0x10F47, 0x10F48, 0x10F49, 0x10F4A, 0x10F4B, 0x10F4C, 0x10F4D,
	0x10F4E, 0x10F4F, 0x10F50, 0x10F82, 0x10F83, 0x10F84, 0x10F85, 0x11070,
	0x1133B, 0x1145E, 0x11839, 0x1183A, 0x1193D, 0x1193E, 0x11943, 0x119E0,
	0x11A34, 0x11A47, 0x11A99, 0x11D42, 0x11D44, 0x11D45, 0x11D97, 0x11F41,
	0x11F42, 0x16FF0, 0x16FF1, 0x1E08F, 0x1E130, 0x1E131, 0x1E132, 0x1E133,
	0x1E134, 0x1E135, 0x1E136, 0x1E2AE, 0x1E2EC, 0x1E2ED, 0x1E2EE, 0x1E2EF,
	0x1E4EC, 0x1E4ED, 0x1E4EE, 0x1E4EF,
}

// hasStaleCCC reports whether s contains a codepoint from staleCCC.
func hasStaleCCC(s string) bool {
	return strings.ContainsFunc(s, func(r rune) bool {
		_, ok := slices.BinarySearch(staleCCC[:], r)
		return ok
	})
}
