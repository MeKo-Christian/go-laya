// Package lang is the dependency-free script and language detection that
// routes a state between the Laya checkpoints. It is a verbatim port of
// original/laya/lang.py.
//
// Routing needs one decision: is this English Latin text, or is it something
// the English checkpoint cannot read? On MASSIVE the English checkpoint
// collapses to near-random on non-Latin scripts (Hindi 0.100 at 20 options,
// where random is 0.050) while holding up on Latin-script languages (French
// 0.487), so script is the signal that matters and "is this Latin text
// English" is the secondary one.
//
// Script detection is exact. The Latin-script language guess is a
// stopword/diacritic heuristic and is explicitly best effort.
//
// # Order is the behaviour
//
// Three orderings in the Python are load-bearing and none of them survives a Go
// map, so this package uses ordered slices throughout:
//
//   - scriptRanges is scanned first-match-wins (invariant #55),
//   - DetectScript breaks an exact tie by *encounter order in the text*, with
//     latin appended last (invariant #56),
//   - ScriptProfile seeds latin first instead, and drops it again when it is
//     zero (invariant #58) -- a different order from DetectScript's, and the one
//     that reaches the wire inside routing.detection.
package lang

import (
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/MeKo-Christian/go-laya/jsonx"
)

// The two script names the control flow branches on, rather than just counts.
const (
	scriptLatin   = "latin"
	scriptUnknown = "unknown"
)

// codeRange is one inclusive code-point interval of a script block.
type codeRange struct{ lo, hi rune }

// scriptBlock is one entry of the Python _SCRIPT_RANGES list.
type scriptBlock struct {
	name   string
	ranges []codeRange
}

// scriptRanges is the Unicode blocks the English (ModernBERT-large, 50k English
// BPE) checkpoint cannot read, in the upstream order. The order is the
// classification: the scan stops at the first block that contains the
// character, so moving an entry silently reclassifies every character an
// earlier block also covers (lang.py:17-42, invariant #55).
//
// Replacing the 24 names with 24 constants would make this unreadable as the
// transcription it is, which is the only thing it is for. goconst no longer
// needs suppressing here -- it stopped counting the parity tests' copies of
// these names when .golangci.yml set ignore-tests -- but the reasoning stands
// if it ever fires again.
var scriptRanges = []scriptBlock{
	{"greek", []codeRange{{0x0370, 0x03FF}, {0x1F00, 0x1FFF}}},
	{"cyrillic", []codeRange{{0x0400, 0x052F}, {0x2DE0, 0x2DFF}, {0xA640, 0xA69F}}},
	{"hebrew", []codeRange{{0x0590, 0x05FF}}},
	{"arabic", []codeRange{
		{0x0600, 0x06FF},
		{0x0750, 0x077F},
		{0x08A0, 0x08FF},
		{0xFB50, 0xFDFF},
		{0xFE70, 0xFEFF},
	}},
	{"devanagari", []codeRange{{0x0900, 0x097F}, {0xA8E0, 0xA8FF}}},
	{"bengali", []codeRange{{0x0980, 0x09FF}}},
	{"gurmukhi", []codeRange{{0x0A00, 0x0A7F}}},
	{"gujarati", []codeRange{{0x0A80, 0x0AFF}}},
	{"oriya", []codeRange{{0x0B00, 0x0B7F}}},
	{"tamil", []codeRange{{0x0B80, 0x0BFF}}},
	{"telugu", []codeRange{{0x0C00, 0x0C7F}}},
	{"kannada", []codeRange{{0x0C80, 0x0CFF}}},
	{"malayalam", []codeRange{{0x0D00, 0x0D7F}}},
	{"sinhala", []codeRange{{0x0D80, 0x0DFF}}},
	{"thai", []codeRange{{0x0E00, 0x0E7F}}},
	{"lao", []codeRange{{0x0E80, 0x0EFF}}},
	{"tibetan", []codeRange{{0x0F00, 0x0FFF}}},
	{"myanmar", []codeRange{{0x1000, 0x109F}}},
	{"georgian", []codeRange{{0x10A0, 0x10FF}}},
	{"ethiopic", []codeRange{{0x1200, 0x137F}}},
	{"khmer", []codeRange{{0x1780, 0x17FF}}},
	{"hangul", []codeRange{{0x1100, 0x11FF}, {0x3130, 0x318F}, {0xAC00, 0xD7AF}}},
	{"kana", []codeRange{{0x3040, 0x309F}, {0x30A0, 0x30FF}, {0x31F0, 0x31FF}}},
	{"han", []codeRange{{0x3400, 0x4DBF}, {0x4E00, 0x9FFF}, {0xF900, 0xFAFF}}},
}

// isLatin reproduces the Latin test at lang.py:99: the Latin blocks plus Latin
// Extended Additional, as two disjoint ranges rather than a Unicode property.
func isLatin(cp rune) bool {
	return cp < 0x0250 || (cp >= 0x1E00 && cp <= 0x1EFF)
}

// blockOf returns the first scriptRanges entry containing cp. A letter matching
// nothing -- Runic, the IPA extensions -- is counted nowhere at all, so it can
// neither win nor lift the total off zero.
func blockOf(cp rune) (string, bool) {
	for _, b := range scriptRanges {
		for _, r := range b.ranges {
			if cp >= r.lo && cp <= r.hi {
				return b.name, true
			}
		}
	}
	return "", false
}

// scriptCount is one entry of the ordered counter that stands in for Python's
// insertion-ordered dict.
type scriptCount struct {
	name string
	n    int
}

// counter is a Python dict of script counts with its iteration order intact.
// It is a slice and not a map because the order *is* the tie-break: linear
// lookup over at most 25 entries is also faster than hashing here.
type counter []scriptCount

// add increments name, appending it in encounter order the first time.
func (c *counter) add(name string) {
	for i := range *c {
		if (*c)[i].name == name {
			(*c)[i].n++
			return
		}
	}
	*c = append(*c, scriptCount{name: name, n: 1})
}

// total is Python's sum(counts.values()).
func (c *counter) total() int {
	sum := 0
	for _, e := range *c {
		sum += e.n
	}
	return sum
}

// tally counts the alphabetic characters of text per script, leaving latin to
// the caller: DetectScript and ScriptProfile disagree about where latin sits in
// the order, and that disagreement is observable.
func tally(text string, c *counter) (latin int) {
	for _, ch := range text {
		// str.isalpha() is exactly the Unicode L categories (invariant #54).
		if !unicode.IsLetter(ch) {
			continue
		}
		if isLatin(ch) {
			latin++
			continue
		}
		if name, ok := blockOf(ch); ok {
			c.add(name)
		}
	}
	return latin
}

// DetectScript returns the dominant script of text: "latin", "han",
// "devanagari", ... or "unknown" when there are no letters at all.
//
// On an exact tie the first non-Latin script *encountered in the text* wins,
// because Python appends counts["latin"] after the loop and max() keeps the
// first maximum in iteration order (lang.py:106-110, invariant #56).
func DetectScript(text string) string {
	// Two entries covers the common case: one script plus the latin slot.
	counts := make(counter, 0, 2)
	latin := tally(text, &counts)

	// Appended, not seeded: latin is last, so it loses every tie.
	counts = append(counts, scriptCount{name: scriptLatin, n: latin})

	if counts.total() == 0 {
		return scriptUnknown
	}

	best := counts[0]
	for _, e := range counts[1:] {
		if e.n > best.n {
			best = e
		}
	}
	return best.name
}

// ScriptProfile returns each detected script's share of the alphabetic
// characters, in the order Python's dict holds them, or nil when text has no
// letters.
//
// The order is not DetectScript's. script_profile seeds counts = {"latin": 0}
// (lang.py:116), so latin comes *first*, and the `if v` filter drops it again
// when nothing Latin was seen (invariant #58). This order reaches the wire
// inside routing.detection, so it is part of the payload bytes.
func ScriptProfile(text string) Profile {
	// Seeded, not appended: latin is first. tally reallocates as it appends,
	// so the count lands in counts[0] only after it has returned.
	counts := make(counter, 1, 2)
	counts[0].name = scriptLatin
	latin := tally(text, &counts)
	counts[0].n = latin

	total := counts.total()
	if total == 0 {
		return nil
	}

	out := make(Profile, 0, len(counts))
	for _, e := range counts {
		if e.n == 0 {
			continue
		}
		out = append(out, ScriptShare{Script: e.name, Fraction: float64(e.n) / float64(total)})
	}
	return out
}

// ---------------------------------------------------------------- language

// stopList is one language's function words. The languages are a slice and not
// a map because _STOP's insertion order breaks ties between equally scoring
// languages (invariant #63).
type stopList struct {
	code  string
	words map[string]struct{}
}

// stopWords is _STOP in its insertion order: en first, then the tie-break order
// fr, de, es, pt, it, nl (lang.py:46-62). Latin-script languages overlap
// heavily -- de, la, le, un, e, que -- so each hit is weighted and a margin is
// required before calling something non-English.
var stopWords = []stopList{
	{"en", wordSet(
		"the", "and", "is", "are", "was", "were", "to", "of", "in", "for", "with", "that",
		"this", "it", "you", "have", "has", "not", "but", "on", "at", "be", "as", "from",
		"will", "can", "would", "there", "their", "what", "which", "please", "we", "i",
	)},
	{"fr", wordSet(
		"le", "la", "les", "des", "une", "est", "pour", "dans", "que", "qui", "avec", "sur",
		"pas", "plus", "nous", "vous", "être", "cette", "mais", "sont", "ont", "aux", "ce",
	)},
	{"de", wordSet(
		"der", "die", "das", "und", "ist", "ein", "eine", "den", "dem", "nicht", "mit", "für",
		"auf", "von", "zu", "sich", "auch", "werden", "wurde", "haben", "sind", "oder", "aber",
	)},
	{"es", wordSet(
		"el", "los", "las", "que", "por", "con", "para", "una", "es", "se", "del", "como",
		"pero", "son", "está", "este", "esta", "todo", "más", "muy", "hay", "sus",
	)},
	{"pt", wordSet(
		"os", "as", "que", "em", "um", "uma", "para", "com", "não", "é", "se", "do", "da",
		"dos", "das", "mas", "são", "está", "este", "esta", "muito", "pelo", "pela",
	)},
	{"it", wordSet(
		"il", "lo", "gli", "che", "di", "per", "con", "non", "è", "si", "del", "della", "sono",
		"questo", "questa", "anche", "come", "più", "sono", "nella", "alla",
	)},
	{"nl", wordSet(
		"het", "een", "van", "is", "op", "te", "dat", "niet", "met", "voor", "zijn", "aan",
		"door", "maar", "ook", "worden", "deze", "naar", "wordt",
	)},
}

func wordSet(words ...string) map[string]struct{} {
	set := make(map[string]struct{}, len(words))
	for _, w := range words {
		set[w] = struct{}{}
	}
	return set
}

// nonEnglishDiacritics is _NON_EN_DIACRITICS (lang.py:63). The upstream string
// repeats å, ä and ö; a set swallows the duplicates the same way.
var nonEnglishDiacritics = runeSet("àâäãáåçéèêëíìîïñóòôöõøúùûüýÿßæœđłşţğıåäö")

func runeSet(s string) map[rune]struct{} {
	set := make(map[rune]struct{}, len(s))
	for _, r := range s {
		set[r] = struct{}{}
	}
	return set
}

// words splits text into lowercased letter runs, reproducing
// re.findall(r"[^\W\d_]+", text, re.UNICODE).
//
// A direct translation is not available: RE2's \w is ASCII-only, so
// [^\W\d_]+ in Go would match nothing outside a-zA-Z. Splitting on
// unicode.IsLetter gives the same runs for every input laya sees. It differs
// only for the numeric characters Python's \w admits and \d does not -- ½, Ⅷ --
// which would join a neighbouring letter run there and break it here.
func words(text string) []string {
	var out []string
	start, in := 0, false

	for i, r := range text {
		switch {
		case unicode.IsLetter(r) && !in:
			start, in = i, true
		case !unicode.IsLetter(r) && in:
			out = append(out, strings.ToLower(text[start:i]))
			in = false
		}
	}
	if in {
		out = append(out, strings.ToLower(text[start:]))
	}
	return out
}

// minWords is the floor at lang.py:139-140: fewer than four letter runs and the
// guess is refused outright. It counts runs, not stop-word hits.
const minWords = 4

// GuessLatinLanguage returns a best-effort language code for Latin-script text.
// The second result is false when the heuristic is undecided, which is what
// Python returns None for and what short inputs get on purpose.
//
// It scores function-word hits per language and requires the winner to beat
// English by a margin, so ordinary English is never misrouted (lang.py:133-156,
// invariants #61-#64).
func GuessLatinLanguage(text string) (string, bool) {
	ws := words(text)
	if len(ws) < minWords {
		return "", false
	}

	// max over {fr, de, es, pt, it, nl} keeping the FIRST maximum, so an exact
	// tie goes to whichever language _STOP lists first (invariant #63).
	en := 0
	bestLang, best := "", 0
	for _, sl := range stopWords {
		score := 0
		for _, w := range ws {
			if _, ok := sl.words[w]; ok {
				score++
			}
		}
		switch {
		case sl.code == "en":
			en = score
		case bestLang == "" || score > best:
			bestLang, best = sl.code, score
		}
	}

	diacRate := diacriticRate(text)

	// The ladder of lang.py:149-156, rung by rung.
	if best == 0 && diacRate < 0.02 {
		return orUndecided(en)
	}
	// A non-English language needs a clear margin over English function words.
	if bestLang != "" && best >= max(2, en+2) {
		return bestLang, true
	}
	if diacRate >= 0.04 && bestLang != "" && best >= en {
		return bestLang, true
	}
	return orUndecided(en)
}

// orUndecided is Python's `return "en" if en else None`.
func orUndecided(en int) (string, bool) {
	if en > 0 {
		return "en", true
	}
	return "", false
}

// diacriticRate is lang.py:143-145. The denominator is the length of the whole
// lowered text -- spaces and punctuation included, not just letters -- so
// padding a state with whitespace really does move the rate (invariant #62).
//
// Python measures that length in code points, so this counts runes. The one
// input where the two languages disagree is U+0130 (İ), which CPython lowers to
// two code points and Go lowers to one.
func diacriticRate(text string) float64 {
	lowered := strings.ToLower(text)

	diac := 0
	for _, r := range lowered {
		if _, ok := nonEnglishDiacritics[r]; ok {
			diac++
		}
	}
	return float64(diac) / float64(max(1, utf8.RuneCountInString(lowered)))
}

// ---------------------------------------------------------------- analyse

// Analyse returns the full detection result for a state: the script, the script
// profile, the best-effort language, whether the English checkpoint can read it
// and how much of it is non-Latin (lang.py:159-177, invariant #65).
func Analyse(state any) Detection {
	text := StateText(state)
	prof := ScriptProfile(text)
	script := DetectScript(text)

	nonLatin := 0.0
	if len(prof) > 0 {
		latin, _ := prof.Fraction(scriptLatin) // absent means 0.0, as dict.get does
		// Python's round(), which is half-to-even; math.Round is not.
		nonLatin = jsonx.Round4(1.0 - latin)
	}

	switch {
	case script == scriptUnknown:
		// The unknown branch hard-codes 0.0 rather than reusing nonLatin.
		return Detection{Script: scriptUnknown, ScriptProfile: prof, IsEnglish: true}
	case script != scriptLatin:
		return Detection{Script: script, ScriptProfile: prof, NonLatinFraction: nonLatin}
	}

	det := Detection{Script: scriptLatin, ScriptProfile: prof, NonLatinFraction: nonLatin}
	if lg, ok := GuessLatinLanguage(text); ok {
		det.Language = &lg
		det.IsEnglish = lg == "en"
	} else {
		// `lang in (None, "en")`: undecided Latin text counts as English.
		det.IsEnglish = true
	}
	return det
}

// IsEnglish reports whether the English checkpoint can be expected to read this
// state.
func IsEnglish(state any) bool {
	return Analyse(state).IsEnglish
}
