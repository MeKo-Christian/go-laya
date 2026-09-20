//nolint:gosmopolitan // the CJK literals are the fixtures under test
package lang

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/MeKo-Christian/go-laya/jsonx"
)

// The expectations in this file were read off the frozen upstream by loading
// original/laya/lang.py directly (its package __init__ imports torch). Where a
// case is one of upstream's own, the comment cites the line in
// original/tests/test_router.py that carries it.

// ------------------------------------------------------------------ script

// TestDetectScript covers the 14 cases of test_router.py:29-46 (task 2.2.5)
// plus the classification edges invariants #54 and #55 turn on. A wrong script
// routes the state to a checkpoint that cannot read it, and the English
// checkpoint answers near-random rather than failing.
func TestDetectScript(t *testing.T) {
	cases := []struct {
		name string
		text string
		want string
	}{
		// test_router.py:30-43.
		{"english", "The customer was charged twice and wants a refund.", "latin"},
		{"french", "Le client a été facturé deux fois et demande un remboursement.", "latin"},
		{"hindi", "ग्राहक से दो बार शुल्क लिया गया और वह धनवापसी चाहता है।", "devanagari"},
		{"japanese", "お客様は二重に請求されたため返金を希望しています。", "kana"},
		{"chinese", "客户被重复扣款要求退款", "han"},
		{"korean", "고객이 두 번 청구되어 환불을 원합니다", "hangul"},
		{"arabic", "تم خصم المبلغ مرتين من العميل ويريد استرداد الأموال", "arabic"},
		{"tamil", "வாடிக்கையாளரிடம் இருமுறை கட்டணம் வசூலிக்கப்பட்டது", "tamil"},
		{"russian", "С клиента дважды сняли деньги и он хочет возврат", "cyrillic"},
		{"thai", "ลูกค้าถูกเรียกเก็บเงินสองครั้งและต้องการเงินคืน", "thai"},
		{"greek", "Ο πελάτης χρεώθηκε δύο φορές και θέλει επιστροφή χρημάτων", "greek"},
		{"hebrew", "הלקוח חויב פעמיים ורוצה החזר כספי", "hebrew"},
		{"empty", "", "unknown"},
		{"digits only", "12345 6789", "unknown"},

		// Invariant #54: the Latin test is two disjoint ranges, and only
		// characters str.isalpha() accepts are counted at all.
		{"latin extended additional", "ẞ", "latin"},
		{"punctuation only", "!?-- ...", "unknown"},

		// Invariant #55, hangul before kana before han: the three CJK blocks
		// are the ones a reordered table would visibly confuse.
		{"hangul jamo", "ᄀ", "hangul"},
		{"hiragana", "あ", "kana"},
		{"cjk unified", "一", "han"},

		// A letter that is neither Latin nor in any listed block is counted
		// nowhere, so it cannot win and cannot lift the total off zero.
		{"runic alone is unknown", "ᚠ", "unknown"},
		{"ipa extensions alone is unknown", "ɐ", "unknown"},
		{"runic does not outvote latin", "ᚠabc", "latin"},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := DetectScript(c.text); got != c.want {
				t.Errorf("DetectScript(%q) = %q, want %q: the wrong script sends this "+
					"state to a checkpoint that cannot read it", c.text, got, c.want)
			}
		})
	}
}

// TestDetectScriptTieBreak pins invariant #56. `counts["latin"]` is assigned
// after the loop, so latin is last in the dict and `max` keeps the first
// maximum: on an exact tie the first non-Latin script *encountered in the text*
// wins. A Go map here would return a different answer on different runs.
func TestDetectScriptTieBreak(t *testing.T) {
	cases := []struct {
		name string
		text string
		want string
	}{
		{"non-latin beats latin on a tie", "aא", "hebrew"},
		{"non-latin beats latin, latin first in text", "אa", "hebrew"},
		{"first encountered non-latin wins, hebrew first", "אα", "hebrew"},
		{"first encountered non-latin wins, greek first", "αא", "greek"},
		{"latin still wins a strict majority", "aaא", "latin"},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := DetectScript(c.text); got != c.want {
				t.Errorf("DetectScript(%q) = %q, want %q: the encounter-order tie-break "+
					"is what keeps this answer stable across runs", c.text, got, c.want)
			}
		})
	}
}

// TestScriptRangeOrder pins the table order itself (task 2.2.4, invariant #55).
// First match wins, so reordering the table silently reclassifies every
// character in an overlapping block — and nothing else in the suite would say
// so, because the shipped blocks happen not to overlap.
func TestScriptRangeOrder(t *testing.T) {
	want := []string{
		"greek", "cyrillic", "hebrew", "arabic", "devanagari", "bengali",
		"gurmukhi", "gujarati", "oriya", "tamil", "telugu", "kannada",
		"malayalam", "sinhala", "thai", "lao", "tibetan", "myanmar",
		"georgian", "ethiopic", "khmer", "hangul", "kana", "han",
	}

	got := make([]string, 0, len(scriptRanges))
	for _, b := range scriptRanges {
		got = append(got, b.name)
	}

	if !slices.Equal(got, want) {
		t.Errorf("scriptRanges order = %v, want %v: first match wins, so the order is "+
			"the classification", got, want)
	}
}

// TestScriptProfile pins invariant #58 and the ordering task 2.2.7 needs.
//
// Note that script_profile's order is NOT detect_script's: it seeds
// `counts = {"latin": 0}` (lang.py:116), so latin comes *first*, and the `if v`
// comprehension drops it again when nothing Latin was seen.
func TestScriptProfile(t *testing.T) {
	cases := []struct {
		name string
		text string
		want []ScriptShare
	}{
		{"no letters is empty", "12345", nil},
		{"pure latin", "abc", []ScriptShare{{"latin", 1.0}}},
		{"zero latin is dropped", "א", []ScriptShare{{"hebrew", 1.0}}},
		{
			"latin comes first when present",
			"aא",
			[]ScriptShare{{"latin", 0.5}, {"hebrew", 0.5}},
		},
		{
			"non-latin keys follow in encounter order",
			"אα",
			[]ScriptShare{{"hebrew", 0.5}, {"greek", 0.5}},
		},
		{
			"non-latin keys follow in encounter order, reversed",
			"αא",
			[]ScriptShare{{"greek", 0.5}, {"hebrew", 0.5}},
		},
		{
			"mixed japanese",
			"お客様は二重に請求されたため返金を希望しています。",
			[]ScriptShare{{"kana", 0.5833333333333334}, {"han", 0.4166666666666667}},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := ScriptProfile(c.text)
			if !slices.Equal([]ScriptShare(got), c.want) {
				t.Errorf("ScriptProfile(%q) = %v, want %v: this order is emitted inside "+
					"routing.detection, so it is part of the payload bytes", c.text, got, c.want)
			}
		})
	}
}

// ------------------------------------------------------------------ english

// TestIsEnglish covers the 7 cases of test_router.py:49-61 (task 2.2.5).
func TestIsEnglish(t *testing.T) {
	cases := []struct {
		name string
		text string
		want bool
	}{
		{"plain english", "Please refund the duplicate charge on invoice 4411 today.", true},
		{"english short", "refund me", true},
		{"hindi", "ग्राहक से दो बार शुल्क लिया गया", false},
		{"japanese", "お客様は二重に請求されました", false},
		{"russian", "С клиента дважды сняли деньги", false},
		{
			"french long",
			"Le client a été facturé deux fois et il demande un remboursement pour la " +
				"facture qui a été payée le mois dernier avec la carte de crédit",
			false,
		},
		{
			"german long",
			"Der Kunde wurde zweimal belastet und möchte eine Rückerstattung für die " +
				"Rechnung die nicht korrekt ist und auch nicht bezahlt wurde",
			false,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := IsEnglish(c.text); got != c.want {
				t.Errorf("IsEnglish(%q) = %v, want %v: a false positive here routes "+
					"non-English text to the English checkpoint", c.text, got, c.want)
			}
		})
	}
}

// TestGuessLatinLanguage covers the 6 cases of test_router.py:65-77 (task
// 2.2.5) plus the tie-break and each rung of the decision ladder (#63, #64).
func TestGuessLatinLanguage(t *testing.T) {
	cases := []struct {
		name string
		text string
		want string // "" means undecided
	}{
		// test_router.py:66-77.
		{"english", "The customer was charged twice and wants a refund for this invoice", "en"},
		{
			"french",
			"Le client a ete facture deux fois et il demande un remboursement pour la facture",
			"fr",
		},
		{
			"german",
			"Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung fuer die Rechnung",
			"de",
		},
		{
			"spanish",
			"El cliente fue cobrado dos veces y quiere que le devuelvan el dinero por la factura",
			"es",
		},
		{"too short", "refund", ""},
		{
			"long english stays en",
			"Please refund the duplicate charge on invoice 4411 today because " +
				"we have been waiting for three days and nobody has replied to us",
			"en",
		},

		// Invariant #61: the four-word floor counts letter runs, not stop-word
		// hits, and a run of unknown words decides nothing.
		{"three words is undecided", "one two three", ""},
		{"four unknown words is undecided", "zzz qqq xxx www", ""},

		// Invariant #63: fr scores 2 (est, les) and de scores 2 (die, der);
		// _STOP's insertion order puts fr first.
		{"fr wins a tie with de", "die est der les", "fr"},

		// Invariant #64 rung (b): a non-English guess needs en+2.
		{"german clears the margin", "der die das und ist the and is", "de"},
		{"german misses the margin", "der die das und the and is are was were", "en"},

		// Invariant #64 rung (c): diacritics alone can decide, and the loser of
		// an all-zero max is still fr.
		{"diacritics alone pick fr", "ééé ààà ööö ñññ", "fr"},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, ok := GuessLatinLanguage(c.text)
			if !ok {
				got = ""
			}
			if got != c.want {
				t.Errorf("GuessLatinLanguage(%q) = %q, want %q: this code is what picks "+
					"the multilingual checkpoint over the English one", c.text, got, c.want)
			}
		})
	}
}

// TestGuessLatinLanguageDiacRateDenominator pins invariant #62: diac_rate
// divides by the length of the whole lowered text, punctuation and spaces
// included, not by the letter count. Padding the same four words with spaces
// walks the rate down past both thresholds and changes the answer, which is the
// only way to see which denominator is in use.
func TestGuessLatinLanguageDiacRateDenominator(t *testing.T) {
	const words = "ééé ààà ööö ñññ" // 12 diacritics, 15 runes

	cases := []struct {
		name string
		pad  int
		want string
	}{
		{"rate above 0.04 picks fr", 200, "fr"},     // 12/215 = 0.0558
		{"rate between the thresholds", 385, ""},    // 12/400 = 0.03
		{"rate below 0.02 is still none", 1000, ""}, // 12/1015 = 0.0118, en == 0
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			text := words + strings.Repeat(" ", c.pad)
			got, ok := GuessLatinLanguage(text)
			if !ok {
				got = ""
			}
			if got != c.want {
				t.Errorf("GuessLatinLanguage(%d spaces of padding) = %q, want %q: the "+
					"denominator is len(text), so padding is not inert", c.pad, got, c.want)
			}
		})
	}
}

// ------------------------------------------------------------------ state

// TestStateText covers the 5 flattening cases of test_router.py:80-86 (task
// 2.2.5) and the depth cap of invariant #60.
func TestStateText(t *testing.T) {
	cases := []struct {
		name  string
		state any
		want  string
	}{
		// test_router.py:80-84.
		{
			"dict",
			jsonx.Obj{{Key: "body", Value: "charged twice"}, {Key: "n", Value: 3}},
			"charged twice",
		},
		{
			"nested",
			jsonx.Obj{{Key: "a", Value: jsonx.Obj{{Key: "b", Value: []any{"deep"}}}}},
			"deep",
		},
		{"list", []any{"x", jsonx.Obj{{Key: "y", Value: "z"}}}, "x z"},
		{"none", nil, ""},
		{"bare string", "hello", "hello"},

		// Invariant #60: keys are ignored, values are joined with one space in
		// depth-first order, and non-string leaves contribute nothing.
		{
			"keys are ignored",
			jsonx.Obj{{Key: "subject", Value: 1}, {Key: "body", Value: "only this"}},
			"only this",
		},
		{"scalar state", 3, ""},
		{"null leaf", jsonx.Obj{{Key: "k", Value: nil}}, ""},
		{"typed slice", []string{"a", "b"}, "a b"},

		// The depth cap is `_depth > 6` on the value, so a leaf at depth 6 is
		// still collected and one nesting level deeper is not.
		{"leaf at the depth cap", nest(6, "edge"), "edge"},
		{"leaf past the depth cap", nest(7, "too deep"), ""},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := StateText(c.state); got != c.want {
				t.Errorf("StateText(%#v) = %q, want %q: detection only ever sees what "+
					"this returns", c.state, got, c.want)
			}
		})
	}
}

// nest wraps leaf in depth levels of single-key object.
func nest(depth int, leaf string) any {
	var v any = leaf
	for range depth {
		v = jsonx.Obj{{Key: "k", Value: v}}
	}
	return v
}

// TestStateTextTruncatesRunes pins the code-point half of invariant #60.
// Python's [:4000] slices code points; Go's s[:4000] slices bytes, which both
// truncates far too early and can cut a rune in half.
func TestStateTextTruncatesRunes(t *testing.T) {
	for _, r := range []string{"é", "あ", "𝄞"} {
		t.Run(r, func(t *testing.T) {
			got := StateText(strings.Repeat(r, 5000))
			if n := utf8.RuneCountInString(got); n != 4000 {
				t.Errorf("StateText(5000×%q) kept %d runes, want 4000: byte slicing "+
					"truncates a multi-byte state to a fraction of what Python sees", r, n)
			}
			if !utf8.ValidString(got) {
				t.Errorf("StateText(5000×%q) is not valid UTF-8: the cut landed inside "+
					"a rune", r)
			}
		})
	}
}

// TestStateTextKeysIgnored is test_router.py:86: English keys wrapped around
// Hindi content must not make the state look English.
func TestStateTextKeysIgnored(t *testing.T) {
	state := jsonx.Obj{
		{Key: "subject", Value: "नमस्ते"},
		{Key: "body", Value: "ग्राहक से दो बार शुल्क लिया गया"},
	}
	if IsEnglish(state) {
		t.Error("IsEnglish(Hindi body under English keys) = true, want false: keys are " +
			"almost always English and must not drive routing")
	}
}

// ------------------------------------------------------------------ analyse

// TestAnalyseJSON is the parity assertion for the whole package: every field of
// Detection, in Python's key order, with Python's float formatting. The want
// strings are json.dumps(analyse(text), ensure_ascii=False) from the frozen
// upstream. Covers invariants #59, #65 and task 2.2.7.
func TestAnalyseJSON(t *testing.T) {
	cases := []struct {
		name string
		text string
		want string
	}{
		{
			"english", "The customer was charged twice and wants a refund.",
			`{"script": "latin", "script_profile": {"latin": 1.0}, "language": "en", "is_english": true, "non_latin_fraction": 0.0}`,
		},
		{
			"french", "Le client a été facturé deux fois et demande un remboursement.",
			`{"script": "latin", "script_profile": {"latin": 1.0}, "language": "fr", "is_english": false, "non_latin_fraction": 0.0}`,
		},
		{
			"hindi", "ग्राहक से दो बार शुल्क लिया गया और वह धनवापसी चाहता है।",
			`{"script": "devanagari", "script_profile": {"devanagari": 1.0}, "language": null, "is_english": false, "non_latin_fraction": 1.0}`,
		},
		{
			"japanese", "お客様は二重に請求されたため返金を希望しています。",
			`{"script": "kana", "script_profile": {"kana": 0.5833333333333334, "han": 0.4166666666666667}, "language": null, "is_english": false, "non_latin_fraction": 1.0}`,
		},
		{
			"chinese", "客户被重复扣款要求退款",
			`{"script": "han", "script_profile": {"han": 1.0}, "language": null, "is_english": false, "non_latin_fraction": 1.0}`,
		},
		{
			"korean", "고객이 두 번 청구되어 환불을 원합니다",
			`{"script": "hangul", "script_profile": {"hangul": 1.0}, "language": null, "is_english": false, "non_latin_fraction": 1.0}`,
		},
		{
			"arabic", "تم خصم المبلغ مرتين من العميل ويريد استرداد الأموال",
			`{"script": "arabic", "script_profile": {"arabic": 1.0}, "language": null, "is_english": false, "non_latin_fraction": 1.0}`,
		},
		{
			"russian", "С клиента дважды сняли деньги и он хочет возврат",
			`{"script": "cyrillic", "script_profile": {"cyrillic": 1.0}, "language": null, "is_english": false, "non_latin_fraction": 1.0}`,
		},
		{
			"empty", "",
			`{"script": "unknown", "script_profile": {}, "language": null, "is_english": true, "non_latin_fraction": 0.0}`,
		},
		{
			"digits only", "12345 6789",
			`{"script": "unknown", "script_profile": {}, "language": null, "is_english": true, "non_latin_fraction": 0.0}`,
		},
		// Invariant #59: non_latin_fraction is Python's round(x, 4), and latin
		// leads script_profile whenever it is non-zero.
		{
			"one third latin", "aאא",
			`{"script": "hebrew", "script_profile": {"latin": 0.3333333333333333, "hebrew": 0.6666666666666666}, "language": null, "is_english": false, "non_latin_fraction": 0.6667}`,
		},
		{
			"two thirds latin", "aaא",
			`{"script": "latin", "script_profile": {"latin": 0.6666666666666666, "hebrew": 0.3333333333333333}, "language": null, "is_english": true, "non_latin_fraction": 0.3333}`,
		},
		{
			"three sevenths latin", "aaaאאאא",
			`{"script": "hebrew", "script_profile": {"latin": 0.42857142857142855, "hebrew": 0.5714285714285714}, "language": null, "is_english": false, "non_latin_fraction": 0.5714}`,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := jsonx.Marshal(Analyse(c.text).Map())
			if err != nil {
				t.Fatalf("marshalling the detection failed: %v", err)
			}
			if string(got) != c.want {
				t.Errorf("Analyse(%q) marshals to\n  %s\nwant\n  %s\nthe payload bytes are "+
					"what a caller reads back out of routing.detection", c.text, got, c.want)
			}
		})
	}
}

// TestDetectionMarshalJSON checks the encoding/json path. encoding/json
// compacts whatever MarshalJSON returns, so the Python separators are gone --
// but the key order and Python's float repr, the two things a Go map or
// encoding/json's own float formatting would destroy, must survive.
func TestDetectionMarshalJSON(t *testing.T) {
	got, err := json.Marshal(Analyse("aא"))
	if err != nil {
		t.Fatalf("json.Marshal(Detection) failed: %v", err)
	}

	const want = `{"script":"hebrew","script_profile":{"latin":0.5,"hebrew":0.5},` +
		`"language":null,"is_english":false,"non_latin_fraction":0.5}`
	if string(got) != want {
		t.Errorf("json.Marshal(Detection) = %s, want %s: latin must lead script_profile "+
			"and 1.0 must not collapse to 1", got, want)
	}
}

// TestDetectionMarshalJSONPureLatin is the float-formatting half on its own:
// encoding/json writes a whole float as `1`, Python writes `1.0`.
func TestDetectionMarshalJSONPureLatin(t *testing.T) {
	got, err := json.Marshal(Analyse("abc"))
	if err != nil {
		t.Fatalf("json.Marshal(Detection) failed: %v", err)
	}
	if !strings.Contains(string(got), `{"latin":1.0}`) {
		t.Errorf("json.Marshal(Detection) = %s, want a script_profile of {\"latin\":1.0}: "+
			"Python never writes a float without a fractional part", got)
	}
}

// TestAnalyseLanguageIsOptional pins the `lang in (None, "en")` half of
// invariant #65: an undecided Latin state counts as English.
func TestAnalyseLanguageIsOptional(t *testing.T) {
	d := Analyse("refund me")
	if d.Language != nil {
		t.Errorf("Analyse(short latin).Language = %q, want nil: short inputs are "+
			"undecided on purpose", *d.Language)
	}
	if !d.IsEnglish {
		t.Error("Analyse(short latin).IsEnglish = false, want true: an undecided Latin " +
			"state stays on the English checkpoint")
	}
}

// ------------------------------------------------------------------ determinism

// TestDetectionIsDeterministic is task 2.2.6. Python's dicts are ordered and
// Go's maps are not, so the encounter-order tie-breaks in detect_script and
// script_profile are exactly the places a map-based port would answer
// differently on different runs -- and usually only on some runs, which is why
// this needs a loop rather than a second assertion.
func TestDetectionIsDeterministic(t *testing.T) {
	// Every text here has at least one exact tie, so a randomized iteration
	// order has something to get wrong.
	texts := []string{
		"aא",
		"אα",
		"αא",
		"one א two α three अ four",
		"Le client a été facturé deux fois et demande un remboursement.",
	}

	for _, text := range texts {
		t.Run(text, func(t *testing.T) {
			first, err := jsonx.Marshal(Analyse(text).Map())
			if err != nil {
				t.Fatalf("marshalling the detection failed: %v", err)
			}
			wantScript := DetectScript(text)

			for i := range 1000 {
				if got := DetectScript(text); got != wantScript {
					t.Fatalf("DetectScript(%q) = %q on iteration %d, %q on the first: the "+
						"tie-break is reading a randomized map order", text, got, i, wantScript)
				}
				got, err := jsonx.Marshal(Analyse(text).Map())
				if err != nil {
					t.Fatalf("marshalling the detection failed on iteration %d: %v", i, err)
				}
				if string(got) != string(first) {
					t.Fatalf("Analyse(%q) marshals to %s on iteration %d and %s on the "+
						"first: the payload bytes are not stable", text, got, i, first)
				}
			}
		})
	}
}

// TestProfileMarshalJSON checks that a profile marshalled on its own is still
// an ordered object rather than the array its Go type suggests.
func TestProfileMarshalJSON(t *testing.T) {
	got, err := json.Marshal(ScriptProfile("aא"))
	if err != nil {
		t.Fatalf("json.Marshal(Profile) failed: %v", err)
	}
	const want = `{"latin":0.5,"hebrew":0.5}`
	if string(got) != want {
		t.Errorf("json.Marshal(Profile) = %s, want %s: a Profile is a Python dict, "+
			"not a list", got, want)
	}
}

// TestStateTextGoMap covers the one state shape Python has no counterpart for.
// A Go map cannot reproduce a dict's insertion order, so this package walks it
// in sorted key order: the content still reaches detection and the answer is
// the same on every run, which is the property a randomized walk would lose.
func TestStateTextGoMap(t *testing.T) {
	state := map[string]any{"z": "last", "a": "first", "m": []any{"middle"}}

	want := "first middle last"
	for i := range 100 {
		if got := StateText(state); got != want {
			t.Fatalf("StateText(map) = %q on iteration %d, want %q: a map state must "+
				"still detect the same way twice", got, i, want)
		}
	}

	if got := StateText(map[int]string{2: "b", 1: "a"}); got != "a b" {
		t.Errorf("StateText(map[int]string) = %q, want %q: a non-string key must still "+
			"sort to something stable", got, "a b")
	}
}
