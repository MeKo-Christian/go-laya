package tokenizer

import (
	"bufio"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"testing"

	"github.com/MeKo-Christian/go-laya/internal/golden"
)

// corpusEnv points at the directory scripts/dump_stages.py wrote its
// stages_{en,ml}.jsonl.gz into. There is no default: the dumps are hundreds of
// megabytes, they live in gitignored build/, and a test that silently found a
// stale one would be worse than a test that skipped.
const corpusEnv = "LAYA_CORPUS"

// maxRepros is how many distinct minimal cases each stage reports. The point of
// the run is to name defect *classes*; 114k lines of the same bug is one finding
// printed 114k times.
const maxRepros = 10

// stage names the pipeline step a divergence is attributed to. Order matters:
// the first one that disagrees is the one reported, because everything after it
// is downstream of a wrong input.
type stage string

const (
	stageNormalize    stage = "normalize"
	stageAdded        stage = "added-split"
	stagePretokenize  stage = "pre-tokenize"
	stageBPE          stage = "bpe"
	stageDropped      stage = "dropped-character"
	stageAmbiguous    stage = "pre-bpe (ambiguous)"
	stageUnattributed stage = "unattributed"
)

// stageSegment is one segment of scripts/dump_stages.py's derived split. An
// added segment carries only the token string and the raw span it claimed; an
// unclaimed one additionally carries the normalizer's and pre-tokenizer's output.
type stageSegment struct {
	Added string   `json:"added"`
	Raw   string   `json:"raw"`
	Norm  string   `json:"norm"`
	Pre   []string `json:"pre"`
}

// stageRecord is one corpus line as the Python oracle saw it.
type stageRecord struct {
	I        int            `json:"i"`
	Text     string         `json:"text"`
	Derived  bool           `json:"derived"`
	Segments []stageSegment `json:"segments"`
	IDs      []int64        `json:"ids"`
	Tokens   []string       `json:"tokens"`
}

// TestDifferentialCorpus is PLAN.md task 4.5.3: ~100k lines of real multilingual
// corpus through both the Python oracle and this package, diffing the id streams
// and attributing every mismatch to a stage.
//
// It is the one test that can find Unicode-version skew between Go's `unicode`
// tables and Oniguruma's (R1), because that skew only shows on codepoints the
// 103-case committed corpus never reaches. It is a one-off gate, not a CI test:
// it needs the checkpoints and a multi-hundred-megabyte dump, so it skips unless
// both are present.
//
// It passes when every mismatch is PLAN.md B.2's documented deviation (B.11):
// an NFC mismatch that tokenizers 0.23.2's missing combining classes explain in
// full, see explainedByStaleCCC. Any other mismatch fails it, including one on
// a line that merely carries a staleCCC codepoint. The report still counts the
// accepted ones per stage.
//
// Build its inputs with:
//
//	just corpus && just dump-stages
func TestDifferentialCorpus(t *testing.T) {
	root := golden.SkipWithoutModels(t)
	dir := os.Getenv(corpusEnv)
	if dir == "" {
		t.Skipf("no corpus: set %s to the directory holding stages_{en,ml}.jsonl.gz "+
			"(build it with `just corpus && just dump-stages`)", corpusEnv)
	}

	for _, c := range []struct{ name, checkpoint string }{
		{"en", golden.English},
		{"ml", golden.Multilingual},
	} {
		t.Run(c.name, func(t *testing.T) {
			path := filepath.Join(dir, "stages_"+c.name+".jsonl.gz")
			if _, err := os.Stat(path); err != nil {
				t.Skipf("no dump at %s: %v", path, err)
			}
			runDifferential(t, openCheckpoint(t, root, c.checkpoint), path)
		})
	}
}

// findings accumulates one run's mismatches: a count per stage and the shortest
// distinct repros for each. accepted counts the mismatches that are PLAN.md
// B.2's documented deviation (see acceptedDeviation); they stay in the
// per-stage table but do not fail the run, and their repros are kept apart so
// they cannot crowd out a failing one.
type findings struct {
	lines          int
	checked        int
	accepted       int
	counts         map[stage]int
	repros         map[stage][]string
	acceptedRepros map[stage][]string
	seen           map[string]bool
}

// newFindings starts a run.
func newFindings() *findings {
	return &findings{
		counts:         map[stage]int{},
		repros:         map[stage][]string{},
		acceptedRepros: map[stage][]string{},
		seen:           map[string]bool{},
	}
}

// acceptedDeviation reports whether a mismatch is B.2's decided deviation:
// attributed to the normalizer, and explained (explainedByStaleCCC says the
// whole line agrees with the oracle once the stale combining classes are
// emulated). Nothing else is: a mismatch first seen at a later stage is a new
// divergence even when the emulation would account for it.
func (f *findings) acceptedDeviation(s stage, explained bool) bool {
	return s == stageNormalize && explained
}

// record files one mismatch. explained is explainedByStaleCCC's verdict on the
// line. Repros are kept shortest-first and deduplicated on their own text, so a
// defect that fires on 40 000 sentences is reported once.
func (f *findings) record(s stage, explained bool, repro string) {
	f.counts[s]++
	repros, tag := f.repros, "failing"
	if f.acceptedDeviation(s, explained) {
		f.accepted++
		repros, tag = f.acceptedRepros, "accepted"
	}
	key := tag + "\x00" + string(s) + "\x00" + repro
	if f.seen[key] {
		return
	}
	f.seen[key] = true
	repros[s] = append(repros[s], repro)
	list := repros[s]
	sort.SliceStable(list, func(i, j int) bool { return len(list[i]) < len(list[j]) })
	if len(list) > maxRepros {
		list = list[:maxRepros]
	}
	repros[s] = list
}

func (f *findings) total() int {
	n := 0
	for _, v := range f.counts {
		n += v
	}
	return n
}

// failing is the number of mismatches that are not B.2's accepted deviation.
func (f *findings) failing() int {
	return f.total() - f.accepted
}

// report renders the per-stage table task 4.5.3 asks for.
func (f *findings) report() string {
	var b strings.Builder
	fmt.Fprintf(&b, "corpus lines: %d, compared: %d, mismatching: %d\n",
		f.lines, f.checked, f.total())
	fmt.Fprintf(&b, "accepted (B.2 NFC deviation): %d, failing: %d\n", f.accepted, f.failing())
	stages := make([]stage, 0, len(f.counts))
	for s := range f.counts {
		stages = append(stages, s)
	}
	sort.Slice(stages, func(i, j int) bool { return f.counts[stages[i]] > f.counts[stages[j]] })
	for _, s := range stages {
		fmt.Fprintf(&b, "  %-22s %8d\n", s, f.counts[s])
		for _, r := range f.repros[s] {
			fmt.Fprintf(&b, "      %q\n", r)
		}
		for _, r := range f.acceptedRepros[s] {
			fmt.Fprintf(&b, "      accepted %q\n", r)
		}
	}
	return b.String()
}

// runDifferential streams one checkpoint's dump and compares every line.
func runDifferential(t *testing.T, tok *HF, path string) {
	t.Helper()

	fh, err := os.Open(path) // #nosec G304 -- a developer-supplied path to their own dump
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	defer func() { _ = fh.Close() }()

	zr, err := gzip.NewReader(fh)
	if err != nil {
		t.Fatalf("gzip %s: %v", path, err)
	}
	defer func() { _ = zr.Close() }()

	f := newFindings()
	sc := bufio.NewScanner(zr)
	// Same 16 MiB ceiling internal/golden uses: one probe line can be long.
	sc.Buffer(make([]byte, 0, 64*1024), 16<<20)

	first := true
	for sc.Scan() {
		line := sc.Bytes()
		if first { // the header record
			first = false
			t.Logf("header: %s", truncate(string(line), 300))
			continue
		}
		var rec stageRecord
		if err := json.Unmarshal(line, &rec); err != nil {
			t.Fatalf("line %d: %v", f.lines+2, err)
		}
		f.lines++
		compareOne(tok, &rec, f)
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("read %s: %v", path, err)
	}

	t.Log("\n" + f.report())
	if f.failing() != 0 {
		t.Errorf("%d of %d corpus lines diverge from the Python oracle outside B.2's NFC deviation",
			f.failing(), f.lines)
	}
}

// compareOne checks one corpus line end to end and, only when that fails, walks
// the stages to attribute the divergence.
//
// End-to-end first because it is the assertion that matters and the one that is
// unambiguous: the per-stage values are a diagnostic, and the derived
// segmentation behind them (scripts/dump_stages.py) is inferred from offsets
// rather than read from AddedVocabulary. A line that agrees on ids is not
// examined further, which is also what makes a 114k-line run cheap.
func compareOne(tok *HF, rec *stageRecord, f *findings) {
	f.checked++

	gotIDs := tok.Encode(rec.Text)
	if len(gotIDs) == len(rec.IDs) && slices.Equal(gotIDs, rec.IDs) &&
		slices.Equal(tokensOf(tok, gotIDs), rec.Tokens) {
		return
	}
	st, detail := attribute(tok, rec)
	f.record(st, st == stageNormalize && explainedByStaleCCC(tok, rec), detail)
}

// explainedByStaleCCC reports whether B.2's deviation accounts for every
// difference on the line: with this package's NFC swapped for staleStarterNFC,
// tokenizers 0.23.2's, each unclaimed segment normalizes to exactly what the
// oracle's did and the line encodes to exactly its ids and tokens.
//
// Carrying a staleCCC codepoint is not enough. A line can hold one that both
// sides agree on and diverge somewhere else, or diverge on it in one segment
// and for an unrelated reason in another, and attribute names only the first
// segment that differs. Replaying the whole line under the emulation catches
// both: any divergence it does not explain survives the swap.
//
// Only an NFC checkpoint can be explained this way: B.2 is about NFC, and
// multilingual's normalizer is a Replace that never looks at combining classes.
func explainedByStaleCCC(tok *HF, rec *stageRecord) bool {
	if _, ok := tok.norm.(nfcNormalizer); !ok || !hasStaleCCC(rec.Text) {
		return false
	}
	for _, seg := range rec.Segments {
		if seg.Added == "" && (staleStarterNFC{}).normalize(seg.Raw) != seg.Norm {
			return false
		}
	}
	// The added-token patterns were normalized at load time with the real NFC,
	// which agrees with the emulation on any pattern free of staleCCC.
	for i := range tok.added {
		if hasStaleCCC(tok.added[i].content) {
			return false
		}
	}
	// A shallow copy shares the vocabulary and merges, which Encode only reads.
	rust := *tok
	rust.norm = staleStarterNFC{}
	ids := rust.Encode(rec.Text)
	return slices.Equal(ids, rec.IDs) && slices.Equal(tokensOf(&rust, ids), rec.Tokens)
}

// attribute names the first stage whose output disagrees with the oracle's, and
// renders the disagreement.
//
// The detail is escaped to codepoints rather than printed raw: every finding
// here is by construction a combining mark, a format character or a whitespace
// run, and a terminal renders all three as something other than what they are.
// A repro you cannot retype is not a repro.
func attribute(tok *HF, rec *stageRecord) (stage, string) {
	if !rec.Derived {
		// The oracle could not prove its own segmentation, so neither can we.
		return stageAmbiguous, esc(rec.Text)
	}

	// 1. The normalizer, per unclaimed segment. Checked before the split,
	//    because a wrong normalization moves the phase-2 boundaries and would
	//    otherwise be reported as a split defect.
	for _, seg := range rec.Segments {
		if seg.Added != "" {
			continue
		}
		if got := tok.norm.normalize(seg.Raw); got != seg.Norm {
			return stageNormalize, diff(seg.Raw, got, seg.Norm)
		}
	}

	// 2. The added-token split. Both sides are rendered as the sequence of
	//    added tokens and normalized text between them.
	if want, got := renderStageSegments(rec.Segments), renderSegments(tok.splitAdded(rec.Text)); want != got {
		return stageAdded, diff(rec.Text, got, want)
	}

	// 3. The pre-tokenizer, per normalized segment.
	for _, seg := range rec.Segments {
		if seg.Added != "" {
			continue
		}
		if got := tok.pretok.preTokenize(seg.Norm); !slices.Equal(got, seg.Pre) {
			return stagePretokenize, diff(seg.Norm, strings.Join(got, "|"), strings.Join(seg.Pre, "|"))
		}
	}

	// 4. BPE, replayed over the oracle's own pieces so the input is identical.
	//    A non-empty piece that yields no ids is the English `unk_token: null`
	//    path at bpe.go:41-43 -- the character is dropped rather than replaced,
	//    and a dropped character shifts every marker after it, so it gets its
	//    own name instead of being counted as an ordinary BPE difference.
	for _, seg := range rec.Segments {
		if seg.Added != "" {
			continue
		}
		for _, piece := range seg.Pre {
			if piece != "" && len(tok.bpe(piece)) == 0 {
				return stageDropped, diff(rec.Text, "<no ids>", esc(piece))
			}
		}
	}
	if got := replayIDs(tok, rec.Segments); !slices.Equal(got, rec.IDs) {
		return stageBPE, diff(rec.Text, fmt.Sprint(got), fmt.Sprint(rec.IDs))
	}

	// Every stage agrees yet Encode does not: the composition differs from the
	// oracle's, which is a finding in itself and must not be filed under a
	// stage that was proven equal.
	return stageUnattributed, diff(rec.Text, fmt.Sprint(tok.Encode(rec.Text)), fmt.Sprint(rec.IDs))
}

// esc renders a string as codepoints, leaving printable ASCII alone so the
// shape of the input stays readable.
func esc(s string) string {
	var b strings.Builder
	for _, r := range s {
		if r >= 0x20 && r < 0x7f {
			b.WriteRune(r)
			continue
		}
		fmt.Fprintf(&b, "<U+%04X>", r)
	}
	return b.String()
}

// diff is one finding: the input, what this package produced, and what the
// oracle produced.
func diff(in, got, want string) string {
	return fmt.Sprintf("in=%s got=%s want=%s", esc(in), esc(got), esc(want))
}

// replayIDs runs the Go BPE over the oracle's segmentation, so a difference
// here is the merge loop and not an earlier stage.
func replayIDs(tok *HF, segs []stageSegment) []int64 {
	var ids []int64
	for _, seg := range segs {
		if seg.Added != "" {
			// The added vocabulary is authoritative: an added token may sit
			// above the contiguous vocabulary (88 of English's do), and one
			// that also has a vocab entry must still resolve to its added id.
			ids = append(ids, addedIDOf(tok, seg.Added))
			continue
		}
		for _, piece := range seg.Pre {
			ids = append(ids, tok.bpe(piece)...)
		}
	}
	return ids
}

// addedIDOf resolves an added token's id by content, falling back to the
// vocabulary for a token that exists only there.
func addedIDOf(tok *HF, content string) int64 {
	for i := range tok.added {
		if tok.added[i].content == content {
			return tok.added[i].id
		}
	}
	if id, ok := tok.vocab[content]; ok {
		return int64(id)
	}
	return -1
}

// renderStageSegments renders the oracle's split the way renderSegments
// (added_test.go) renders the Go one, so the two are directly comparable.
func renderStageSegments(segs []stageSegment) string {
	parts := make([]string, 0, len(segs))
	for _, seg := range segs {
		if seg.Added != "" {
			parts = append(parts, "<"+seg.Added+">")
			continue
		}
		parts = append(parts, seg.Norm)
	}
	return strings.Join(parts, "|")
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// TestFindingsAcceptOnlyDecidedClass checks the classifier behind
// TestDifferentialCorpus on synthetic findings, so it runs without the corpus.
// Only B.2's deviation is accepted: a normalize mismatch that
// explainedByStaleCCC explains. A mismatch first seen at any other stage still
// fails the run, explained or not.
func TestFindingsAcceptOnlyDecidedClass(t *testing.T) {
	for _, c := range []struct {
		name              string
		stages            []stage
		explained         []bool
		accepted, failing int
		normalizeCount    int
	}{
		{
			name:   "explained normalize",
			stages: []stage{stageNormalize}, explained: []bool{true},
			accepted: 1, failing: 0, normalizeCount: 1,
		},
		{
			name:   "unexplained normalize",
			stages: []stage{stageNormalize}, explained: []bool{false},
			accepted: 0, failing: 1, normalizeCount: 1,
		},
		{
			name: "every other stage, explained",
			stages: []stage{
				stageAdded, stagePretokenize, stageBPE,
				stageDropped, stageAmbiguous, stageUnattributed,
			},
			explained: []bool{true, true, true, true, true, true},
			accepted:  0, failing: 6,
		},
		{
			name: "mix",
			stages: []stage{
				stageNormalize, stageNormalize, stageNormalize,
				stagePretokenize, stageAdded, stageUnattributed,
			},
			explained: []bool{true, true, false, true, true, false},
			accepted:  2, failing: 4, normalizeCount: 3,
		},
	} {
		t.Run(c.name, func(t *testing.T) {
			f := newFindings()
			for i, s := range c.stages {
				f.record(s, c.explained[i], fmt.Sprintf("repro %d", i))
			}
			if f.accepted != c.accepted || f.failing() != c.failing {
				t.Errorf("accepted %d, failing %d; want %d, %d",
					f.accepted, f.failing(), c.accepted, c.failing)
			}
			// The per-stage table still counts every mismatch, accepted or not.
			if f.total() != len(c.stages) || f.counts[stageNormalize] != c.normalizeCount {
				t.Errorf("total %d, normalize %d; want %d, %d",
					f.total(), f.counts[stageNormalize], len(c.stages), c.normalizeCount)
			}
			want := fmt.Sprintf("accepted (B.2 NFC deviation): %d, failing: %d\n", c.accepted, c.failing)
			if r := f.report(); !strings.Contains(r, want) {
				t.Errorf("report lacks %q:\n%s", want, r)
			}
		})
	}
}

// TestFindingsReportFailingRepro checks that accepted mismatches cannot crowd a
// failing one out of the report. The accepted repros are short and outnumber
// the slots, which is exactly the run in which a new divergence must still be
// printed, not only counted.
func TestFindingsReportFailingRepro(t *testing.T) {
	f := newFindings()
	for i := range 2 * maxRepros {
		f.record(stageNormalize, true, esc(strings.Repeat("a", i)+"\u07fd"))
	}
	failing := esc("a new and longer divergence\u0301\u0316")
	f.record(stageNormalize, false, failing)

	if f.accepted != 2*maxRepros || f.failing() != 1 {
		t.Fatalf("accepted %d, failing %d; want %d, 1", f.accepted, f.failing(), 2*maxRepros)
	}
	if r := f.report(); !strings.Contains(r, failing) {
		t.Errorf("report omits the failing repro %q:\n%s", failing, r)
	}
}

// oracleRecord builds the record scripts/dump_stages.py would write for a line
// whose unclaimed segments the oracle normalized to segs[i].Norm. Pre, IDs and
// Tokens follow from Norm through this package's pre-tokenizer and BPE, which
// TestGolden pins to the oracle's, so the only thing a test chooses is the
// normalization.
func oracleRecord(tok *HF, segs ...stageSegment) *stageRecord {
	rec := &stageRecord{Derived: true, Segments: segs}
	var text strings.Builder
	for i := range rec.Segments {
		seg := &rec.Segments[i]
		text.WriteString(seg.Raw)
		if seg.Added == "" {
			seg.Pre = tok.pretok.preTokenize(seg.Norm)
		}
	}
	rec.Text = text.String()
	rec.IDs = replayIDs(tok, rec.Segments)
	rec.Tokens = tokensOf(tok, rec.IDs)
	return rec
}

// TestDifferentialAcceptsOnlyStaleCCCCause runs compareOne on synthetic oracle
// records, so it needs no corpus. A normalize mismatch is accepted only when
// tokenizers 0.23.2's missing combining classes explain all of it: a line that
// merely carries a stale-CCC codepoint, with the divergence somewhere else,
// must still fail the run (B.11).
func TestDifferentialAcceptsOnlyStaleCCCCause(t *testing.T) {
	tok := mustOpenMini(t, "mini_en")
	for _, c := range []struct {
		name     string
		rec      *stageRecord
		accepted int
	}{
		{
			// Go reorders U+0334 (ccc 1) before U+07FD (ccc 220); Rust, which
			// sees U+07FD as a starter, leaves the order alone.
			name:     "reordering around a stale mark",
			rec:      oracleRecord(tok, stageSegment{Raw: "x̴߽", Norm: "x̴߽"}),
			accepted: 1,
		},
		{
			// U+07FD (220) does not block U+0301 (230) from composing with e in
			// Go; as a Rust starter it does.
			name:     "composition blocked by a stale mark",
			rec:      oracleRecord(tok, stageSegment{Raw: "é߽", Norm: "é߽"}),
			accepted: 1,
		},
		{
			name:     "stale mark present, divergence elsewhere in the segment",
			rec:      oracleRecord(tok, stageSegment{Raw: "x߽ é", Norm: "x߽ é"}),
			accepted: 0,
		},
		{
			name: "stale divergence plus an unrelated one in the same segment",
			rec: oracleRecord(tok,
				stageSegment{Raw: "x̴߽ é", Norm: "x̴߽ é"}),
			accepted: 0,
		},
		{
			// attribute reports the first segment, which B.2 explains; the
			// second diverges for a reason B.2 has nothing to do with.
			name: "stale divergence plus an unrelated one in another segment",
			rec: oracleRecord(tok,
				stageSegment{Raw: "x̴߽", Norm: "x̴߽"},
				stageSegment{Added: "[CLS]", Raw: "[CLS]"},
				stageSegment{Raw: "é", Norm: "é"}),
			accepted: 0,
		},
	} {
		t.Run(c.name, func(t *testing.T) {
			f := newFindings()
			compareOne(tok, c.rec, f)
			if f.total() != 1 || f.counts[stageNormalize] != 1 {
				t.Fatalf("want one normalize mismatch, got %v", f.counts)
			}
			if f.accepted != c.accepted || f.failing() != 1-c.accepted {
				t.Errorf("accepted %d, failing %d; want %d, %d\n%s",
					f.accepted, f.failing(), c.accepted, 1-c.accepted, f.report())
			}
		})
	}
	// Multilingual normalizes with a Replace, not NFC, so B.2 cannot explain
	// anything there, whatever the line carries.
	ml := mustOpenMini(t, "mini_ml")
	if rec := oracleRecord(ml, stageSegment{Raw: "x\u07fd\u0334", Norm: "x\u07fd\u0334"}); explainedByStaleCCC(ml, rec) {
		t.Error("explainedByStaleCCC accepted a line on a checkpoint without NFC")
	}
}
