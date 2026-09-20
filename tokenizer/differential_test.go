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
// distinct repros for each.
type findings struct {
	lines   int
	checked int
	counts  map[stage]int
	repros  map[stage][]string
	seen    map[string]bool
}

func newFindings() *findings {
	return &findings{
		counts: map[stage]int{},
		repros: map[stage][]string{},
		seen:   map[string]bool{},
	}
}

// record files one mismatch. Repros are kept shortest-first and deduplicated on
// their own text, so a defect that fires on 40 000 sentences is reported once.
func (f *findings) record(s stage, repro string) {
	f.counts[s]++
	key := string(s) + "\x00" + repro
	if f.seen[key] {
		return
	}
	f.seen[key] = true
	f.repros[s] = append(f.repros[s], repro)
	list := f.repros[s]
	sort.SliceStable(list, func(i, j int) bool { return len(list[i]) < len(list[j]) })
	if len(list) > maxRepros {
		list = list[:maxRepros]
	}
	f.repros[s] = list
}

func (f *findings) total() int {
	n := 0
	for _, v := range f.counts {
		n += v
	}
	return n
}

// report renders the per-stage table task 4.5.3 asks for.
func (f *findings) report() string {
	var b strings.Builder
	fmt.Fprintf(&b, "corpus lines: %d, compared: %d, mismatching: %d\n",
		f.lines, f.checked, f.total())
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
	if f.total() != 0 {
		t.Errorf("%d of %d corpus lines diverge from the Python oracle", f.total(), f.lines)
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
	f.record(st, detail)
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
