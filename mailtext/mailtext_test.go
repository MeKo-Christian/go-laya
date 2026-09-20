package mailtext

import (
	"encoding/json"
	"strconv"
	"strings"
	"testing"

	"github.com/MeKo-Christian/go-laya/internal/golden"
	"github.com/MeKo-Christian/go-laya/jsonx"
)

// filler is a body long enough that the signature scan actually reaches the
// last line: with n lines the scan starts at max(1, min(int(n*0.6), n-8)), so a
// twenty-line body starts looking at index 12 and a two-line body at index 1
// (invariant #70). Every signature case below relies on the first shape.
func filler(lines int) []string {
	out := make([]string, lines)
	for i := range out {
		out[i] = "line " + strconv.Itoa(i) + " of the actual message body"
	}
	return out
}

// TestCleanBodyGolden is task 2.3.1 against the 19 recorded cases: the fixture
// was produced by running original/laya/email.py, so a disagreement here is the
// port being wrong, never the corpus (PLAN.md R6).
func TestCleanBodyGolden(t *testing.T) {
	for _, c := range golden.ByFn(t, "mailtext", "clean_email_body") {
		t.Run(c.Name, func(t *testing.T) {
			var rec struct {
				Input    string `json:"input"`
				Output   string `json:"output"`
				MaxChars *int   `json:"max_chars"`
			}
			c.Unmarshal(t, &rec)

			maxChars := DefaultMaxChars
			if rec.MaxChars != nil {
				maxChars = *rec.MaxChars
			}

			if got := CleanBodyLimit(rec.Input, maxChars); got != rec.Output {
				t.Errorf("CleanBodyLimit(%q, %d)\n got: %q\nwant: %q\n"+
					"The cleaned body is what gets serialized into the prompt, so a "+
					"differing string means the model sees different tokens.",
					rec.Input, maxChars, got, rec.Output)
			}
		})
	}
}

// TestStateGolden is invariant #73 against the 5 recorded cases. It compares
// the serialized form, because key order is the part that is observable: the
// state is json.dumps'd into the prompt, so a reordered "from" is a different
// prompt, not a cosmetic difference.
func TestStateGolden(t *testing.T) {
	for _, c := range golden.ByFn(t, "mailtext", "email_state") {
		t.Run(c.Name, func(t *testing.T) {
			var rec struct {
				Input  jsonx.Obj `json:"input"`
				Output jsonx.Obj `json:"output"`
			}
			c.Unmarshal(t, &rec)

			subject, body, opts := stateArgs(t, rec.Input)
			got := StateWith(subject, body, opts)

			gotJSON := mustMarshal(t, got)
			wantJSON := mustMarshal(t, rec.Output)
			if gotJSON != wantJSON {
				t.Errorf("StateWith(%q, %q, %+v)\n got: %s\nwant: %s\n"+
					"The state is serialized straight into the prompt, so both the "+
					"keys and their order have to match Python's dict.",
					subject, body, opts, gotJSON, wantJSON)
			}
		})
	}
}

// stateArgs turns a recorded call's keyword arguments back into the Go call.
// Anything that is not one of the four named Python parameters is an entry of
// **extra, and its position in the recorded object is its position in the
// result, so the ordered jsonx.Obj is decoded rather than a map.
func stateArgs(tb testing.TB, in jsonx.Obj) (subject, body string, opts StateOptions) {
	tb.Helper()

	for _, f := range in {
		switch f.Key {
		case "subject":
			subject, _ = f.Value.(string)
		case "body":
			body, _ = f.Value.(string)
		case "sender":
			opts.Sender, _ = f.Value.(string)
		case "clean":
			clean, ok := f.Value.(bool)
			if !ok {
				tb.Fatalf("recorded clean=%v is not a bool; the case cannot be replayed", f.Value)
			}
			opts.Raw = !clean
		default:
			opts.Extra = append(opts.Extra, f)
		}
	}
	return subject, body, opts
}

func mustMarshal(tb testing.TB, v any) string {
	tb.Helper()

	b, err := jsonx.Marshal(v)
	if err != nil {
		tb.Fatalf("marshal %v: %v", v, err)
	}
	return string(b)
}

// TestCleanBodyPythonSemantics covers what the corpus does not: the places
// where a literal Go translation of the Python quietly produces a different
// string. Every want below was read off the reference implementation in
// original/laya/email.py; none of them needs Python at test time.
func TestCleanBodyPythonSemantics(t *testing.T) {
	long := strings.Join(filler(20), "\n")

	tests := []struct {
		name     string
		body     string
		maxChars int
		want     string
		why      string
	}{
		{
			name:     "german signature is cut",
			body:     long + "\nBeste Gr\u00fc\u00dfe,\nChristian",
			maxChars: DefaultMaxChars,
			want:     long,
			why:      "Python's [\\w ,!.] is Unicode-aware, so 'Beste Grüße,' matches; Go's ASCII \\w would keep the whole signature block",
		},
		{
			name:     "40 stripped chars still counts as a signature",
			body:     long + "\nThanks " + strings.Repeat("a", 33),
			maxChars: DefaultMaxChars,
			want:     long,
			why:      "the length test is <= 40 on the stripped line",
		},
		{
			name:     "41 stripped chars is body text",
			body:     long + "\nThanks " + strings.Repeat("a", 35),
			maxChars: DefaultMaxChars,
			want:     long + "\nThanks " + strings.Repeat("a", 35),
			why:      "the length test is <= 40 on the stripped line",
		},
		{
			name:     "a short body starts the scan at index 1, not at 60%",
			body:     "--\nChristian",
			maxChars: DefaultMaxChars,
			want:     "--\nChristian",
			why:      "with n=2 the window is [1, 2), so the '--' at index 0 is never examined (invariant #70)",
		},
		{
			name:     "U+001C is whitespace to Python's rstrip",
			body:     "Hello there\x1c\x1f\n\nSecond para\x1e",
			maxChars: DefaultMaxChars,
			want:     "Hello there\n\nSecond para",
			why:      "unicode.IsSpace excludes U+001C-U+001F, so strings.TrimSpace would leave the control characters in the prompt",
		},
		{
			name:     "a line of U+001C is an empty line",
			body:     "a\n\x1c\nb",
			maxChars: DefaultMaxChars,
			want:     "a\n\nb",
			why:      "rstrip empties the line, and the paragraph split then sees a blank line",
		},
		{
			name:     "NBSP separates paragraphs",
			body:     "a\n\u00a0\nb",
			maxChars: DefaultMaxChars,
			want:     "a\n\nb",
			why:      "Python's \\s in the r\"\\n\\s*\\n\" split is Unicode-aware; Go RE2's is ASCII-only (invariant #71)",
		},
		{
			name:     "ideographic space separates paragraphs",
			body:     "a\n\u3000\nb",
			maxChars: DefaultMaxChars,
			want:     "a\n\nb",
			why:      "as above, for U+3000",
		},
		{
			name:     "NBSP runs are not collapsed",
			body:     "a\u00a0\u00a0b",
			maxChars: DefaultMaxChars,
			want:     "a\u00a0\u00a0b",
			why:      "the collapse is r\"[ \\t]+\", which is ASCII by construction in both languages",
		},
		{
			name:     "NBSP-indented From: still breaks the scan",
			body:     "My reply.\n\u00a0From: x@y.z\nold",
			maxChars: DefaultMaxChars,
			want:     "My reply.",
			why:      "the ^\\s* of the quote header is Unicode-aware too (invariant #74)",
		},
		{
			name:     "truncation counts code points",
			body:     "\u00e4\u00f6\u00fc\u00df\u4e2d\u6587",
			maxChars: 3,
			want:     "\u00e4\u00f6\u00fc",
			why:      "Python slices str by code point; a Go byte slice would cut a rune in half (invariant #72)",
		},
		{
			name:     "a zero limit empties the body",
			body:     "hello",
			maxChars: 0,
			want:     "",
			why:      "text[:0] is the empty string",
		},
		{
			name:     "a negative limit slices from the end",
			body:     "hello",
			maxChars: -2,
			want:     "hel",
			why:      "Python's text[:-2]; reproduced rather than rejected so no deviation has to be recorded",
		},
		{
			name:     "forwarded header breaks the scan",
			body:     "My note.\n---- Forwarded Message ----\nold",
			maxChars: DefaultMaxChars,
			want:     "My note.",
			why:      "invariant #67",
		},
		{
			name:     "On ... wrote: allows 300 characters between",
			body:     "Hi.\nOn " + strings.Repeat("x", 299) + " wrote:\nq",
			maxChars: DefaultMaxChars,
			want:     "Hi.",
			why:      "{0,300} counts the run plus the trailing space",
		},
		{
			name:     "On ... wrote: gives up past 300 characters",
			body:     "Hi.\nOn " + strings.Repeat("x", 301) + " wrote:\nq",
			maxChars: DefaultMaxChars,
			want:     "Hi.\nOn " + strings.Repeat("x", 301) + " wrote:\nq",
			why:      "RE2 supports {0,300}, and the bound has to stay a bound",
		},
		{
			name:     "an indented quote line is dropped, not a break",
			body:     "My reply.\n   > their text\nmine",
			maxChars: DefaultMaxChars,
			want:     "My reply.\nmine",
			why:      "invariant #68",
		},
		{
			name:     "a literal backslash-n hides a quote header",
			body:     `a\nFrom: q@r.s\nb`,
			maxChars: DefaultMaxChars,
			want:     "a",
			why:      "the third replacement turns the two-character \\n into a newline first (invariant #66)",
		},
		{
			name:     "a lone CR is a line break",
			body:     "a\rb\r\nc",
			maxChars: DefaultMaxChars,
			want:     "a\nb\nc",
			why:      "invariant #66: CRLF first, then the bare CR",
		},
		{
			name:     "tabs collapse but newlines survive",
			body:     "a\t \tb\n\nc\td",
			maxChars: DefaultMaxChars,
			want:     "a b\n\nc d",
			why:      "invariant #72",
		},
		{
			name:     "a disclaimer-only body cleans to nothing",
			body:     "This message is confidential.",
			maxChars: DefaultMaxChars,
			want:     "",
			why:      "the disclaimer regex searches anywhere in the paragraph (invariant #71)",
		},
		{
			name:     "an empty body stays empty",
			body:     "",
			maxChars: DefaultMaxChars,
			want:     "",
			why:      "the loop over a single empty line keeps nothing",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := CleanBodyLimit(tt.body, tt.maxChars); got != tt.want {
				t.Errorf("CleanBodyLimit(%q, %d)\n got: %q\nwant: %q\nwhy: %s",
					tt.body, tt.maxChars, got, tt.want, tt.why)
			}
		})
	}
}

// TestCleanBodyUsesTheDefaultLimit pins DefaultMaxChars to the Python default,
// which is the limit every caller that does not pass one gets.
func TestCleanBodyUsesTheDefaultLimit(t *testing.T) {
	if DefaultMaxChars != 3000 {
		t.Fatalf("DefaultMaxChars = %d, want 3000 (email.py:23)", DefaultMaxChars)
	}

	// Multi-byte on purpose: a byte-wise truncation would cut at 3000 bytes,
	// i.e. after 1500 runes, and would also leave a broken rune behind.
	body := strings.Repeat("\u00e4", 4000)
	got := CleanBody(body)
	if n := len([]rune(got)); n != 3000 {
		t.Errorf("CleanBody kept %d runes, want 3000; truncation must count code "+
			"points, not bytes (invariant #72)", n)
	}
	if got != CleanBodyLimit(body, DefaultMaxChars) {
		t.Error("CleanBody disagrees with CleanBodyLimit at the default limit, so " +
			"callers get a different prompt depending on which one they reach for")
	}
}

// TestStateSemantics covers the parts of invariant #73 the corpus cannot reach,
// because Python's **extra cannot carry a key that collides with a named
// parameter unless that key is "from".
func TestStateSemantics(t *testing.T) {
	tests := []struct {
		name    string
		subject string
		body    string
		opts    StateOptions
		want    string
		why     string
	}{
		{
			name:    "no sender means no from key",
			subject: "S",
			body:    "B",
			want:    `{"subject": "S", "body": "B"}`,
			why:     "Python tests the sender for truthiness, so \"\" adds nothing",
		},
		{
			name:    "subject is stripped the Python way",
			subject: "  S\x1c",
			body:    "B\n",
			want:    `{"subject": "S", "body": "B"}`,
			why:     "str.strip() also removes U+001C-U+001F",
		},
		{
			name:    "raw keeps the body verbatim",
			subject: "S",
			body:    "B\n> quoted",
			opts:    StateOptions{Raw: true},
			want:    `{"subject": "S", "body": "B\n> quoted"}`,
			why:     "clean=False skips the pipeline entirely",
		},
		{
			name:    "nil extras are dropped",
			subject: "S",
			body:    "B",
			opts: StateOptions{Extra: jsonx.Obj{
				{Key: "priority", Value: "high"},
				{Key: "dropped", Value: nil},
			}},
			want: `{"subject": "S", "body": "B", "priority": "high"}`,
			why:  "the dict comprehension filters v is not None",
		},
		{
			name:    "an extra from replaces the sender in place",
			subject: "S",
			body:    "B",
			opts: StateOptions{Sender: "a@b", Extra: jsonx.Obj{
				{Key: "from", Value: "override"},
				{Key: "z", Value: json.Number("1")},
			}},
			want: `{"subject": "S", "body": "B", "from": "override", "z": 1}`,
			why:  "dict.update replaces the value and keeps the original position",
		},
		{
			name:    "an extra from without a sender appends",
			subject: "S",
			body:    "B",
			opts: StateOptions{Extra: jsonx.Obj{
				{Key: "z", Value: json.Number("1")},
				{Key: "from", Value: "only-extra"},
			}},
			want: `{"subject": "S", "body": "B", "z": 1, "from": "only-extra"}`,
			why:  "a key the dict does not have yet goes to the end",
		},
		{
			name:    "extras keep their own order",
			subject: "S",
			body:    "B",
			opts: StateOptions{Sender: "a@b", Extra: jsonx.Obj{
				{Key: "b", Value: json.Number("2")},
				{Key: "a", Value: json.Number("1")},
			}},
			want: `{"subject": "S", "body": "B", "from": "a@b", "b": 2, "a": 1}`,
			why:  "**extra preserves the caller's keyword order; it is never sorted",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := mustMarshal(t, StateWith(tt.subject, tt.body, tt.opts))
			if got != tt.want {
				t.Errorf("StateWith(%q, %q, %+v)\n got: %s\nwant: %s\nwhy: %s",
					tt.subject, tt.body, tt.opts, got, tt.want, tt.why)
			}
		})
	}
}

// TestStateDefaultsToCleaning guards the one option whose Go zero value has to
// mean the opposite of its Python default: clean=True.
func TestStateDefaultsToCleaning(t *testing.T) {
	got := mustMarshal(t, State(" Invoice ", "Please refund.\n> old"))
	want := `{"subject": "Invoice", "body": "Please refund."}`
	if got != want {
		t.Errorf("State()\n got: %s\nwant: %s\nCleaning is on unless the caller "+
			"opts out, so the quoted history must already be gone here.", got, want)
	}
}

// TestStateExtraIsNotAliased makes sure a caller's slice is not written through:
// StateWith appends to its own object, and a hidden append into the caller's
// backing array would corrupt a reused Options value.
func TestStateExtraIsNotAliased(t *testing.T) {
	extra := jsonx.Obj{{Key: "priority", Value: "high"}}
	_ = StateWith("S", "B", StateOptions{Extra: extra})

	if len(extra) != 1 || extra[0].Key != "priority" || extra[0].Value != "high" {
		t.Errorf("StateWith mutated the caller's Extra: %v", extra)
	}
}
