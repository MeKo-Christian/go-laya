// Package mailtext strips quoted history, signatures and legal disclaimers out
// of an email body, and builds the state object an email question is asked
// about. It is a port of original/laya/email.py, lines 23-53.
//
// Three Python behaviours here have no direct Go equivalent, and each one
// silently changes the cleaned text rather than failing:
//
//   - Python's `\w` and `\s` are Unicode-aware for str, Go RE2's are ASCII-only.
//     A German "Beste Grüße," matches the signature marker in Python and would
//     not in a literal translation, leaving the whole signature block in the
//     prompt (invariant #74).
//   - Python's str.strip() treats U+001C-U+001F as whitespace; unicode.IsSpace
//     does not, so strings.TrimSpace is not a drop-in (invariants #69, #72).
//   - Python slices str by code point. A Go byte slice truncates in the middle
//     of a rune (invariant #72).
//
// The package deliberately does not port email.email_questions: it is a
// byte-identical duplicate of presets.email_questions (PLAN.md task 2.3.2).
package mailtext

import (
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/MeKo-Christian/go-laya/jsonx"
)

// DefaultMaxChars is the max_chars default of clean_email_body (email.py:23).
// It counts code points, not bytes.
const DefaultMaxChars = 3000

// senderKey is the state key a truthy sender lands under (email.py:50). The
// key is part of the prompt the model was trained on, so it is not renameable.
const senderKey = "from"

// pySpace is the character class Python's `\s` denotes for str: the Unicode
// White_Space property plus U+001C-U+001F, which Python's str.isspace() reports
// as whitespace although they do not carry the property. Go's `\s` is
// [\t\n\f\r ], so every `\s` of the upstream patterns is spelled out with this.
const pySpace = `[\t\n\v\f\r \x{1C}-\x{1F}\x{85}\p{Z}]`

// pyWord is Python's `\w` for str: anything str.isalnum() accepts, plus the
// underscore. Go's `\w` is [0-9A-Za-z_], which would make the signature marker
// below miss every non-English sign-off.
const pyWord = `\p{L}\p{N}_`

// quoteHeaders end the scan: everything from the first matching line onwards is
// quoted history (email.py:5-10). Matched per line with the pattern anchored at
// the start, which is what Python's re.match does.
var quoteHeaders = []*regexp.Regexp{
	regexp.MustCompile(`(?i)^` + pySpace + `*On .{0,300}wrote:` + pySpace + `*$`),
	regexp.MustCompile(`(?i)^` + pySpace + `*-{2,}` + pySpace + `*(Original|Forwarded) Message` + pySpace + `*-{2,}`),
	regexp.MustCompile(`^` + pySpace + `*_{8,}` + pySpace + `*$`),
	regexp.MustCompile(`(?i)^` + pySpace + `*From:` + pySpace + `.+$`),
}

// signatureMarkers open a sign-off block (email.py:11-15).
var signatureMarkers = []*regexp.Regexp{
	regexp.MustCompile(`^` + pySpace + `*--` + pySpace + `*$`),
	regexp.MustCompile(`(?i)^` + pySpace +
		`*(best|kind|warm|many thanks|thanks|thank you|regards|cheers|sincerely)[` + pyWord + ` ,!.]*$`),
	regexp.MustCompile(`(?i)^` + pySpace + `*sent from my (iphone|android|mobile|ipad)`),
}

// disclaimer drops a whole paragraph wherever it hits, anchored nowhere
// (email.py:16-20); Python uses .search, not .match.
var disclaimer = regexp.MustCompile(`(?i)(confidential|intended (solely )?for the (use of the )?(named )?` +
	`(addressee|recipient)|if you (have )?received this (e-?mail|message) in error)`)

// paragraphBreak is email.py:39's r"\n\s*\n" with Python's Unicode `\s`.
var paragraphBreak = regexp.MustCompile(`\n` + pySpace + `*\n`)

// horizontalRun is email.py:40's r"[ \t]+": ASCII by construction, so newlines
// between paragraphs survive the collapse.
var horizontalRun = regexp.MustCompile(`[ \t]+`)

// CleanBody removes quoted history, signature blocks and legal disclaimers,
// keeping at most DefaultMaxChars code points. It is clean_email_body with its
// default limit.
func CleanBody(body string) string {
	return CleanBodyLimit(body, DefaultMaxChars)
}

// CleanBodyLimit is CleanBody with an explicit limit, measured in code points.
//
// A negative maxChars counts back from the end of the cleaned text, because
// that is what Python's text[:max_chars] does; reproducing it keeps the port
// free of a deviation nobody would benefit from.
func CleanBodyLimit(body string, maxChars int) string {
	// The third replacement is not redundant: it turns the two-character
	// sequence backslash-n into a newline, which is how a JSON-escaped body
	// arrives (invariant #66).
	text := strings.ReplaceAll(body, "\r\n", "\n")
	text = strings.ReplaceAll(text, "\r", "\n")
	text = strings.ReplaceAll(text, `\n`, "\n")

	lines := keptLines(text)
	lines = lines[:signatureCut(lines)]

	var paragraphs []string
	for _, p := range paragraphBreak.Split(strings.Join(lines, "\n"), -1) {
		if disclaimer.MatchString(p) {
			continue
		}
		if s := strings.TrimFunc(p, isPySpace); s != "" {
			paragraphs = append(paragraphs, s)
		}
	}

	joined := horizontalRun.ReplaceAllString(strings.Join(paragraphs, "\n\n"), " ")
	return truncate(joined, maxChars)
}

// keptLines walks the body once, dropping quoted lines and stopping at the
// first quote header. The header only ends the scan once something has been
// kept, so a mail that opens with "From: …" survives whole (invariant #67),
// while a ">" line is dropped on its own without ending anything (#68).
func keptLines(text string) []string {
	var lines []string
	for line := range strings.SplitSeq(text, "\n") {
		if len(lines) > 0 && matchesAny(quoteHeaders, line) {
			break
		}
		if strings.HasPrefix(strings.TrimLeftFunc(line, isPySpace), ">") {
			continue
		}
		lines = append(lines, strings.TrimRightFunc(line, isPySpace))
	}
	return lines
}

// signatureCut returns the index the body is cut at, or len(lines) for no cut.
//
// The window reads backwards from what it does: max(1, min(int(n*0.6), n-8)).
// For n < 9 the n-8 term is negative, the min picks it and the max clamps the
// start to 1 -- so a *short* body is scanned almost in full while a long one is
// only scanned from 60% on. That is upstream's behaviour and reproducing it is
// the point (invariant #70, PLAN.md:777-783).
func signatureCut(lines []string) int {
	n := len(lines)
	start := max(1, min(int(float64(n)*0.6), n-8))

	for i := start; i < n; i++ {
		// The length test runs on the stripped line, the pattern on the raw
		// one, and both count code points.
		if utf8.RuneCountInString(strings.TrimFunc(lines[i], isPySpace)) <= 40 &&
			matchesAny(signatureMarkers, lines[i]) {
			return i
		}
	}
	return n
}

// StateOptions carries email_state's optional arguments. Its zero value is
// Python's default call: no sender, cleaning on, no extra fields.
type StateOptions struct {
	// Sender becomes the "from" field. An empty string adds no field at all,
	// matching Python's truthiness test on the parameter.
	Sender string

	// Raw keeps the body exactly as passed. It is Python's clean=False,
	// inverted so that the useful default is the zero value.
	Raw bool

	// Extra are the **extra keyword arguments, in the caller's order. A field
	// whose value is nil is dropped, and a field whose key is already present
	// replaces that value *in place*, keeping the original position -- both are
	// what dict.update does, and the position is observable in the prompt.
	Extra jsonx.Obj
}

// State builds the state object for an email question with email_state's
// defaults: the subject stripped, the body cleaned, and nothing else.
func State(subject, body string) jsonx.Obj {
	return StateWith(subject, body, StateOptions{})
}

// StateWith builds the state object for an email question.
//
// The result is an ordered jsonx.Obj and never a Go map: it is serialized
// straight into the prompt, so the key order -- subject, body, from, then the
// extras -- is part of what the model sees (invariant #73).
func StateWith(subject, body string, opts StateOptions) jsonx.Obj {
	cleaned := body
	if !opts.Raw {
		cleaned = CleanBody(body)
	}

	state := jsonx.Obj{
		{Key: "subject", Value: strings.TrimFunc(subject, isPySpace)},
		{Key: "body", Value: cleaned},
	}
	if opts.Sender != "" {
		state = append(state, jsonx.Field{Key: senderKey, Value: opts.Sender})
	}
	for _, f := range opts.Extra {
		if f.Value == nil {
			continue
		}
		state = setField(state, f)
	}
	return state
}

// setField is dict.update for one key: replace in place, or append.
func setField(o jsonx.Obj, f jsonx.Field) jsonx.Obj {
	for i := range o {
		if o[i].Key == f.Key {
			o[i].Value = f.Value
			return o
		}
	}
	return append(o, f)
}

// isPySpace reports what Python's str.isspace() reports. Go's unicode.IsSpace
// covers the Unicode White_Space property; Python additionally treats the file
// and unit separators U+001C-U+001F as whitespace, and a stray one of those at
// the end of a line would otherwise survive into the prompt.
func isPySpace(r rune) bool {
	return unicode.IsSpace(r) || (r >= 0x1C && r <= 0x1F)
}

// truncate is Python's text[:n], counting code points and accepting a negative
// n as an offset from the end.
func truncate(s string, n int) string {
	if n >= 0 && len(s) <= n {
		// A string never has more runes than bytes, so this is safe and skips
		// the conversion for the overwhelmingly common case.
		return s
	}

	runes := []rune(s)
	if n < 0 {
		n += len(runes)
	}
	if n <= 0 {
		return ""
	}
	if n >= len(runes) {
		return s
	}
	return string(runes[:n])
}

func matchesAny(patterns []*regexp.Regexp, line string) bool {
	for _, p := range patterns {
		if p.MatchString(line) {
			return true
		}
	}
	return false
}
