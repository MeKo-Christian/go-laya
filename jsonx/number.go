package jsonx

import (
	"math"
	"strconv"
	"strings"
)

// Round4 rounds to four decimal places the way Python's round(x, 4) does, which
// is how every probability, score and confidence laya emits is produced
// (invariant #29).
//
// This is deliberately not math.Round(x*1e4)/1e4. Python rounds half to even on
// the exact binary value of the double; Go's math.Round rounds half away from
// zero. They disagree on ties -- 0.00125 goes to 0.0012, not 0.0013 -- and a tie
// is not exotic: it is what a softmax over small integers keeps producing.
// strconv rounds the same way Python's float_repr does, so formatting to four
// places and parsing back reproduces it exactly rather than approximately.
//
// Negative zero survives: Round4(-2.5e-05) is -0.0, not 0.0. Python emits "-0.0"
// and testdata/round4.jsonl pins it.
func Round4(x float64) float64 {
	if math.IsInf(x, 0) || math.IsNaN(x) {
		return x
	}
	r, err := strconv.ParseFloat(strconv.FormatFloat(x, 'f', 4, 64), 64)
	if err != nil {
		// Unreachable: FormatFloat's output is always valid input to ParseFloat.
		return x
	}
	return r
}

// Repr formats a float the way Python's repr does, which is what json.dumps
// writes for a float. Go's strconv and encoding/json both disagree with it, in
// four of the five ways that matter (see the package comment).
//
// The shortest round-tripping digits are the same in both languages; what
// differs is the presentation. Python keeps fixed notation over a much wider
// range, and always writes a fractional part, so 1.0 is "1.0" and not "1", and
// 1e15 is "1000000000000000.0" and not "1e+15".
func Repr(f float64) string {
	switch {
	case math.IsNaN(f):
		return "NaN"
	case math.IsInf(f, 1):
		return "Infinity"
	case math.IsInf(f, -1):
		return "-Infinity"
	}

	// CPython's format_float_short picks exponential notation when the decimal
	// point sits at or before -4, or past 16 significant places. Recovering the
	// decimal exponent from the 'e' form is the cheapest way to ask the same
	// question with the same shortest-digits rounding.
	e := strconv.FormatFloat(f, 'e', -1, 64)
	decpt := decimalPoint(e)
	if decpt <= -4 || decpt > 16 {
		return e
	}

	s := strconv.FormatFloat(f, 'f', -1, 64)
	if !strings.ContainsRune(s, '.') {
		s += ".0"
	}
	return s
}

// decimalPoint returns the position of the decimal point in a 'e'-formatted
// float, i.e. CPython's decpt: 1 for 1e0, 17 for 1e16, -4 for 1e-5.
func decimalPoint(e string) int {
	_, after, found := strings.Cut(e, "e")
	if !found {
		return 1
	}
	exp, err := strconv.Atoi(after)
	if err != nil {
		return 1
	}
	return exp + 1
}
