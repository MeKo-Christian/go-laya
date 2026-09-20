package lang

import (
	"fmt"

	"github.com/MeKo-Christian/go-laya/jsonx"
)

// ScriptShare is one script's share of the alphabetic characters of a state.
type ScriptShare struct {
	// Script is the script name, e.g. "latin" or "devanagari".
	Script string
	// Fraction is the share of alphabetic characters, unrounded. Python emits
	// it through repr, so 1.0 stays "1.0" and 7/12 stays
	// "0.5833333333333334".
	Fraction float64
}

// Profile is the `script_profile` of a detection: an *ordered* object, not a
// map.
//
// The order is Python's dict order, which for script_profile is latin first
// (lang.py:116 seeds it) followed by the non-Latin scripts in the order the
// text encounters them, with zero counts dropped (invariant #58). It is
// observable: the profile is emitted inside routing.detection, and a sorted Go
// map would change the payload bytes without changing any answer.
type Profile []ScriptShare

// Fraction returns the share recorded for script. The second result is false
// when the script is absent, which is what Python's dict.get default covers.
func (p Profile) Fraction(script string) (float64, bool) {
	for _, s := range p {
		if s.Script == script {
			return s.Fraction, true
		}
	}
	return 0, false
}

// Obj returns the profile as the ordered JSON object Python builds. An empty
// profile becomes {}, which is what script_profile returns for a state with no
// letters.
func (p Profile) Obj() jsonx.Obj {
	obj := make(jsonx.Obj, 0, len(p))
	for _, s := range p {
		obj = append(obj, jsonx.Field{Key: s.Script, Value: s.Fraction})
	}
	return obj
}

// MarshalJSON emits the profile as an ordered object.
func (p Profile) MarshalJSON() ([]byte, error) {
	b, err := jsonx.Marshal(p.Obj())
	if err != nil {
		return nil, fmt.Errorf("lang: marshal script profile: %w", err)
	}
	return b, nil
}

// Detection is the result of Analyse, and the `detection` block of a routing
// payload. Its JSON shape is lang.py:170-177:
//
//	{"script": "latin", "script_profile": {"latin": 1.0}, "language": "en",
//	 "is_english": true, "non_latin_fraction": 0.0}
//
// The five keys are always present, in that order, and Language serialises as
// null when the guess was undecided.
type Detection struct {
	// Script is the dominant script, or "unknown" when the state has no
	// letters.
	Script string
	// ScriptProfile is each script's share of the alphabetic characters, in
	// Python's key order.
	ScriptProfile Profile
	// Language is the best-effort Latin-script language code, nil when
	// undecided and always nil off the Latin path.
	Language *string
	// IsEnglish reports whether the English checkpoint can read this state.
	// An undecided Latin state counts as English, and so does "unknown".
	IsEnglish bool
	// NonLatinFraction is round(1 - script_profile["latin"], 4), or 0.0 when
	// the profile is empty.
	NonLatinFraction float64
}

// Map returns the detection as the ordered object Python's analyse() builds.
// This is the form to hand to jsonx.Marshal when the bytes have to match
// Python's json.dumps exactly, separators included.
func (d Detection) Map() jsonx.Obj {
	var language any
	if d.Language != nil {
		language = *d.Language
	}

	return jsonx.Obj{
		{Key: "script", Value: d.Script},
		{Key: "script_profile", Value: d.ScriptProfile.Obj()},
		{Key: "language", Value: language},
		{Key: "is_english", Value: d.IsEnglish},
		{Key: "non_latin_fraction", Value: d.NonLatinFraction},
	}
}

// MarshalJSON emits the five keys in Python's order, with Python's float
// formatting.
//
// encoding/json compacts whatever a MarshalJSON returns, so it strips the
// ", " and ": " separators Python writes. Use jsonx.Marshal(d.Map()) where
// byte equality with json.dumps is the requirement; everything else that makes
// the bytes differ -- key order and 1.0 versus 1 -- survives the compaction.
func (d Detection) MarshalJSON() ([]byte, error) {
	b, err := jsonx.Marshal(d.Map())
	if err != nil {
		return nil, fmt.Errorf("lang: marshal detection: %w", err)
	}
	return b, nil
}
