// Package jsonx produces JSON byte-identical to what Python's json.dumps
// writes, which is what the laya checkpoints were trained and evaluated on.
//
// This package exists because encoding/json gets three things wrong at once,
// and laya feeds json.dumps(state, ensure_ascii=False) straight into the
// tokenizer (common.py:15-18). Different bytes mean different tokens, which
// means different marker positions and different probabilities -- silently, with
// a plausible answer rather than an error. Invariant #18:
//
//   - separators: Go emits {"a":1}, Python emits {"a": 1}
//   - key order: Go sorts map keys, Python preserves insertion order
//   - HTML escaping: Go escapes < > & by default, Python does not
//
// Two more surfaced while porting, and they are not in the invariant list:
//
//   - Go escapes U+2028 and U+2029 for JSONP safety; Python emits them literally
//     under ensure_ascii=False.
//   - Go and Python format floats differently almost everywhere. Python writes
//     1.0, 1000000000000000.0, 1e+16, 1e-05 and -0.0 where Go writes 1,
//     1000000000000000, 10000000000000000, 0.00001 and 0. See Repr.
//
// Ordered objects are Obj, never a Go map. A map cannot express what Python
// guarantees, so this package refuses to encode one rather than sorting its keys
// and producing bytes that look right and are not.
package jsonx

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"unicode/utf8"
)

// ErrUnorderedMap is returned when a Go map reaches the encoder. Python dicts
// preserve insertion order and laya's prompts depend on it, so a map is an
// ambiguity the caller has to resolve with Obj rather than one this package can
// resolve by guessing.
var ErrUnorderedMap = errors.New("jsonx: cannot encode a Go map; key order is observable, use Obj")

// ErrUnsupportedType is returned for values Python's json module would also
// refuse and laya never produces, such as channels and functions.
var ErrUnsupportedType = errors.New("jsonx: unsupported type")

// Field is one key/value pair of an Obj.
type Field struct {
	Key   string
	Value any
}

// Obj is an ordered JSON object: the Go stand-in for a Python dict. Python
// dicts preserve insertion order and laya builds prompts from serialized state,
// so a Go map would silently change the bytes the model sees -- and the key
// order of the probabilities it emits. Every place the Python code uses a dict
// whose order is observable, Go uses Obj.
//
// The split receivers are deliberate and cannot be avoided: UnmarshalJSON has to
// take a pointer to replace the slice, while MarshalJSON has to be on the value
// so that an Obj stored in an interface field still marshals.
//
//nolint:recvcheck // see above
type Obj []Field

// Get returns the first value stored under key. A duplicate key is possible in
// hand-written JSON but not in a Python dict, so the first wins the same way a
// dict literal's last assignment does not reorder the key.
func (o Obj) Get(key string) (any, bool) {
	for _, f := range o {
		if f.Key == key {
			return f.Value, true
		}
	}
	return nil, false
}

// MarshalJSON reproduces json.dumps(x, ensure_ascii=False) for an object:
// separators ", " and ": ", insertion order, no HTML escaping, non-ASCII
// emitted literally.
func (o Obj) MarshalJSON() ([]byte, error) {
	var b strings.Builder
	if err := encodeObj(&b, o, false); err != nil {
		return nil, err
	}
	return []byte(b.String()), nil
}

// UnmarshalJSON decodes a JSON object while preserving key order, which
// encoding/json's map[string]any cannot. The golden fixtures record their
// inputs as objects whose key order is the assertion -- testdata/render.jsonl's
// state/dict-unsorted-keys is {"z": 1, "a": 2, "m": 3} -- so a decoder that
// loses the order destroys exactly what the test checks.
//
// Numbers decode as json.Number so an integer stays an integer: Python writes 3
// for an int and 3.0 for a float, and a round trip through float64 would turn
// every int in the corpus into the wrong bytes.
func (o *Obj) UnmarshalJSON(data []byte) error {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()

	v, err := decodeValue(dec)
	if err != nil {
		return err
	}
	got, ok := v.(Obj)
	if !ok {
		return fmt.Errorf("jsonx: cannot unmarshal %s into Obj", firstToken(data))
	}
	*o = got
	return nil
}

// Decode parses arbitrary JSON, mapping objects to Obj and numbers to
// json.Number so that both key order and int-versus-float survive.
func Decode(data []byte) (any, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	return decodeValue(dec)
}

// Marshal reproduces json.dumps(v, ensure_ascii=False), which is what
// common.serialize_state calls (invariant #18). It has no default= fallback, so
// a value Python's json module would refuse is an error here too.
func Marshal(v any) ([]byte, error) {
	var b strings.Builder
	if err := encodeValue(&b, v, false); err != nil {
		return nil, err
	}
	return []byte(b.String()), nil
}

// Compact reproduces
// json.dumps(v, ensure_ascii=False, separators=(", ", ": "), default=str),
// which is common.render_criterion (invariant #17). The separators are the same
// ones Marshal uses -- Python passes them explicitly there and relies on the
// defaults here -- so the two differ only in the fallback.
//
// default=str means an unserializable value is stringified rather than raising,
// per value and not per document. Go's nearest equivalent is fmt.Sprint. Note
// that a Go map stringifies rather than encoding, because ErrUnorderedMap
// applies here too: the result will not match Python, and it is meant to be
// visibly wrong rather than quietly plausible.
func Compact(v any) string {
	var b strings.Builder
	if err := encodeValue(&b, v, true); err != nil {
		// Unreachable while fallback is true: every refusal becomes a string.
		return fmt.Sprint(v)
	}
	return b.String()
}

// encodeValue writes v. When fallback is true, anything the encoder refuses is
// written as its Go string form, reproducing Python's default=str.
func encodeValue(b *strings.Builder, v any, fallback bool) error {
	if v == nil {
		b.WriteString("null")
		return nil
	}

	switch x := v.(type) {
	case bool:
		b.WriteString(strconv.FormatBool(x))
		return nil
	case string:
		encodeString(b, x)
		return nil
	case json.Number:
		b.WriteString(x.String())
		return nil
	case float64:
		b.WriteString(Repr(x))
		return nil
	case float32:
		b.WriteString(Repr(float64(x)))
		return nil
	case Obj:
		return encodeObj(b, x, fallback)
	case []any:
		return encodeSlice(b, x, fallback)
	case json.RawMessage:
		b.Write(x)
		return nil
	}

	return encodeReflect(b, v, fallback)
}

// encodeReflect handles the integer widths and the slice types the explicit
// switch does not enumerate, and refuses everything else.
func encodeReflect(b *strings.Builder, v any, fallback bool) error {
	rv := reflect.ValueOf(v)
	// The kinds not listed are handled by default: bool, string, the floats and
	// the structural kinds are either caught by the type switch in encodeValue
	// or have no Python counterpart.
	//exhaustive:ignore
	switch rv.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		b.WriteString(strconv.FormatInt(rv.Int(), 10))
		return nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		b.WriteString(strconv.FormatUint(rv.Uint(), 10))
		return nil
	case reflect.Slice, reflect.Array:
		items := make([]any, rv.Len())
		for i := range items {
			items[i] = rv.Index(i).Interface()
		}
		return encodeSlice(b, items, fallback)
	case reflect.Pointer, reflect.Interface:
		if rv.IsNil() {
			b.WriteString("null")
			return nil
		}
		return encodeValue(b, rv.Elem().Interface(), fallback)
	case reflect.Map:
		return refuse(b, v, fallback, ErrUnorderedMap)
	default:
		return refuse(b, v, fallback, fmt.Errorf("%w: %T", ErrUnsupportedType, v))
	}
}

// refuse applies Python's default=str when fallback is on, and reports the
// error when it is off.
func refuse(b *strings.Builder, v any, fallback bool, err error) error {
	if !fallback {
		return err
	}
	encodeString(b, fmt.Sprint(v))
	return nil
}

func encodeObj(b *strings.Builder, o Obj, fallback bool) error {
	b.WriteByte('{')
	for i, f := range o {
		if i > 0 {
			b.WriteString(", ")
		}
		encodeString(b, f.Key)
		b.WriteString(": ")
		if err := encodeValue(b, f.Value, fallback); err != nil {
			return err
		}
	}
	b.WriteByte('}')
	return nil
}

func encodeSlice(b *strings.Builder, items []any, fallback bool) error {
	b.WriteByte('[')
	for i, it := range items {
		if i > 0 {
			b.WriteString(", ")
		}
		if err := encodeValue(b, it, fallback); err != nil {
			return err
		}
	}
	b.WriteByte(']')
	return nil
}

// encodeString writes a Python json string literal under ensure_ascii=False.
//
// Python escapes exactly the two structural characters, the five shorthand
// control codes, and the remaining C0 controls as \u00xx. Everything else is
// emitted literally, including DEL and U+2028/U+2029 -- the last of which
// encoding/json always escapes and Python never does.
func encodeString(b *strings.Builder, s string) {
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\b':
			b.WriteString(`\b`)
		case '\f':
			b.WriteString(`\f`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		default:
			switch {
			case r < 0x20:
				b.WriteString(`\u00`)
				const hex = "0123456789abcdef"
				b.WriteByte(hex[(r>>4)&0xf])
				b.WriteByte(hex[r&0xf])
			case r == utf8.RuneError:
				// A lone surrogate or invalid byte reached the encoder. Python's
				// str cannot hold one, so there is no Python behaviour to match;
				// PLAN.md task 4.4.11 sanitizes at the API edge instead.
				b.WriteRune(r)
			default:
				b.WriteRune(r)
			}
		}
	}
	b.WriteByte('"')
}

// decodeValue reads one value, mapping objects to Obj to keep key order.
func decodeValue(dec *json.Decoder) (any, error) {
	tok, err := dec.Token()
	if err != nil {
		return nil, fmt.Errorf("jsonx: decode: %w", err)
	}
	return decodeFrom(dec, tok)
}

func decodeFrom(dec *json.Decoder, tok json.Token) (any, error) {
	delim, ok := tok.(json.Delim)
	if !ok {
		return tok, nil
	}

	switch delim {
	case '{':
		obj := Obj{}
		for dec.More() {
			kt, err := dec.Token()
			if err != nil {
				return nil, fmt.Errorf("jsonx: decode key: %w", err)
			}
			key, ok := kt.(string)
			if !ok {
				return nil, fmt.Errorf("jsonx: object key is %T, not a string", kt)
			}
			v, err := decodeValue(dec)
			if err != nil {
				return nil, err
			}
			obj = append(obj, Field{Key: key, Value: v})
		}
		return obj, closeDelim(dec)
	case '[':
		arr := []any{}
		for dec.More() {
			v, err := decodeValue(dec)
			if err != nil {
				return nil, err
			}
			arr = append(arr, v)
		}
		return arr, closeDelim(dec)
	default:
		return nil, fmt.Errorf("jsonx: unexpected delimiter %q", delim)
	}
}

func closeDelim(dec *json.Decoder) error {
	if _, err := dec.Token(); err != nil {
		return fmt.Errorf("jsonx: decode close: %w", err)
	}
	return nil
}

// firstToken is used only to build a readable type error.
func firstToken(data []byte) string {
	s := strings.TrimSpace(string(data))
	if len(s) > 20 {
		s = s[:20] + "..."
	}
	return s
}
