// Package checkpoint reads and validates what a laya checkpoint directory
// holds besides the graph: today its rl_agent_config.json.
package checkpoint

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/MeKo-Christian/go-laya/backend"
)

// ConfigFile is the checkpoint config's name inside a checkpoint directory
// (agent.py:139).
const ConfigFile = "rl_agent_config.json"

// maxConfigSize caps how much of a config is read. It is a downloaded
// artefact; the shipped ones are under 1 KiB.
const maxConfigSize = 1 << 20

// requiredKeys are the keys _verify_compatibility demands (agent.py:52), in
// its order.
var requiredKeys = []string{"encoder", "head_layers"}

// Config is a checkpoint's rl_agent_config.json, kept as its top-level
// fields so later readers can decode the ones they need.
type Config struct {
	path   string
	fields map[string]json.RawMessage
}

// Field returns the raw JSON of the top-level key, and whether it is present.
// The result is a copy; writing to it does not change the Config.
func (c *Config) Field(key string) (json.RawMessage, bool) {
	v, ok := c.fields[key]
	return bytes.Clone(v), ok
}

// LoadConfig reads dir's rl_agent_config.json and requires the keys
// "encoder" and "head_layers" to be present, whatever their value, as
// `k not in cfg` does (agent.py:51-57). Every failure wraps
// backend.ErrIncompatibleCheckpoint; a missing file also wraps
// fs.ErrNotExist.
func LoadConfig(dir string) (*Config, error) {
	path := filepath.Join(dir, ConfigFile)
	// #nosec G304 -- dir is the checkpoint directory the caller chose to load;
	// reading its config is the point.
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("config: %w: %w", err, backend.ErrIncompatibleCheckpoint)
	}
	defer f.Close()

	raw, err := io.ReadAll(io.LimitReader(f, maxConfigSize+1))
	if err != nil {
		return nil, fmt.Errorf("config: %s: %w: %w", path, err, backend.ErrIncompatibleCheckpoint)
	}
	if len(raw) > maxConfigSize {
		return nil, fmt.Errorf("config: %s is larger than %d bytes: %w",
			path, maxConfigSize, backend.ErrIncompatibleCheckpoint)
	}
	// json.Unmarshal decodes null into a nil map without complaint, so the
	// object check comes first.
	if !bytes.HasPrefix(bytes.TrimSpace(raw), []byte("{")) {
		return nil, fmt.Errorf("config: %s is not a JSON object: %w", path, backend.ErrIncompatibleCheckpoint)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return nil, fmt.Errorf("config: %s: %w: %w", path, err, backend.ErrIncompatibleCheckpoint)
	}

	var missing []string
	for _, k := range requiredKeys {
		if _, ok := fields[k]; !ok {
			missing = append(missing, k)
		}
	}
	if len(missing) > 0 {
		return nil, fmt.Errorf("config: %s: missing keys %s: %w",
			path, strings.Join(missing, ", "), backend.ErrIncompatibleCheckpoint)
	}
	return &Config{path: path, fields: fields}, nil
}

// ActWidth is the act head's output width, len(cfg.get("act_costs", {})) + 1
// (common.py:137): 1 without the key, else one more than Python's len() of
// the value, which counts an object's keys, an array's elements or a
// string's code points. Any other value is where len() raises, and fails with
// backend.ErrIncompatibleCheckpoint.
func (c *Config) ActWidth() (int, error) {
	raw, ok := c.fields["act_costs"]
	if !ok {
		return 1, nil
	}
	var n int
	var err error
	switch v := bytes.TrimSpace(raw); {
	case bytes.HasPrefix(v, []byte("{")):
		var m map[string]json.RawMessage
		err = json.Unmarshal(v, &m)
		n = len(m)
	case bytes.HasPrefix(v, []byte("[")):
		var a []json.RawMessage
		err = json.Unmarshal(v, &a)
		n = len(a)
	case bytes.HasPrefix(v, []byte(`"`)):
		var s string
		err = json.Unmarshal(v, &s)
		n = utf8.RuneCountInString(s)
	default:
		err = fmt.Errorf("%s has no len()", v)
	}
	if err != nil {
		return 0, fmt.Errorf("config: %s: act_costs: %w: %w", c.path, err, backend.ErrIncompatibleCheckpoint)
	}
	return n + 1, nil
}

// Temperatures returns the two temperature tables agent.py:194-195 reads:
// byQType is cfg.get("temperature", [1.0, 1.0, 1.0]), indexed by qtype id
// (agent.py:304), and byOptions is cfg.get("temperature_by_options", {}),
// keyed by temp_bucket. An absent key yields the upstream default; byOptions
// is never nil. Numbers are decoded as JSON float64, unrounded, since they
// divide the logits (agent.py:305).
//
// A present value that is not the expected shape fails with
// backend.ErrIncompatibleCheckpoint: temperature must be a list of exactly
// three numbers, temperature_by_options an object of numbers. A present null
// fails too, since Python indexes or .get()s it and raises.
func (c *Config) Temperatures() (byQType [3]float64, byOptions map[string]float64, err error) {
	byQType = [3]float64{1, 1, 1}
	if raw, ok := c.fields["temperature"]; ok {
		var elems []json.RawMessage
		if err := decodeContainer(raw, '[', &elems); err != nil {
			return byQType, nil, c.fieldErr("temperature", err)
		}
		if len(elems) != len(byQType) {
			return byQType, nil, c.fieldErr("temperature",
				fmt.Errorf("has %d elements, want %d", len(elems), len(byQType)))
		}
		for i, e := range elems {
			if byQType[i], err = decodeNumber(e); err != nil {
				return byQType, nil, c.fieldErr("temperature", fmt.Errorf("element %d: %w", i, err))
			}
		}
	}

	byOptions = map[string]float64{}
	if raw, ok := c.fields["temperature_by_options"]; ok {
		var m map[string]json.RawMessage
		if err := decodeContainer(raw, '{', &m); err != nil {
			return byQType, nil, c.fieldErr("temperature_by_options", err)
		}
		for k, v := range m {
			if byOptions[k], err = decodeNumber(v); err != nil {
				return byQType, nil, c.fieldErr("temperature_by_options", fmt.Errorf("%q: %w", k, err))
			}
		}
	}
	return byQType, byOptions, nil
}

// MaxLen is the full sequence budget, cfg.get("max_len", 512)
// (agent.py:256). See intField for what fails.
func (c *Config) MaxLen() (int, error) { return c.intField("max_len", 512) }

// HeadMaxLen is the options budget, cfg.get("head_max_len", 192)
// (agent.py:257). See intField for what fails.
func (c *Config) HeadMaxLen() (int, error) { return c.intField("head_max_len", 192) }

// HeadLayers is the decision head's layer count, cfg.get("head_layers", 2)
// (common.py:137). LoadConfig has required the key. DecisionModel tests
// head_layers > 0 first and builds no head otherwise (common.py:98), so
// every number that is not positive, a float such as -0.5 included, is 0.
// Only a positive value reaches range() in nn.TransformerEncoder, so it must
// be a JSON integer. Anything else fails with
// backend.ErrIncompatibleCheckpoint, as it does in Python: null and a string
// in the comparison, a positive float in range(). The one exception is a
// bool, which Python takes as 0 or 1 and this refuses.
func (c *Config) HeadLayers() (int, error) {
	raw := bytes.TrimSpace(c.fields["head_layers"])
	if n, err := strconv.Atoi(string(raw)); err == nil {
		return max(n, 0), nil
	}
	f, err := decodeNumber(raw)
	if err != nil {
		return 0, c.fieldErr("head_layers", err)
	}
	if f <= 0 {
		return 0, nil
	}
	return 0, c.fieldErr("head_layers", fmt.Errorf("%s is not an integer", raw))
}

// intField reads key as a positive JSON integer, or dflt when it is absent.
// build_sequence slices with the value (common.py:82-86), so Python needs an
// int: a float literal such as 512.0, a string, a bool (which Go cannot tell
// apart from Python's int subclass anyway) or null fails with
// backend.ErrIncompatibleCheckpoint. So does a non-positive integer, which
// Python accepts. A max_len ≤ 0 leaves build_sequence no option marker
// (common.py:86), so the first prediction raises a ValueError that blames
// head_max_len (agent.py:262-263); a head_max_len ≤ 0 silently cuts every
// option to 4 tokens and the question head to 8 (common.py:70-75). Failing
// loudly at load is a deliberate deviation.
func (c *Config) intField(key string, dflt int) (int, error) {
	raw, ok := c.fields[key]
	if !ok {
		return dflt, nil
	}
	n, err := strconv.Atoi(string(bytes.TrimSpace(raw)))
	if err != nil {
		return 0, c.fieldErr(key, fmt.Errorf("%s is not an integer", raw))
	}
	if n <= 0 {
		return 0, c.fieldErr(key, fmt.Errorf("%d is not positive", n))
	}
	return n, nil
}

// fieldErr wraps err as a failure of the config's key.
func (c *Config) fieldErr(key string, err error) error {
	return fmt.Errorf("config: %s: %s: %w: %w", c.path, key, err, backend.ErrIncompatibleCheckpoint)
}

// decodeContainer decodes raw into dst after checking it opens with open ('['
// or '{'); json.Unmarshal would otherwise accept null as an empty value.
func decodeContainer(raw json.RawMessage, open byte, dst any) error {
	v := bytes.TrimSpace(raw)
	if len(v) == 0 || v[0] != open {
		kind := "array"
		if open == '{' {
			kind = "object"
		}
		return fmt.Errorf("%s is not a JSON %s", v, kind)
	}
	return json.Unmarshal(v, dst)
}

// decodeNumber decodes raw as a JSON number. Unmarshal into a float64 would
// accept null silently, so it is rejected first; strings and bools already
// fail there.
func decodeNumber(raw json.RawMessage) (float64, error) {
	v := bytes.TrimSpace(raw)
	if bytes.Equal(v, []byte("null")) {
		return 0, errors.New("null is not a number")
	}
	var f float64
	if err := json.Unmarshal(v, &f); err != nil {
		return 0, fmt.Errorf("%s is not a number", v)
	}
	return f, nil
}
