// Package checkpoint reads and validates what a laya checkpoint directory
// holds besides the graph: today its rl_agent_config.json.
package checkpoint

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
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
