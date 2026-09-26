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
	fields map[string]json.RawMessage
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
	return &Config{fields: fields}, nil
}
