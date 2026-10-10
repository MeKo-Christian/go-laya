package native

import (
	"fmt"
	"path/filepath"

	"github.com/MeKo-Christian/go-laya/backend"
	"github.com/MeKo-Christian/go-laya/internal/head"
	"github.com/MeKo-Christian/go-laya/internal/safetensors"
)

// HeadShape is what a checkpoint's model.safetensors fixes of the decision
// head, and Open builds: the head's layer count and its act width.
// Upstream takes both from rl_agent_config.json instead (head_layers and
// len(act_costs)+1, common.py:137) and its strict load_state_dict refuses a
// file that disagrees, so a caller holds this to the config.
type HeadShape struct {
	// Layers is how many head.layers.N the file numbers.
	Layers int
	// ActWidth is act_head.2.weight's row count, the act logits per row.
	ActWidth int
}

// ReadHeadShape reads dir/model.safetensors' header, and no tensor, and
// returns the HeadShape Open would build from it. Open counts the layers and
// the act rows from the same names and shape, and refuses a file whose
// numbering has gaps, so a shape this returns is the one an Open of the same
// file either builds or refuses.
//
// A missing or unreadable file is the error safetensors.Open gives; an
// act_head.2.weight that is missing or not of rank 2 wraps
// backend.ErrIncompatibleCheckpoint.
func ReadHeadShape(dir string) (HeadShape, error) {
	path := filepath.Join(dir, weightsFile)
	f, err := safetensors.Open(path)
	if err != nil {
		return HeadShape{}, fmt.Errorf("native backend: %w", err)
	}
	defer f.Close()

	const actName = "act_head.2.weight"
	info, ok := f.Info(actName)
	if !ok || len(info.Shape) != 2 {
		return HeadShape{}, fmt.Errorf("native backend: %s: tensor %q is missing or not [n_act, %d]: %w",
			path, actName, head.ActHidden, backend.ErrIncompatibleCheckpoint)
	}
	return HeadShape{
		Layers:   len(layerIndices(f.Names(), headLayers)),
		ActWidth: int(info.Shape[0]),
	}, nil
}
