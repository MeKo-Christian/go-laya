// Package native is the pure-Go backend.Backend (PLAN.md M8, Task 8.10, D8):
// a laya DecisionModel run without a shared library, from the checkpoint's
// model.safetensors and encoder/config.json.
//
// Open reads the encoder's layer plan through internal/modernbert's Config,
// decodes every weight with internal/safetensors (F16 or F32 to float32), and
// assembles the encoder of internal/modernbert and the decision head of
// internal/head from them by their state_dict names (docs/ARCHITECTURE.md
// §1.2-1.3):
//
//	encoder.embeddings.tok_embeddings.weight, encoder.embeddings.norm.weight
//	encoder.layers.N.{attn_norm (not on layer 0), attn.Wqkv, attn.Wo,
//	                  mlp_norm, mlp.Wi, mlp.Wo}.weight
//	encoder.final_norm.weight
//	type_emb.weight
//	head.layers.N.{self_attn.in_proj_weight, self_attn.in_proj_bias,
//	               self_attn.out_proj, linear1, linear2, norm1, norm2}
//	scorer.{0,1,3}, act_head.{0,2}    (the nn.Sequential indices, common.py:100-101)
//	temperature                       (checked, never decoded)
//
// Loading is strict: a checkpoint whose tensors are not exactly these, at the
// shapes encoder/config.json implies, is backend.ErrIncompatibleCheckpoint.
//
// Forward runs the encoder and then the head. Both only read their weights
// and allocate every intermediate per call, so concurrent Forward calls share
// one model; an RWMutex guards only the lifecycle against Close, as in
// internal/backend/onnx.
//
// laya.WithRuntime(laya.RuntimeNative) selects this backend (D30). The
// root's loader downloads model.safetensors and encoder/config.json for it,
// and holds ReadHeadShape's head to rl_agent_config.json before Open.
package native
