package laya

import (
	"context"
	"fmt"
)

// Predict routes the request, loads the chosen checkpoint and answers every
// question in one forward pass on it: Router.predict (router.py:293-309).
//
// The result is the agent's SystemOne result with the decision under Routing,
// as upstream sets result["routing"] = dict(decision). Result.Map and
// Result.MarshalJSON write it last, through jsonx.
//
// A failed route loads nothing. The load and the forward pass take ctx.
//
// The agent is leased for the length of the pass, so it outlives its own
// eviction as it does in Python, where the local reference keeps it
// (router.py:303-309). A concurrent request that evicts it, or an Unload or
// Close, drops it from the cache at once and closes it when this pass ends,
// waiting for that. An Agent whose SystemOne calls back into the same Router
// and drops itself would therefore wait on its own pass.
func (r *Router) Predict(ctx context.Context, state any, qs Questions, opts ...RouteOption) (*Result, error) {
	decision, err := r.Route(state, qs, opts...)
	if err != nil {
		return nil, err
	}
	agent, release, err := r.lease(ctx, decision.Model)
	if err != nil {
		return nil, err
	}
	defer release()

	res, err := agent.SystemOne(ctx, state, qs)
	if err != nil {
		return nil, fmt.Errorf("laya: %s: %w", decision.Model, err)
	}
	if res == nil {
		return nil, fmt.Errorf("laya: %s: SystemOne returned no result", decision.Model)
	}
	res.Routing = &decision
	return res, nil
}

// SystemOne is Predict under upstream's other name (router.py:311).
func (r *Router) SystemOne(ctx context.Context, state any, qs Questions, opts ...RouteOption) (*Result, error) {
	return r.Predict(ctx, state, qs, opts...)
}
