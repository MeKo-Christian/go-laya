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
// Two goroutines on one Router may evict each other's agent: at max_loaded=1,
// a request in another language can close the agent between the load and its
// forward pass. Close waits for a pass already running, so nothing is freed
// under it, but a pass that has not started fails, as the ONNX backend fails
// every call after Close. Python has the same window and GC to hide it.
func (r *Router) Predict(ctx context.Context, state any, qs Questions, opts ...RouteOption) (*Result, error) {
	decision, err := r.Route(state, qs, opts...)
	if err != nil {
		return nil, err
	}
	agent, err := r.Load(ctx, decision.Model)
	if err != nil {
		return nil, err
	}
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
