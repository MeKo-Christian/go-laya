package laya

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// Task 7.7.6: Predict leases the agent it runs, and whatever drops that agent
// from the cache closes it only once the pass is over. Python gets this from
// the local reference router.predict holds (router.py:303-309); eviction there
// only pops the dicts (router.py:188-195).

// leaseWait bounds every wait below, so a lease that is never released fails a
// test instead of hanging it.
const leaseWait = 5 * time.Second

// gateAgent records whether a pass ever ran on it after Close. A gated one
// holds its first pass until release is closed.
type gateAgent struct {
	name    string
	gated   bool
	started chan struct{}
	release chan struct{}
	once    sync.Once
	closed  atomic.Int32
}

func newGateAgent(name string, gated bool) *gateAgent {
	return &gateAgent{name: name, gated: gated, started: make(chan struct{}), release: make(chan struct{})}
}

func (g *gateAgent) SystemOne(context.Context, any, Questions) (*Result, error) {
	if g.closed.Load() != 0 {
		return nil, fmt.Errorf("%s: pass started on a closed agent", g.name)
	}
	if g.gated {
		g.once.Do(func() { close(g.started) })
		<-g.release
	}
	if g.closed.Load() != 0 {
		return nil, fmt.Errorf("%s: closed under a running pass", g.name)
	}
	return &Result{Model: g.name, Answers: AnswerSet{}}, nil
}

func (g *gateAgent) Close() error {
	g.closed.Add(1)
	return nil
}

// gateLoader hands out gate for english, if set, and a fresh ungated agent
// for everything else, keeping every agent it built.
type gateLoader struct {
	gate *gateAgent
	mu   sync.Mutex
	made []*gateAgent
}

func (l *gateLoader) fn(_ context.Context, name string, _ ModelSpec) (Predictor, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	a := newGateAgent(name, false)
	if name == ModelEnglish && l.gate != nil {
		a = l.gate
	}
	l.made = append(l.made, a)
	return a, nil
}

func (l *gateLoader) agents() []*gateAgent {
	l.mu.Lock()
	defer l.mu.Unlock()
	return slices.Clone(l.made)
}

func gatedRouter(t *testing.T, gate *gateAgent) (*Router, *gateLoader) {
	t.Helper()
	l := &gateLoader{gate: gate}
	r, err := NewRouter(WithMaxLoaded(1), WithLoader(l.fn))
	if err != nil {
		t.Fatalf("NewRouter: %v", err)
	}
	return r, l
}

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(leaseWait)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(time.Millisecond)
	}
}

func receive(t *testing.T, what string, ch <-chan error) error {
	t.Helper()
	select {
	case err := <-ch:
		return err
	case <-time.After(leaseWait):
		t.Fatalf("timed out waiting for %s", what)
		return nil
	}
}

// duringPass runs drop while a Predict on english is inside its pass, and
// asserts the agent leaves the cache at once but is closed only after the
// pass, exactly once, with both calls succeeding.
func duringPass(t *testing.T, drop func(*Router) error) {
	t.Helper()
	ctx := context.Background()
	gate := newGateAgent(ModelEnglish, true)
	r, _ := gatedRouter(t, gate)

	passDone := make(chan error, 1)
	go func() {
		_, err := r.Predict(ctx, english, predictQs)
		passDone <- err
	}()
	select {
	case <-gate.started:
	case <-time.After(leaseWait):
		t.Fatal("the english pass never started")
	}

	dropDone := make(chan error, 1)
	go func() { dropDone <- drop(r) }()
	waitFor(t, "english to leave the cache", func() bool { return !slices.Contains(r.Loaded(), ModelEnglish) })

	select {
	case err := <-dropDone:
		t.Fatalf("the drop returned (%v) while the pass was still running", err)
	case <-time.After(50 * time.Millisecond):
	}
	if n := gate.closed.Load(); n != 0 {
		t.Fatalf("english closed %d times under a running pass", n)
	}

	close(gate.release)
	if err := receive(t, "the english pass", passDone); err != nil {
		t.Errorf("Predict(english) = %v, want it to finish on its leased agent", err)
	}
	if err := receive(t, "the drop", dropDone); err != nil {
		t.Errorf("drop = %v", err)
	}
	if n := gate.closed.Load(); n != 1 {
		t.Errorf("english closed %d times after its pass, want 1", n)
	}
}

func TestRouterPredictSurvivesEviction(t *testing.T) {
	duringPass(t, func(r *Router) error {
		res, err := r.Predict(context.Background(), hindi, predictQs)
		if err == nil && res.Model != ModelMultilingual {
			err = fmt.Errorf("the evicting request ran %q", res.Model)
		}
		return err
	})
}

func TestRouterPredictSurvivesUnload(t *testing.T) {
	duringPass(t, func(r *Router) error { return r.Unload(ModelEnglish) })
}

func TestRouterPredictSurvivesClose(t *testing.T) {
	duringPass(t, (*Router).Close)
}

// Requests in two languages at max_loaded=1 evict each other constantly; none
// may fail, and every agent is closed once, after the last pass on it.
func TestRouterPredictConcurrent(t *testing.T) {
	r, l := gatedRouter(t, nil)
	states := []any{english, hindi}

	var failures atomic.Int32
	var wg sync.WaitGroup
	for i := range 8 {
		wg.Go(func() {
			for j := range 25 {
				if _, err := r.Predict(context.Background(), states[(i+j)%2], predictQs); err != nil {
					failures.Add(1)
					t.Error(err)
					return
				}
			}
		})
	}
	wg.Wait()
	if failures.Load() != 0 {
		return
	}

	if got := len(r.Loaded()); got != 1 {
		t.Errorf("%d resident at max_loaded=1, want 1", got)
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	for _, a := range l.agents() {
		if n := a.closed.Load(); n != 1 {
			t.Errorf("%s agent closed %d times, want 1", a.name, n)
		}
	}
}

// A close error still reaches the call that dropped the agent, after the pass.
func TestRouterPredictReportsDeferredCloseError(t *testing.T) {
	ctx := context.Background()
	errClose := errors.New("close failed")
	gate := &failingGate{gateAgent: newGateAgent(ModelEnglish, true), err: errClose}
	r, err := NewRouter(WithMaxLoaded(1), WithLoader(func(context.Context, string, ModelSpec) (Predictor, error) {
		return gate, nil
	}))
	if err != nil {
		t.Fatal(err)
	}

	passDone := make(chan error, 1)
	go func() {
		_, err := r.Predict(ctx, english, predictQs)
		passDone <- err
	}()
	select {
	case <-gate.started:
	case <-time.After(leaseWait):
		t.Fatal("the english pass never started")
	}
	unloaded := make(chan error, 1)
	go func() { unloaded <- r.Unload(ModelEnglish) }()
	waitFor(t, "english to leave the cache", func() bool { return len(r.Loaded()) == 0 })

	close(gate.release)
	if err := receive(t, "the pass", passDone); err != nil {
		t.Errorf("Predict = %v", err)
	}
	if err := receive(t, "Unload", unloaded); !errors.Is(err, errClose) {
		t.Errorf("Unload = %v, want the close error", err)
	}
}

// failingGate is a gateAgent whose Close fails.
type failingGate struct {
	*gateAgent

	err error
}

func (f *failingGate) Close() error {
	_ = f.gateAgent.Close()
	return f.err
}
