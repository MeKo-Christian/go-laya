//nolint:gosmopolitan // the CJK literal is a routing fixture, not UI text
package laya

import (
	"context"
	"errors"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
)

// stubAgent stands in for a loaded checkpoint, as _Stub does upstream
// (test_router.py:167-172). The whole of what the Router asks of an agent is
// Close, so the whole of what a stub needs is to record it.
type stubAgent struct {
	name   string
	closed atomic.Int32
}

func (s *stubAgent) Close() error {
	s.closed.Add(1)
	return nil
}

// stubLoader is the Go form of upstream's `rr.load = lambda n: ...`
// monkeypatch. WithLoader is what makes the injection expressible without
// reflection (task 3.2.3); it also records the call order, because "was it
// built again?" is the question most of these tests are really asking.
type stubLoader struct {
	mu    sync.Mutex
	built []string
	made  map[string]*stubAgent
}

func newStubLoader() *stubLoader {
	return &stubLoader{made: map[string]*stubAgent{}}
}

func (l *stubLoader) fn(_ context.Context, name string, _ ModelSpec) (Agent, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.built = append(l.built, name)
	a := &stubAgent{name: name}
	l.made[name] = a
	return a, nil
}

func (l *stubLoader) calls() []string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return slices.Clone(l.built)
}

func stubbedRouter(t *testing.T, maxLoaded int, extra ...RouterOption) (*Router, *stubLoader) {
	t.Helper()
	l := newStubLoader()
	opts := append([]RouterOption{WithMaxLoaded(maxLoaded), WithLoader(l.fn)}, extra...)
	r, err := NewRouter(opts...)
	if err != nil {
		t.Fatalf("NewRouter: %v", err)
	}
	return r, l
}

func mustLoad(t *testing.T, r *Router, name string) Agent {
	t.Helper()
	a, err := r.Load(context.Background(), name)
	if err != nil {
		t.Fatalf("Load(%q): %v", name, err)
	}
	return a
}

// A hit returns the cached agent and does not build a second one
// (router.py:171-173).
func TestRouterLoadCaches(t *testing.T) {
	r, l := stubbedRouter(t, 2)

	first := mustLoad(t, r, "english")
	second := mustLoad(t, r, "en") // an alias resolves to the same entry

	if first != second {
		t.Error("a cache hit returned a different agent")
	}
	if got := l.calls(); len(got) != 1 {
		t.Errorf("loader called %d times (%v), want 1", len(got), got)
	}
}

// _order is least-recently-used first, load appends and _evict pops from the
// front (invariant #49). The cap-1 case is the one worth stating: load appends
// *before* evicting, so the newcomer survives and the incumbent is dropped.
func TestRouterEvictsLeastRecentlyUsed(t *testing.T) {
	t.Run("cap-1-keeps-newest", func(t *testing.T) {
		r, _ := stubbedRouter(t, 1)
		mustLoad(t, r, "english")
		mustLoad(t, r, "multilingual")

		if got := r.Loaded(); !slices.Equal(got, []string{"multilingual"}) {
			t.Errorf("Loaded() = %v, want [multilingual]", got)
		}
	})

	t.Run("cap-2-evicts-oldest", func(t *testing.T) {
		r, _ := stubbedRouter(t, 2)
		mustLoad(t, r, "english")
		mustLoad(t, r, "multilingual")
		mustLoad(t, r, "typed-decisions")

		if got := r.Loaded(); !slices.Equal(got, []string{"multilingual", "typed-decisions"}) {
			t.Errorf("Loaded() = %v, want [multilingual typed-decisions]", got)
		}
	})

	t.Run("a-hit-protects-from-the-next-eviction", func(t *testing.T) {
		r, _ := stubbedRouter(t, 2)
		mustLoad(t, r, "english")
		mustLoad(t, r, "multilingual")
		mustLoad(t, r, "english") // touch: english becomes most recent
		mustLoad(t, r, "typed-decisions")

		if got := r.Loaded(); !slices.Equal(got, []string{"english", "typed-decisions"}) {
			t.Errorf("Loaded() = %v, want [english typed-decisions]", got)
		}
	})
}

// Task 3.2.4, and a deviation from upstream: Python's _evict drops the victim
// from both dicts and lets refcounting free it, which in Go frees nothing. A
// leaked ONNX Runtime session is hundreds of megabytes, so eviction closes.
func TestRouterEvictionClosesTheVictim(t *testing.T) {
	r, l := stubbedRouter(t, 1)
	mustLoad(t, r, "english")
	victim := l.made["english"]

	mustLoad(t, r, "multilingual")

	if got := victim.closed.Load(); got != 1 {
		t.Errorf("evicted agent closed %d times, want exactly 1", got)
	}
	if survivor := l.made["multilingual"]; survivor.closed.Load() != 0 {
		t.Error("the surviving agent was closed")
	}
}

// attach registers a pre-built agent and raises max_loaded so the LRU cannot
// immediately evict it (invariant #50, test_router.py:259-270). It accepts
// aliases.
func TestRouterAttach(t *testing.T) {
	r, _ := stubbedRouter(t, 1)
	sentinel := &stubAgent{name: "sentinel"}

	if err := r.Attach("en", sentinel); err != nil {
		t.Fatalf("Attach: %v", err)
	}
	if got := r.Loaded(); !slices.Equal(got, []string{"english"}) {
		t.Errorf("Loaded() = %v, want [english]", got)
	}

	r.SetMaxLoaded(2)
	mustLoad(t, r, "multilingual")

	if got := r.Loaded(); !slices.Equal(got, []string{"english", "multilingual"}) {
		t.Errorf("Loaded() = %v, want [english multilingual]", got)
	}
	got, err := r.Load(context.Background(), "english")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got != Agent(sentinel) {
		t.Error("the attached agent was replaced")
	}
}

// An attached agent belongs to whoever attached it. The Router closes what its
// own loader built and nothing else, so a caller can attach one checkpoint to
// several routers, or keep using it after the router is done.
func TestRouterNeverClosesAnAttachedAgent(t *testing.T) {
	r, _ := stubbedRouter(t, 1)
	sentinel := &stubAgent{name: "sentinel"}
	if err := r.Attach("english", sentinel); err != nil {
		t.Fatalf("Attach: %v", err)
	}

	r.SetMaxLoaded(1)
	mustLoad(t, r, "multilingual") // evicts english
	if err := r.Unload("multilingual"); err != nil {
		t.Fatalf("Unload: %v", err)
	}
	if err := r.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	if got := sentinel.closed.Load(); got != 0 {
		t.Errorf("attached agent closed %d times, want 0", got)
	}
}

// preload raises max_loaded to fit what it builds, otherwise the LRU would
// evict what it just made, and it skips names that are already resident
// (invariant #51).
func TestRouterPreload(t *testing.T) {
	t.Run("raises-the-cap", func(t *testing.T) {
		r, _ := stubbedRouter(t, 1)
		if err := r.Preload(context.Background()); err != nil {
			t.Fatalf("Preload: %v", err)
		}
		if got := r.MaxLoaded(); got < 3 {
			t.Errorf("MaxLoaded() = %d, want at least 3", got)
		}
		if got := len(r.Loaded()); got != 3 {
			t.Errorf("%d resident, want 3", got)
		}
	})

	t.Run("skips-what-is-already-there", func(t *testing.T) {
		r, l := stubbedRouter(t, 3)
		mustLoad(t, r, "english")
		if err := r.Preload(context.Background(), "english", "multilingual"); err != nil {
			t.Fatalf("Preload: %v", err)
		}
		if got := l.calls(); !slices.Equal(got, []string{"english", "multilingual"}) {
			t.Errorf("loader calls = %v, want [english multilingual]", got)
		}
	})

	// Python walks its models dict, whose order is the DEFAULT_MODELS literal.
	// A Go map has no order, so preloading everything would otherwise give a
	// different Loaded() on every run -- which is exactly what -count=10 is
	// there to catch.
	t.Run("all-models-in-a-stable-order", func(t *testing.T) {
		r, _ := stubbedRouter(t, 3)
		if err := r.Preload(context.Background()); err != nil {
			t.Fatalf("Preload: %v", err)
		}
		want := []string{"english", "multilingual", "typed-decisions"}
		if got := r.Loaded(); !slices.Equal(got, want) {
			t.Errorf("Loaded() = %v, want %v", got, want)
		}
	})
}

// unload() with no argument clears everything; with a name it removes just
// that one (invariant #52).
func TestRouterUnload(t *testing.T) {
	r, _ := stubbedRouter(t, 3)
	r.SetMaxLoaded(3)
	mustLoad(t, r, "english")
	mustLoad(t, r, "multilingual")

	if err := r.Unload("english"); err != nil {
		t.Fatalf("Unload: %v", err)
	}
	if got := r.Loaded(); !slices.Equal(got, []string{"multilingual"}) {
		t.Errorf("Loaded() = %v, want [multilingual]", got)
	}

	if err := r.Unload("nothing-is-loaded-under-this"); err != nil {
		t.Fatalf("Unload of an unknown name: %v", err)
	}
	if err := r.Unload(); err != nil {
		t.Fatalf("Unload all: %v", err)
	}
	if got := r.Loaded(); len(got) != 0 {
		t.Errorf("Loaded() = %v, want empty", got)
	}
}

// max_loaded = max(1, int(max_loaded)) (invariant #48), and the upstream tests
// raise it after construction, which a constructor-only API cannot express
// (task 3.1.6, test_router.py:241,249,266).
func TestRouterMaxLoadedClampsToOne(t *testing.T) {
	for _, n := range []int{0, -1, -100} {
		r, err := NewRouter(WithMaxLoaded(n))
		if err != nil {
			t.Fatalf("NewRouter: %v", err)
		}
		if got := r.MaxLoaded(); got != 1 {
			t.Errorf("WithMaxLoaded(%d) gave MaxLoaded() = %d, want 1", n, got)
		}
		r.SetMaxLoaded(n)
		if got := r.MaxLoaded(); got != 1 {
			t.Errorf("SetMaxLoaded(%d) gave MaxLoaded() = %d, want 1", n, got)
		}
	}
}

// Task 3.2.7. Until M6 supplies a default loader there is nothing to build an
// agent from, and caching a nil agent that panics at first use would be the
// worst of the available answers.
func TestRouterLoadWithoutALoader(t *testing.T) {
	r, err := NewRouter()
	if err != nil {
		t.Fatalf("NewRouter: %v", err)
	}
	if _, err := r.Load(context.Background(), "english"); !errors.Is(err, ErrNoLoader) {
		t.Errorf("error = %v, want ErrNoLoader", err)
	}
}

// Task 3.2.6. Close releases everything the Router built, and is idempotent:
// a second call must not close the same session twice.
func TestRouterClose(t *testing.T) {
	r, l := stubbedRouter(t, 3)
	mustLoad(t, r, "english")
	mustLoad(t, r, "multilingual")

	if err := r.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := r.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}

	for name, a := range l.made {
		if got := a.closed.Load(); got != 1 {
			t.Errorf("%s closed %d times, want exactly 1", name, got)
		}
	}
	if got := r.Loaded(); len(got) != 0 {
		t.Errorf("Loaded() = %v after Close, want empty", got)
	}
}

// Task 3.2.2. Python's Router is not concurrency-safe and says so; a Go server
// would reach for one from several goroutines on day one. Run under -race.
func TestRouterConcurrentRouteAndEvict(t *testing.T) {
	r, _ := stubbedRouter(t, 1, WithAutoTaskDetection(true))
	names := []string{"english", "multilingual", "typed-decisions"}
	states := []any{"Please refund", "मुझसे दो बार", "12345", map[string]any{"body": "二重に請求"}}

	var wg sync.WaitGroup
	for i := range 8 {
		wg.Go(func() {
			for j := range 25 {
				if _, err := r.Route(states[(i+j)%len(states)], idQuestions("a")); err != nil {
					t.Errorf("Route: %v", err)
					return
				}
				if _, err := r.Load(context.Background(), names[(i+j)%len(names)]); err != nil {
					t.Errorf("Load: %v", err)
					return
				}
				r.Loaded()
			}
		})
	}
	wg.Wait()

	if got := len(r.Loaded()); got != 1 {
		t.Errorf("%d resident at max_loaded=1, want 1", got)
	}
}

// Invariant #50's actual content, which upstream does not test.
//
// test_router.py:264 asserts `ra.max_loaded >= 1` on a cap-1 router holding one
// attached agent -- true whatever attach does, since the cap already is 1. The
// raise only becomes observable with a second agent, and without it the next
// load evicts what was just attached, which is the whole thing attach exists to
// prevent. Deleting the raise leaves the upstream port entirely green.
func TestRouterAttachRaisesTheCapEnoughToHoldEverything(t *testing.T) {
	r, _ := stubbedRouter(t, 1)
	mustLoad(t, r, "english")

	sentinel := &stubAgent{name: "sentinel"}
	if err := r.Attach("multilingual", sentinel); err != nil {
		t.Fatalf("Attach: %v", err)
	}

	if got := r.MaxLoaded(); got != 2 {
		t.Errorf("MaxLoaded() = %d after attaching a second agent, want 2", got)
	}

	// With the cap raised, the next load evicts only the oldest. With the cap
	// still at 1 it would evict both, taking the attached agent with it.
	mustLoad(t, r, "typed-decisions")
	if got := r.Loaded(); !slices.Equal(got, []string{"multilingual", "typed-decisions"}) {
		t.Errorf("Loaded() = %v, want [multilingual typed-decisions]", got)
	}
}
