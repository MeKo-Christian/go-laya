package native

import (
	"fmt"
	"math"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"github.com/MeKo-Christian/go-laya/internal/safetensors"
)

// TestShortList: a list holds at most maxListed names however many are
// added, and they are the least ones whatever order they come in, so the
// error is the same for the same file.
func TestShortList(t *testing.T) {
	const n = 1000
	var l shortList
	for i := n - 1; i >= 0; i-- {
		l.add(fmt.Sprintf("n%04d", i))
		if len(l.least) > maxListed || cap(l.least) > maxListed+1 {
			t.Fatalf("after %d names the list holds %d (cap %d), want at most %d",
				n-i, len(l.least), cap(l.least), maxListed)
		}
	}
	want := `"n0000", "n0001", "n0002", "n0003", "n0004", "n0005", "n0006", "n0007", "n0008", "n0009" and 990 more`
	if got := l.format(strconv.Quote); got != want {
		t.Errorf("format = %s, want %s", got, want)
	}

	var short shortList
	for _, name := range []string{"b", "a"} {
		short.add(name)
	}
	if got := short.format(strconv.Quote); got != `"a", "b"` {
		t.Errorf("format = %s, want \"a\", \"b\"", got)
	}
}

// TestCheckNamesBoundedAllocs: a header of many unexpected names costs
// checkNames no more allocations than one of a few, so a hostile file
// cannot make the error itself large.
func TestCheckNamesBoundedAllocs(t *testing.T) {
	allocs := func(n int) float64 {
		s := newSynth(5)
		for i := range n {
			s.zeros(fmt.Sprintf("decoder.%06d.weight", i), 1)
		}
		dir := s.write(t)
		cfg, err := readConfig(dir)
		if err != nil {
			t.Fatal(err)
		}
		if cfg.plan, err = cfg.Layers(); err != nil {
			t.Fatal(err)
		}
		f, err := safetensors.Open(filepath.Join(dir, weightsFile))
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		var msg string
		a := testing.AllocsPerRun(3, func() {
			msg = checkNames(f, cfg, tinyHeadLayers).Error()
		})
		if want := fmt.Sprintf("and %d more", n-maxListed); !strings.Contains(msg, want) {
			t.Errorf("checkNames = %s, want it to say %s", msg, want)
		}
		return a
	}
	few, many := allocs(20), allocs(20000)
	t.Logf("%v allocations for 20 unexpected names, %v for 20000", few, many)
	// A list that grew with the header would cost at least one allocation
	// per name, 20000 here. The bound leaves room for the few allocations
	// the runtime makes on its own, which AllocsPerRun counts too: the
	// macOS race runner saw 255 against 273.
	if many > 2*few {
		t.Errorf("%v allocations for 20000 unexpected names, %v for 20: the error grows with the header", many, few)
	}
}

// TestLoadKeepsOneCopy: load hands each decoded weight to its tensor rather
// than copying it (PLAN.md B.16). The checkpoint's embedding table is 16 MiB
// and every other tensor together a few KiB, so what one load allocates is
// that table, decoded once, plus Float32's read buffer of at most chunkBytes,
// 1 MiB. A copy of the table would add 16 MiB more. The table is large next
// to the buffer, so the bound still holds if chunkBytes grows several times.
func TestLoadKeepsOneCopy(t *testing.T) {
	const vocab = 1 << 19
	const table = vocab * tinyHidden * 4
	s := newSynth(1)
	s.config["vocab_size"] = vocab
	s.zeros("encoder.embeddings.tok_embeddings.weight", vocab, tinyHidden)
	dir := s.write(t)
	cfg, err := readConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, weightsFile)
	f, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	// The least of a few loads, so a goroutine allocating elsewhere can
	// only make one load look larger, never pass the bound for it.
	least := uint64(math.MaxUint64)
	for range 3 {
		var before, after runtime.MemStats
		runtime.ReadMemStats(&before)
		_, err := load(t.Context(), f, path, cfg)
		runtime.ReadMemStats(&after)
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		least = min(least, after.TotalAlloc-before.TotalAlloc)
	}
	t.Logf("one load allocates %d bytes for a %d-byte embedding table", least, table)
	switch {
	case least < table:
		t.Fatalf("one load allocates %d bytes, less than the %d-byte embedding table it decodes: the measurement is broken",
			least, table)
	case least >= table+table/2:
		t.Errorf("one load allocates %d bytes for a %d-byte embedding table, over 1.5 tables: load holds a copy of each weight",
			least, table)
	}
}
