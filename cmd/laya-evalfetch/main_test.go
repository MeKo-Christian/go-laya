package main

import (
	"io"
	"testing"
)

func TestParseFlags(t *testing.T) {
	o, err := parseFlags(nil, io.Discard)
	if err != nil || o.n != 200 || o.update || o.out != "build/eval" {
		t.Fatalf("parseFlags() = %+v, %v; want the defaults", o, err)
	}
	if o, err = parseFlags([]string{"-update", "-n=5"}, io.Discard); err != nil || o.n != 5 || !o.update {
		t.Errorf("parseFlags(-update -n=5) = %+v, %v", o, err)
	}
	// -n=-1 used to reach SampleOffsets and panic there; zero drew an empty set.
	for _, args := range [][]string{{"-update", "-n=-1"}, {"-update", "-n=0"}, {"extra"}} {
		if o, err := parseFlags(args, io.Discard); err == nil {
			t.Errorf("parseFlags(%q) = %+v, want an error", args, o)
		}
	}
}
