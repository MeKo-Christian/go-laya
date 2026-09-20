package tokenizer

import (
	"fmt"
	"strings"
)

// GPT-2's bytes_to_unicode: a bijection from the 256 byte values onto 256
// printable runes, so that arbitrary bytes survive a text-shaped vocabulary.
// The 188 bytes that are already printable map to themselves; the other 68 are
// shifted into U+0100 and up, in byte order.
//
// This is why a space appears as U+0120 and a tab as U+0109 in the English
// vocabulary, and why the vocabulary's alphabet is missing exactly 13 of these
// runes (0xC0, 0xC1, 0xF5-0xFF) -- those bytes never occur in valid UTF-8.
var (
	byteToRune [256]rune
	runeToByte map[rune]byte
)

func init() {
	printable := func(b int) bool {
		return (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)
	}
	next := rune(0x100)
	runeToByte = make(map[rune]byte, 256)
	for b := range 256 {
		r := rune(b)
		if !printable(b) {
			r = next
			next++
		}
		byteToRune[b] = r
		runeToByte[r] = byte(b)
	}
}

// byteEncode maps each byte of s through the table.
func byteEncode(s string) string {
	var b strings.Builder
	b.Grow(len(s) * 2)
	for i := range len(s) {
		b.WriteRune(byteToRune[s[i]])
	}
	return b.String()
}

// byteDecode inverts byteEncode. Inference never decodes -- this exists for
// task 4.5.1's round-trip invariant and for readable test failures.
func byteDecode(s string) (string, error) {
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		v, ok := runeToByte[r]
		if !ok {
			return "", fmt.Errorf("tokenizer: %q is not a byte-level rune", r)
		}
		b.WriteByte(v)
	}
	return b.String(), nil
}
