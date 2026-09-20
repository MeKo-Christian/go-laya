package laya

import (
	"regexp"
	"testing"
)

// semverRE is the official SemVer 2.0.0 pattern, anchored. Version is compared
// against a git tag with its "v" prefix stripped, so anything this rejects would
// make the release workflow's equality check meaningless.
var semverRE = regexp.MustCompile(
	`^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)` +
		`(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)` +
		`(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?` +
		`(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$`,
)

func TestVersionIsSemver(t *testing.T) {
	if Version == "" {
		t.Fatal("Version is empty; release.yml has nothing to compare the tag against")
	}

	if !semverRE.MatchString(Version) {
		t.Errorf("Version = %q, which is not valid semver; the release tag gate compares it to $GITHUB_REF_NAME with the leading v removed", Version)
	}
}

func TestVersionHasNoTagPrefix(t *testing.T) {
	if len(Version) > 0 && Version[0] == 'v' {
		t.Errorf("Version = %q must not carry the tag's leading v; release.yml strips it from the tag, not from the constant", Version)
	}
}
