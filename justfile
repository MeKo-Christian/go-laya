set shell := ["bash", "-uc"]

# Default recipe - show available commands
default:
    @just --list

# Build all packages
build:
    go build -v ./...

# Run all tests
test:
    go test -v -count=1 ./...

# Run tests with the race detector
test-race:
    go test -race -count=1 ./...

# Run linters
lint:
    golangci-lint run --timeout=5m ./...

# Run linters and apply the fixes they can make
lint-fix:
    golangci-lint run --fix --timeout=5m ./...

# Format everything treefmt knows about
fmt:
    treefmt . --allow-missing-formatter

# Fail if anything is unformatted
fmt-check:
    treefmt --allow-missing-formatter --fail-on-change

# Lint Markdown (checker only -- prettier owns Markdown formatting, see treefmt.toml)
lint-md:
    markdownlint '**/*.md' --ignore original --ignore models --ignore node_modules

# Generate a coverage report
cover:
    go test -coverprofile=coverage.txt -covermode=atomic ./...
    go tool cover -html=coverage.txt -o coverage.html

# Ensure go.mod/go.sum are tidy
check-tidy:
    go mod tidy
    # Not `git diff --exit-code go.mod go.sum`: that errors out when go.sum does not
    # exist yet, and it cannot see a go.sum that `go mod tidy` has just created.
    # --porcelain covers modified, created and deleted alike.
    if [ -n "$(git status --porcelain -- go.mod go.sum)" ]; then \
        git status --short -- go.mod go.sum; \
        git diff -- go.mod go.sum; \
        echo "go.mod/go.sum are not tidy - run 'go mod tidy' and commit the result"; \
        exit 1; \
    fi

# Spike S2: run the ONNX-binding spike (skips unless an ORT library and the S1 export exist)
spike-onnx:
    CGO_ENABLED=0 go test -count=1 -v ./internal/onnxspike/

# The S2.2 finalizer hunt. -race needs cgo, so this cannot also assert CGO_ENABLED=0.
spike-onnx-race count="200":
    go test -race -count={{count}} ./internal/onnxspike/

# Spike S3: the ORT-CPU latency sweep. An iteration costs 0.3-20 s, so the count is
# fixed rather than left to `go test` to calibrate against a wall-clock budget.
bench-onnx reps="5" out="build/bench-onnx.txt":
    mkdir -p build
    CGO_ENABLED=0 go test -run '^$' -bench 'BenchmarkForward' \
        -benchtime={{reps}}x -timeout 120m -v ./internal/onnxspike/ 2>&1 | tee {{out}}
    # One process per checkpoint: the reported RSS growth is only meaningful the first
    # time a session is built, and a shared process reports the second one as ~zero.
    for ck in english multilingual typed-decisions; do \
        CGO_ENABLED=0 go test -run '^$' -bench "BenchmarkSessionLoad/$ck\$" \
            -benchtime={{reps}}x -timeout 20m ./internal/onnxspike/ 2>&1 | tee -a {{out}}; \
    done

# Report known vulnerabilities in the dependency graph
vuln:
    govulncheck ./...

# Developer check: tests, linters, coverage
check: test lint cover

# What CI runs
ci: fmt-check lint-md test-race lint check-tidy

# lint-fix then fmt
fix:
    just lint-fix
    just fmt

# Remove build and coverage artefacts
clean:
    rm -f coverage.txt coverage.html
    rm -rf bin/
    go clean -cache -testcache
