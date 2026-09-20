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
    markdownlint '**/*.md' --ignore original --ignore node_modules

# Generate a coverage report
cover:
    go test -coverprofile=coverage.txt -covermode=atomic ./...
    go tool cover -html=coverage.txt -o coverage.html

# Ensure go.mod/go.sum are tidy
check-tidy:
    go mod tidy
    git diff --exit-code go.mod go.sum

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
