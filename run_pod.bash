#!/usr/bin/env bash
#
# Build the opencode-agent-pod image and run it as a container.
#
# Usage:
#   ./run_pod.bash              # build + (re)start the pod
#   ./run_pod.bash build        # build the image only
#
# Configuration (override via environment variables):
#   ENV_FILE        Env file passed to the container     (default: .env)
#   POD_PORT        Host port for the pod                (default: 8080)
#   IMAGE           Image name:tag                       (default: opencode-agent-pod)
#   CONTAINER       Container name                       (default: agent-pod)
#   HEALTH_TIMEOUT  Seconds to wait for /health          (default: 60)
#   DOCKER          Docker command                       (default: "docker")

set -euo pipefail

MODE="${1:-run}"
if [[ "$MODE" != "run" && "$MODE" != "build" ]]; then
    echo "Usage: $0 [run|build]  (default: run)" >&2
    exit 2
fi

# --- Configuration --------------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env}"
POD_PORT="${POD_PORT:-8080}"
IMAGE="${IMAGE:-opencode-agent-pod}"
CONTAINER="${CONTAINER:-agent-pod}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-60}"
DOCKER="${DOCKER:-docker}"

# --- Build ----------------------------------------------------------------
echo "Building $IMAGE..."
$DOCKER build -t "$IMAGE" "$SCRIPT_DIR"

if [[ "$MODE" == "build" ]]; then
    echo "Build complete."
    exit 0
fi

# --- Run ------------------------------------------------------------------
if [[ ! -f "$ENV_FILE" ]]; then
    echo "Env file not found: $ENV_FILE (cp env.example .env and set AGENT_POD_TOKEN)" >&2
    exit 1
fi

$DOCKER rm -f "$CONTAINER" &> /dev/null || true

$DOCKER run -d \
    --name "$CONTAINER" \
    --env-file "$ENV_FILE" \
    -p "$POD_PORT:8080" \
    "$IMAGE"

# --- Health wait ----------------------------------------------------------
echo "Waiting for the pod to become healthy (timeout: ${HEALTH_TIMEOUT}s)..."
deadline=$((SECONDS + HEALTH_TIMEOUT))
until curl -fsS "http://localhost:$POD_PORT/health" &> /dev/null; do
    if (( SECONDS >= deadline )); then
        echo "Pod did not become healthy within ${HEALTH_TIMEOUT}s; recent logs:" >&2
        $DOCKER logs --tail 40 "$CONTAINER" >&2
        exit 1
    fi
    sleep 2
done

echo "Pod is up: http://localhost:$POD_PORT"
echo
echo "Smoke test:"
echo "  curl -N -X POST http://localhost:$POD_PORT/agent/run \\"
echo "    -H \"Authorization: Bearer \$AGENT_POD_TOKEN\" -H 'Content-Type: application/json' \\"
echo "    -d '{\"prompt\": \"Say hello.\"}'"
