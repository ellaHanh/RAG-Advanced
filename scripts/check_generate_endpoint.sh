#!/usr/bin/env bash
# Quick integration check for POST /generate.
# Usage: ./scripts/check_generate_endpoint.sh [BASE_URL]
# Example: ./scripts/check_generate_endpoint.sh http://localhost:8000
#
# One-liners (from host, API on localhost:8000):
#   curl -s http://localhost:8000/openapi.json | grep -o '"/[^"]*"' | tr -d '"' | sort -u
#   curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8000/generate -H "Content-Type: application/json" -d '{"query":"test"}'
# If 404 = route missing. If 422/502/503 = route exists.
#
# If route is missing, check API startup logs:
#   docker compose logs api
# And test import inside container:
#   docker compose run --rm api python -c "from api.routes.generate import generate_endpoint; print('OK')"

set -e
BASE_URL="${1:-http://localhost:8000}"

echo "=== 1. Check if /generate is in OpenAPI ==="
if curl -s "${BASE_URL}/openapi.json" | grep -q '"/generate"'; then
    echo "OK: /generate is registered in OpenAPI."
else
    echo "FAIL: /generate is NOT in OpenAPI. Route not registered."
    echo "Listed paths:"
    curl -s "${BASE_URL}/openapi.json" | python3 -c "import json,sys; d=json.load(sys.stdin); print('\n'.join(sorted(d.get('paths',{}).keys())))" 2>/dev/null || true
    exit 1
fi

echo ""
echo "=== 2. POST /generate (expect 200, 422, or 502/503) ==="
HTTP_CODE=$(curl -s -o /tmp/generate_response.json -w "%{http_code}" \
    -X POST "${BASE_URL}/generate" \
    -H "Content-Type: application/json" \
    -d '{"query": "What is RAG?", "strategy": "standard", "limit": 2}')
echo "HTTP status: $HTTP_CODE"

if [ "$HTTP_CODE" = "404" ]; then
    echo "FAIL: 404 Not Found - /generate route does not exist."
    exit 1
fi

if [ "$HTTP_CODE" = "200" ]; then
    echo "OK: 200. Response keys:"
    python3 -c "import json; d=json.load(open('/tmp/generate_response.json')); print(list(d.keys()))" 2>/dev/null || cat /tmp/generate_response.json | head -c 300
    echo ""
fi

if [ "$HTTP_CODE" = "422" ]; then
    echo "OK: Route exists (422 = validation error, e.g. missing body)."
fi

if [ "$HTTP_CODE" = "502" ] || [ "$HTTP_CODE" = "503" ]; then
    echo "Route exists; server returned $HTTP_CODE (retrieval/generation error or dependency)."
fi

echo ""
echo "Done. POST /generate is available."
