#!/usr/bin/env bash
# Jules API trigger script
# Usage: bash trigger_jules.sh <job_file.json>
# Requires: JULES_API_KEY environment variable

set -e

API_KEY="${JULES_API_KEY:?Need to set JULES_API_KEY environment variable}"
JOB_FILE="$1"

if [ -z "$JOB_FILE" ]; then
    echo "Usage: bash trigger_jules.sh <job_file.json>"
    echo ""
    echo "Available jobs:"
    ls jules_jobs/*.json | while read f; do
        echo "  $(basename $f)"
    done
    exit 1
fi

echo "📤 Send task to Jules..."
echo "    Job: $JOB_FILE"

RESULT=$(curl -s -X POST https://jules.googleapis.com/v1alpha/sessions \
  -H "Content-Type: application/json" \
  -H "X-Goog-Api-Key: $API_KEY" \
  -d @"$JOB_FILE")

SESSION_ID=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('name',''))" 2>/dev/null)
SESSION_URL=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('url',''))" 2>/dev/null)

echo ""
echo "✅ Task has been sent!"
echo "   Session: $SESSION_ID"
echo "Monitor: $SESSION_URL"
