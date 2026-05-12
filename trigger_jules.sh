#!/usr/bin/env bash
# Jules API 触发脚本
# 用法: bash trigger_jules.sh <job_file.json>
# 需要: JULES_API_KEY 环境变量

set -e

API_KEY="${JULES_API_KEY:?需设置 JULES_API_KEY 环境变量}"
JOB_FILE="$1"

if [ -z "$JOB_FILE" ]; then
    echo "用法: bash trigger_jules.sh <job_file.json>"
    echo ""
    echo "可用 jobs:"
    ls jules_jobs/*.json | while read f; do
        echo "  $(basename $f)"
    done
    exit 1
fi

echo "📤 发送任务到 Jules..."
echo "    Job: $JOB_FILE"

RESULT=$(curl -s -X POST https://jules.googleapis.com/v1alpha/sessions \
  -H "Content-Type: application/json" \
  -H "X-Goog-Api-Key: $API_KEY" \
  -d @"$JOB_FILE")

SESSION_ID=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('name',''))" 2>/dev/null)
SESSION_URL=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('url',''))" 2>/dev/null)

echo ""
echo "✅ 任务已发送！"
echo "   Session: $SESSION_ID"
echo "   监控: $SESSION_URL"
