#!/usr/bin/env bash
# Jules API 触发脚本
# 用法: bash trigger_jules.sh
# 需要: JULES_API_KEY 环境变量

set -e

API_KEY="${JULES_API_KEY:?需设置 JULES_API_KEY 环境变量}"
JOB_FILE="$1"
[ -z "$JOB_FILE" ] && JOB_FILE="jules_jobs/generalized_bouc_wen.json"

echo "📤 发送任务到 Jules..."
echo "    Job: $JOB_FILE"

curl -s -X POST "https://jules.googleapis.com/v1alpha/sessions" \
  -H "Content-Type: application/json" \
  -H "X-Goog-Api-Key: $API_KEY" \
  -d @"$JOB_FILE" | jq .

echo ""
echo "✅ 任务已发送！"
echo "   监控: https://askjules.ai/dashboard"
