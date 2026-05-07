## Jules API 工作流

### 当前可触发的任务

| # | 任务 | 文件 | 状态 |
|---|------|------|------|
| 1 | Generalized Bouc-Wen 迟滞模型 | `jules_jobs/generalized_bouc_wen.json` | ✅ 就绪 |

### 触发方式

```bash
# 方式1: 使用脚本
export JULES_API_KEY="你的API_KEY"
bash trigger_jules.sh jules_jobs/generalized_bouc_wen.json

# 方式2: 直接调用 API
curl -X POST https://jules.googleapis.com/v1alpha/sessions \
  -H "Content-Type: application/json" \
  -H "X-Goog-Api-Key: $JULES_API_KEY" \
  -d @jules_jobs/generalized_bouc_wen.json
```

### 工作流说明
1. Hermes 分析论文 → 生成 Markdown 报告
2. 提取数学模型 → 构建 Jules prompt
3. 触发 Jules → 生成工程代码 → 自动创建 PR
