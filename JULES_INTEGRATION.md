## Jules API workflow

### Currently triggerable tasks

| # | Task | File | Status |
|---|------|------|------|
| 1 | Generalized Bouc-Wen hysteresis model | `jules_jobs/generalized_bouc_wen.json` | ✅ Ready |

### Trigger mode

```bash
# Method 1: Use script
export JULES_API_KEY="yourAPI_KEY"
bash trigger_jules.sh jules_jobs/generalized_bouc_wen.json

# Method 2: Directly call the API
curl -X POST https://jules.googleapis.com/v1alpha/sessions \
  -H "Content-Type: application/json" \
  -H "X-Goog-Api-Key: $JULES_API_KEY" \
  -d @jules_jobs/generalized_bouc_wen.json
```

### Workflow description
1. Hermes analysis paper → Generate Markdown report
2. Extract mathematical model → Build Jules prompt
3. Trigger Jules → Generate project code → Automatically create PR
