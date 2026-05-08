# IEEE 控制算法研究枢纽 (IEEE Control Algorithm Research Hub)

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![Jules API](https://img.shields.io/badge/AI_Agent-Jules-orange.svg)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-brightgreen.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

本项目是一个...
本项目是一个自动化控制算法研究与代码生成平台，通过 **Jules AI** 将从学术论文中提取的数学模型直接转化为生产级 Python 代码，并自动创建 PR、运行 CI/CD 进行质量验证。

## 架构概览

```
研究层 (hermes)         翻译层 (Jules API)        代码层 (GitHub)
ieee_search.py  ──→  paper_analyzer.py  ──→  Jules job JSON  ──→  生成代码 + PR  ──→  合并 + CI/CD
                    (提取数学模型)          (含 prompt + source)
```

核心组件：
- **`hub.py`** – 中央调度脚本（搜索、分析、触发 Jules）
- **`agents/hermes/`** – 论文搜索与深度分析工具（ieee-search 技能）
- **`<project_name>/jobs/`** – 预构建的 Jules 任务 JSON 文件
- **`<project_name>/src/`** – Jules 生成的最终代码
- **`.github/workflows/pipeline.yml`** – CI/CD 流水线（代码风格、类型检查、单元测试）

## 仓库结构

```
ieee-research/
├── .github/workflows/pipeline.yml      # CI/CD 流水线
├── agents/hermes/                      # 研究代理脚本
│   ├── ieee_search.py                  # 多源论文搜索 (Semantic Scholar + OpenAlex)
│   ├── paper_analyzer.py               # 深度分析 → Markdown 报告
│   ├── citation_network.py             # 引用网络分析
│   ├── daily_digest.py                 # 每日/每周摘要生成
│   ├── notion_exporter.py              # 导出分析结果到 Notion
│   └── research_domains.yaml           # 研究方向配置 (关键词/优先级)
├── generalized_bouc_wen/               # 迟滞模型项目
│   ├── src/                            # GBW 模型 + PSO 辨识代码
│   ├── tests/                          # 测试代码
│   └── jobs/                           # Jules 任务定义
├── piezo_stepper_eso/                  # 压电步进电机控制项目
│   ├── src/                            # 压电步进电机 + ADRC
│   ├── tests/                          # 测试代码
│   └── jobs/                           # Jules 任务定义
├── piezo_walking_motor/                # 压电行走电机项目
│   ├── src/                            # 控制与绘图代码
│   ├── tests/                          # 测试代码
│   ├── assets/                         # 图像资源
│   └── jobs/                           # Jules 任务定义
├── pmn_pt_shear_actuator/              # PMN-PT 剪切执行器项目
│   ├── src/                            # ADRC 算法代码
│   ├── tests/                          # 测试代码
│   └── jobs/                           # Jules 任务定义
├── rmso_bw/                            # RMSO-BW 综合项目
│   ├── src/                            # RMSO-BW 补偿器, Fuzzy NN 控制
│   ├── tests/                          # 单元测试
│   ├── examples/                       # Jupyter 示例
│   ├── docs/                           # 论文文献
│   └── jobs/                           # Jules 任务定义
├── voltage_waveform/                   # 电压波形生成与优化项目
│   ├── src/                            # 梯形波、锯齿波生成器, 遗传优化器
│   ├── tests/                          # 测试代码
│   └── docs/                           # 论文文献
├── research/templates/                 # 研究领域模板 (YAML)
├── hub.py                              # 中央调度器
├── trigger_jules.sh                    # 发送任务到 Jules API 的辅助脚本
├── JULES_INTEGRATION.md                # Jules API 集成指南
└── AGENTS.md                           # AI 代理角色定义
```

## 研究→代码流水线

完整流程分为 6 步，可通过 `hub.py pipeline` 自动执行或手动按步骤操作。

### 1. 搜索论文

```bash
python agents/hermes/ieee_search.py --query "Bouc-Wen piezoelectric hysteresis" --max 5
```

输出 JSON 结果（含 DOI、标题、摘要、引用数）。

### 2. 深度分析

```bash
python agents/hermes/paper_analyzer.py --doi "10.1109/ACCESS.2020.2984645" --lang zh --output research/reports/analysis.md
```

生成 Markdown 报告，包含：
- 摘要原文、方法论分类、创新点
- 与用户研究领域的相关度评分（1–5 星）
- 可借鉴的技术点

### 3. 构建 Jules Job JSON

从分析报告中提取：
- **Prompt**：包含论文背景、数学模型、工程要求（文件路径、类名、方法签名、依赖约束、测试要求）
- **sourceContext**：指定 GitHub 仓库和起始分支
- **automationMode** = `"AUTO_CREATE_PR"`

示例结构 (参考 `jules_jobs/rmso_bw_fuzzy_nn.json`)：

```json
{
  "prompt": "## Task: Implement RMSO-BW Hysteresis Model...\n...",
  "sourceContext": {
    "source": "sources/github/Mada0m0/ieee-research",
    "githubRepoContext": { "startingBranch": "main" }
  },
  "title": "RMSO-BW Hysteresis Model with Fuzzy-NN Controller",
  "automationMode": "AUTO_CREATE_PR"
}
```

**⚠️ 重要**：`sourceContext` 必须包含 `githubRepoContext.startingBranch`，否则 API 返回 400 错误（旧格式已废弃）。

### 4. 上传参考文档并创建分支

将作业文件放入 `jules_jobs/`，论文引用放入 `references/`，然后推送到远程仓库的新分支：

```bash
git checkout -b feat/new-model
git add references/paper.md jules_jobs/new_model.json
git commit -m "feat: add Jules job for XXX model"
git push origin feat/new-model
```

**强阻塞校验**：等待远程分支完全可用（使用 `hub.py` 中的 `wait_for_remote_branch` 函数，避免 Jules 因分支未就绪而克隆失败）。

### 5. 触发 Jules API

```bash
curl -X POST https://jules.googleapis.com/v1alpha/sessions \
  -H "Content-Type: application/json" \
  -H "X-Goog-Api-Key: $JULES_API_KEY" \
  -d @jules_jobs/new_model.json
```

返回 `session` ID 和 web URL。Jules 将：
- 从 `main` 分支创建新工作分支（名称格式：`feat/model-name-<session_id>`）
- 生成所有文件（代码、测试、示例）
- 提交并创建 Pull Request

### 6. 合并 PR 并运行 CI/CD

PR 创建后，GitHub Actions 自动执行：
- `pycodestyle` 代码风格检查
- `mypy` 类型注解验证
- `pytest` 单元测试（必须全部通过）
- 若通过，可安全合并。

合并后，生成的代码即纳入主分支。

## 已有实现清单

| PR # | 标题 | 核心组件 | 论文来源 | 状态 |
|------|------|----------|----------|------|
| 1 | Initialize IEEE Research Hub | 基础结构 | – | ✅ 合并 |
| 2 | Generalized Bouc-Wen + PSO | `generalized_bouc_wen.py`, PSO 辨识 | Zhou 2024 (GBW) | ✅ 合并 |
| 3 | Piezo Stepper ESO | `piezo_stepper_eso.py`, ADRC | – | ✅ 合并 |
| 4 | Walking Piezo Motor (ADRC) | `piezo_walking_motor.py`, ADRC | – | ✅ 合并 |
| 5 | PMN-PT Shear Actuator | `pmn_pt_shear_actuator.py`, ADRC | Berik 2018 | ✅ 合并 |
| 6 | Waveform Generators + GA | `voltage_waveform/generators.py`, GA 优化器 | – | ✅ 合并 |
| 10 | **RMSO-BW + Fuzzy‑NN** | `rmso_bw_model.py`, `rmso_optimizer.py`, `fuzzy_nn_controller.py`, `rmso_bw_compensator.py`, `test_rmso_bw.py`, `example_rmso_bw_control.ipynb` | Liu 2020 (IEEE Access) | ✅ 合并 |

> **注**：PR #8 和 #9 是中间文档/工具更新，不包含新算法实现。

## 如何添加一篇新论文

1. **搜索与分析**：
   ```bash
   python agents/hermes/ieee_search.py --doi "10.1109/XXX.2024.1234567" --max 1
   python agents/hermes/paper_analyzer.py --doi "10.1109/XXX.2024.1234567" --lang zh
   ```

2. **创建参考文档**：将摘要、方法论、关键公式写入 `references/paper_author_year.md`。

3. **编写 Jules Job**：使用模板 `jules_jobs/template.json`，填写 prompt（参考已有作业）。

4. **本地测试**（可选）：可以在仿真环境中先验证 prompt 是否清晰。

5. **触发流水线**：
   ```bash
   git checkout -b feat/new-paper-model
   git add references/... jules_jobs/...
   git commit -m "feat: add Jules job for XXX"
   git push origin feat/new-paper-model
   # 等待分支可见后触发 API
   curl -X POST ... -d @jules_jobs/new_model.json
   ```

6. **审查 PR**：Jules 生成的 PR 会自动分配给你，检查代码质量后合并。

## Jules API 集成注意事项

- **认证**：使用 `X-Goog-Api-Key` 头，值格式为 `AQ.xxxx...`（Google Cloud API 风格）。
- **端点**：`https://jules.googleapis.com/v1alpha/sessions`
- **source 名称**：必须与 `listSources` API 返回的完全一致，例如 `sources/github/Mada0m0/ieee-research`。
- **来源上下文**：必须提供 `githubRepoContext.startingBranch`，否则请求被拒。
- **作业文件编码**：prompt 中的换行必须通过 `\n` 转义（或使用 `json.dump` 自动处理），避免 JSON 解析错误。
- **分支可见性**：推送后需等待 2–3 秒再触发 Jules，否则可能导致 `fatal: reference is not a tree`。

## CI/CD 流水线说明

文件：`.github/workflows/pipeline.yml`

触发条件：
- 推送到 `main`
- 任何 PR 创建/更新

执行步骤：
1. **代码风格**：`pycodestyle */src/ */tests/ */examples/`
2. **类型检查**：`mypy --ignore-missing-imports */src/`
3. **单元测试**：`pytest */tests/ -v`
4. **构建报告**：生成测试覆盖率和风格报告（仅 `main` 分支）。

所有阶段必须通过才能合并 PR。

## 快速开始（开发环境）

```bash
# 克隆仓库
git clone git@github.com:Mada0m0/ieee-research.git
cd ieee-research

# 安装依赖
pip install -r requirements.txt   # 若存在，否则手动安装 numpy scipy matplotlib pytest pycodestyle mypy

# 设置 Jules API 密钥（可选，仅当需要触发新任务时）
export JULES_API_KEY="AQ.xxxx..."

# 运行现有的分析示例
python hub.py pipeline --doi "10.1109/ACCESS.2020.2984645"
```

## 贡献指南

- 新算法实现应通过 Jules 自动生成，避免手动提交代码到 `implementations/`。
- 若发现生成代码有缺陷，请修改对应的 Jules Job JSON 中的 prompt，并重新触发任务（新分支）。
- 更新研究领域配置：编辑 `research/templates/research_domains.yaml`。

---

**最后更新**：2026-05-08  
**维护者**：@Mada0m0
