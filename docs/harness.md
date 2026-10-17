# Agent Harness 参考文档

本文描述本仓库的 agent harness：两个项目级 MCP server、它们驱动的辅助脚本、共享 skill，以及 Codex 子 agent 定义。`AGENTS.md` 是操作入口（什么时候用哪个工具、委派策略）；本文是 harness 内部结构的参考。

## 分层单一出处

- **委派策略**（路由表、跳过门槛、父 agent 保留职责）：`AGENTS.md` 的「Codex Subagent Policy」小节，是该策略的唯一出处。
- **Agent 运行时行为**（每个子 agent 的自我简报）：`.codex/agents/*.toml`。
- **工作流阶段映射**（哪个重构阶段委派给哪个 agent）：`.agents/skills/refactor-code/SKILL.md`；它只引用策略、不复述。

同一条规则出现在第二层即是缺陷：更新 owning 层与引用处，而不是维护两份事实。

## 文件图

```
.agents/
  mcp/
    build_and_test_server.py   — build-and-test MCP server（CMake/Ninja/ctest 封装）
    openfoam_server.py         — openfoam MCP server（WSL OpenFOAM-14 驱动）
    openfoam_fields.py         — 纯 ascii 解析 + 字段重映射
                                 （无 WSL、无 subprocess、无 I/O）
    openfoam/                  — server 驱动的辅助脚本：
                                 make_of_case.sh（case 搭建 + 启动）、
                                 of_log_summary.py（日志 -> JSON）
    tests/                     — 计划中（步骤 C）：标准库 unittest 测试套件，
                                 覆盖解析器、字段重映射与日志摘要
  skills/refactor-code/SKILL.md — 四阶段代码改动工作流（中文）
.codex/
  config.toml                  — MCP server 注册 + 审批模式
  agents/*.toml                — fvm-explorer / implementer / reviewer /
                                 test-triage / docs（glm-5.3-flash）
opencode.json                  — OpenCode：skill 路径、同一组 MCP server、
                                 bash 权限块
```

两个 agent 访问的是**同一份** server 实现；注册位置在 `.codex/config.toml`（Codex）和 `opencode.json`（OpenCode）。这些 server 运行在 **agent 沙箱之外**——这正是 WSL 和 vcpkg 无需逐次提权就能工作的原因，也是它们存在的意义（而不是让 agent 直接敲 shell 命令）。

## 环境与行为要点

- `VCPKG_ROOT` 与 `Path` 中的 `ninja.exe` 必须在 agent 会话启动**之前**就绪；运行中的会话不会感知后来的变更。
- `FVM_OF_ALLOWED_ROOTS` —— 冒号分隔的绝对 WSL 路径，是 `run_case` 的 `template`/`root` 额外允许的根；默认运行根（`~/OpenFOAM/<user>-14/run`）始终允许。候选路径会在 WSL 内经 `realpath` 规范化，符号链接与 `..` 无法逃逸。该检查不是 TOCTOU-free 的；威胁模型是混淆的 agent，不是本地竞争者。
- `extract_fields` 仅支持 ascii：`system/controlDict` 声明 `writeFormat binary` 会被显式拒绝；含 NUL/替换符的字段文件跳过解析并告警，而不是解析出垃圾数值。
- 工具副作用：`build_and_test`（写 `build/<config>-agent[-py]/`）、`run_case`（在 WSL 中创建 case 目录并启动 `foamRun`）——其余工具全部只读。
- 审批模式记录在 `.codex/config.toml`；当前 `openfoam` 除 `list_cases` 外的每个工具都需要审批，`build_and_test` 亦然。
- openfoam 字段映射假定结构网格，要求 `nx*ny` 等于网格单元数；`grid[j][i] = cell j*nx+i`，面通量从 OpenFOAM 的 owner->neighbour 约定转换到本项目的 `+x/+y` 约定。

## 验证（当前状态）

harness 自身还没有自动化测试（计划为步骤 C：字典解析器、`extract_fields` 重映射、`of_log_summary.py` 的 fixture，以及一个 CI job）。在此之前，手工验证：

```powershell
python -m py_compile .agents\mcp\openfoam_server.py .agents\mcp\openfoam_fields.py
# 纯函数冒烟：直接导入模块，不需要 WSL
python -c "import sys; sys.path.insert(0, r'.agents/mcp'); import openfoam_fields, openfoam_server; print('import ok')"
```

真实数据回归：对一个已有 case（如 16x16 的 `mcptest_*`）以 `nx=ny=16` 运行 `extract_fields`，要求零警告。

## 扩展规则

- 新的 MCP 工具与其 schema 一起加进所属 server；可复用的 OpenFOAM 文本解析放进 `openfoam_fields.py`，且必须保持无 WSL/subprocess/I/O，以维持可单测性。
- server 保持纯标准库 Python、stdio 上的换行分隔 JSON-RPC；`.codex/config.toml` / `opencode.json` 注册的入口（`python .agents/mcp/<server>.py`）必须始终可用。
- harness 源码仅 ASCII；面向 agent 的文档跟随所在文件的语言（AGENTS.md 与本文：中文以外的术语保留英文；skill：中文）。

## 已知局限 / 计划项

- 只读子 agent（`fvm-explorer`、`fvm-reviewer`、`fvm-test-triage`）自身命令受沙箱约束，但 MCP 工具在沙箱外运行，仍可能调用 `build_and_test`/`run_case`（计划：指令收口或 per-agent 工具白名单）。
- server 进程重启后，`run_status` 无法识别过期的运行状态（计划：把启动器 PID 记入 case 的 `mcp_run`）。
- `build_and_test` 返回文本报告而非结构化 JSON（计划中）。
