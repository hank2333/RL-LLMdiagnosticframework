# RL-LLM Diagnostic Framework

Minimal closed-loop PPO diagnosis framework for testing whether history-aware LLM diagnosis outperforms single-shot diagnosis in a lightweight discrete-action environment.

## Project Goal

Build a small research-oriented codebase that can:

- train PPO on a custom discrete-action environment
- summarize policy behavior compactly
- generate structured diagnoses in `single_shot` and `history_aware` modes
- apply guarded parameter updates over multiple rounds
- compare against a `fixed` baseline

The implementation is intentionally minimal. Architecture and testability take priority over completeness.

## Project Tree

```text
project/
  README.md
  configs/
    fixed.yaml
    single_shot.yaml
    history_aware.yaml
  artifacts/
    history.json
    runs/
  docs/
    design_notes.md
  src/
    envs/
      __init__.py
      mini_defense_env.py
    core/
      __init__.py
      action_mapper.py
      diagnosis_types.py
      guardrail.py
      history_manager.py
      llm_client.py
      loop.py
      prompt_builder.py
      summarizer.py
      train_runner.py
    experiments/
      __init__.py
      run_fixed.py
      run_history_aware.py
      run_single_shot.py
```

## Core Module Responsibilities

- `src/envs/mini_defense_env.py`
  Defines `MiniDefenseEnv`, a fast custom Gymnasium environment with 6 discrete actions, short episodes, low-dimensional observations, and reward dynamics that can induce action collapse.
- `src/core/train_runner.py`
  Holds PPO training configuration, parameter bounds, and the training entrypoint abstraction.
- `src/core/summarizer.py`
  Builds the compact training summary used by the diagnosis step.
- `src/core/diagnosis_types.py`
  Defines structured diagnosis schemas, allowed fault types, and outcome labels.
- `src/core/history_manager.py`
  Stores and retrieves the last 3 round records in JSON format.
- `src/core/prompt_builder.py`
  Assembles inputs for `single_shot` and `history_aware` diagnosis modes.
- `src/core/llm_client.py`
  Provides a mock LLM interface that returns structured diagnosis payloads.
- `src/core/action_mapper.py`
  Converts symbolic adjustment actions like `increase_small` into bounded numeric PPO updates.
- `src/core/guardrail.py`
  Enforces allowed keys, parameter ranges, and per-round change limits.
- `src/core/loop.py`
  Coordinates train -> summarize -> diagnose -> guard -> record history.
- `src/experiments/run_fixed.py`
  Runs the baseline with no adaptive updates.
- `src/experiments/run_single_shot.py`
  Runs closed-loop diagnosis using only the current summary.
- `src/experiments/run_history_aware.py`
  Runs closed-loop diagnosis using the current summary plus recent history.

## Initial Implementation Notes

- Environment:
  `MiniDefenseEnv` exposes 6 actions:
  `monitor`, `scan`, `isolate_host`, `patch`, `deploy_decoy`, `do_nothing`
- PPO tunable parameters:
  `learning_rate`, `ent_coef`, `clip_range`
- Diagnosis modes:
  `single_shot`, `history_aware`
- Experiment groups:
  `fixed`, `single_shot`, `history_aware`
- History length:
  3 rounds

## Short Implementation Plan

1. Finalize the custom environment and verify that PPO can train on it quickly.
2. Connect `train_runner.py` to Stable-Baselines3 PPO and collect rollout metrics.
3. Implement deterministic summary generation from evaluation traces and policy stats.
4. Improve the mock diagnosis policy, then swap in a real LLM client behind the same interface.
5. Run side-by-side experiments for `fixed`, `single_shot`, and `history_aware`.

## Status

Current state: project skeleton and core interfaces are in place. PPO integration and end-to-end experiment execution are the next implementation steps.

## Notes
 RL-LLM 诊断框架
  ├─ 实验入口层
  │  ├─ run_fixed.py
  │  │  ├─ 作用：启动固定参数基线实验
  │  │  ├─ 输入：命令行参数 + fixed.yaml
  │  │  └─ 输出：调用 run_fixed_baseline(...)
  │  ├─ run_single_shot.py
  │  │  ├─ 作用：启动 single_shot 实验
  │  │  ├─ 输入：命令行参数 + single_shot.yaml
  │  │  └─ 输出：调用 run_closed_loop(mode="single_shot")
  │  ├─ run_history_aware.py
  │  │  ├─ 作用：启动 history_aware 实验
  │  │  ├─ 输入：命令行参数 + history_aware.yaml
  │  │  └─ 输出：调用 run_closed_loop(mode="history_aware")
  │  └─ common.py
  │     ├─ 作用：统一解析 CLI 和配置文件
  │     └─ 输出：标准化后的运行参数 settings
  ├─ 配置层
  │  ├─ config_loader.py
  │  │  ├─ 作用：读取简化版 YAML 配置
  │  │  ├─ 输入：配置文件路径
  │  │  └─ 输出：Python dict
  │  ├─ fixed.yaml
  │  ├─ single_shot.yaml
  │  └─ history_aware.yaml
  │     └─ 保存默认实验参数
  ├─ 主控制循环
  │  └─ loop.py
  │     ├─ 作用：组织整套闭环流程
  │     ├─ 输入
  │     │  ├─ 模式 mode
  │     │  ├─ 轮数 rounds
  │     │  ├─ PPO 参数
  │     │  ├─ seed
  │     │  ├─ artifact 目录
  │     │  └─ Ollama 设置
  │     ├─ 每轮流程
  │     │  ├─ 训练 PPO
  │     │  ├─ 生成 summary
  │     │  ├─ 读取历史
  │     │  ├─ 构造诊断输入
  │     │  ├─ 调用 LLM
  │     │  ├─ 将符号调整映射成数值
  │     │  ├─ 通过 guardrail 校验
  │     │  ├─ 标记 outcome
  │     │  └─ 写入 history 和 artifact
  │     └─ 输出
  │        ├─ HistoryRecord 列表
  │        ├─ artifacts/history.json
  │        └─ 每轮 round_summary.json
  ├─ 训练层
  │  └─ train_runner.py
  │     ├─ 作用
  │     │  ├─ 定义 PPOConfig
  │     │  ├─ 调用 Stable-Baselines3 PPO 训练
  │     │  └─ 训练后评估策略
  │     ├─ 输入
  │     │  ├─ learning_rate
  │     │  ├─ ent_coef
  │     │  ├─ clip_range
  │     │  ├─ train_steps
  │     │  ├─ eval_episodes
  │     │  └─ seed
  │     └─ 输出
  │        └─ TrainingArtifacts
  │           ├─ summary_inputs
  │           ├─ raw_metrics
  │           └─ config
  ├─ 环境层
  │  └─ mini_defense_env.py
  │     ├─ 作用：自定义 Gymnasium 环境 MiniDefenseEnv
  │     ├─ 动作空间
  │     │  ├─ monitor
  │     │  ├─ scan
  │     │  ├─ isolate_host
  │     │  ├─ patch
  │     │  ├─ deploy_decoy
  │     │  └─ do_nothing
  │     ├─ 观测空间
  │     │  ├─ threat_level
  │     │  ├─ host_health
  │     │  ├─ visibility
  │     │  ├─ decoy_level
  │     │  ├─ patch_level
  │     │  └─ steps_remaining_ratio
  │     ├─ 奖励设计
  │     │  ├─ 安全动作前期有收益
  │     │  ├─ monitor/scan 重复过多会惩罚
  │     │  ├─ patch/isolate/decoy 是条件性高价值动作
  │     │  └─ do_nothing 是坏动作
  │     └─ 输出
  │        ├─ next obs
  │        ├─ reward
  │        ├─ terminated/truncated
  │        └─ info
  ├─ 总结层
  │  └─ summarizer.py
  │     ├─ 作用：把训练评估结果压缩成诊断 summary
  │     ├─ 输入：summary_inputs
  │     └─ 输出：TrainingSummary
  │        ├─ avg_return
  │        ├─ std_return
  │        ├─ success_rate
  │        ├─ avg_episode_length
  │        ├─ reward_trend
  │        ├─ entropy_trend
  │        ├─ policy_stability
  │        ├─ action_distribution
  │        ├─ dominant_action
  │        ├─ action_collapse
  │        └─ alerts
  ├─ 诊断输入层
  │  └─ prompt_builder.py
  │     ├─ 作用：构造发给 LLM 的 payload
  │     ├─ single_shot：只用 current_summary
  │     └─ history_aware：current_summary + recent_history
  ├─ LLM 诊断层
  │  └─ llm_client.py
  │     ├─ 作用
  │     │  ├─ 构造严格 JSON prompt
  │     │  ├─ 调用本地 Ollama
  │     │  ├─ 解析并规范化输出
  │     │  └─ 出错时退回 heuristic 规则诊断
  │     ├─ 输入
  │     │  ├─ summary payload
  │     │  ├─ model
  │     │  ├─ base_url
  │     │  └─ timeout
  │     └─ 输出：DiagnosisOutput
  │        ├─ fault_type
  │        ├─ observed_symptoms
  │        ├─ reasoning
  │        ├─ proposed_adjustments
  │        ├─ confidence
  │        ├─ risk_level
  │        └─ rollback_trigger
  ├─ 调整映射层
  │  └─ action_mapper.py
  │     ├─ 作用：把 increase_small / decrease_small / keep 转成数值参数修改
  │     ├─ learning_rate / ent_coef
  │     │  ├─ increase_small => ×1.5
  │     │  └─ decrease_small => ÷1.5
  │     └─ clip_range
  │        ├─ increase_small => +0.02
  │        └─ decrease_small => -0.02
  ├─ 守护规则层
  │  └─ guardrail.py
  │     ├─ 作用：检查参数更新是否合法
  │     ├─ 规则
  │     │  ├─ 只允许 3 个 PPO 参数
  │     │  ├─ 必须在范围内
  │     │  ├─ 每轮最多改 2 个
  │     │  └─ high risk 时最多改 1 个
  │     └─ 输出
  │        ├─ accepted config
  │        ├─ alerts
  │        └─ is_valid
  ├─ 历史层
  │  └─ history_manager.py
  │     ├─ 作用：存储最近 3 轮 JSON 历史
  │     ├─ 每条记录
  │     │  ├─ round_id
  │     │  ├─ config
  │     │  ├─ summary
  │     │  ├─ diagnosis
  │     │  └─ outcome
  │     └─ 输出：artifacts/history.json
  ├─ 类型定义层
  │  └─ diagnosis_types.py
  │     ├─ 作用：定义全系统共享 schema
  │     └─ 包含
  │        ├─ TrainingSummary
  │        ├─ DiagnosisOutput
  │        ├─ HistoryRecord
  │        ├─ OutcomeLabel
  │        └─ DiagnosisMode
  └─ 全流程数据流
     ├─ 用户运行实验脚本
     ├─ 读取 config + CLI 覆盖
     ├─ 进入 loop
     ├─ PPO 在 MiniDefenseEnv 上训练
     ├─ 评估并生成 summary_inputs
     ├─ summarizer 生成 TrainingSummary
     ├─ prompt_builder 组装诊断输入
     ├─ llm_client 调用 Ollama 得到 DiagnosisOutput
     ├─ action_mapper 变成数值 proposal
     ├─ guardrail 校验 proposal
     ├─ loop 决定下一轮 config 和 outcome
     ├─ history_manager 保存历史
     └─ 写出 round artifact