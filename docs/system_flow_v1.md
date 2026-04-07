# System Flow v1

## High-Level Flow

1. train_runner.py 运行一次训练
2. 输出 raw metrics / log
3. summarize_metrics.py 生成 training summary
4. memory_manager.py 提供最近几轮历史
5. call_llm.py 读取 summary + history，生成 diagnosis
6. constraint_guard.py 检查 proposed changes
7. loop_controller.py 决定接受或回滚，并进入下一轮