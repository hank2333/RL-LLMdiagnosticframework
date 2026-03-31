# System Flow v1

## High-Level Flow

1. `train_runner.py` executes a training run and emits raw artifacts.
2. `summarize_metrics.py` converts raw outputs into a compact summary.
3. `call_llm.py` sends the summary and context to an LLM for diagnosis.
4. `constraint_guard.py` validates any proposed config changes.
5. `memory_manager.py` stores important history for future iterations.
6. `loop_controller.py` coordinates repeated execution until stopping criteria are met.
