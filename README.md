# Sidhart Throwing Box Counter

YOLO-based automated throwing box counting with momentum tracking and dual-line crossing detection.

## Features
- **Momentum-Based Tracking**: Predicts box velocity with exponential smoothing to track high-speed box movement reliably.
- **Dual-Line Triggering**: Primary and backup crossing line geometry to capture directional box transitions.
- **Minimal Clean Output**: Suppresses verbose model and inference logs, outputting clean timestamps in Indian Standard Time (IST, UTC+5:30) whenever a box count increases.
- **Dedicated IST Count Log**: Automatically records each increment to a count log file (`_count_log_ist.txt`).

## Output Format
When running the pipeline, logs are emitted only when the count increases:
```text
[2026-11-02 16:56:45 IST] Count: 1
[2026-11-02 16:56:50 IST] Count: 2
```

## Running the Script
```bash
python main.py
```

