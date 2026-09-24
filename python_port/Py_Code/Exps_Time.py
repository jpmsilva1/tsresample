"""
Port of Exps_Time.R: the same experiments as Exps.R, keeping the training
and train+prediction times of every iteration (R: proc.time() differences).

In this port every workflow always records the times, so this script is
Exps.py writing to results/exp.time_ds<i>.pkl; summarise with
GetResults_RunTime.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from Exps import main  # noqa: E402

if __name__ == "__main__":
    main(time_run=True)
