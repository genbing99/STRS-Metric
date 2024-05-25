## Spot-Then-Recognize Score (STRS) Metric for Micro-Expression

### Overview
This repository contains a Python script designed to calculate the metrics based on given prediction and ground-truth data. The metrics include:
- F1-score for Spotting Task.
- F1-score for Analysis Task.
- Spot-then-Recognize Score (STRS): This is the product of the F1-scores from Spotting Task and Analysis Task.

## Installation
``` pip install -r requirements.txt ```

## Usage
``` python main.py ```

## Brief Example
```python
import pandas as pd
from strs_metric import Evaluator

pred_df = pd.DataFrame(
    columns=["subject", "video", "onset", "offset", "emotion"],
    data=[
        [1, "EP02_01f", 10, 70, "happiness"],
        [1, "EP03_02", 100, 170, "others"],
        [1, "EP04_02", 0, 70, "others"],
        [2, "EP01_11f", 20, 90, "happiness"],
        [2, "EP02_04f", 70, 140, "happiness"],
    ]
)

gt_df = pd.DataFrame(
    columns=["subject", "video", "onset", "offset", "emotion"],
    data=[
        [1, "EP02_01f", 45, 85, "happiness"],
        [1, "EP03_02", 130, 160, "others"],
        [1, "EP04_02", 20, 75, "others"],
        [2, "EP01_11f", 45, 95, "repression"],
        [2, "EP02_04f", 30, 140, "repression"],
    ]
)

evaluator = Evaluator()
res = evaluator.evaluate_STRS(pred_df, gt_df)
print("STRS:", round(res["STRS"], 4)) # STRS: 0.2
```

## Reference Paper
```bibtex
@article{liong2023spot,
  title={Spot-then-recognize: A micro-expression analysis network for seamless evaluation of long videos},
  author={Liong, Gen-Bing and See, John and Chan, Chee-Seng},
  journal={Signal Processing: Image Communication},
  volume={110},
  pages={116875},
  year={2023},
  publisher={Elsevier}
}
```