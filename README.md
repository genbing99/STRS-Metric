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