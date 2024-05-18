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
```
Liong, G-B., See, J. and C.S. Chan (2023). Spot-then-recognize: A micro-expression analysis network for seamless evaluation of long videos. Signal Processing: Image Communication, vol. 110, pp. 116875, January 2023, doi: 10.1016/j.image.2022.116875
```