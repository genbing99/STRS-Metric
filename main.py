import pandas as pd
from strs_metric import Evaluator

pred_df = pd.read_csv("data/pred.csv")
gt_df = pd.read_csv("data/gt.csv")

evaluator = Evaluator()
res = evaluator.evaluate_STRS(pred_df, gt_df)
for metric, value in res.items():
    print(metric, ":", round(value, 4))