import pandas as pd
from strs_metric import Evaluator

pred_df = pd.read_csv("data/pred.csv")
gt_df = pd.read_csv("data/gt.csv")

evaluator = Evaluator()
# modify the analysis_emotion to evaluate specific emotions for analysis task, ex. ['negative', 'positive', 'surprise']
res = evaluator.evaluate_STRS(pred_df, gt_df, analysis_emotion=None) 
for metric, value in res.items():
    print(metric, ":", round(value, 4))