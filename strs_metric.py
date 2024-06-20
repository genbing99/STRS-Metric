import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

class Evaluator:
    def __init__(self):
        pass

    def _calculate_iou(self, pred, gt):
        """ Calculate intersection over union. """
        intersection = np.intersect1d(np.arange(pred["onset"], pred["offset"]+1), np.arange(gt["onset"], gt["offset"]+1)).size
        union = np.union1d(np.arange(pred["onset"], pred["offset"]+1), np.arange(gt["onset"], gt["offset"]+1)).size
        return intersection / union if union != 0 else 0

    def calculate_metrics(self, TP, FP, FN):
        """ Calculate precision, recall and f1-score. """
        precision = TP / (TP + FP) if TP + FP != 0 else 0
        recall = TP / (TP + FN) if TP + FN != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        return precision, recall, f1
    
    def calculate_analysis_metrics(self, TP_df, analysis_emotion):
        """ Calculate metrics for the analysis task. """
        precision_list, recall_list, f1_list = [], [], []

        new_TP_df = pd.DataFrame(columns=TP_df.columns)
        for row_index, row in TP_df.iterrows():
            if row["gt"]["emotion"] in analysis_emotion:
                new_TP_df = new_TP_df.append(row, ignore_index=True)

        for emotion in analysis_emotion:
            gt = [1 if x["emotion"]==emotion else 0 for x in new_TP_df["gt"]]
            pred = [1 if x["emotion"]==emotion else 0 for x in new_TP_df["pred"]]
            try:
                _, FP, FN, TP = confusion_matrix(gt, pred).ravel()
            except:
                FP = FN = TP = 0
            precision, recall, f1 = self.calculate_metrics(TP, FP, FN)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

        precision_all = np.mean(precision_list)
        recall_all = np.mean(recall_list)
        f1_all = (2 * precision_all * recall_all) / (precision_all + recall_all)
        uf1 = np.mean(f1_list)
        uar = np.mean(precision_list)

        return precision_all, recall_all, f1_all, uf1, uar
    
    def evaluate_spotting(self, pred_df_video, gt_df_video, iou_threshold=0.5):
        """ Evaluate spotting task for a given video. """
        TP_spot_df, FP_spot_df, FN_spot_df = (pd.DataFrame() for _ in range(3))
        matched_gt_indices = set()

        for _, pred in pred_df_video.iterrows():
            matched = False
            for gt_row, gt in gt_df_video.iterrows():
                iou = self._calculate_iou(pred, gt)
                if iou >= iou_threshold and gt_row not in matched_gt_indices:
                    TP_spot_df = TP_spot_df.append({"pred": pred, "gt": gt, "iou": iou}, ignore_index=True)
                    matched_gt_indices.add(gt_row)
                    matched = True

            if not matched:
                FP_spot_df = FP_spot_df.append(pred, ignore_index=True)

        FN_spot_df = gt_df_video.loc[~gt_df_video.index.isin(matched_gt_indices)]

        return TP_spot_df, FP_spot_df, FN_spot_df
    
    def evaluate_analysis(self, TP_spot_df, analysis_emotion):
        """ Evaluate analysis task from the spotted TP intervals. """
        TP_video_df, FP_video_df, FN_video_df = (pd.DataFrame() for _ in range(3))
        TP_spot_df = TP_spot_df[TP_spot_df['gt'].apply(lambda x: x['emotion'] in analysis_emotion)] if len(TP_spot_df) > 0 else TP_spot_df
        for emotion in analysis_emotion:
            TP_analysis_df, FP_analysis_df, FN_analysis_df = (pd.DataFrame() for _ in range(3))
            TP_indices = set()
            for row_index, TP_row in TP_spot_df.iterrows():
                if TP_row["pred"]["emotion"] == TP_row["gt"]["emotion"] and TP_row["pred"]["emotion"] == emotion:
                    TP_row["emotion"] = emotion
                    TP_indices.add(row_index)
                    TP_analysis_df = TP_analysis_df.append(TP_row, ignore_index=True)
                elif TP_row["pred"]["emotion"] != TP_row["gt"]["emotion"] and TP_row["pred"]["emotion"] == emotion:
                    FP_analysis_df = FP_analysis_df.append(TP_row["pred"], ignore_index=True)
            for row_index, TP_row in TP_spot_df.iterrows():
                if TP_row["pred"]["emotion"] == emotion and row_index not in TP_indices:
                    FN_analysis_df = FN_analysis_df.append(TP_row["gt"], ignore_index=True)

            TP_video_df = TP_video_df.append(TP_analysis_df, ignore_index=True)
            FP_video_df = FP_video_df.append(FP_analysis_df, ignore_index=True)
            FN_video_df = FN_video_df.append(FN_analysis_df, ignore_index=True)

        return TP_video_df, FP_video_df, FN_video_df
    
    def evaluate_STRS(self, pred_df, gt_df, analysis_emotion=None):
        """ Evaluate STRS metric. """
        analysis_emotion = set(gt_df["emotion"].unique()) if analysis_emotion is None else analysis_emotion
        subject_list = pred_df["subject"].unique()
        subject_dict = {}
        for subject in subject_list:
            subject_dict[subject] = {
                "pred": pred_df[pred_df["subject"] == subject],
                "gt": gt_df[gt_df["subject"] == subject]
            }

        TP_analysis_all_df, FP_analysis_all_df, FN_analysis_all_df = (pd.DataFrame() for _ in range(3))
        TP_spot_all_df, FP_spot_all_df, FN_spot_all_df = (pd.DataFrame() for _ in range(3))
        for subject_name, subject_data in subject_dict.items():
            video_list = subject_data["gt"]["video"].unique()
            video_dict = {}
            for video in video_list:
                video_dict[video] = {
                    "pred": pred_df[(pred_df["video"] == video) & (pred_df["subject"] == subject_name)],
                    "gt": gt_df[(gt_df["video"] == video) & (gt_df["subject"] == subject_name)]
                }

            for _, video_data in video_dict.items():
                TP_spot_df, FP_spot_df, FN_spot_df = self.evaluate_spotting(video_data["pred"], video_data["gt"])
                TP_video_df, FP_video_df, FN_video_df = self.evaluate_analysis(TP_spot_df, analysis_emotion)

                TP_spot_all_df = TP_spot_all_df.append(TP_spot_df, ignore_index=True)
                FP_spot_all_df = FP_spot_all_df.append(FP_spot_df, ignore_index=True)
                FN_spot_all_df = FN_spot_all_df.append(FN_spot_df, ignore_index=True)
                TP_analysis_all_df = TP_analysis_all_df.append(TP_video_df, ignore_index=True)
                FP_analysis_all_df = FP_analysis_all_df.append(FP_video_df, ignore_index=True)
                FN_analysis_all_df = FN_analysis_all_df.append(FN_video_df, ignore_index=True)

        TP_spot, FP_spot, FN_spot = len(TP_spot_all_df), len(FP_spot_all_df), len(FN_spot_all_df)
        TP_analysis, FP_analysis, FN_analysis = len(TP_analysis_all_df), len(FP_analysis_all_df), len(FN_analysis_all_df)
        precision_spot, recall_spot, f1_spot = self.calculate_metrics(TP_spot, FP_spot, FN_spot)
        precision_analysis, recall_analysis, f1_analysis, uf1_analysis, uar_analysis = self.calculate_analysis_metrics(TP_spot_all_df, analysis_emotion)
        
        res = {
            "STRS": f1_spot * f1_analysis,
            "TP spotting": TP_spot,
            "FP spotting": FP_spot,
            "FN spotting": FN_spot,
            "Precision spotting": precision_spot,
            "Recall spotting": recall_spot,
            "F1-score spotting": f1_spot,
            "TP analysis": TP_analysis,
            "FP analysis": FP_analysis,
            "FN analysis": FN_analysis,
            "Precision analysis": precision_analysis,
            "Recall analysis": recall_analysis,
            "F1-score analysis": f1_analysis,
            "UF1 analysis": uf1_analysis,
            "UAR analysis": uar_analysis
        }
        return res