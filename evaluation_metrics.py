import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
import wandb
import pandas as pd
import torch
import random
import os

# Configure device and seed everithing for reproducibility
seed = 19980125

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class MetricsEvaluator:

    def calculate_metrics(self, preds, gts, metrics):
        """Calculate metrics based on predictions and ground truths using sklearn."""
        if len(preds) != len(gts):
            raise ValueError("Length of predictions and ground truths do not match.")

        results = {}
        for metric in metrics:
            if metric == "accuracy":
                results[metric] = accuracy_score(gts, preds)
            elif metric == "precision":
                results[metric] = precision_score(gts, preds, average='macro', zero_division=0)
            elif metric == "recall":
                results[metric] = recall_score(gts, preds, average='macro', zero_division=0)
            elif metric == "jaccard":
                results[metric] = jaccard_score(gts, preds, average='macro')
            elif metric == "f1_score":
                results[metric] = f1_score(gts, preds, average='macro')
        return results


    def evaluation_per_dataset(self, preds, gts, videos_names, dataset_name):
        """Evaluate metrics per dataset."""

        results = {}
        dataset_preds = np.concatenate(preds).tolist()
        dataset_gts = np.concatenate(gts).tolist()
        names = []

        for i in range(len(preds)):
            names.extend([videos_names[i]] * len(preds[i]))
        

        if not len(dataset_preds) == len(dataset_gts):
            AssertionError(f"Length of predictions and ground truths do not match for {dataset_name}. ")

        if dataset_name == "AutoLaparo":
            results[dataset_name] = self.calculate_metrics(dataset_preds, dataset_gts, ["accuracy", "precision", "recall", "jaccard", 'f1_score'])
        
        elif dataset_name == "Cholec80":

            dataset_info = {'Video_id': names, 'Pred': dataset_preds, 'GT': dataset_gts}
            df = pd.DataFrame(dataset_info)
            video_results = []
            
            # Iterate through each video to calculate metrics
            for video in videos_names:
                filtered_df = df[df['Video_id'] == video]
                video_preds = filtered_df['Pred'].tolist()
                video_gts = filtered_df['GT'].tolist()

                if video_preds and video_gts:
                    video_results.append(self.calculate_metrics(video_preds, video_gts, ["accuracy", "precision", "recall", 'f1_score']))
                else:
                    raise ValueError(f"No predictions or ground truths found for video {video} in dataset {dataset_name}.")
                
            metrics_mean = {metric: np.mean([res[metric] for res in video_results]) for metric in ["accuracy", "precision", "recall", 'f1_score']}
            metrics_std = {metric: np.std([res[metric] for res in video_results]) for metric in ["accuracy", "precision", "recall", 'f1_score']}
            results[dataset_name] = {"mean": metrics_mean, "std": metrics_std}
        
        elif dataset_name == "HeiChole":
            results[dataset_name] = self.calculate_metrics(dataset_preds, dataset_gts, ["accuracy", "precision", "recall", "jaccard", "f1_score"])
        
        elif dataset_name == "HeiCo":
            results[dataset_name] = self.calculate_metrics(dataset_preds, dataset_gts, ["accuracy", "precision", "recall", "jaccard", "f1_score"])
        
        elif dataset_name == "M2CAI":
            results[dataset_name] = self.calculate_metrics(dataset_preds, dataset_gts, ["accuracy", "precision", "recall"])

        wandb.log({dataset_name: results[dataset_name]})

        return results
