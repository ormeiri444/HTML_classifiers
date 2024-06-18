import torch
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import scipy.stats
import pandas as pd
import gc


class ModelEvaluator:

    def __init__(self, model, tokenizer, x_test, y_test, device):
        self.model = model
        self.tokenizer = tokenizer
        self.x_test = x_test
        self.y_test = y_test
        self.device = device

    def predict(self, text_list, batch_size=1):
        # Ensure text_list is a list of strings
        if isinstance(text_list, pd.Series):
            text_list = text_list.tolist()

        all_predictions = []
        for i in range(0, len(text_list), batch_size):
            batch_texts = text_list[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = outputs.logits.squeeze().cpu().numpy()
                if predictions.ndim == 0:
                    all_predictions.append(predictions.item())  # Handle single prediction case
                else:
                    all_predictions.extend(predictions)
            # Clear MPS cache and trigger garbage collection
            torch.mps.empty_cache()
            gc.collect()
        return np.array(all_predictions)

    def evaluate(self):
        y_pred = self.predict(self.x_test)

        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        pearson_corr, _ = scipy.stats.pearsonr(self.y_test, y_pred)

        return mse, r2, pearson_corr
