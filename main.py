import torch
from DataPreprocessor import DataPreprocessor
from sklearn.model_selection import train_test_split
from Models.MBERT import MBERTModel
from transformers import BertTokenizer, BertForSequenceClassification
from ModelEvaluator import ModelEvaluator
import os
import numpy as np

# Load the data
data = DataPreprocessor('combined_data_no_html.csv')

# Preprocess the data
df = data.preprocess()


# Split the data into training and temporary sets
X_train, X_temp, y_train, y_temp = train_test_split(df['cleaned_text'], df['quality_mean'], test_size=0.4,
                                                    random_state=42)

# Further split the temporary set into validation and test sets
X_eval, X_test, y_eval, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# Check if MPS is available
device = torch.device("mps") if torch.backends.mps.is_built() else torch.device("cpu")
print(f"Using device: {device}")

# Paths to save and load model and tokenizer
model_path = "Models/saved_mbert_model"
tokenizer_path = "Models/saved_mbert_model"

# Check if the model directory exists
if not os.path.exists(model_path):
    os.makedirs(model_path)


# Function to check if any model files exist
def model_files_exist(path):
    files = ["pytorch_model.bin", "model.safetensors", "tf_model.h5"]
    return any(os.path.exists(os.path.join(path, file)) for file in files)


# Hyperparameter grid
param_grid = {
    'batch_size': [8, 16],
    'learning_rate': [5e-5, 3e-5, 0.0001],
    'num_train_epochs': [10, 15, 20]
}


# Function to train and evaluate the model
def train_and_evaluate(model, batch_size, learning_rate, num_train_epochs, X_eval, y_eval):

    model.train(batch_size=batch_size, learning_rate=learning_rate, num_train_epochs=num_train_epochs)

    # Evaluate the model
    model_evaluator = ModelEvaluator(model.model, model.tokenizer, X_eval, y_eval, device)
    mse, r2, pearson_corr = model_evaluator.evaluate()

    return mse, r2, pearson_corr


# Number of iterations for averaging
num_iterations = 5

# Grid search over hyperparameters
best_mse = float('inf')
best_params = None

for batch_size in param_grid['batch_size']:
    for learning_rate in param_grid['learning_rate']:
        for num_train_epochs in param_grid['num_train_epochs']:
            mse_list, r2_list, pearson_list = [], [], []
            print(
                f"Training with batch_size={batch_size}, learning_rate={learning_rate}, num_train_epochs={num_train_epochs}")
            for iteration in range(num_iterations):
                print(f"Iteration {iteration + 1}/{num_iterations}")
                # Initialize the model
                model = MBERTModel(X_train, y_train, X_eval, y_eval, device)

                mse, r2, pearson_corr = train_and_evaluate(model, batch_size, learning_rate, num_train_epochs, X_eval,
                                                           y_eval)

                print(f"Iteration {iteration + 1} Results: MSE={mse}, R2={r2}, Pearson Correlation={pearson_corr}")

                mse_list.append(mse)
                r2_list.append(r2)
                pearson_list.append(pearson_corr)

            mean_mse = np.mean(mse_list)
            mean_r2 = np.mean(r2_list)
            mean_pearson = np.mean(pearson_list)

            print(f"Averaged Results: MSE={mean_mse}, R2={mean_r2}, Pearson Correlation={mean_pearson}")

            # if mean_mse < best_mse:
            #     best_mse = mean_mse
            #     best_params = (batch_size, learning_rate, num_train_epochs)
            #
            #     # Save the best model
            #     model.model.save_pretrained(os.path.join(model_path, "best_model"))
            #     model.tokenizer.save_pretrained(os.path.join(tokenizer_path, "best_tokenizer"))
print(
    f"Best parameters: batch_size={best_params[0]}, learning_rate={best_params[1]}, num_train_epochs={best_params[2]}")
print(f"Best MSE: {best_mse}")

# # Load the best model and tokenizer
# best_model = BertForSequenceClassification.from_pretrained(os.path.join(model_path, "best_model")).to(device)
# best_tokenizer = BertTokenizer.from_pretrained(os.path.join(tokenizer_path, "best_tokenizer"))
#
# # Evaluate the best model
# best_model_evaluator = ModelEvaluator(best_model, best_tokenizer, X_test, y_test, device)
# best_mse, best_r2, best_pearson_corr = best_model_evaluator.evaluate()
#
# print(f"Best Model - Mean Squared Error (MSE): {best_mse}")
# print(f"Best Model - R-squared (R2): {best_r2}")
# print(f"Best Model - Pearson Correlation: {best_pearson_corr}")

# # Train and save the model if it doesn't exist
# if not model_files_exist(model_path):
#     # Train the model
#     model = MBERTModel(X_train, y_train, X_eval, y_eval, device)
#     model.train(batch_size=8, learning_rate=5e-5, num_train_epochs=10)
#
#     # Save the model and tokenizer
#     model.model.save_pretrained(model_path)
#     model.tokenizer.save_pretrained(tokenizer_path)
# else:
#     # Load the model and tokenizer
#     model = MBERTModel(X_train, y_train, X_eval, y_eval, device)
#     model.model = BertForSequenceClassification.from_pretrained(model_path).to(device)
#     model.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
#     print("Model and tokenizer loaded from disk.")
#
# # Evaluate the model
# model_evaluator = ModelEvaluator(model.model, model.tokenizer, X_test, y_test, device)
# mse, r2, pearson_corr = model_evaluator.evaluate()
#
# print(f"Mean Squared Error (MSE): {mse}")
# print(f"R-squared (R2): {r2}")
# print(f"Pearson Correlation: {pearson_corr}")
