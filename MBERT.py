import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, Features, Value, ClassLabel


class MBERTModel:
    def __init__(self, x_train, y_train, x_test, y_test, device=None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=1)
        self.device = device
        self.model.to(self.device)
        self.model.config.problem_type = "regression"

    def tokenize_function(self, examples):
        return self.tokenizer(examples['text'], padding="max_length", truncation=True)

    def prepare_dataset(self, texts, labels):
        features = Features({
            'text': Value('string'),
            'label': Value('float')
        })

        data = [{'text': text, 'label': label} for text, label in zip(texts, labels)]
        dataset = Dataset.from_list(data, features=features)
        return dataset

    def train(self, batch_size=8, learning_rate=5e-5, num_train_epochs=3):
        train_dataset = self.prepare_dataset(self.x_train, self.y_train)
        train_dataset = train_dataset.map(self.tokenize_function, batched=True)
        # Rename the 'label' column to 'labels' because Hugging Face expects the target column to be named 'labels'
        train_dataset = train_dataset.rename_column("label", "labels")
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        eval_dataset = self.prepare_dataset(self.x_test, self.y_test)
        eval_dataset = eval_dataset.map(self.tokenize_function, batched=True)

        eval_dataset = eval_dataset.rename_column("label", "labels")
        eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        # Move dataset tensors to MPS device
        def to_mps(batch):
            batch["input_ids"] = batch["input_ids"].to(self.device)
            batch["attention_mask"] = batch["attention_mask"].to(self.device)
            batch["labels"] = batch["labels"].to(self.device)
            return batch

        train_dataset = train_dataset.map(to_mps, batched=True)
        eval_dataset = eval_dataset.map(to_mps, batched=True)

        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="no",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            weight_decay=0.01,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        trainer.train()

