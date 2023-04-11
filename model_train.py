"""
https://colab.research.google.com/drive/1XnVhrd9YiaL9sreMP8p9AsGlvcClpOg1#scrollTo=fnq8n_KP_3aX
"""
import os
import numpy as np
from datasets import DatasetDict
from transformers import BertTokenizer
from transformers import (BertConfig, 
                          RobertaConfig,
                          RobertaTokenizer,
                          RobertaModelWithHeads, 
                          BertModelWithHeads)
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction

def encode_batch(batch):
  """Encodes a batch of input data using the model tokenizer."""
  return tokenizer(
      batch["query"],
      batch["answer"],
      max_length=180,
      truncation=True,
      padding="max_length"
  )

def compute_accuracy(p: EvalPrediction):
  preds = np.argmax(p.predictions, axis=1)
  return {"acc": (preds == p.label_ids).mean()}


# nn_name = "bert-base-uncased"
nn_name = "roberta-base"

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
dataset = DatasetDict.load_from_disk(os.path.join("data", "datasets_litle.huggingface"))

print(dataset)

# id2label = {id: label for (id, label) in enumerate(dataset["train"].features["label"])}
'''
config = BertConfig.from_pretrained(
    nn_name,
    # id2label=id2label,
)'''

config = RobertaConfig.from_pretrained("roberta-base", num_labels=2, )
dataset = dataset.map(encode_batch, batched=True)
'''
model = BertModelWithHeads.from_pretrained(
    nn_name,
    config=config,
)'''

model = RobertaModelWithHeads.from_pretrained(nn_name, config=config, )

model.add_adapter("nli/multinli@ukp")
# Add a matching classification head
model.add_classification_head("nli/multinli@ukp", num_labels=2, id2label={0: "👎", 1: "👍"})
print(0, "👎", 1, "👍")
# Activate the adapter
model.train_adapter("nli/multinli@ukp")


# model.load_adapter("nli/multinli@ukp", load_as="multinli", with_head=False)
'''
training_args = TrainingArguments(
    learning_rate=5e-5,
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_steps=200,
    output_dir="./training_output",
    overwrite_output_dir=True,
    # The next line is important to ensure the dataset labels are properly passed to the model
    remove_unused_columns=False,
)


trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_accuracy,
)'''

training_args = TrainingArguments(
    learning_rate=1e-4,
    num_train_epochs=6,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_steps=200,
    output_dir="./training_output",
    overwrite_output_dir=True,
    # The next line is important to ensure the dataset labels are properly passed to the model
    remove_unused_columns=False,
)

trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_accuracy,
)


trainer.train()
trainer.evaluate()