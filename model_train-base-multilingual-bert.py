"""
https://colab.research.google.com/drive/1XnVhrd9YiaL9sreMP8p9AsGlvcClpOg1#scrollTo=fnq8n_KP_3aX
https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
ValueError: Unrecognized configuration class <class 'transformers.models.deberta_v2.configuration_deberta_v2.DebertaV2Config'> for this kind of AutoModel: AutoModelWithHeads.
Model type should be one of BartConfig, BertConfig, DistilBertConfig, GPT2Config, MBartConfig, RobertaConfig, T5Config, XLMRobertaConfig.


как использовать nli/multinli@ukp:
https://colab.research.google.com/github/Adapter-Hub/adapter-transformers/blob/master/notebooks/03_Adapter_Fusion.ipynb


если результаты обучения из этого файла использовать с from transformers import (BertTokenizer, BertModelWithHeads), то модель как минимум 
"узнает" примеры из обучающей выборки (с AutoModelWithHeads не узнает)

"""
import os
import numpy as np
from datasets import DatasetDict
from transformers import (
                          BertTokenizer, 
                          BertModelWithHeads)
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction

model_name = "bert-base-multilingual-cased"
adapter_name = "nli/qnli@ukp"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModelWithHeads.from_pretrained(model_name)


def encode_batch(batch):
  """Encodes a batch of input data using the model tokenizer."""
  return tokenizer(
      batch["query"],
      batch["answer"],
      max_length=512,
      truncation=True,
      padding="max_length"
  )

def compute_accuracy(p: EvalPrediction):
  preds = np.argmax(p.predictions, axis=1)
  return {"acc": (preds == p.label_ids).mean()}


dataset = DatasetDict.load_from_disk(os.path.join("data", "datasets_litle.huggingface"))
print(dataset)

dataset = dataset.map(encode_batch, batched=True)
model.add_adapter(adapter_name)

# Add a matching classification head
model.add_classification_head(adapter_name, num_labels=2)
# Activate the adapter
model.train_adapter(adapter_name)


training_args = TrainingArguments(
    learning_rate=1e-4,
    num_train_epochs=15,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=100,
    save_steps=160000,
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

adapter_path = os.path.join(os.getcwd(), "models", "nli-adapter-bert-e15")
model.save_adapter(adapter_path, adapter_name)

# как предсказывать:
import torch

def predict(premise, hypothesis):
  encoded = tokenizer(premise, hypothesis, return_tensors="pt")
  if torch.cuda.is_available():
    encoded.to("cuda")
  logits = model(**encoded)[0]
  pred_class = torch.argmax(logits).item()
  return pred_class

txs = [("срок оплаты страховых взносов с заработной платы в 2023 году", 
        "Срок сдачи декларации по налогу на прибыль за I квартал и январь–март: 25 апреля 2023 года.Куда сдавать: в ИФНС.Форма: не изменилась. Скачать бланк >>. Образцы заполнения для разных случаев можно скачать ниже."),
        ("единый налоговый платеж с 2023 года - видео", 
         "С 2023 года налоги и взносы перечисляют единым налоговым платежом, который инспекция сама засчитает в счет текущих платежей, недоимок, пеней и штрафов. В рекомендации — как перейти на ЕНП, в какие сроки его платить, как подать уведомление о начислениях, как заполнить платежку и отражать платеж в учете. Образцы документов и видеоинструкции помогут быстро разобраться в новых правилах работы.")]

for q, a in txs:
  prd = predict(q, a)
  print(prd)