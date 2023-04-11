import os
import numpy as np
import pandas as pd
from datasets import DatasetDict
from transformers import (BertTokenizer, 
                          BertModelWithHeads,
                          )
from transformers.adapters.composition import Fuse
import torch



def predict(premise, hypothesis):
  encoded = tokenizer(premise, hypothesis, return_tensors="pt")
  # if torch.cuda.is_available():
  #  encoded.to("cuda")
  logits = model(**encoded)[0]
  tanh = torch.tanh(logits)
  pred_class = torch.argmax(logits).item()
  print("sigmoid:", torch.sigmoid(logits))
  return pred_class




# model_name = "cointegrated/rubert-tiny2"
model_name = "bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained(model_name, output_attentions=True)
model = BertModelWithHeads.from_pretrained(model_name)


adapter_name = "nli/qnli@ukp"
# mode_name =  "nli-adapter-bert-base-e15"
mode_name =  "nli-adapter-bert-e15"
# mode_name = "nli-adapter-rubert-tiny2-e5"
# adapter_name = "my_adapter"

adapter_path = os.path.join(os.getcwd(), "models", mode_name)

model.load_adapter(adapter_path)
model.set_active_adapters(adapter_name)


df = pd.read_csv(os.path.join("data", "validate_litle.csv"), sep="\t")

k = 1
for q, a, l in list(zip(df["query"], df["answer"], df["label"]))[:25]:
    prd = predict(q, a)
    print(k, "true:", l, "predict:", prd, "val:", l - prd)
    k += 1