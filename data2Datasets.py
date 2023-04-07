import os
import pandas as pd
from datasets import Dataset, DatasetDict

# https://discuss.huggingface.co/t/how-to-split-main-dataset-into-train-dev-test-as-datasetdict/1090/3

df0 = pd.read_feather(os.path.join("data", "queries_with_answers_0.feather"))
df0.drop(["id1", "id2"], axis=1, inplace=True)
df1 = pd.read_feather(os.path.join("data", "queries_with_answers_1.feather"))
df1.drop("ID", axis=1, inplace=True)

dataset_df = pd.concat((df0, df1), axis=0)
dataset_df = dataset_df.sample(frac=1)
print(dataset_df[:30])

train_df = dataset_df[:380000]
validate_df = dataset_df[380000:450000]
test_df = dataset_df[450000:]
print(train_df, validate_df, test_df)

ds_dict = {}

for nm, df in [("train", train_df), ("validation", validate_df), ("test", test_df)]:
    dataset = Dataset.from_dict({"query": list(df["query"]), "answer": list(df["answer"]), "label": list(df["label"])})
    ds_dict[nm] = dataset

datasets = DatasetDict(ds_dict)
print(datasets)

datasets.save_to_disk(os.path.join("data", "datasets.huggingface"))