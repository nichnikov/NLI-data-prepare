import os
import re
import pandas as pd
from random import shuffle
from datasets import Dataset
from sentence_transformers import SentenceTransformer, util

patterns = re.compile(r"\xa0")

qrs_df = pd.read_csv(os.path.join("data", "not_similar_queries.csv"), sep="\t")
print(qrs_df)

answers_df = pd.read_feather(os.path.join("data", "queries.feather"))
print(answers_df)

u_answers = list(set([(d["ID"], patterns.sub(" ", d["ShortAnswerText"])) for d in answers_df.to_dict(orient="records")]))
unique_answers_df = pd.DataFrame(u_answers, columns=["ID", "answer"])


print(unique_answers_df)

queries_answers_df = pd.merge(qrs_df, unique_answers_df, on="ID")
queries_answers_df["label"] = 1
queries_answers_df.to_feather(os.path.join("data", "queries_with_answers_1.feather"))
queries_answers_df.to_csv(os.path.join("data", "queries_with_answers_1.csv"), sep="\t", index=False)
print(queries_answers_df)

"""для каждого вопроса сделать ответ с другим АйДи максимально близкий к ответу на этот вопрос (но не равный, например, с simScore в районе  0.9) 
и относительно далекий с simScore, например, меньше 0.7"""
# queries_answers_dicts = queries_answers_df.to_dict(orient="records")
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

u_ids_ans = list(set([(i, anw) for i, anw in zip(list(queries_answers_df["ID"]), list(queries_answers_df["answer"]))]))
print("quantity of unique answers:", len(u_ids_ans))
u_ids_ans_dicts = {i: a for i, a in u_ids_ans}
u_ids = [i for i, a in u_ids_ans]
u_answers = [a for i, a in u_ids_ans]
paraphrases = util.paraphrase_mining(model, u_answers)

diff_ids_pairs = [(u_ids[i], u_ids[j]) for sc, i, j in paraphrases if sc < 0.8]
print(diff_ids_pairs[:10])
print(len(diff_ids_pairs))

diff_qrs_ans = []
for i1, i2 in diff_ids_pairs:
    queries = list(queries_answers_df["query"][queries_answers_df["ID"] == i1])
    shuffle(queries)
    diff_qrs_ans.append((i1, i2, queries[0], u_ids_ans_dicts[i2], 0))

diff_qrs_ans_df = pd.DataFrame(diff_qrs_ans, columns=["id1", "id2", "query", "answer", "label"])
print(diff_qrs_ans_df)
diff_qrs_ans_df.to_feather(os.path.join("data", "queries_with_answers_0.feather"))
diff_qrs_ans_df.to_csv(os.path.join("data", "queries_with_answers_0.csv"), sep="\t", index=False)