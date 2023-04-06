import os
import pandas as pd
from config import PROJECT_ROOT_DIR
from sentence_transformers import SentenceTransformer, util


model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
df = pd.read_feather(os.path.join(PROJECT_ROOT_DIR, "data", "queries.feather"))

for cl in df:
    print(cl)

unique_ids = list(set(list(df["ID"])))
print(unique_ids)
print(len(unique_ids))


not_similar_questions = []
for num, a_id in enumerate(unique_ids):
    print(num, "/", len(unique_ids))
    queries = list(df["Cluster"][df["ID"] == a_id])
    paraphrases = util.paraphrase_mining(model, queries)
    len1 = len(queries)
    sim_qrs = []
    for paraphrase in paraphrases:
        score, i, j = paraphrase
        if score > 0.95:
            sim_qrs.append(queries[j])
    not_similar_questions += [{"ID": a_id, "query": q} for q in queries if q not in sim_qrs]

not_similar_questions_df = pd.DataFrame(not_similar_questions)
print(df)
print(not_similar_questions_df)
not_similar_questions_df.to_csv(os.path.join(PROJECT_ROOT_DIR, "data", "not_similar_queries.csv"),
                                sep="\t", index=False)