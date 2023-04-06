import os
import re
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from config import PROJECT_ROOT_DIR


model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

q_df = pd.read_feather(os.path.join(PROJECT_ROOT_DIR, "data", "queries.feather"))
print(q_df)

for cl in q_df:
    print(cl)

uniques_answers = set(list(q_df["ID"]))

patterns = re.compile(r"\xa0")
ids_answers = []
for a_id in uniques_answers:
    ans_texts = list(q_df["ShortAnswerText"][q_df["ID"] == a_id])[0]
    ans_texts = patterns.sub(" ", ans_texts)
    ids_answers.append((a_id, ans_texts))

print(ids_answers[:10])
print(len(ids_answers))
paraphrases_answers = []

answers = [tx for i, tx in ids_answers]
paraphrases = util.paraphrase_mining(model, answers)
for paraphrase in paraphrases:
    score, i, j = paraphrase
    if score >= 0.9:
        print("{} \t\t {} \t\t Score: {:.4f}".format(ids_answers[i], ids_answers[j], score))
        paraphrases_answers.append((ids_answers[i][0], ids_answers[i][1], ids_answers[j][0], ids_answers[j][1], score))

paraphrases_answers_df = pd.DataFrame(paraphrases_answers, columns=["id1", "query1", "id2", "query2", "Score"])
print(paraphrases_answers_df)
paraphrases_answers_df.to_csv(os.path.join(PROJECT_ROOT_DIR, "data", "paraphrases_queries.csv"), sep="\t", index=False)