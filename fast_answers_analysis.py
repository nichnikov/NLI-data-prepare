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

answers = [tx for i, tx in ids_answers]
paraphrases = util.paraphrase_mining(model, answers)
for paraphrase in paraphrases[0:10]:
    score, i, j = paraphrase
    print("{} \t\t {} \t\t Score: {:.4f}".format(ids_answers[i], ids_answers[j], score))
