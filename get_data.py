import os
import json
import pandas as pd
from datetime import datetime
from storage import DataFromDB
from config import PROJECT_ROOT_DIR


with open(os.path.join(PROJECT_ROOT_DIR, "data", "statistics_parameters.json")) as stat_config_file:
    statistic_parameters = json.load(stat_config_file)


print(statistic_parameters)
today = datetime.today().strftime('%Y-%m-%d')
print(today)

db_ = DataFromDB(**statistic_parameters["db_credentials"])
rows = db_.fetch_from_db(1, today)
print(len(rows))

rows_df = pd.DataFrame(rows)
rows_df.to_feather(os.path.join(PROJECT_ROOT_DIR, "data", "queries.feather"))

