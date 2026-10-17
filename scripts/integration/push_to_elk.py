import os

import pandas as pd
import urllib3
from elasticsearch import Elasticsearch

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ES_HOST = os.getenv("ES_HOST", "https://localhost:9200")
ES_USER = os.getenv("ES_USER", "elastic")
ES_PASS = os.environ["ES_PASS"]  # required, no default
VERIFY_CERTS = os.getenv("ES_VERIFY_CERTS", "false").lower() == "true"

es = Elasticsearch(ES_HOST, basic_auth=(ES_USER, ES_PASS), verify_certs=VERIFY_CERTS)

df = pd.read_csv("ueba_scored.csv", parse_dates=["time_window"])

for _, row in df.iterrows():
    doc = row.to_dict()
    doc["is_anomaly"] = (row["anomaly_label"] == "ANOMALY")
    doc["@timestamp"] = row["time_window"].isoformat()
    es.index(index="ueba-results", document=doc)
