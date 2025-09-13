from sentence_transformers import SentenceTransformer
import pandas as pd

data = pd.read_csv("SMS_HAM_SPAM.csv")

model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(data["Text"].tolist())

data["embeddings"] = embeddings.tolist()

data.to_csv("SMS_HAM_SPAM_with_embeddings.csv", index=False)