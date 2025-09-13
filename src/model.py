import pandas as pd
import ast
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Lire les données
data = pd.read_csv("SMS_HAM_SPAM_with_embeddings.csv")

# Convertir les embeddings de str à liste
data["embeddings"] = data["embeddings"].apply(ast.literal_eval)

# Extraire X et Y
Y = data["Label"]
X = np.array(data["embeddings"].tolist())  # <- IMPORTANT ! Convertir en 2D numpy array

# Split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.75, random_state=0)

# Classificateur
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)

# Résultat
print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))
