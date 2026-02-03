import os
import logging
import argparse
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from src.pipeline.build_pipeline import build_pipeline
from src.models.train_evaluate import evaluate_model



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("application.log"), logging.StreamHandler()],
)



# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Récupérer ton jeton
jetonapi = os.getenv("JETONAPI")

print(f"Jeton API chargé : {jetonapi}")


# titanic.py


# --- 1. Paramètres en ligne de commande ---
parser = argparse.ArgumentParser(description="Random Forest Titanic")
parser.add_argument(
    "--n_trees", type=int, default=20, help="Nombre d'arbres dans la Random Forest"
)
args = parser.parse_args()

print(f"Nombre d'arbres : {args.n_trees}")

# --- 2. Créer le modèle en utilisant le paramètre ---
clf = RandomForestClassifier(n_estimators=args.n_trees)

# --- 3. Reste du code : chargement des données, pipeline, etc. ---


# # os.chdir('/home/coder/work/ensae-reproductibilite-application')
TrainingData = pd.read_csv("data/raw/data.csv")


N_TREES = 20
MAX_DEPTH = None
MAX_FEATURES = "sqrt"




## Un peu d'exploration et de feature engineering

# ### Statut socioéconomique


## Encoder les données imputées ou transformées.



pipe = build_pipeline(n_trees=args.n_trees)
# splitting samples
y = TrainingData["Survived"]
X = TrainingData.drop("Survived", axis="columns")

# On _split_ notre _dataset_ d'apprentisage pour faire de la validation croisée une partie pour apprendre une partie pour regarder le score.
# Prenons arbitrairement 10% du dataset en test et 90% pour l'apprentissage.
X_TRAIN, X_TEST, y_train, y_test = train_test_split(X, y, test_size=0.1)
pd.concat([X_TRAIN, y_train], axis=1).to_csv("data/derived/train.csv")
pd.concat([X_TEST, y_test], axis=1).to_csv("data/derived/test.csv")

JETONAPI = "$trotskitueleski1917"


# Random Forest


# Ici demandons d'avoir 20 arbres
pipe.fit(X_TRAIN, y_train)


# calculons le score sur le dataset d'apprentissage et sur le dataset de test (10% du dataset d'apprentissage mis de côté)
# le score étant le nombre de bonne prédiction


y_pred = pipe.predict(X_TEST)

score, conf_matrix = evaluate_model(y_test, y_pred)
print(f"{score:.1%} de bonnes réponses sur les données de test pour validation")


print(20 * "-")
print("matrice de confusion")
print(confusion_matrix(y_test, pipe.predict(X_TEST)))
