import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

def train_and_save_models():
    print("Chargement des données...")
    df = pd.read_excel('Dataset_Malnutrition_Cameroun_2018.xlsx', sheet_name='data')
    
    # Fill any remaining NaNs in numeric fields just in case
    df.fillna(0, inplace=True)

    quant_vars = ['age_mere', 'imc_mere', 'poids_mere', 'taille_mere', 'age_enfant_mois', 'rang_naissance', 'intervalle_naissance']
    qual_vars = ['region', 'milieu_residence', 'education_mere', 'source_eau', 'type_toilettes', 'index_richesse', 'sexe_enfant', 'religion']

    print("Préparation des features...")
    X_raw = df[quant_vars + qual_vars]
    
    # Targets
    y1 = df['Y1_retard_croissance']
    y2 = df['Y2_amaigrissement']
    y3 = df['Y3_insuffisance_ponderale']

    X_encoded = pd.get_dummies(X_raw, columns=qual_vars, drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    
    MODELS_DIR = Path('models')
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(scaler, MODELS_DIR / 'scaler.joblib')
    print("Scaler sauvegardé.")

    # Base estimators
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))
    ]
    
    # Train model 1 (Stunting / Retard de croissance)
    print("Entraînement Modèle 1 : Retard de croissance...")
    clf1 = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(class_weight='balanced'))
    clf1.fit(X_scaled, y1)
    joblib.dump(clf1, MODELS_DIR / 'model_stunting.joblib')
    
    # Train model 2 (Wasting / Amaigrissement)
    print("Entraînement Modèle 2 : Amaigrissement...")
    clf2 = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(class_weight='balanced'))
    clf2.fit(X_scaled, y2)
    joblib.dump(clf2, MODELS_DIR / 'model_wasting.joblib')
    
    # Train model 3 (Underweight / Insuffisance pondérale)
    print("Entraînement Modèle 3 : Insuffisance pondérale...")
    clf3 = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(class_weight='balanced'))
    clf3.fit(X_scaled, y3)
    joblib.dump(clf3, MODELS_DIR / 'model_underweight.joblib')
    
    # Global model (to be consistent, though we can just take max(y1, y2, y3))
    # Let's train a global model as well just in case, or we can just infer global risk from the 3 models.
    # We will infer it in app_eds.py.
    
    print("Entraînement des modèles terminé avec succès.")

if __name__ == "__main__":
    train_and_save_models()
