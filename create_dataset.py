import pandas as pd
import numpy as np
import os
from pathlib import Path

def create_malnutrition_dataset():
    print("Chargement des données EDS...")
    # Read the dataset
    input_file = "CMKR71FL.xlsx"
    df = pd.read_excel(input_file)
    
    # Convert all columns to lower case to ensure standard matching
    df.columns = df.columns.str.lower()
    
    # Variables of interest
    vars_interest = {
        'v012': 'age_mere',
        'v024': 'region',
        'v025': 'milieu_residence',
        'v106': 'education_mere',
        'v113': 'source_eau',
        'v116': 'type_toilettes',
        'v190': 'index_richesse',
        'v130': 'religion',
        'v437': 'poids_mere',
        'v438': 'taille_mere',
        'v445': 'imc_mere',
        'b4': 'sexe_enfant',
        'hw1': 'age_enfant_mois',
        'bord': 'rang_naissance',
        'b11': 'intervalle_naissance',
        'hw70': 'zscore_taille_age',
        'hw71': 'zscore_poids_taille',
        'hw72': 'zscore_poids_age'
    }
    
    # Check which variables are present
    available_vars = [v for v in vars_interest.keys() if v in df.columns]
    
    # Select available columns
    df_selected = df[available_vars].copy()
    
    # Rename columns to be more readable
    df_selected.rename(columns={k: vars_interest[k] for k in available_vars}, inplace=True)
    
    print("Dimensions initiales:", df_selected.shape)
    
    # Fill missing intervals for first-borns with 0 or a specific category
    if 'intervalle_naissance' in df_selected.columns:
        df_selected['intervalle_naissance'] = df_selected['intervalle_naissance'].fillna(0)
    
    # Filter Z-scores:
    # 1. Remove missing/flagged values (>= 9996)
    # 2. Exclude over-nutrition (Z-score > 200)
    
    zscore_cols = ['zscore_taille_age', 'zscore_poids_taille', 'zscore_poids_age']
    for col in zscore_cols:
        if col in df_selected.columns:
            # Drop NaN
            df_selected = df_selected.dropna(subset=[col])
            # Keep valid numeric scores (DHS flags are >= 9996)
            df_selected = df_selected[df_selected[col] < 9000]
            # Exclude over-nutrition (score > 200)
            df_selected = df_selected[df_selected[col] <= 200]
            
    # Clean mother BMI, Weight and Height
    if 'imc_mere' in df_selected.columns:
        # 9998/9999 are flags
        df_selected = df_selected[df_selected['imc_mere'] < 9000]
        # DHS stores BMI * 100, so we convert it
        df_selected['imc_mere'] = df_selected['imc_mere'] / 100.0

    if 'poids_mere' in df_selected.columns:
        df_selected = df_selected[df_selected['poids_mere'] < 9000]
        df_selected['poids_mere'] = df_selected['poids_mere'] / 10.0

    if 'taille_mere' in df_selected.columns:
        df_selected = df_selected[df_selected['taille_mere'] < 9000]
        df_selected['taille_mere'] = df_selected['taille_mere'] / 10.0

    print("Dimensions après filtrage de la sur-nutrition et valeurs manquantes :", df_selected.shape)
    
    # Create the binary target variables: Under-nutrition if Z-score < -200
    if 'zscore_taille_age' in df_selected.columns:
        df_selected['Y1_retard_croissance'] = (df_selected['zscore_taille_age'] < -200).astype(int)
    if 'zscore_poids_taille' in df_selected.columns:
        df_selected['Y2_amaigrissement'] = (df_selected['zscore_poids_taille'] < -200).astype(int)
    if 'zscore_poids_age' in df_selected.columns:
        df_selected['Y3_insuffisance_ponderale'] = (df_selected['zscore_poids_age'] < -200).astype(int)
        
    # Global CIAF (Composite Index of Anthropometric Failure)
    df_selected['Y_global_malnutrition'] = df_selected[['Y1_retard_croissance', 'Y2_amaigrissement', 'Y3_insuffisance_ponderale']].max(axis=1)

    # Drop any remaining NA values
    df_selected = df_selected.dropna()
    print("Dimensions finales :", df_selected.shape)

    # Creation of metadata sheet
    metadata = {
        'Variable_Finale': [
            'age_mere', 'region', 'milieu_residence', 'education_mere', 'source_eau', 'type_toilettes',
            'index_richesse', 'religion', 'poids_mere', 'taille_mere', 'imc_mere', 'sexe_enfant', 'age_enfant_mois', 'rang_naissance', 'intervalle_naissance',
            'zscore_taille_age', 'zscore_poids_taille', 'zscore_poids_age',
            'Y1_retard_croissance', 'Y2_amaigrissement', 'Y3_insuffisance_ponderale', 'Y_global_malnutrition'
        ],
        'Variable_EDS_Origine': [
            'v012', 'v024', 'v025', 'v106', 'v113', 'v116', 'v190', 'v130', 'v437', 'v438', 'v445', 'b4', 'hw1', 'bord', 'b11',
            'hw70', 'hw71', 'hw72', 'Calculé (hw70<-200)', 'Calculé (hw71<-200)', 'Calculé (hw72<-200)', 'Calculé (Max(Y1, Y2, Y3))'
        ],
        'Description_et_Encodage': [
            'Age en annees (Continu)',
            'Region (1=Adamaoua, 2=Centre, 3=Douala, 4=Est, 5=Extrême-Nord, 6=Littoral, 7=Nord, 8=Nord-Ouest, 9=Ouest, 10=Sud, 11=Sud-Ouest, 12=Yaoundé)',
            'Milieu de residence (1=Urbain, 2=Rural)',
            'Niveau education (0=Aucune, 1=Primaire, 2=Secondaire, 3=Superieur)',
            'Source principale eau (Categoriel EDS)',
            'Type de toilettes (Categoriel EDS)',
            'Indice de richesse (1=Le plus pauvre, 2=Plus pauvre, 3=Moyen, 4=Plus riche, 5=Le plus riche)',
            'Religion (Categoriel EDS: 1=Catholique, 2=Protestant, etc.)',
            'Poids de la mere (Continu, kg)',
            'Taille de la mere (Continu, cm)',
            'IMC de la mere (Continu, kg/m2)',
            'Sexe de l enfant (1=Masculin, 2=Feminin)',
            'Age de l enfant en mois (Continu)',
            'Rang de naissance (Continu)',
            'Intervalle naissance precedent en mois (Continu)',
            'Z-score taille/age (Continu * 100)',
            'Z-score poids/taille (Continu * 100)',
            'Z-score poids/age (Continu * 100)',
            'Retard de croissance (0=Non, 1=Oui)',
            'Amaigrissement (0=Non, 1=Oui)',
            'Insuffisance ponderale (0=Non, 1=Oui)',
            'Indice global de sous-nutrition (0=Non, 1=Oui)'
        ]
    }
    
    df_meta = pd.DataFrame(metadata)
    
    # Save to Excel
    output_file = 'Dataset_Malnutrition_Cameroun_2018.xlsx'
    print(f"Sauvegarde du jeu de donnees dans {output_file}...")
    with pd.ExcelWriter(output_file) as writer:
        df_selected.to_excel(writer, sheet_name='data', index=False)
        df_meta.to_excel(writer, sheet_name='metadonnees', index=False)
    
    print("Processus terminé avec succès.")

if __name__ == "__main__":
    create_malnutrition_dataset()
