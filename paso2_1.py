import pandas as pd
import os

# Cargar las anotaciones de entrenamiento
train_df = pd.read_csv('./train_metadata.csv')

# Crear nuevas anotaciones para las imágenes aumentadas
augmented_annotations = []

for idx in range(len(train_df)):
    isic_id = train_df.iloc[idx]['isic_id']
    benign_malignant = train_df.iloc[idx]['benign_malignant']
    for i in range(5):  # Número de imágenes aumentadas por imagen original
        augmented_annotations.append({
            'isic_id': f'augmented_{idx}_{i}',
            'benign_malignant': benign_malignant
        })

# Convertir a DataFrame y guardar
augmented_df = pd.DataFrame(augmented_annotations)
augmented_df.to_csv('./augmented_train_metadata.csv', index=False)

# Combinar con el DataFrame original
combined_df = pd.concat([train_df, augmented_df])
combined_df.to_csv('./combined_train_metadata.csv', index=False)
