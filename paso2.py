# Cargar las anotaciones originales y aumentadas
import pandas as pd


original_annotations = pd.read_csv('./metadata.csv')
augmented_annotations = pd.read_csv('./augmented_train_metadata.csv')

# Combinar ambos conjuntos de anotaciones
combined_annotations = pd.concat([original_annotations, augmented_annotations])
combined_annotations.to_csv('./combined_train_metadata.csv', index=False)
