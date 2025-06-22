import pandas as pd
import joblib
import pickle

# Load the metadata to see what features the model expects
with open('model_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

print("Features used in training:")
for i, feature in enumerate(metadata['feature_names']):
    print(f"{i+1}. {feature}")

print(f"\nTotal features: {len(metadata['feature_names'])}")

print("\nCategorical features:")
for feature in metadata['categorical_features']:
    print(f"- {feature}")
