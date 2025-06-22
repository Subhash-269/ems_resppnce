#!/usr/bin/env python3
"""
Test script to check if pickle models can be loaded and rebuild them if needed.
"""
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def test_and_fix_models():
    """Test loading models and rebuild if needed."""
    print("Testing model loading...")
    
    # Try to load each model
    models_to_test = [
        'linear_regression_model.pkl',
        'random_forest_model.pkl', 
        'xgboost_model.pkl',
        'target_encoder.pkl',
        'model_metadata.pkl'
    ]
    
    broken_models = []
    working_models = []
    
    for model_file in models_to_test:
        try:
            print(f"Testing {model_file}...")
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            print(f"✅ {model_file} loaded successfully")
            working_models.append(model_file)
        except Exception as e:
            print(f"❌ {model_file} failed to load: {e}")
            broken_models.append(model_file)
    
    print(f"\nWorking models: {len(working_models)}")
    print(f"Broken models: {len(broken_models)}")
    
    if broken_models:
        print(f"\nAttempting to rebuild broken models...")
        rebuild_models()
    
    return len(broken_models) == 0

def rebuild_models():
    """Rebuild the models from the processed data."""
    print("Loading processed data...")
    
    # Load the processed data
    try:
        df = pd.read_csv('Processed_Fatality.csv')
        print(f"Loaded data with shape: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return False
    
    # Define features based on what we know the model expects
    # (from the previous dashboard debugging)
    feature_columns = [
        'HOUR', 'DAY_WEEK', 'MONTH', 'YEAR', 'FATALS', 'DRUNK_DR',
        'STATE', 'COUNTY', 'CITY', 'MAN_COLL', 'RELJCT2', 'TYP_INT',
        'WRK_ZONE', 'REL_ROAD', 'LGT_COND', 'WEATHER1', 'WEATHER2',
        'WEATHER', 'SCH_BUS', 'RAIL', 'NOT_HOUR', 'NOT_MIN', 'ARR_HOUR',
        'ARR_MIN', 'HOSP_HR', 'HOSP_MN', 'CF1', 'CF2', 'CF3'
    ]
    
    # Check which features exist in the data
    available_features = [col for col in feature_columns if col in df.columns]
    missing_features = [col for col in feature_columns if col not in df.columns]
    
    print(f"Available features: {len(available_features)}")
    print(f"Missing features: {missing_features}")
    
    if len(available_features) < 10:  # Need at least some features
        print("Not enough features available to rebuild models")
        return False
    
    # Prepare the data
    X = df[available_features].copy()
    
    # Create a simple target variable (EMS delay)
    # Based on notification to arrival time difference
    if 'NOT_HOUR' in df.columns and 'ARR_HOUR' in df.columns:
        df['EMS_DELAY'] = (df['ARR_HOUR'] - df['NOT_HOUR']) * 60 + (df.get('ARR_MIN', 0) - df.get('NOT_MIN', 0))
        # Clip to reasonable values
        df['EMS_DELAY'] = df['EMS_DELAY'].clip(0, 120)  # 0 to 120 minutes
        y = df['EMS_DELAY'].fillna(df['EMS_DELAY'].median())
    else:
        # Fallback: use a random target for demonstration
        print("Warning: Creating synthetic target variable")
        y = np.random.normal(15, 5, len(df))  # Average 15 min delay
    
    # Handle missing values
    X = X.fillna(X.median() if X.select_dtypes(include=[np.number]).shape[1] > 0 else 0)
    
    # Encode categorical variables
    label_encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    print(f"Final feature shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build and save models
    models = {}
    
    # Linear Regression
    print("Building Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models['linear_regression'] = lr
    
    # Random Forest
    print("Building Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models['random_forest'] = rf
    
    # XGBoost
    print("Building XGBoost...")
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    models['xgboost'] = xgb_model
    
    # Save models
    print("Saving models...")
    
    # Save individual models
    with open('linear_regression_model.pkl', 'wb') as f:
        pickle.dump(lr, f)
    
    with open('random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf, f)
    
    with open('xgboost_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    
    # Save label encoders as target encoder
    with open('target_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    
    # Save metadata
    metadata = {
        'feature_names': available_features,
        'feature_count': len(available_features),
        'target_name': 'EMS_DELAY',
        'label_encoders': list(label_encoders.keys())
    }
    
    with open('model_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print("✅ All models rebuilt and saved successfully!")
    
    # Test predictions
    print("\nTesting predictions...")
    for name, model in models.items():
        pred = model.predict(X_test[:5])
        print(f"{name}: {pred}")
    
    return True

if __name__ == "__main__":
    success = test_and_fix_models()
    if success:
        print("\n🎉 All models are working correctly!")
    else:
        print("\n❌ Some issues remain")
