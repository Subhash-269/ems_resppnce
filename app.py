from flask import Flask, render_template, request, jsonify
import joblib
import pickle
import pandas as pd
import numpy as np
import json
from datetime import datetime

app = Flask(__name__)

# Load models and metadata
def load_models():
    try:
        linear_model = joblib.load('linear_regression_model.pkl')
        rf_model = joblib.load('random_forest_model.pkl')
        xgb_model = joblib.load('xgboost_model.pkl')
        encoder = joblib.load('target_encoder.pkl')
        
        with open('model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        return linear_model, rf_model, xgb_model, encoder, metadata
    except FileNotFoundError as e:
        print(f"Model files not found: {e}")
        return None, None, None, None, None

# Load models globally
linear_model, rf_model, xgb_model, encoder, metadata = load_models()

# Load county data and feature options
def load_county_data():
    try:
        df = pd.read_csv('Processed_Fatality.csv')
        
        # Get unique values for dropdown options
        feature_options = {
            'MONTHNAME': sorted(df['MONTHNAME'].dropna().unique().tolist()),
            'DAY_WEEKNAME': sorted(df['DAY_WEEKNAME'].dropna().unique().tolist()),
            'WEATHERNAME': sorted(df['WEATHERNAME'].dropna().unique().tolist()),
            'LGT_CONDNAME': sorted(df['LGT_CONDNAME'].dropna().unique().tolist()),
            'RUR_URBNAME': sorted(df['RUR_URBNAME'].dropna().unique().tolist()),
            'COUNTYNAME': sorted(df['COUNTYNAME'].dropna().unique().tolist()),
            'HOURNAME': sorted(df['HOURNAME'].dropna().unique().tolist()),
            'RD_OWNERNAME': sorted(df['RD_OWNERNAME'].dropna().unique().tolist()),
            'TYP_INTNAME': sorted(df['TYP_INTNAME'].dropna().unique().tolist()),
            'NHSNAME': sorted(df['NHSNAME'].dropna().unique().tolist()),
            'SP_JURNAME': sorted(df['SP_JURNAME'].dropna().unique().tolist()),
            'REL_ROADNAME': sorted(df['REL_ROADNAME'].dropna().unique().tolist()),
            'FUNC_SYSNAME': sorted(df['FUNC_SYSNAME'].dropna().unique().tolist())
        }
        
        # Create realistic scattered coordinates for Alabama counties
        realistic_coords = [
            (32.3584, -86.7998),  # Jefferson County (Birmingham area)
            (34.7304, -86.5861),  # Madison County (Huntsville area)
            (30.6944, -88.0431),  # Mobile County (Mobile area)
            (32.4167, -85.7000),  # Lee County (Auburn area)
            (33.2098, -87.5692),  # Tuscaloosa County
            (32.3668, -86.2999),  # Montgomery County
            (34.0522, -85.8306),  # Calhoun County (Anniston area)
            (33.6500, -85.8300),  # Talladega County
            (31.3000, -85.3900),  # Houston County (Dothan area)
            (34.6059, -87.6773),  # Lauderdale County (Florence area)
            (33.7940, -86.8074),  # Walker County
            (32.8065, -87.1236),  # Bibb County
            (33.4734, -86.8075),  # Shelby County
            (32.4500, -87.2400),  # Perry County
            (33.2100, -86.4500),  # St. Clair County
        ]
        
        alabama_counties = df[df['STATENAME'] == 'Alabama']['COUNTYNAME'].unique()[:15]
        
        sample_data = {
            'zip_code': [f'35{i:03d}' for i in range(100, 115)],
            'latitude': [coord[0] for coord in realistic_coords[:15]],
            'longitude': [coord[1] for coord in realistic_coords[:15]],
            'area_name': alabama_counties[:15].tolist() if len(alabama_counties) >= 15 else 
                        list(alabama_counties) + [f'Area_{i}' for i in range(len(alabama_counties), 15)],
            'county_name': alabama_counties[:15].tolist() if len(alabama_counties) >= 15 else 
                          list(alabama_counties) + [alabama_counties[0]] * (15 - len(alabama_counties))
        }
        
        county_df = pd.DataFrame(sample_data)
        return county_df, feature_options
        
    except Exception as e:
        print(f"Error loading data: {e}")
        # Fallback data
        sample_data = {
            'zip_code': ['35201', '35007', '35801', '36101', '35004'],
            'latitude': [33.5207, 33.3734, 34.7304, 32.3668, 33.6803],
            'longitude': [-86.8025, -86.6397, -86.5861, -86.2999, -86.5861],
            'area_name': ['Birmingham Area', 'McCalla Area', 'Huntsville Area', 'Montgomery Area', 'Cullman Area'],
            'county_name': ['JEFFERSON', 'JEFFERSON', 'MADISON', 'MONTGOMERY', 'CULLMAN']
        }
        feature_options = {
            'MONTHNAME': ['January', 'February', 'March', 'April', 'May', 'June', 
                         'July', 'August', 'September', 'October', 'November', 'December'],
            'DAY_WEEKNAME': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            'WEATHERNAME': ['Clear', 'Rain', 'Snow', 'Fog', 'Cloudy'],
            'LGT_CONDNAME': ['Daylight', 'Dark - Not Lighted', 'Dark - Lighted', 'Dawn', 'Dusk'],
            'RUR_URBNAME': ['Rural', 'Urban'],
            'COUNTYNAME': ['JEFFERSON', 'MADISON', 'MOBILE'],
            'HOURNAME': ['12:00am-12:59am', '1:00am-1:59am', '2:00am-2:59am'],
            'RD_OWNERNAME': ['State Highway Agency', 'County Highway Agency'],
            'TYP_INTNAME': ['Not an Intersection', 'Four-Way Intersection'],
            'NHSNAME': ['This Section IS NOT on the NHS', 'This Section IS on the NHS'],
            'SP_JURNAME': ['No Special Jurisdiction'],
            'REL_ROADNAME': ['On Roadway', 'On Roadside'],
            'FUNC_SYSNAME': ['Interstate', 'Principal Arterial']
        }
        return pd.DataFrame(sample_data), feature_options

# Load data globally
county_data, feature_options = load_county_data()

def create_area_prediction_input(area_data, user_inputs, encoder, feature_names):
    """Create prediction input for a specific area"""
    
    try:
        # Create a mapping of all possible feature values
        feature_mapping = {
            'STATE': user_inputs.get('state', 1),
            'COUNTYNAME': user_inputs.get('countyname', 'JEFFERSON'),
            'CITYNAME': 'NOT APPLICABLE',
            'MONTHNAME': user_inputs.get('monthname', 'June'),
            'DAY': user_inputs.get('day', 15),
            'DAY_WEEKNAME': user_inputs.get('day_weekname', 'Wednesday'),
            'HOURNAME': user_inputs.get('hourname', '12:00pm-12:59pm'),
            'TWAY_ID': user_inputs.get('tway_id', 1),
            'ROUTENAME': 'Interstate',
            'RUR_URBNAME': user_inputs.get('rur_urbname', 'Rural'),
            'FUNC_SYSNAME': user_inputs.get('func_sysname', 'Interstate'),
            'RD_OWNERNAME': user_inputs.get('rd_ownername', 'State Highway Agency'),
            'NHSNAME': user_inputs.get('nhsname', 'This Section IS NOT on the NHS'),
            'SP_JURNAME': 'No Special Jurisdiction',
            'MILEPTNAME': '0',
            'RELJCT1NAME': 'No',
            'RELJCT2NAME': 'No',
            'TYP_INTNAME': user_inputs.get('typ_intname', 'Not an Intersection'),
            'REL_ROADNAME': user_inputs.get('rel_roadname', 'On Roadway'),
            'WRK_ZONE': user_inputs.get('wrk_zone', 0),
            'LGT_CONDNAME': user_inputs.get('lgt_condname', 'Daylight'),
            'WEATHERNAME': user_inputs.get('weathername', 'Clear'),
            'SCH_BUS': user_inputs.get('sch_bus', 0),
            'RAIL': user_inputs.get('rail', 0)
        }
        
        # Create input row in the exact order expected by the model
        input_row = []
        for feature in feature_names:
            if feature in feature_mapping:
                input_row.append(feature_mapping[feature])
            else:
                input_row.append(0)
        
        # Create DataFrame with features in correct order
        input_df = pd.DataFrame([input_row], columns=feature_names)
        
        # Apply target encoding
        input_encoded = encoder.transform(input_df)
        
        return input_encoded
        
    except Exception as e:
        print(f"Error in prediction input: {e}")
        return None

@app.route('/')
def index():
    """Render the main dashboard page"""
    return render_template('index.html', 
                         feature_options=feature_options,
                         model_features=metadata['feature_names'] if metadata else [],
                         model_performance=metadata['model_performance'] if metadata else {})

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint to generate predictions"""
    try:
        data = request.json
        user_inputs = data.get('inputs', {})
        model_choice = data.get('model', 'Random Forest')
        
        predictions = []
        
        for _, row in county_data.iterrows():
            area_data = {
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'county_name': row.get('county_name', 'JEFFERSON')
            }
            
            input_data = create_area_prediction_input(
                area_data, user_inputs, encoder, metadata['feature_names']
            )
            
            if input_data is not None:
                try:
                    if model_choice == "Random Forest":
                        prediction = rf_model.predict(input_data)[0]
                    elif model_choice == "XGBoost":
                        prediction = xgb_model.predict(input_data)[0]
                    elif model_choice == "Linear Regression":
                        prediction = linear_model.predict(input_data)[0]
                    else:  # Ensemble
                        pred_rf = rf_model.predict(input_data)[0]
                        pred_xgb = xgb_model.predict(input_data)[0]
                        pred_linear = linear_model.predict(input_data)[0]
                        prediction = (pred_rf + pred_xgb + pred_linear) / 3
                    
                    predictions.append({
                        'zip_code': row['zip_code'],
                        'area_name': row['area_name'],
                        'latitude': row['latitude'],
                        'longitude': row['longitude'],
                        'predicted_delay': float(prediction)
                    })
                except Exception as e:
                    print(f"Prediction failed for {row['zip_code']}: {e}")
                    predictions.append({
                        'zip_code': row['zip_code'],
                        'area_name': row['area_name'],
                        'latitude': row['latitude'],
                        'longitude': row['longitude'],
                        'predicted_delay': 2.0
                    })
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'model_used': model_choice
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/feature-options')
def get_feature_options():
    """API endpoint to get feature options"""
    return jsonify(feature_options)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
