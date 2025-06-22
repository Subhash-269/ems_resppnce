import streamlit as st
import joblib
import pickle
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
from datetime import datetime, time
import requests
import json

# Set page config
st.set_page_config(
    page_title="EMS Delay Hotspot Dashboard",
    page_icon="🚨",
    layout="wide"
)

# Load models and metadata
@st.cache_resource
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
        st.error(f"Model files not found: {e}")
        return None, None, None, None, None

# Load sample county data and feature options from actual data
@st.cache_data
def load_county_data():
    # Load actual data to get feature options
    try:
        df = pd.read_csv('Processed_Fatality.csv')
        
        # Get unique values for dropdown options
        feature_options = {
            'MONTHNAME': sorted(df['MONTHNAME'].dropna().unique()),
            'DAY_WEEKNAME': sorted(df['DAY_WEEKNAME'].dropna().unique()),
            'WEATHERNAME': sorted(df['WEATHERNAME'].dropna().unique()),
            'LGT_CONDNAME': sorted(df['LGT_CONDNAME'].dropna().unique()),
            'RUR_URBNAME': sorted(df['RUR_URBNAME'].dropna().unique()),
            'COUNTYNAME': sorted(df['COUNTYNAME'].dropna().unique()),
            'HOURNAME': sorted(df['HOURNAME'].dropna().unique()),
            'RD_OWNERNAME': sorted(df['RD_OWNERNAME'].dropna().unique()),
            'TYP_INTNAME': sorted(df['TYP_INTNAME'].dropna().unique()),
            'NHSNAME': sorted(df['NHSNAME'].dropna().unique()),
            'SP_JURNAME': sorted(df['SP_JURNAME'].dropna().unique()),
            'REL_ROADNAME': sorted(df['REL_ROADNAME'].dropna().unique()),
            'FUNC_SYSNAME': sorted(df['FUNC_SYSNAME'].dropna().unique())
        }
          # Sample county with ZIP codes (using Alabama counties from actual data)
        alabama_counties = df[df['STATENAME'] == 'Alabama']['COUNTYNAME'].unique()[:15]
        
        # Create realistic scattered coordinates for Alabama counties
        # Using actual approximate coordinates for different Alabama counties
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
          # Ensure we have enough coordinates
        while len(realistic_coords) < 15:
            # Add some random variations around Birmingham area
            base_lat, base_lon = 32.3584, -86.7998
            import random
            realistic_coords.append((
                base_lat + random.uniform(-1.5, 1.5),
                base_lon + random.uniform(-1.5, 1.5)
            ))
        
        sample_data = {
            'zip_code': [f'35{i:03d}' for i in range(100, 115)],
            'latitude': [coord[0] for coord in realistic_coords[:15]],
            'longitude': [coord[1] for coord in realistic_coords[:15]],
            'area_name': alabama_counties[:15] if len(alabama_counties) >= 15 else 
                        list(alabama_counties) + [f'Area_{i}' for i in range(len(alabama_counties), 15)],
            'county_name': alabama_counties[:15] if len(alabama_counties) >= 15 else 
                          list(alabama_counties) + [alabama_counties[0]] * (15 - len(alabama_counties))
        }
        
        county_df = pd.DataFrame(sample_data)
        return county_df, feature_options
        
    except Exception as e:
        st.error(f"Error loading data: {e}")        # Fallback data with scattered coordinates
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

# Function to create prediction input based on area characteristics
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
        
        # Create input row in the EXACT order expected by the model
        # Expected order: ['STATE', 'COUNTYNAME', 'CITYNAME', 'MONTHNAME', 'DAY', 'DAY_WEEKNAME', 'HOURNAME', 'TWAY_ID', 'ROUTENAME', 'RUR_URBNAME', 'FUNC_SYSNAME', 'RD_OWNERNAME', 'NHSNAME', 'SP_JURNAME', 'MILEPTNAME', 'RELJCT1NAME', 'RELJCT2NAME', 'TYP_INTNAME', 'REL_ROADNAME', 'WRK_ZONE', 'LGT_CONDNAME', 'WEATHERNAME', 'SCH_BUS', 'RAIL']
        
        input_row = []
        for feature in feature_names:
            if feature in feature_mapping:
                input_row.append(feature_mapping[feature])
            else:
                # Default value if feature not found
                input_row.append(0)
        
        # Create DataFrame with features in correct order
        input_df = pd.DataFrame([input_row], columns=feature_names)
        
        # Apply target encoding
        input_encoded = encoder.transform(input_df)
        
        # Debug output
        st.sidebar.write("🔍 Debug Info:")
        st.sidebar.write(f"Input shape: {input_encoded.shape}")
        st.sidebar.write(f"Expected features: {len(feature_names)}")
        st.sidebar.write(f"Input columns: {list(input_encoded.columns)}")
        
        return input_encoded
        
    except Exception as e:
        st.error(f"Error in prediction input: {e}")
        st.write(f"Expected features: {feature_names}")
        st.write(f"Feature mapping keys: {list(feature_mapping.keys()) if 'feature_mapping' in locals() else 'Not created'}")
        return None

# Main dashboard
def main():
    # Load models
    linear_model, rf_model, xgb_model, encoder, metadata = load_models()
    
    if not all([linear_model, rf_model, xgb_model, encoder, metadata]):
        st.error("Failed to load models. Please ensure all model files are present.")
        return
      # Get feature information
    feature_names = metadata['feature_names']
    categorical_features = metadata['categorical_features']
      # Load county data and feature options
    county_data, feature_options = load_county_data()
    
    # Debug: Show model features
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Model Info")
    st.sidebar.write(f"Features: {len(feature_names)}")
    with st.sidebar.expander("View Features"):
        for i, feature in enumerate(feature_names):
            st.sidebar.write(f"{i+1}. {feature}")
    
    # Dashboard title
    st.title("🚨 EMS Delay Hotspot Prediction Dashboard")
    st.markdown("---")
    
    # Sidebar for inputs
    st.sidebar.header("🔧 Scenario Configuration")
    
    # County selection
    county = st.sidebar.selectbox(
        "Select County:",
        ["Alabama (Demo)"]  # Add more states as needed
    )
    
    # Time-related inputs
    st.sidebar.subheader("⏰ Time Parameters")
    monthname = st.sidebar.selectbox("Month", feature_options['MONTHNAME'], index=5)
    day_weekname = st.sidebar.selectbox("Day of Week", feature_options['DAY_WEEKNAME'], index=2)
    hourname = st.sidebar.selectbox("Hour Range", feature_options['HOURNAME'], index=12)
    day = st.sidebar.number_input("Day of Month", 1, 31, 15)
    
    # Location-related inputs
    st.sidebar.subheader("📍 Location Parameters")
    countyname = st.sidebar.selectbox("County Name", feature_options['COUNTYNAME'])
    rur_urbname = st.sidebar.selectbox("Rural/Urban", feature_options['RUR_URBNAME'])
    
    # Road/Infrastructure inputs
    st.sidebar.subheader("🛣️ Road Infrastructure")
    rd_ownername = st.sidebar.selectbox("Road Owner", feature_options['RD_OWNERNAME'])
    func_sysname = st.sidebar.selectbox("Functional System", feature_options['FUNC_SYSNAME'])
    nhsname = st.sidebar.selectbox("NHS Status", feature_options['NHSNAME'])
    typ_intname = st.sidebar.selectbox("Intersection Type", feature_options['TYP_INTNAME'])
    rel_roadname = st.sidebar.selectbox("Relation to Road", feature_options['REL_ROADNAME'])
    
    # Environmental conditions
    st.sidebar.subheader("🌤️ Environmental Conditions")
    weathername = st.sidebar.selectbox("Weather Condition", feature_options['WEATHERNAME'])
    lgt_condname = st.sidebar.selectbox("Light Condition", feature_options['LGT_CONDNAME'])
    
    # Special conditions
    st.sidebar.subheader("🚨 Special Conditions")
    sch_bus = st.sidebar.selectbox("School Bus Related", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    rail = st.sidebar.selectbox("Railroad Related", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    wrk_zone = st.sidebar.selectbox("Work Zone", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    tway_id = st.sidebar.number_input("Trafficway ID", 0, 99999, 1)
    
    # Model selection
    model_choice = st.sidebar.selectbox(
        "Prediction Model:",
        ["Random Forest", "XGBoost", "Linear Regression", "Ensemble Average"]
    )
    
    # Collect user inputs
    user_inputs = {
        'monthname': monthname,
        'day_weekname': day_weekname,
        'hourname': hourname,
        'day': day,
        'countyname': countyname,
        'rur_urbname': rur_urbname,
        'rd_ownername': rd_ownername,
        'func_sysname': func_sysname,
        'nhsname': nhsname,
        'typ_intname': typ_intname,
        'rel_roadname': rel_roadname,
        'weathername': weathername,
        'lgt_condname': lgt_condname,
        'sch_bus': sch_bus,
        'rail': rail,
        'wrk_zone': wrk_zone,
        'tway_id': tway_id,
        'state': 1,  # Alabama
        'mileptname': '0',
        'reljct1name': 'No',
        'routename': 'Interstate',
        'cityname': 'NOT APPLICABLE',
        'sp_jurname': 'No Special Jurisdiction',
        'reljct2name': 'No'
    }    # Initialize session state for predictions
    if 'predictions_generated' not in st.session_state:
        st.session_state.predictions_generated = False
        st.session_state.pred_df = None
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("🗺️ EMS Delay Hotspot Map")
        
        # Buttons in columns for better layout
        btn_col1, btn_col2 = st.columns([2, 1])
        
        with btn_col1:
            generate_btn = st.button("🔮 Generate Hotspot Predictions", type="primary")
        
        with btn_col2:
            if st.button("🗑️ Clear Results"):
                st.session_state.predictions_generated = False
                st.session_state.pred_df = None
                st.rerun()
        
        if generate_btn:
            # Create predictions for each ZIP code
            predictions = []
            
            progress_bar = st.progress(0)
            
            for idx, row in county_data.iterrows():
                progress_bar.progress((idx + 1) / len(county_data))
                
                # Create area data dictionary from row
                area_data = {
                    'latitude': row['latitude'],
                    'longitude': row['longitude'],
                    'county_name': row.get('county_name', 'JEFFERSON')
                }
                
                # Create prediction input for this area
                input_data = create_area_prediction_input(
                    area_data, user_inputs, encoder, feature_names                )
                
                if input_data is not None:
                    try:
                        # Make prediction based on selected model
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
                            'predicted_delay': prediction
                        })
                    except Exception as e:
                        st.warning(f"Prediction failed for {row['zip_code']}: {e}")
                        predictions.append({
                            'zip_code': row['zip_code'],
                            'area_name': row['area_name'],
                            'latitude': row['latitude'],
                            'longitude': row['longitude'],
                            'predicted_delay': 2.0  # Default value
                        })
            
            progress_bar.empty()
            
            # Store results in session state
            st.session_state.pred_df = pd.DataFrame(predictions)
            st.session_state.predictions_generated = True
        
        # Display results if they exist in session state
        if st.session_state.predictions_generated and st.session_state.pred_df is not None:
            pred_df = st.session_state.pred_df
            
            # Check if predictions were successful
            if pred_df.empty:
                st.error("No predictions were generated. Please check the model files and try again.")
                return
              # Debug info
            st.write(f"Generated {len(pred_df)} predictions")
            st.write("Prediction columns:", pred_df.columns.tolist())
              # Create map
            center_lat = pred_df['latitude'].mean()
            center_lon = pred_df['longitude'].mean()
            
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=7,  # Reduced zoom to show more of Alabama
                tiles="OpenStreetMap"
            )
            
            # Define color scale for hotspots
            min_delay = pred_df['predicted_delay'].min()
            max_delay = pred_df['predicted_delay'].max()
            
            # Create color mapping
            def get_color(delay):
                # Normalize delay to 0-1 scale
                normalized = (delay - min_delay) / (max_delay - min_delay) if max_delay > min_delay else 0
                
                if normalized < 0.2:
                    return 'green'  # Low delay
                elif normalized < 0.4:
                    return 'yellow'  # Moderate delay
                elif normalized < 0.6:
                    return 'orange'  # High delay
                elif normalized < 0.8:
                    return 'red'     # Very high delay
                else:
                    return 'darkred'  # Extreme delay (HOTSPOT)
              # Add markers for each ZIP code
            for _, row in pred_df.iterrows():
                color = get_color(row['predicted_delay'])
                
                # Create popup text
                popup_text = f"""
                <b>{row['area_name']}</b><br>
                ZIP: {row['zip_code']}<br>
                Predicted Delay: {row['predicted_delay']:.2f} hours<br>
                Status: {'🔴 HOTSPOT' if color == 'darkred' else '⚠️ High Risk' if color == 'red' else '🟡 Moderate' if color in ['orange', 'yellow'] else '🟢 Low Risk'}
                """
                
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=15,
                    popup=popup_text,
                    color='black',
                    weight=1,
                    fillColor=color,
                    fillOpacity=0.7
                ).add_to(m)
              # Create legend using Streamlit columns instead of HTML overlay
            st.subheader("🗺️ Map Legend")
            
            # Show actual delay ranges from the data
            st.write(f"**Delay Range:** {min_delay:.2f} - {max_delay:.2f} hours")
            
            leg_cols = st.columns(5)
            
            with leg_cols[0]:
                st.markdown("🟢 **Low**")
                st.caption(f"< {min_delay + (max_delay - min_delay) * 0.2:.1f}h")
            with leg_cols[1]:
                st.markdown("🟡 **Moderate**") 
                st.caption(f"{min_delay + (max_delay - min_delay) * 0.2:.1f} - {min_delay + (max_delay - min_delay) * 0.4:.1f}h")
            with leg_cols[2]:
                st.markdown("🟠 **High**")
                st.caption(f"{min_delay + (max_delay - min_delay) * 0.4:.1f} - {min_delay + (max_delay - min_delay) * 0.6:.1f}h")
            with leg_cols[3]:
                st.markdown("🔴 **Very High**")
                st.caption(f"{min_delay + (max_delay - min_delay) * 0.6:.1f} - {min_delay + (max_delay - min_delay) * 0.8:.1f}h")
            with leg_cols[4]:
                st.markdown("🔴 **HOTSPOT**")
                st.caption(f"> {min_delay + (max_delay - min_delay) * 0.8:.1f}h")
            
            # Display map
            st_folium(m, width=700, height=500)
            
            # Display results table
            st.subheader("📊 Prediction Results by ZIP Code")
            
            # Sort by predicted delay (highest first)
            display_df = pred_df.sort_values('predicted_delay', ascending=False)
            display_df['Status'] = display_df['predicted_delay'].apply(
                lambda x: '🔴 HOTSPOT' if x > 4 else '⚠️ High Risk' if x > 3 else '🟡 Moderate' if x > 2 else '🟢 Low Risk'
            )
            display_df['Delay (Hours)'] = display_df['predicted_delay'].round(2)
            
            st.dataframe(
                display_df[['zip_code', 'area_name', 'Delay (Hours)', 'Status']].rename(columns={
                    'zip_code': 'ZIP Code',
                    'area_name': 'Area Name'
                }),
                hide_index=True
            )
            
            # Hotspot summary
            hotspots = display_df[display_df['predicted_delay'] > 4]
            if not hotspots.empty:
                st.error(f"🚨 {len(hotspots)} HOTSPOT(S) IDENTIFIED:")
                for _, hotspot in hotspots.iterrows():
                    st.write(f"• **{hotspot['area_name']}** (ZIP: {hotspot['zip_code']}) - {hotspot['predicted_delay']:.2f} hours")
            else:
                st.success("✅ No critical hotspots identified in this scenario.")
    
    with col2:
        st.header("📈 Analytics")
        
        # Model performance
        st.subheader("🏆 Model Performance")
        perf_data = []
        for model_name, metrics in metadata['model_performance'].items():
            perf_data.append({
                'Model': model_name.title(),
                'R²': f"{metrics['R2']:.3f}",
                'RMSE': f"{metrics['RMSE']:.3f}"
            })
        
        perf_df = pd.DataFrame(perf_data)
        st.dataframe(perf_df, hide_index=True)
          # Current scenario summary
        st.subheader("🔧 Current Scenario")
        st.write(f"**Time:** {hourname}, {day_weekname}")
        st.write(f"**Month:** {monthname}")
        st.write(f"**Weather:** {weathername}")
        st.write(f"**Light:** {lgt_condname}")
        st.write(f"**County:** {countyname}")
        st.write(f"**Model:** {model_choice}")
        
        # Instructions
        st.subheader("ℹ️ How to Use")
        st.write("""
        1. Adjust scenario parameters in the sidebar
        2. Click 'Generate Hotspot Predictions'
        3. View the map for visual hotspots
        4. Check the table for detailed results
        5. **Dark red areas** are critical hotspots
        """)

if __name__ == "__main__":
    main()