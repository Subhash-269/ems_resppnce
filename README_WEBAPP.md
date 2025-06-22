# EMS Delay Hotspot Dashboard

A professional web application for predicting and visualizing Emergency Medical Service (EMS) delay hotspots across Alabama counties using machine learning models.

## Features

- **Professional Web Interface**: Modern, responsive design with Bootstrap-style components
- **Interactive Map**: Leaflet-based map with color-coded markers showing delay predictions
- **Real-time Predictions**: Generate predictions using Random Forest, XGBoost, Linear Regression, or Ensemble models
- **Dynamic Legend**: Shows actual delay ranges based on predictions
- **Detailed Results**: Sortable table with ZIP codes, area names, delays, and risk levels
- **Hotspot Detection**: Automatically identifies and highlights critical delay areas

## Technology Stack

- **Backend**: Python Flask
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Mapping**: Leaflet.js
- **Machine Learning**: scikit-learn, XGBoost
- **Styling**: Custom CSS with modern design patterns

## Installation

1. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Model Files are Present**:
   Make sure these files are in the project directory:
   - `linear_regression_model.pkl`
   - `random_forest_model.pkl`
   - `xgboost_model.pkl`
   - `target_encoder.pkl`
   - `model_metadata.pkl`
   - `Processed_Fatality.csv`

3. **Run the Application**:
   ```bash
   python app.py
   ```

4. **Access the Dashboard**:
   Open your browser and go to: `http://localhost:5000`

## Project Structure

```
├── app.py                 # Flask backend application
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Main HTML template
├── static/
│   ├── css/
│   │   └── style.css     # Custom CSS styles
│   └── js/
│       └── app.js        # Frontend JavaScript
└── [model files]         # ML models and data
```

## Usage

1. **Configure Scenario**: Use the sidebar to set time parameters, location details, road infrastructure, environmental conditions, and special conditions

2. **Select Model**: Choose from Random Forest, XGBoost, Linear Regression, or Ensemble Average

3. **Generate Predictions**: Click "Generate Predictions" to create delay forecasts for Alabama counties

4. **View Results**: 
   - Interactive map shows color-coded markers (green=low risk, red=hotspot)
   - Legend displays actual delay ranges
   - Results table shows detailed predictions sorted by delay
   - Hotspot summary alerts you to critical areas

5. **Clear Results**: Use "Clear Results" to reset the map and start over

## API Endpoints

- `GET /`: Main dashboard page
- `POST /api/predict`: Generate predictions (JSON API)
- `GET /api/feature-options`: Get available feature values

## Model Information

The dashboard uses pre-trained machine learning models that consider 24 features including:
- Time factors (month, day, hour)
- Location characteristics (county, rural/urban)
- Road infrastructure (ownership, functional system, intersection type)
- Environmental conditions (weather, lighting)
- Special conditions (school bus, railroad, work zones)

## Deployment

For production deployment:

1. **Use a Production WSGI Server**:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:8000 app:app
   ```

2. **Environment Variables**: Set `FLASK_ENV=production`

3. **Reverse Proxy**: Use nginx or Apache for serving static files

## Advantages over Streamlit

✅ **Professional UI**: Custom CSS with modern design patterns
✅ **Better Performance**: Faster loading and rendering
✅ **Full Control**: Complete customization of layout and interactions
✅ **Scalability**: Can handle multiple concurrent users
✅ **Production Ready**: Easy to deploy with standard web servers
✅ **SEO Friendly**: Better for search engine optimization
✅ **Mobile Responsive**: Optimized for all device sizes

## Browser Compatibility

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is part of ALY6150 Healthcare Analytics coursework.
