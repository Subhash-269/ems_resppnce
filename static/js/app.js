// Global variables
let map;
let predictions = [];
let markers = [];

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeMap();
    setupEventListeners();
});

// Initialize the Leaflet map
function initializeMap() {
    // Center on Alabama
    map = L.map('map').setView([32.8067, -86.7911], 7);
    
    // Add dark theme tiles (CartoDB Dark Matter)
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '© OpenStreetMap contributors © CARTO',
        subdomains: 'abcd',
        maxZoom: 19
    }).addTo(map);
}

// Setup event listeners
function setupEventListeners() {
    const form = document.getElementById('prediction-form');
    const clearBtn = document.getElementById('clear-btn');
    
    form.addEventListener('submit', handlePredictionSubmit);
    clearBtn.addEventListener('click', clearResults);
}

// Handle form submission
async function handlePredictionSubmit(event) {
    event.preventDefault();
    
    const submitBtn = document.getElementById('predict-btn');
    const loading = document.getElementById('loading');
    
    // Show loading state
    submitBtn.disabled = true;
    loading.style.display = 'flex';
    
    try {        // Collect form data
        const formData = new FormData(event.target);
        const inputs = {};
        
        for (const [key, value] of formData.entries()) {
            // Convert numeric fields
            if (['day', 'tway_id', 'sch_bus', 'rail', 'wrk_zone'].includes(key)) {
                inputs[key] = parseInt(value);
            } else {
                inputs[key] = value;
            }
        }
        
        // Add default values
        inputs.state = 1; // Alabama
        inputs.mileptname = '0';
        inputs.reljct1name = 'No';
        inputs.routename = 'Interstate';
        inputs.cityname = 'NOT APPLICABLE';
        inputs.sp_jurname = 'No Special Jurisdiction';
        inputs.reljct2name = 'No';
        
        // Get selected model
        const model = formData.get('model');
        
        // Make prediction request
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                inputs: inputs,
                model: model
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            predictions = data.predictions;
            displayResults(predictions, model);
            updateMap(predictions);
        } else {
            showError('Prediction failed: ' + data.error);
        }
        
    } catch (error) {
        console.error('Error:', error);
        showError('Failed to generate predictions. Please try again.');
    } finally {
        // Hide loading state
        submitBtn.disabled = false;
        loading.style.display = 'none';
    }
}

// Display prediction results
function displayResults(predictions, modelUsed) {
    const resultsSection = document.getElementById('results-section');
    const mapLegend = document.getElementById('map-legend');
    const hotspotSummary = document.getElementById('hotspot-summary');
    const resultsTable = document.getElementById('results-tbody');
    const delayRange = document.getElementById('delay-range');
    
    // Show results section and legend
    resultsSection.style.display = 'block';
    mapLegend.style.display = 'block';
    
    // Calculate delay statistics
    const delays = predictions.map(p => p.predicted_delay);
    const minDelay = Math.min(...delays);
    const maxDelay = Math.max(...delays);
    
    // Update delay range
    delayRange.textContent = `Delay Range: ${minDelay.toFixed(2)} - ${maxDelay.toFixed(2)} hours (Model: ${modelUsed})`;
    
    // Sort predictions by delay (highest first)
    const sortedPredictions = [...predictions].sort((a, b) => b.predicted_delay - a.predicted_delay);
    
    // Check for hotspots
    const hotspots = sortedPredictions.filter(p => p.predicted_delay > 4);
    
    // Update hotspot summary
    if (hotspots.length > 0) {
        hotspotSummary.className = 'hotspot-summary error';
        hotspotSummary.innerHTML = `
            <h3><i class="fas fa-exclamation-triangle"></i> ${hotspots.length} HOTSPOT(S) IDENTIFIED:</h3>
            ${hotspots.map(h => `<div>• <strong>${h.area_name}</strong> (ZIP: ${h.zip_code}) - ${h.predicted_delay.toFixed(2)} hours</div>`).join('')}
        `;
    } else {
        hotspotSummary.className = 'hotspot-summary success';
        hotspotSummary.innerHTML = `
            <h3><i class="fas fa-check-circle"></i> No Critical Hotspots Identified</h3>
            <div>All areas show acceptable EMS delay levels in this scenario.</div>
        `;
    }
    
    // Update results table
    resultsTable.innerHTML = sortedPredictions.map(prediction => {
        const delay = prediction.predicted_delay;
        let status, statusClass;
        
        if (delay > 4) {
            status = '🔴 HOTSPOT';
            statusClass = 'status-hotspot';
        } else if (delay > 3) {
            status = '⚠️ High Risk';
            statusClass = 'status-high';
        } else if (delay > 2) {
            status = '🟡 Moderate';
            statusClass = 'status-moderate';
        } else {
            status = '🟢 Low Risk';
            statusClass = 'status-low';
        }
        
        return `
            <tr>
                <td>${prediction.zip_code}</td>
                <td>${prediction.area_name}</td>
                <td>${delay.toFixed(2)}</td>
                <td class="${statusClass}">${status}</td>
            </tr>
        `;
    }).join('');
}

// Update map with predictions
function updateMap(predictions) {
    // Clear existing markers
    markers.forEach(marker => map.removeLayer(marker));
    markers = [];
    
    // Calculate delay range for color mapping
    const delays = predictions.map(p => p.predicted_delay);
    const minDelay = Math.min(...delays);
    const maxDelay = Math.max(...delays);
    
    // Add new markers
    predictions.forEach(prediction => {
        const color = getMarkerColor(prediction.predicted_delay, minDelay, maxDelay);
        
        // Create popup content
        const popupContent = `
            <div style="font-family: 'Segoe UI', sans-serif;">
                <h4 style="margin: 0 0 8px 0; color: #2c3e50;">${prediction.area_name}</h4>
                <p style="margin: 4px 0;"><strong>ZIP:</strong> ${prediction.zip_code}</p>
                <p style="margin: 4px 0;"><strong>Predicted Delay:</strong> ${prediction.predicted_delay.toFixed(2)} hours</p>
                <p style="margin: 4px 0;"><strong>Status:</strong> ${getStatusText(prediction.predicted_delay)}</p>
            </div>
        `;
          // Create marker
        const marker = L.circleMarker([prediction.latitude, prediction.longitude], {
            radius: 12,
            fillColor: color,
            color: '#fff',
            weight: 2,
            opacity: 1,
            fillOpacity: 0.8
        })
        .bindPopup(popupContent)
        .addTo(map);
        
        markers.push(marker);
    });
    
    // Fit map to show all markers
    if (markers.length > 0) {
        const group = new L.featureGroup(markers);
        map.fitBounds(group.getBounds().pad(0.1));
    }
}

// Get marker color based on delay
function getMarkerColor(delay, minDelay, maxDelay) {
    const normalized = maxDelay > minDelay ? (delay - minDelay) / (maxDelay - minDelay) : 0;
    
    if (normalized < 0.2) return '#28a745';      // Green - Low
    if (normalized < 0.4) return '#ffc107';      // Yellow - Moderate
    if (normalized < 0.6) return '#fd7e14';      // Orange - High
    if (normalized < 0.8) return '#dc3545';      // Red - Very High
    return '#6f0000';                            // Dark Red - Hotspot
}

// Get status text based on delay
function getStatusText(delay) {
    if (delay > 4) return '🔴 HOTSPOT';
    if (delay > 3) return '⚠️ High Risk';
    if (delay > 2) return '🟡 Moderate';
    return '🟢 Low Risk';
}

// Clear all results
function clearResults() {
    // Hide results
    document.getElementById('results-section').style.display = 'none';
    document.getElementById('map-legend').style.display = 'none';
    
    // Clear markers
    markers.forEach(marker => map.removeLayer(marker));
    markers = [];
    
    // Reset map view
    map.setView([32.8067, -86.7911], 7);
    
    // Clear predictions
    predictions = [];
}

// Show error message
function showError(message) {
    const hotspotSummary = document.getElementById('hotspot-summary');
    hotspotSummary.className = 'hotspot-summary error';
    hotspotSummary.style.display = 'block';
    hotspotSummary.innerHTML = `
        <h3><i class="fas fa-exclamation-triangle"></i> Error</h3>
        <div>${message}</div>
    `;
    
    // Show the results section so error is visible
    document.getElementById('results-section').style.display = 'block';
}

// Utility functions for form handling
function resetForm() {
    document.getElementById('prediction-form').reset();
}

function validateForm() {
    const form = document.getElementById('prediction-form');
    return form.checkValidity();
}
