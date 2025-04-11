import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import uuid
from math import radians, sin, cos, sqrt, atan2

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def haversine(lat1, lon1, lat2, lon2):
    # Radius of the Earth in km
    R = 6371.0
    
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Difference in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    # Distance in km
    distance = R * c
    return distance

def generate_emergency_data(num_records):
    # Define boroughs with their approximate bounds and hospitals
    boroughs = {
        'Manhattan': {
            'bounds': (40.700, 40.880, -74.020, -73.910),
            'hospitals': [
                (40.7420, -73.9760, 'Bellevue Hospital'),
                (40.7894, -73.9418, 'Mount Sinai Hospital'),
                (40.7505, -73.9773, 'NYU Langone')
            ],
            'population_density': 70000,
            'cluster_characteristics': {
                'healthcare_intensity': 0.9,
                'emergency_frequency': 0.85,
                'socioeconomic_complexity': 0.75
            }
        },
        'Brooklyn': {
            'bounds': (40.570, 40.740, -74.040, -73.860),
            'hospitals': [
                (40.6945, -73.9565, 'Brooklyn Hospital Center'),
                (40.6508, -73.9589, 'Kings County Hospital')
            ],
            'population_density': 40000,
            'cluster_characteristics': {
                'healthcare_intensity': 0.7,
                'emergency_frequency': 0.65,
                'socioeconomic_complexity': 0.6
            }
        },
        'Queens': {
            'bounds': (40.580, 40.780, -73.960, -73.700),
            'hospitals': [
                (40.7419, -73.8711, 'Elmhurst Hospital'),
                (40.7127, -73.7924, 'Queens Hospital Center')
            ],
            'population_density': 20000,
            'cluster_characteristics': {
                'healthcare_intensity': 0.6,
                'emergency_frequency': 0.55,
                'socioeconomic_complexity': 0.5
            }
        },
        'Bronx': {
            'bounds': (40.790, 40.900, -73.940, -73.800),
            'hospitals': [
                (40.8437, -73.8444, 'Jacobi Medical Center'),
                (40.8401, -73.9060, 'Lincoln Medical Center')
            ],
            'population_density': 30000,
            'cluster_characteristics': {
                'healthcare_intensity': 0.5,
                'emergency_frequency': 0.7,
                'socioeconomic_complexity': 0.4
            }
        },
        'Staten Island': {
            'bounds': (40.490, 40.650, -74.260, -74.050),
            'hospitals': [
                (40.6172, -74.0830, 'Staten Island University Hospital')
            ],
            'population_density': 8000,
            'cluster_characteristics': {
                'healthcare_intensity': 0.4,
                'emergency_frequency': 0.3,
                'socioeconomic_complexity': 0.3
            }
        }
    }

    # Enhanced emergency types with more clustering potential
    emergency_types = [
        ('Medical Emergency', 2),
        ('Trauma', 1),
        ('Cardiac Arrest', 1),
        ('Respiratory Distress', 1),
        ('Traffic Accident', 2),
        ('Fall', 2),
        ('Stroke', 1),
        ('Allergic Reaction', 2),
        ('Burns', 2),
        ('Poisoning', 1),
        ('Fire', 1),
        ('Chemical Spill', 1),
        ('Pediatric Emergency', 1),
        ('Geriatric Emergency', 2),
        ('Mental Health Crisis', 2)
    ]

    # More detailed noise and time factors for clustering
    noise_factors = [
        ('Low Noise', 1.0),
        ('Moderate Noise', 1.2),
        ('High Noise', 1.5),
        ('Construction Noise', 1.3),
        ('Traffic Noise', 1.4)
    ]

    # Initialize lists for DataFrame
    data = []
    
    # Generate data for 5 years
    base_time = datetime.now() - timedelta(days=365 * 5)
    for _ in range(num_records):
        # Generate unique incident ID
        incident_id = str(uuid.uuid4())
        
        # Select borough based on population density with more nuanced weighting
        borough_weights = [70, 40, 20, 30, 8]  # normalized population densities
        borough_weights = np.array(borough_weights) / sum(borough_weights)
        borough = np.random.choice(list(boroughs.keys()), p=borough_weights)
        
        # Get borough details
        borough_info = boroughs[borough]
        lat_min, lat_max, lon_min, lon_max = borough_info['bounds']
        
        # Generate timestamp (last 5 years)
        timestamp = base_time + timedelta(
            days=np.random.randint(0, 365 * 5),
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60)
        )
        
        # Time of day effects on response time and call volume
        hour = timestamp.hour
        is_rush_hour = 7 <= hour <= 9 or 16 <= hour <= 18
        is_night = 23 <= hour or hour <= 5
        
        # Generate location with clustering potential
        cluster_noise = random.choice(noise_factors)
        latitude = np.random.uniform(lat_min, lat_max)
        longitude = np.random.uniform(lon_min, lon_max)
        
        # Emergency type and priority with more granular selection
        emergency_type, priority = random.choice(emergency_types)
        
        # Calculate nearest hospital using Haversine formula
        hospitals = borough_info['hospitals']
        distances = [
            haversine(latitude, longitude, h_lat, h_lon)
            for h_lat, h_lon, _ in hospitals
        ]
        nearest_hospital = hospitals[np.argmin(distances)][2]
        distance_to_hospital = min(distances)  # Distance in km
        
        # Generate response time based on various factors
        base_response_time = np.random.normal(8, 2)  # base response time in minutes
        
        # Adjust response time based on conditions
        if is_rush_hour:
            base_response_time *= 1.3
        if is_night:
            base_response_time *= 0.8
        if priority == 1:
            base_response_time *= 0.9
        
        # Add clustering-relevant noise factor
        base_response_time *= cluster_noise[1]
        
        # Add some random variation
        response_time = max(2, base_response_time + np.random.normal(0, 1))
        
        # Weather conditions with clustering considerations
        weather_conditions = ['Clear', 'Rain', 'Snow', 'Fog']
        weather_weights = [0.7, 0.15, 0.1, 0.05]
        weather = np.random.choice(weather_conditions, p=weather_weights)
        
        # Traffic conditions
        traffic_conditions = ['Light', 'Moderate', 'Heavy']
        if is_rush_hour:
            traffic_weights = [0.1, 0.3, 0.6]
        else:
            traffic_weights = [0.5, 0.3, 0.2]
        traffic = np.random.choice(traffic_conditions, p=traffic_weights)
        
        # Ambulance unit info with additional tracking potential
        unit_id = f"AMB-{np.random.randint(1, 51):03d}"
        
        # Add cluster-specific metadata
        cluster_info = borough_info['cluster_characteristics']
        
        # Combine all information
        record = {
            'incident_id': incident_id,
            'timestamp': timestamp,
            'borough': borough,
            'latitude': latitude,
            'longitude': longitude,
            'emergency_type': emergency_type,
            'priority': priority,
            'response_time_minutes': round(response_time, 2),
            'nearest_hospital': nearest_hospital,
            'distance_to_hospital_km': round(distance_to_hospital, 2),
            'weather_condition': weather,
            'traffic_condition': traffic,
            'is_rush_hour': is_rush_hour,
            'is_night': is_night,
            'ambulance_unit': unit_id,
            'population_density': borough_info['population_density'],
            # New cluster-related features
            'noise_level': cluster_noise[0],
            'healthcare_intensity': cluster_info['healthcare_intensity'],
            'emergency_frequency': cluster_info['emergency_frequency'],
            'socioeconomic_complexity': cluster_info['socioeconomic_complexity']
        }
        
        data.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Save to CSV
    df.to_csv('data/emergency_service_data.csv', index=False)
    
    # Print basic statistics
    print("\nDataset Summary:")
    print(f"Total records: {len(df)}")
    print("\nResponse Time Statistics:")
    print(df['response_time_minutes'].describe())
    print("\nEmergency Types Distribution:")
    print(df['emergency_type'].value_counts(normalize=True))
    print("\nBorough Distribution:")
    print(df['borough'].value_counts(normalize=True))
    
    return df

# Generate the dataset
emergency_data = generate_emergency_data(10000)  # Increased sample size for more robust clustering