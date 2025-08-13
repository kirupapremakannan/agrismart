import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import requests
import os
from datetime import datetime

class CropPredictionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder_crop = LabelEncoder()
        
        # Define feature names based on the farming_data.csv and our parsing strategy
        # NOTE: 'soil_type_encoded' is intentionally NOT in this list.
        self.feature_names = [
            'temperature', 'humidity', 'ph', 'nitrogen', 'phosphorus', 'potassium'
        ]
        
        self.model_path = 'crop_prediction_model.pkl'
        self.scaler_path = 'feature_scaler.pkl'
        self.label_encoder_crop_path = 'label_encoder_crop.pkl'

    def load_data(self, file_path='farming_data.csv'):
        """Loads and preprocesses the farming_data.csv dataset.
        Will raise an error if the file is not found or unreadable."""
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded {len(df)} rows from {file_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: {file_path} not found. Please ensure the dataset is in the correct directory.")
        except Exception as e:
            raise Exception(f"Error loading or reading CSV '{file_path}': {e}.")

        # --- Data Preprocessing for farming_data.csv ---

        # 1. Parse NPK_ratio_ppm into separate N, P, K columns
        try:
            df[['nitrogen_str', 'phosphorus_str', 'potassium_str']] = df['NPK_ratio_ppm'].str.split('-', expand=True)
            df['nitrogen'] = pd.to_numeric(df['nitrogen_str'], errors='coerce')
            df['phosphorus'] = pd.to_numeric(df['phosphorus_str'], errors='coerce')
            df['potassium'] = pd.to_numeric(df['potassium_str'], errors='coerce')
            df.drop(columns=['nitrogen_str', 'phosphorus_str', 'potassium_str'], inplace=True)
            
            # Fill NaNs created by 'coerce' with a reasonable default (mean)
            df['nitrogen'].fillna(df['nitrogen'].mean(), inplace=True)
            df['phosphorus'].fillna(df['phosphorus'].mean(), inplace=True)
            df['potassium'].fillna(df['potassium'].mean(), inplace=True)

        except Exception as e:
            print(f"Warning: Could not robustly parse NPK_ratio_ppm or it's not in N-P-K format. Error: {e}. NPK columns will be filled with mean values.")
            df['nitrogen'] = df['nitrogen'].fillna(df['nitrogen'].mean() if 'nitrogen' in df.columns else 0)
            df['phosphorus'] = df['phosphorus'].fillna(df['phosphorus'].mean() if 'phosphorus' in df.columns else 0)
            df['potassium'] = df['potassium'].fillna(df['potassium'].mean() if 'potassium' in df.columns else 0)


        # 2. Parse pH_Range (e.g., "5.6-6.1") to a single pH value (average)
        def parse_range_avg(range_str):
            if isinstance(range_str, str) and '-' in range_str:
                try:
                    parts = range_str.split('-')
                    return (float(parts[0]) + float(parts[1])) / 2
                except ValueError:
                    return np.nan
            return np.nan

        df['ph'] = df['pH_Range'].apply(parse_range_avg)
        df['ph'].fillna(df['ph'].mean(), inplace=True)

        # 3. Use Indoor_Farm_Temperature_C and Indoor_Farm_Humidity_pct directly
        df['temperature'] = df['Indoor_Farm_Temperature_C']
        df['humidity'] = df['Indoor_Farm_Humidity_pct']

        # 4. Select final features and target
        # Ensure all selected features are numeric and cleaned
        df_processed = df[self.feature_names + ['Crop']].copy()
        df_processed = df_processed.dropna() # Drop rows with any remaining NaNs after processing

        # Encode target variable 'Crop'
        df_processed['Crop_encoded'] = self.label_encoder_crop.fit_transform(df_processed['Crop'])

        return df_processed

    def train_model(self):
        """Trains the Random Forest Classifier model."""
        df_processed = self.load_data()
        
        # Define features (X) and target (y)
        X = df_processed[self.feature_names]
        y = df_processed['Crop_encoded']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale numerical features
        self.scaler.fit(X_train) # Scaler learns features from X_train DataFrame (with names)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Initialize and train the model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)

        # Evaluate (optional)
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained with accuracy: {accuracy * 100:.2f}%")
        print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=self.label_encoder_crop.classes_))

    def save_model(self):
        """Saves the trained model, scaler, and label encoders."""
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        joblib.dump(self.label_encoder_crop, self.label_encoder_crop_path)
        print("Model, scaler, and label encoder saved.")

    def load_model(self):
        """Loads the trained model, scaler, and label encoders."""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path) and \
           os.path.exists(self.label_encoder_crop_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.label_encoder_crop = joblib.load(self.label_encoder_crop_path)
            print("Model, scaler, and label encoder loaded.")
        else:
            raise FileNotFoundError("One or more model files not found. Please train the model first.")

    def predict_crops(self, temperature, humidity, ph, nitrogen, phosphorus, potassium):
        """Predicts the best crop given input parameters."""
        if self.model is None or self.scaler is None or self.label_encoder_crop is None:
            raise RuntimeError("Model is not loaded or trained. Call load_model() or train_model() first.")

        # Create a DataFrame for prediction to maintain feature names consistency with StandardScaler
        # This resolves the "X does not have valid feature names" warning
        input_data_df = pd.DataFrame([[
            temperature, humidity, ph, nitrogen, phosphorus, potassium
        ]], columns=self.feature_names) # IMPORTANT: Use self.feature_names for column names

        # Scale the input data using the loaded scaler
        scaled_input = self.scaler.transform(input_data_df)

        # Predict the crop
        probabilities = self.model.predict_proba(scaled_input)[0]
        top_indices = probabilities.argsort()[-3:][::-1] # Top 3 crops

        predictions = []
        for i in top_indices:
            crop_name = self.label_encoder_crop.inverse_transform([i])[0]
            confidence = round(probabilities[i] * 100, 2)
            predictions.append({"crop": crop_name, "confidence": confidence})

        return predictions

class WeatherAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
        # CORRECTED: Ensure geo_url ends with a trailing slash
        self.geo_url = "https://api.openweathermap.org/geo/1.0/" 

    def get_current_weather(self, lat, lon):
        """Fetches current weather data for given coordinates."""
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric'  # Get temperature in Celsius
        }
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            data = response.json()
            return {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data['wind']['speed'],
                'description': data['weather'][0]['description']
            }
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"OpenWeatherMap API Response Content: {e.response.text}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred in get_current_weather: {e}")
            return None

    def get_city_name_from_coords(self, lat, lon):
        """
        Uses reverse geocoding to get a detailed location name (City, State, Country) from coordinates.
        """
        params = {
            'lat': lat,
            'lon': lon,
            'limit': 1, # Request only one result
            'appid': self.api_key
        }
        try:
            # CORRECTED: Use self.geo_url directly without adding another '/'
            response = requests.get(f"{self.geo_url}reverse", params=params) 
            response.raise_for_status()
            data = response.json()
            if data and len(data) > 0:
                city_name = data[0].get('name', 'Unknown City')
                state_name = data[0].get('state')
                country_name = data[0].get('country', 'Unknown Country')
                
                parts = [city_name]
                # Add state only if it exists and is not the same as the city name (to avoid "London, London, UK")
                if state_name and state_name.lower() != city_name.lower():
                    parts.append(state_name)
                parts.append(country_name)
                
                # Join parts, filtering out any empty strings that might result from .get()
                return ", ".join(filter(None, parts))
            return None # Return None if no location is found or data is malformed
        except requests.exceptions.RequestException as e:
            print(f"Error fetching city name from coordinates: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"OpenWeatherMap API Response Content: {e.response.text}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred in get_city_name_from_coords: {e}")
            return None
    
    def get_coords_from_city(self, city_name):
        """
        Retrieves latitude and longitude for a given city name using OpenWeatherMap's Geocoding API.
        """
        # CORRECTED: Use self.geo_url directly without adding another '/'
        url = f"{self.geo_url}direct?q={city_name}&limit=1&appid={self.api_key}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if data and len(data) > 0:
                return {'lat': data[0]['lat'], 'lon': data[0]['lon']}
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error fetching coordinates for city '{city_name}': {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"OpenWeatherMap API Response Content: {e.response.text}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred in get_coords_from_city: {e}")
            return None


# Usage example and testing (only run if this file is executed directly)
if __name__ == "__main__":
    crop_model = CropPredictionModel()
    
    # Try to load existing model, otherwise train a new one
    try:
        crop_model.load_model()
    except FileNotFoundError:
        print("Training new model with farming_data.csv...")
        try:
            crop_model.train_model()
            crop_model.save_model()
        except Exception as train_error:
            print(f"CRITICAL ERROR: Failed to train model from farming_data.csv: {train_error}")
            print("Please ensure 'farming_data.csv' exists and is readable in the same directory.")
            exit()

    except Exception as e:
        print(f"An error occurred loading the model: {e}. Attempting to train new model...")
        try:
            crop_model.train_model()
            crop_model.save_model()
        except Exception as train_error:
            print(f"CRITICAL ERROR: Failed to train model: {train_error}")
            print("Please ensure 'farming_data.csv' exists and is readable in the same directory.")
            exit()

    # Test prediction with example values
    print("\nTest Prediction Results:")
    try:
        test_prediction = crop_model.predict_crops(
            temperature=25.0,
            humidity=70.0,
            ph=6.0,
            nitrogen=12.0,
            phosphorus=5.0,
            potassium=2000.0
        )
        for result in test_prediction:
            print(f"{result['crop']}: {result['confidence']}% confidence")
    except Exception as e:
        print(f"Error during test prediction: {e}")
        print("Please ensure your input values for prediction match the features and ranges the model was trained on.")


    # Example with weather API (requires OPENWEATHER_API_KEY set in environment)
    weather_api_key = os.getenv('OPENWEATHER_API_KEY', 'f9ffd8c5a21c3427694f975abb6bf37d')
    if weather_api_key != 'f9ffd8c5a21c3427694f975abb6bf37d':
        weather_api = WeatherAPI(weather_api_key)
        lat, lon = 11.0168, 76.9558 # Example: Coimbatore coordinates
        
        current_weather = weather_api.get_current_weather(lat, lon)
        if current_weather:
            print(f"\nCurrent Weather at {lat},{lon}:")
            print(f"Temperature: {current_weather['temperature']}°C")
            print(f"Humidity: {current_weather['humidity']}%")
            print(f"Description: {current_weather['description']}")
            
            city_name = weather_api.get_city_name_from_coords(lat, lon)
            print(f"Location Name: {city_name if city_name else 'Could not determine location name'}")
        else:
            print("\nCould not retrieve current weather for testing.")
    else:
        print("\nSkipping weather API test: OPENWEATHER_API_KEY not set in environment or hardcoded placeholder.")