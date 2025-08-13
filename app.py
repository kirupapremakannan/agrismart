from flask import Flask, request, jsonify, render_template
import os
import joblib
from dotenv import load_dotenv

import google.generativeai as genai

load_dotenv()

from crop_prediction_model import CropPredictionModel, WeatherAPI

app = Flask(__name__)

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in environment variables. Please add it to your .env file with the correct key and no leading spaces, and ensure .env is in the same directory as app.py.")
genai.configure(api_key=GEMINI_API_KEY)

# Use 'models/gemini-1.5-flash-latest' for better free-tier performance and general chat
gemini_model = genai.GenerativeModel('models/gemini-1.5-flash-latest')

crop_model = CropPredictionModel()
try:
    crop_model.load_model()
except FileNotFoundError as e:
    print(e)
    print("Attempting to train new model from farming_data.csv...")
    try:
        crop_model.train_model()
        crop_model.save_model()
    except Exception as train_error:
        print(f"CRITICAL ERROR: Failed to train model: {train_error}")
        print("Please ensure 'farming_data.csv' exists and is readable in the same directory.")
        exit()
except Exception as e:
    print(f"An unexpected error occurred during model loading: {e}. Attempting to train new model...")
    try:
        crop_model.train_model()
        crop_model.save_model()
    except Exception as train_error:
        print(f"CRITICAL ERROR: Failed to train new model: {train_error}")
        print("Prediction functionality may be limited or unavailable.")
        crop_model.model = None

OPENWEATHER_API_KEY = os.environ.get('OPENWEATHER_API_KEY', 'f9ffd8c5a21c3427694f975abb6bf37d')
if OPENWEATHER_API_KEY == 'f9ffd8c5a21c3427694f975abb6bf37d':
    print("WARNING: OPENWEATHER_API_KEY is not set. Weather features may not work.")
weather_api = WeatherAPI(OPENWEATHER_API_KEY)

# These default values are for internal use or if data is missing,
# but the client *should* ideally send them if they are part of the input.
DEFAULT_NITROGEN = 12.26
DEFAULT_PHOSPHORUS = 5.36
DEFAULT_POTASSIUM = 2011.72
DEFAULT_PH = 6.23


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict-crops', methods=['POST'])
def predict_crops():
    data = request.get_json()
    
    # Extract data, providing defaults if not present in the request
    temperature = data.get('temperature')
    humidity = data.get('humidity')
    nitrogen = data.get('nitrogen', DEFAULT_NITROGEN)
    phosphorus = data.get('phosphorus', DEFAULT_PHOSPHORUS)
    potassium = data.get('potassium', DEFAULT_POTASSIUM)
    ph = data.get('ph', DEFAULT_PH)
    
    # location is not used in predict_crops directly, but might be passed for context
    # It's important that this field is NOT passed to crop_model.predict_crops
    # as it's not a feature.
    location = data.get('location') 

    if temperature is None or humidity is None: # Ensure essential weather data is always provided by client
        return jsonify({"success": False, "error": "Missing essential environmental data (temperature, humidity). Please provide them."}), 400

    try:
        # Explicitly convert all incoming data to float. This handles cases where
        # data might be sent as strings by the frontend, even if they're numerical.
        temperature = float(temperature)
        humidity = float(humidity)
        nitrogen = float(nitrogen)
        phosphorus = float(phosphorus)
        potassium = float(potassium)
        ph = float(ph)
        
        predictions = crop_model.predict_crops(
            temperature=temperature,
            humidity=humidity,
            ph=ph,
            nitrogen=nitrogen,
            phosphorus=phosphorus,
            potassium=potassium
        )
        
        insights = generate_insights(temperature, humidity, ph, predictions)

        return jsonify({"success": True, "predictions": predictions, "insights": insights})

    except ValueError:
        # This error occurs if float() conversion fails, e.g., if input is 'abc'
        return jsonify({"success": False, "error": "Invalid numerical input for environmental data. Please ensure temperature, humidity, N, P, K, and pH are numbers."}), 400
    except RuntimeError as e:
        # This error catches issues like model not being loaded
        return jsonify({"success": False, "error": str(e)}), 500
    except Exception as e:
        print(f"Error during crop prediction: {e}")
        return jsonify({"success": False, "error": "An unexpected error occurred during prediction."}), 500


@app.route('/api/weather/<float:lat>/<float:lon>')
def get_weather(lat, lon):
    weather_data = weather_api.get_current_weather(lat, lon)
    if weather_data:
        # Get city name using the correct method
        city_name = weather_api.get_city_name_from_coords(lat, lon)
        if city_name:
            weather_data['location_name'] = city_name
        else:
            weather_data['location_name'] = f"Lat: {lat:.2f}, Lon: {lon:.2f}" # Fallback
            print(f"DEBUG: Could not determine city name from coordinates {lat}, {lon}. Using raw coordinates in response.")

        return jsonify({"success": True, "weather": weather_data})
    return jsonify({"success": False, "error": "Could not retrieve weather data."}), 500


@app.route('/api/geocode', methods=['GET'])
def geocode_city():
    city_name_query = request.args.get('cityName')
    if not city_name_query:
        return jsonify({"success": False, "error": "City name is required."}), 400

    # Prevent coordinate strings from being sent as city names to this endpoint
    # This checks for common patterns found in coordinate strings
    if "Lat:" in city_name_query or "Lon:" in city_name_query or (',' in city_name_query and any(c.isdigit() for c in city_name_query)):
        return jsonify({"success": False, "error": "Invalid city name format. Please enter a city name (e.g., 'London', 'New York'), not coordinates."}), 400

    coords = weather_api.get_coords_from_city(city_name_query)
    if coords:
        return jsonify({"success": True, "location": coords})
    return jsonify({"success": False, "error": "Could not find coordinates for the given city. Please check spelling or try a different city."}), 404


@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"success": False, "error": "No message provided."}), 400
    
    try:
        prompt_parts = [
            "You are AgriBot, an AI agricultural assistant designed to help farmers and gardening enthusiasts. ",
            "Provide helpful, concise, and accurate information related to crops, soil, pests, farming techniques, and general agriculture. ",
            "Keep your responses focused on farming topics. If a question is off-topic, politely redirect them to farming-related queries.",
            f"\nUser: {user_message}",
            "\nAgriBot: "
        ]
        
        response = gemini_model.generate_content("".join(prompt_parts))
        
        response_text = response.text
        
        return jsonify({"success": True, "response": response_text})

    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        return jsonify({"success": False, "error": "An error occurred while communicating with the AI. Please try again later."}), 500

def generate_insights(temperature, humidity, ph, predictions):
    """Generates simple insights based on the prediction and input parameters."""
    insights = "Based on the provided conditions:\n"
    
    if predictions:
        top_crop = predictions[0]['crop']
        insights += f"- The top recommended crop is **{top_crop}** with a confidence of {predictions[0]['confidence']}%. \n"
    else:
        insights += "- No specific crop recommendation could be made at this time.\n"

    insights += f"- Your average temperature is {temperature}°C and humidity is {humidity}%. \n"
    insights += f"- The pH is approximately {ph:.2f}. \n"
    
    if temperature < 15:
        insights += "- These cooler temperatures are generally suitable for crops like wheat or barley.\n"
    elif temperature > 30:
        insights += "- Warmer temperatures favor crops such as rice, corn, or sugarcane.\n"
    
    if humidity < 60:
        insights += "- Lower humidity might indicate a need for more irrigation or drought-resistant crops.\n"
    elif humidity > 80:
        insights += "- Higher humidity could increase the risk of fungal diseases, so ensure good air circulation.\n"

    if ph < 5.5:
        insights += "- Acidic soil (low pH) is preferred by crops like potatoes and coffee.\n"
    elif ph > 7.0:
        insights += "- Alkaline soil (high pH) is suitable for crops like cotton and certain legumes.\n"
    elif 5.5 <= ph <= 7.0:
        insights += "- The pH level is generally balanced, suitable for a wide range of crops.\n"

    return insights


if __name__ == '__main__':
    app.run(debug=True)