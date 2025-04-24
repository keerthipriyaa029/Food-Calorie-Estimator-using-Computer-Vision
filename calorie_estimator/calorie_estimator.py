import requests
import os
import json

class CalorieEstimator:
    def __init__(self, api_key=None, app_id=None):
        """
        Initialize the calorie estimator with an API key.
        
        Args:
            api_key (str, optional): API key for Calorie Ninja API.
                If None, uses environment variable NUTRITION_API_KEY.
            app_id (str, optional): Not used for Calorie Ninja API, kept for compatibility.
        """
        # Use provided API key or get from environment
        self.api_key = api_key or os.environ.get("NUTRITION_API_KEY")
        
        # Define endpoint for Calorie Ninja API
        self.api_url = "https://api.calorieninjas.com/v1/nutrition"
        
        # Fallback calorie data for common foods when API is not available
        self.fallback_data = {
            # Original items
            "apple_pie": 237,
            "banana": 89,
            "burger": 295,
            "pizza": 285,
            "salad": 33,
            "steak": 271,
            "sushi": 145,
            "pasta": 131,
            "chicken": 165,
            "soup": 62,
            "rice": 130,
            "bread": 265,
            "french_fries": 312,
            "ice_cream": 207,
            "chocolate_cake": 371,
            
            # Indian cuisine
            "butter_chicken": 240,
            "chicken_tikka_masala": 195,
            "samosa": 252,
            "naan": 260,
            "biryani": 180,
            "dal_makhani": 130,
            "palak_paneer": 180,
            "chole": 165,
            "dosa": 120,
            "idli": 39,
            "vada": 97,
            "paratha": 260,
            "chapati": 120,
            "paneer_tikka": 226,
            "tandoori_chicken": 165,
            "aloo_gobi": 105,
            "pakora": 175,
            "gulab_jamun": 175,
            "jalebi": 310,
            "rasgulla": 186,
            
            # Chinese cuisine
            "dim_sum": 170,
            "spring_rolls": 150,
            "fried_rice": 163,
            "chow_mein": 178,
            "kung_pao_chicken": 153,
            "ma_po_tofu": 110,
            "peking_duck": 225,
            "sweet_and_sour_pork": 231,
            "dumplings": 115,
            "hot_pot": 165,
            
            # Japanese cuisine
            "tempura": 230,
            "yakitori": 195,
            "tonkatsu": 271,
            "okonomiyaki": 170,
            "takoyaki": 210,
            "miso_soup": 40,
            "gyoza": 142,
            "sashimi": 110,
            "onigiri": 140,
            "tamagoyaki": 145,
            "matcha_ice_cream": 120,
            
            # Korean cuisine
            "bulgogi": 182,
            "japchae": 155,
            "tteokbokki": 145,
            "kimbap": 165,
            "samgyeopsal": 240,
            "galbi": 220,
            "bibimbap": 150,
            "kimchi_jjigae": 96,
            "sundubu_jjigae": 110,
            
            # Thai cuisine
            "tom_yum": 95,
            "pad_thai": 170,
            "green_curry": 135,
            "massaman_curry": 195,
            "som_tam": 75,
            "pad_see_ew": 160,
            "tom_kha_gai": 105,
            "mango_sticky_rice": 272,
            
            # Vietnamese cuisine
            "pho": 215,
            "banh_mi": 250,
            "goi_cuon": 105,
            "bun_cha": 170,
            "banh_xeo": 210,
            "ca_kho_to": 180,
            
            # Malaysian & Indonesian cuisine
            "nasi_lemak": 182,
            "satay": 195,
            "rendang": 197,
            "laksa": 160,
            "roti_canai": 301,
            "nasi_goreng": 186,
            "mee_goreng": 175,
            
            # Additional international cuisines
            "tacos": 210,
            "paella": 165,
            "ramen": 190,
            "falafel": 333,
            "hummus": 166,
            "gyro": 217,
            "lasagna": 132,
            "risotto": 174,
            "croissant": 406,
            "enchiladas": 168,
            "tiramisu": 283
        }
    
    def get_calories(self, food_item: str, serving_size: float = 100.0):
        """
        Get calorie estimation for a food item.
        
        Args:
            food_item (str): Food item name
            serving_size (float): Serving size in grams
            
        Returns:
            dict: Dictionary with calorie and nutrition information
        """
        # Replace underscores with spaces for better API matching
        food_query = food_item.replace("_", " ")
        
        # Check if API key is available
        if not self.api_key:
            print("Warning: No API key provided, using fallback data")
            return self._get_fallback_calories(food_item, serving_size)
        
        # Make API request
        try:
            # Format the query with the serving size
            query = f"{serving_size}g {food_query}"
            
            headers = {
                'X-Api-Key': self.api_key
            }
            
            params = {
                'query': query
            }
            
            response = requests.get(self.api_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Parse the response to get calorie information
            if data.get("items") and len(data["items"]) > 0:
                item = data["items"][0]
                
                # Extract nutritional information
                calories = item.get("calories", 0)
                protein = item.get("protein_g", 0)
                fat = item.get("fat_total_g", 0)
                carbs = item.get("carbohydrates_total_g", 0)
                
                return {
                    "food_item": food_query,
                    "serving_size": serving_size,
                    "calories": round(calories, 1),
                    "protein": round(protein, 1),
                    "fat": round(fat, 1),
                    "carbs": round(carbs, 1),
                    "source": "Calorie Ninja API"
                }
            else:
                # Fallback if no results found
                print(f"No nutrition data found for '{food_query}', using fallback data.")
                return self._get_fallback_calories(food_item, serving_size)
                
        except Exception as e:
            print(f"Error fetching nutrition data: {e}")
            return self._get_fallback_calories(food_item, serving_size)
    
    def _get_fallback_calories(self, food_item: str, serving_size: float = 100.0):
        """
        Get calorie estimation from fallback data.
        
        Args:
            food_item (str): Food item name
            serving_size (float): Serving size in grams
            
        Returns:
            dict: Dictionary with calorie information
        """
        # Clean up food item name for lookup
        food_item_clean = food_item.lower().replace("_", " ")
        
        # Find best match in fallback data
        best_match = None
        for key in self.fallback_data:
            if key.replace("_", " ") in food_item_clean or food_item_clean in key.replace("_", " "):
                best_match = key
                break
        
        # Use exact match, best match, or default value
        if food_item in self.fallback_data:
            calories_per_100g = self.fallback_data[food_item]
        elif best_match:
            calories_per_100g = self.fallback_data[best_match]
        else:
            # Default value if no match found
            calories_per_100g = 150
        
        # Adjust for serving size
        calories = (calories_per_100g / 100) * serving_size
        
        return {
            "food_item": food_item.replace("_", " "),
            "serving_size": serving_size,
            "calories": round(calories, 1),
            "protein": None,  # We don't have this data in the fallback
            "fat": None,
            "carbs": None,
            "source": "Fallback database"
        } 