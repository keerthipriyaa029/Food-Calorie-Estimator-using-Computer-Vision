# type: ignore
import torch
import torch.nn as nn
import torchvision.models as models
import os

class FoodClassifier:
    def __init__(self, model_path=None):
        """
        Initialize the food classifier with a pretrained model.
        
        Args:
            model_path (str, optional): Path to a fine-tuned model.
                If None, uses a pretrained MobileNetV2 model.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define the Food101 class labels
        self.food_classes = [
            'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
            'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
            'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
            'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla',
            'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
            'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
            'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict',
            'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras',
            'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice',
            'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich',
            'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup',
            'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna',
            'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup',
            'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters',
            'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck',
            'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib',
            'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto',
            'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits',
            'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake',
            'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles'
        ]
        
        # Load model
        if model_path and os.path.exists(model_path):
            # Load fine-tuned model
            self.model = torch.load(model_path, map_location=self.device)
        else:
            # Use pretrained MobileNetV2
            self.model = models.mobilenet_v2(pretrained=True)
            # Replace the classifier to match the number of food classes
            # Normally this would be fine-tuned on Food-101 dataset
            num_classes = len(self.food_classes)
            in_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features, num_classes)
            )
            print("Using pretrained MobileNetV2 without fine-tuning")
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def predict_food(self, image_tensor):
        """
        Predict food class from an image tensor.
        
        Args:
            image_tensor (torch.Tensor): Preprocessed image tensor
            
        Returns:
            str: Predicted food class label
            float: Confidence score
        """
        # Move tensor to device
        image_tensor = image_tensor.to(self.device)
        
        # Disable gradient calculation for inference
        with torch.no_grad():
            # Forward pass
            outputs = self.model(image_tensor)
            
            # Get predictions
            _, predicted_idx = torch.max(outputs, 1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence = probabilities[0][predicted_idx].item()
            
            # Get class label
            predicted_label = self.food_classes[predicted_idx.item()]
            
        return predicted_label, confidence
    
    def get_top_predictions(self, image_tensor, top_k=5):
        """
        Get top k predictions from an image tensor.
        
        Args:
            image_tensor (torch.Tensor): Preprocessed image tensor
            top_k (int): Number of top predictions to return
            
        Returns:
            list: List of (class_label, confidence) tuples
        """
        # Move tensor to device
        image_tensor = image_tensor.to(self.device)
        
        # Disable gradient calculation for inference
        with torch.no_grad():
            # Forward pass
            outputs = self.model(image_tensor)
            
            # Get top k predictions
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_idx = torch.topk(probabilities, top_k)
            
            # Convert to list of (class_label, confidence) tuples
            top_predictions = [
                (self.food_classes[idx.item()], prob.item())
                for idx, prob in zip(top_idx[0], top_probs[0])
            ]
            
        return top_predictions 