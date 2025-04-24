# Food Calorie Estimator

A Python application that uses computer vision and machine learning to identify food items from images and estimate their calorie content.

## Requirements

- Python 3.12 or newer
- Webcam (for image capture feature)
- Internet connection (for API access, optional)
- System Requirements:
  - CPU: 64-bit processor
  - RAM: 4GB minimum (8GB+ recommended)
  - OS: Windows 10/11, macOS 11+, or Linux (Ubuntu 20.04+, Debian 11+)

## Features

- **Food Classification**: Uses a pre-trained CNN (MobileNetV2) to identify food items from images
- **Calorie Estimation**: Estimates calories using Calorie Ninja API or fallback database
- **User-friendly Interface**: Streamlit-based web interface for easy interaction
- **Nutritional Information**: Provides calorie count and macronutrients (when available)
- **Live Image Capture**: Capture food images directly using your webcam
- **API Key Management**: Store API credentials in a .env file for convenience

## Directory Structure

```
food-calorie-estimator/
├── main.py                # Entry point with Streamlit app
├── classifier.py          # Loads model & predicts food item
├── calorie_estimator.py   # Fetches calories using Calorie Ninja API
├── utils.py               # Preprocessing image (OpenCV)
├── config.py              # API key configuration
├── .env                   # API key storage (not committed to version control)
├── requirements.txt       # Dependencies
├── runtime.txt            # Python version specification
├── setup.py               # Package setup and metadata
├── install.sh             # Installation script for Linux/Mac
├── install.bat            # Installation script for Windows
└── model/                 # Stores downloaded model
    └── food_model.pth     # Fine-tuned model (if available)
```

## Installation

### Quick Installation

1. Ensure you have Python 3.12 or newer installed:
```bash
python --version
```

2. Clone the repository:
```bash
git clone https://github.com/yourusername/food-calorie-estimator.git
cd food-calorie-estimator
```

3. Run the installation script:

   **On Linux/Mac:**
   ```bash
   chmod +x install.sh
   ./install.sh
   ```

   **On Windows:**
   ```
   install.bat
   ```

### Manual Installation

1. Ensure you have Python 3.12 or newer installed
2. Clone the repository
3. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. Install dependencies:
```bash
pip install -e .
```
   Or install directly from requirements file:
```bash
pip install -r requirements.txt
```

5. Configure API Key:
   - The application works without an API key using fallback data
   - For more accurate results, obtain an API key from [Calorie Ninja](https://calorieninjas.com/api)
   - You have two options to provide your API key:
     
     a. Edit the `.env` file (recommended):
     ```
     NUTRITION_API_KEY="your-api-key"
     ```
     
     b. Set environment variable:
     ```bash
     export NUTRITION_API_KEY="your-api-key"
     ```
     
     c. Enter the API key in the Streamlit interface.

## Usage

1. Start the Streamlit app:
```bash
streamlit run main.py
```

2. Open your browser and go to `http://localhost:8501`

3. Use the app in one of three ways:
   - Upload a food image from your device
   - Capture a food image using your webcam
   - Try one of the example images

4. View the classification results and calorie estimation

### Using the Webcam Feature

1. Click on the "Use Webcam" tab
2. Click "Start Camera" to activate your webcam
3. Position your food item in the frame
4. Click "Capture Image" when ready
5. View the results or click "Retake Photo" if needed

### API Key Configuration

For the best experience with accurate nutritional data:

1. Sign up for an account at [Calorie Ninja](https://calorieninjas.com/api)
2. Get your API key from your account dashboard
3. Add your API key to the `.env` file in the project root directory
4. Restart the application

The app will automatically use your API key without requiring manual entry each time.

## Technical Details

- **Image Preprocessing**: OpenCV and PyTorch for image preprocessing (resize, normalize)
- **Food Classification**: Pre-trained MobileNetV2 model that can recognize ~100 food categories
- **Calorie Estimation**: Uses Calorie Ninja API with fallback to local database
- **Data Visualization**: Matplotlib for nutrition pie charts
- **Webcam Integration**: OpenCV for capturing live images from webcam

## Compatibility and Troubleshooting

### Package Compatibility
This project uses packages that are compatible with Python 3.12:
- PyTorch 2.1.0+
- OpenCV 4.8.0+
- Streamlit 1.30.0+
- All other dependencies specified in requirements.txt

### Common Issues
- **Camera access**: On macOS, you may need to grant camera permissions explicitly
- **PyTorch installation**: For CUDA support, refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/)
- **Package installation issues**: Try `pip install --upgrade pip` and `pip install --only-binary=:all: package_name`

## Limitations

- The model can recognize about 100 common food items
- Calorie estimates are approximate and based on standard serving sizes
- Mixed dishes may be identified by their most prominent ingredient
- Webcam functionality requires proper camera permissions and may not work in all environments

## Future Improvements

- Fine-tune model on larger food datasets
- Add portion size estimation using depth sensing or reference objects
- Support for mixed dishes and multiple food items in one image
- Add historical tracking of food consumption
- Mobile app deployment
- Continuous video analysis for real-time calorie estimation

## License

MIT License

## Acknowledgments

- Food-101 dataset
- PyTorch and torchvision
- Streamlit for the web interface
- Calorie Ninja API for nutritional data 