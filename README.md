# Boat Distribution Prediction Using Gaussian Mixture Model (GMM)

This project uses a Gaussian Mixture Model (GMM) to predict the spatial distribution of small boats around large boats. It includes an interactive interface for data collection, model training, and prediction visualization.

## Features
- **Interactive Data Collection**:
  - Phase 1: Record the positions of large boats.
  - Phase 2: Record the positions of small boats relative to the large boats.
- **GMM Training**: Learn the spatial relationships of small boats relative to large boats.
- **Prediction Phase**: Predict small boat positions based on hypothetical large boat placements.
- **Visualization**: Display the recorded data, GMM results, and predicted positions.

## How to Use

### 1. Installation
1. Install dependencies:
   pip install -r requirements.txt

## Dependencies
The following Python libraries are required:
- `matplotlib`
- `numpy`
- `scikit-learn`

### 2. Running the Tool
Run the Python script to start the interactive interface:
python main.py

### 3. Keyboard Controls
- **`Enter`**: Advance to the next phase (e.g., from large boat collection to small boat collection, or to the prediction phase).

### 4. Workflow
1. **Phase 1**: Click on the image to record the positions of large boats.
2. **Phase 2**: Click on the positions of small boats relative to the large boats.
3. **Phase 3**: After training the GMM, click on the blank canvas to place hypothetical large boats. The tool predicts small boat distributions around them.

## Outputs
- **Plots**:
  - Relative coordinates of small boats around large boats.
  - Predicted positions of small boats around hypothetical large boats.
- **Console Logs**: Summaries of the recorded data and trained GMM parameters.
