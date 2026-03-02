import streamlit as st
import cv2
import numpy as np
from PIL import Image
import joblib
import torch
import torch.nn as nn
import os

# ASL letter mapping (excluding J=9 and Z=25 which require motion)
# Kaggle MNIST labels: 0-24 map to A-Z (skipping J and Z)
# Label 0=A, 1=B, ..., 8=I, 9=K (J skipped), ..., 23=X, 24=Y (Z skipped)
LABEL_TO_LETTER = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
    17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'
}

ASL_LETTERS = list(LABEL_TO_LETTER.values())

# Feature size for pixel-based model (28x28 images)
PIXEL_FEATURE_SIZE = 784


class MLPClassifier(nn.Module):
    """Multi-Layer Perceptron for ASL classification."""
    
    def __init__(self, input_size=784, hidden_sizes=[512, 256, 128], num_classes=24,
                 activation='relu', dropout=0.3):
        super(MLPClassifier, self).__init__()
        
        self.hidden_sizes = hidden_sizes
        self.activation_name = activation
        
        layers = []
        prev_size = input_size
        
        activation_fn = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU()
        }.get(activation, nn.ReLU())
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def preprocess_image_for_prediction(image, invert=False):
    """
    Preprocess an uploaded image to match Kaggle MNIST format.
    - Convert to grayscale
    - Resize to 28x28
    - Optionally invert colors to match training data format
    - Flatten to 784 pixels
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Resize to 28x28
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Invert if needed (training data has light background, dark hand)
    if invert:
        resized = 255 - resized
    
    flattened = resized.flatten().astype(np.float64)
    
    return flattened, resized


def create_processed_preview(resized_image):
    """Create a larger preview of the 28x28 processed image."""
    preview = cv2.resize(resized_image, (200, 200), interpolation=cv2.INTER_NEAREST)
    return preview


@st.cache_resource
def load_svm_model():
    """Load the trained SVM model."""
    model_paths = [
        'src/models/asl_svm_model.pkl',
        'src/asl_svm_model.pkl',
        'models/asl_svm_model.pkl',
        'asl_svm_model.pkl'
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            return joblib.load(model_path)
    
    return None


@st.cache_resource
def load_mlp_model():
    """Load the trained MLP model."""
    model_paths = [
        'src/models/asl_mlp_model.pt',
        'src/asl_mlp_model.pt',
        'models/asl_mlp_model.pt',
        'asl_mlp_model.pt'
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
            
            hidden_sizes = checkpoint.get('hidden_sizes', [128, 64])
            num_classes = checkpoint.get('num_classes', 24)
            activation = checkpoint.get('activation', 'relu')
            dropout = checkpoint.get('dropout', 0.3)
            
            model = MLPClassifier(
                input_size=checkpoint.get('input_size', 784), 
                hidden_sizes=hidden_sizes, 
                num_classes=num_classes,
                activation=activation,
                dropout=dropout
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            return model, checkpoint.get('label_encoder', None)
    
    return None, None


def predict_with_svm(model, features):
    """Make prediction using SVM model."""
    features_array = np.array(features).reshape(1, -1)
    prediction_label = model.predict(features_array)[0]
    probabilities = model.predict_proba(features_array)[0]
    confidence = max(probabilities) * 100
    
    # Convert numeric label to letter
    prediction = LABEL_TO_LETTER.get(prediction_label, str(prediction_label))
    
    return prediction, confidence, probabilities


def predict_with_mlp(model, features, label_encoder=None):
    """Make prediction using MLP model."""
    features_tensor = torch.FloatTensor(features).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(features_tensor)
        probabilities = torch.softmax(outputs, dim=1).numpy()[0]
        predicted_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_idx] * 100
        
        if label_encoder is not None:
            prediction = label_encoder.inverse_transform([predicted_idx])[0]
        else:
            prediction = ASL_LETTERS[predicted_idx]
    
    return prediction, confidence, probabilities


def main():
    st.set_page_config(
        page_title="ASL Sign Language Detector",
        page_icon="🤟",
        layout="wide"
    )
    
    st.title("🤟 ASL Sign Language Detector")
    st.markdown("""
    Upload an image of a hand sign (A-Z, excluding J and Z) to predict the letter.
    The image will be converted to 28x28 grayscale format for classification.
    """)
    
    st.sidebar.header("Model Selection")
    
    svm_model = load_svm_model()
    mlp_model, label_encoder = load_mlp_model()
    
    available_models = []
    if svm_model is not None:
        available_models.append("SVM (Classical ML)")
    if mlp_model is not None:
        available_models.append("MLP (Neural Network)")
    
    if not available_models:
        st.error("""
        No trained models found! Please train the models first:
        
        1. **SVM Model**: Run `python src/model_training.py` 
        2. **MLP Model**: Run `python src/train_mlp.py`
        
        Expected model locations:
        - `src/models/asl_svm_model.pkl` or `src/asl_svm_model.pkl`
        - `src/models/asl_mlp_model.pt` or `src/asl_mlp_model.pt`
        """)
        return
    
    selected_model = st.sidebar.radio(
        "Choose a model:",
        available_models,
        help="Select which model to use for prediction"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Status")
    st.sidebar.markdown(f"- SVM: {'✅ Loaded' if svm_model else '❌ Not found'}")
    st.sidebar.markdown(f"- MLP: {'✅ Loaded' if mlp_model else '❌ Not found'}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Image Settings")
    invert_colors = st.sidebar.checkbox(
        "Invert colors",
        value=False,
        help="Try this if predictions are wrong. Inverts image colors to match training data format."
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'webp'],
            help="Upload a clear image of a hand sign"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            if len(image_array.shape) == 2:
                display_image = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            elif image_array.shape[2] == 4:
                display_image = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
            else:
                display_image = image_array
            
            st.image(display_image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.header("🎯 Prediction Results")
        
        if uploaded_file is not None:
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                cv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
                cv_image = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
            else:
                cv_image = image_array
            
            with st.spinner("Processing image..."):
                features, resized_image = preprocess_image_for_prediction(cv_image, invert=invert_colors)
            
            st.success("✅ Image processed!" + (" (inverted)" if invert_colors else ""))
            
            preview = create_processed_preview(resized_image)
            st.image(preview, caption="Processed 28x28 Grayscale Image", use_container_width=True)
            
            with st.spinner("Making prediction..."):
                if "SVM" in selected_model and svm_model is not None:
                    prediction, confidence, probabilities = predict_with_svm(svm_model, features)
                    model_type = "SVM (Classical ML)"
                else:
                    prediction, confidence, probabilities = predict_with_mlp(mlp_model, features, label_encoder)
                    model_type = "MLP (Neural Network)"
            
            st.markdown("---")
            
            pred_col1, pred_col2 = st.columns(2)
            
            with pred_col1:
                st.metric(
                    label="Predicted Letter",
                    value=prediction,
                    help="The predicted ASL letter"
                )
            
            with pred_col2:
                st.metric(
                    label="Confidence",
                    value=f"{confidence:.1f}%",
                    help="Model confidence in the prediction"
                )
            
            if confidence >= 80:
                st.success(f"High confidence prediction using {model_type}")
            elif confidence >= 50:
                st.warning(f"Medium confidence prediction using {model_type}")
            else:
                st.error(f"Low confidence prediction using {model_type}")
            
            with st.expander("📊 View All Class Probabilities"):
                if "SVM" in selected_model and svm_model is not None:
                    # Convert numeric class labels to letters
                    classes = [LABEL_TO_LETTER.get(c, str(c)) for c in svm_model.classes_]
                else:
                    classes = ASL_LETTERS[:len(probabilities)]
                
                prob_data = {
                    'Letter': classes,
                    'Probability (%)': [p * 100 for p in probabilities]
                }
                
                import pandas as pd
                prob_df = pd.DataFrame(prob_data)
                prob_df = prob_df.sort_values('Probability (%)', ascending=False)
                
                st.dataframe(
                    prob_df.head(10),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("👆 Upload an image to see predictions")
    
    st.markdown("---")
    st.markdown("""
    ### ℹ️ About This App
    
    This application is part of an ASL Sign Language Detection project that compares:
    - **Classical ML (SVM)**: Support Vector Machine with RBF kernel
    - **Neural Network (MLP)**: Multi-Layer Perceptron built with PyTorch
    
    The app processes uploaded images by:
    1. Converting to grayscale
    2. Resizing to 28x28 pixels
    3. Flattening to 784 features for classification
    
    **Note**: Letters J and Z are excluded as they require motion to sign.
    """)


if __name__ == "__main__":
    main()
