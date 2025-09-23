import torch
from torch import nn
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io

MODEL_PATH = 'model.pt'
app = FastAPI(debug=True)
model = torch.load(MODEL_PATH)

def predict_single_image(model, image_path_or_bytes, class_names, device='cpu'):
    """
    Predict a single image
    
    Args:
        model: Trained PyTorch model
        image_path_or_bytes: Either path to image or image bytes
        class_names: List of class names
        device: 'cuda' or 'cpu'
    """
    # Set model to evaluation mode
    model.eval()
    model.to(device)
    
    # Load image
    if isinstance(image_path_or_bytes, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image_path_or_bytes)).convert('RGB')
    else:
        image = Image.open(image_path_or_bytes).convert('RGB')
    
    image_tensor = image.unsqueeze(0)  # Adds batch dimension
    image_tensor = image_tensor.to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_prob, predicted_class = torch.max(probabilities, 1)
    
    return {
        'predicted_class': class_names[predicted_class.item()],
        'confidence': predicted_prob.item(),
    }

@app.post("/predict")
def predict_image(file: UploadFile = File(...)):
    # Validate file type
    print('Started')
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image file
        contents = file.read()
        
        # Predict
        return predict_single_image(model, contents, [x for x in range(10)])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")