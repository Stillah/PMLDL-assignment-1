import torch
from torch import nn
from torch.nn import functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
import torchvision.transforms as transforms
import os

# ..\\..\\..\\models\\model.pt if not using docker
MODEL_PATH = "model.pt"
labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

app = FastAPI(debug=True)

# Creating model instance
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, 5),
            nn.MaxPool2d(5, 1),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 16, 3),
            nn.AvgPool2d(3, 2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1024, 64)

        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        return x
    

model = CNN()
ckpt = torch.load(MODEL_PATH)
model.load_state_dict(ckpt)

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
        image = Image.open(io.BytesIO(image_path_or_bytes)).convert('L')
    else:
        image = Image.open(image_path_or_bytes).convert('L')
    
    to_tensor = transforms.ToTensor()
    image_tensor = to_tensor(image).unsqueeze(0)  # Adds batch dimension
    image_tensor = image_tensor.to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_prob, predicted_class = torch.max(probabilities, 1)
    
    return {
        'predicted_class': class_names[predicted_class.item()],
        'confidence': predicted_prob.item(),
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image file
        contents = await file.read()
        
        # Predict
        return predict_single_image(model, contents, labels)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")