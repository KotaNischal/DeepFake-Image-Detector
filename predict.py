import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from model import ModifiedMobileNetV2

class_names = ["Real", "Fake"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModifiedMobileNetV2().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        prediction = class_names[predicted.item()]

    plt.imshow(image)
    plt.title(f"Prediction: {prediction}", fontsize=16)
    plt.axis('off')
    plt.show()

    return prediction
