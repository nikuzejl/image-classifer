from flask import Flask, request, render_template
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

# Load your trained model
model = torch.load("model.pth", map_location=torch.device("cpu"))
model.eval()

# Transform (use same as training)
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Change based on model input
    transforms.ToTensor(),
])

# Class labels (example for CIFAR-10)
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            img = Image.open(image_file).convert("RGB")
            img = transform(img).unsqueeze(0)
            with torch.no_grad():
                output = model(img)
                probs = F.softmax(output, dim=1)
                pred = torch.argmax(probs, 1)
            return f"Predicted class: {classes[pred.item()]}"
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
