import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import transforms, models
from PIL import Image
import cv2

# Definir las transformaciones de imagen
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalización
])

# Crear el modelo y cargar los pesos preentrenados desde el archivo
model = models.resnet101()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(num_ftrs, 2)
)
model.load_state_dict(torch.load('./model/derma_resnet101_2.pth'))  # Asegúrate de la ruta correcta
model.eval()
model = model.cuda()

# Definir la función Grad-CAM
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate_cam(self, input_image, class_idx=None):
        self.model.eval()
        output = self.model(input_image)

        if class_idx is None:
            class_idx = torch.argmax(output)

        self.model.zero_grad()
        output[0, class_idx].backward(retain_graph=True)

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_image.shape[2], input_image.shape[3]))
        cam -= np.min(cam)
        cam /= np.max(cam)
        return cam

# Cargar la imagen de la mancha de piel
uploaded_image_path = './ISIC-imagesp/ISIC_0015351.jpg'  # Reemplaza con la ruta a tu imagen subida
uploaded_image = Image.open(uploaded_image_path).convert('RGB')
image_for_plot = np.array(uploaded_image)  # Guardar la imagen original para la visualización
image_for_plot = cv2.resize(image_for_plot, (224, 224))  # Redimensionar la imagen original

# Transformar y pasar la imagen a través del modelo
uploaded_image = transform(uploaded_image).unsqueeze(0).cuda()

# Generar Grad-CAM
grad_cam = GradCAM(model, model.layer4[2].conv3)
cam = grad_cam.generate_cam(uploaded_image)

# Superponer el mapa de calor en la imagen original
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heatmap = np.float32(heatmap) / 255
overlay = heatmap + np.float32(image_for_plot) / 255
overlay = overlay / np.max(overlay)

# Visualizar el resultado con la barra de color
plt.figure(figsize=(10, 10))
plt.imshow(overlay)
plt.title('Grad-CAM on Skin Lesion')
plt.axis('off')
plt.colorbar(plt.imshow(cam, cmap='jet', alpha=0))  # Agregar la barra de color
plt.show()
