# neural_style_transfer.py

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 512 if torch.cuda.is_available() else 256
STYLE_WEIGHT = 1e6
CONTENT_WEIGHT = 1e0
EPOCHS = 500


# Load and Preprocess Images
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image.to(DEVICE)

def im_convert(tensor):
    image = tensor.cpu().clone().detach()
    image = image.squeeze(0)
    image = transforms.Normalize((-2.118, -2.036, -2.06), (4.367, 4.464, 4.444))(image)
    image = torch.clamp(image, 0, 1)
    return image


# Define Style and Content Loss

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(b * c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram.div(b * c * h * w)

class StyleLoss(nn.Module):
    def __init__(self, target):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target).detach()

    def forward(self, input):
        G = gram_matrix(input)
        loss = nn.functional.mse_loss(G, self.target)
        return loss

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        loss = nn.functional.mse_loss(input, self.target)
        return loss


# Build VGG Model for Style Transfer

def get_vgg_model():
    vgg = models.vgg19(pretrained=True).features.to(DEVICE).eval()
    
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    model = nn.Sequential()
    content_losses = []
    style_losses = []
    i = 0

    for layer in vgg.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            continue

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f'content_loss_{i}', content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target = model(style_img).detach()
            style_loss = StyleLoss(target)
            model.add_module(f'style_loss_{i}', style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:i + 1]
    
    return model, style_losses, content_losses


# Train Model

def train(model, input_img, style_losses, content_losses):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    
    epoch = [0]
    print("[INFO] Starting training...")

    def closure():
        input_img.data.clamp_(0, 1)

        optimizer.zero_grad()
        model(input_img)
        style_score = 0
        content_score = 0

        for sl in style_losses:
            style_score += sl()

        for cl in content_losses:
            content_score += cl()

        loss = STYLE_WEIGHT * style_score + CONTENT_WEIGHT * content_score
        loss.backward()

        epoch[0] += 1
        if epoch[0] % 50 == 0:
            print(f"Epoch: {epoch[0]}, Style Loss: {style_score.item():.4f}, Content Loss: {content_score.item():.4f}")

        return loss

    optimizer.step(closure)
    input_img.data.clamp_(0, 1)

    return input_img


# Save Result
def save_result(output, output_path):
    image = im_convert(output)
    plt.imshow(image.permute(1, 2, 0))
    plt.axis('off')
    plt.savefig(output_path)
    print(f"[INFO] Result saved to {output_path}")


# Main Function

def main():
    parser = argparse.ArgumentParser(description="Neural Style Transfer with VGG-19")
    parser.add_argument('--content', type=str, required=True, help="Path to content image")
    parser.add_argument('--style', type=str, required=True, help="Path to style image")
    parser.add_argument('--output', type=str, default='./output.png', help="Path to save the output image")

    args = parser.parse_args()

    global content_img, style_img
    content_img = load_image(args.content)
    style_img = load_image(args.style)
    input_img = content_img.clone()

    model, style_losses, content_losses = get_vgg_model()

    # Train the model
    output = train(model, input_img, style_losses, content_losses)

    # Save result
    save_result(output, args.output)

if __name__ == "__main__":
    main()
