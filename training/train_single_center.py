from typing import cast

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from data.annotations import BackgroundType, ShapeOutline, ShapeType
from data.synthetic_shapes_dataset import ShapeDataset
from models.simple_center_net import SimpleCenterNet

# Define transformation to convert PIL images to tensors.
transform = transforms.ToTensor()

dataset = ShapeDataset(
    num_images=1000,
    image_size=(256, 256),
    num_shapes_range=(1, 1),  # Exactly one shape per image for this demo.
    shape_types=tuple(ShapeType),
    shape_size_range=(20, 128),
    background=BackgroundType.RANDOM,
    shape_outline=ShapeOutline.RANDOM,
    add_noise=False,
    fixed_dataset=False,
    transform=transform
)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=ShapeDataset.collate_function)

# Set up device, model, optimizer, and loss function.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = SimpleCenterNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5)
criterion = torch.nn.MSELoss()  # Regression loss for predicting center coordinates

num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, annotations in dataloader:
        # images: tensor of shape [batch_size, 3, 256, 256]
        images = images.to(device)

        centers = []
        for ann in annotations:
            center = ann[0]['center']  # Grab the center from the first shape
            centers.append(center)
        centers_tensor = torch.tensor(centers, dtype=torch.float32).to(device)
        optimizer.zero_grad()
        outputs = model(images)  # Model outputs predicted center coordinates
        loss = criterion(outputs, centers_tensor)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataset)
    prev_lr = optimizer.param_groups[0]["lr"]
    scheduler.step(epoch_loss)
    new_lr = optimizer.param_groups[0]["lr"]
    if new_lr != prev_lr:
        print(f"Learning rate reduced from {prev_lr:.6f} to {new_lr:.6f} at epoch {epoch + 1}")
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

print("Training complete.")

# Set the model to evaluation mode.
model.eval()

# Visualize predictions for a few samples from the dataset.
num_samples_to_show = 5
with torch.no_grad():
    data_iter = iter(dataset)
    for sample_num in range(num_samples_to_show):
        image, ann = next(data_iter)
        # Convert annotation to dict and extract the center.
        if isinstance(ann, list):
            # Assume there's one shape per image.
            gt_center = ann[0].center
        else:
            gt_center = ann.center

        # Transform the image if necessary.
        # image_tensor = transform(image).unsqueeze(0).to(device)
        image = cast(torch.Tensor, image).to(device)
        pred_center = model(image.unsqueeze(0)).cpu().numpy()[0]

        # Convert the image tensor back to a NumPy array.
        image_np = image.detach().cpu().permute(1, 2, 0).numpy()

        plt.imshow(image_np)
        plt.scatter(*dataset.convert_center_to_image_coordinates(pred_center),
                    marker='x',
                    color='red',
                    s=100,
                    label='Predicted')
        plt.scatter(*dataset.convert_center_to_image_coordinates(gt_center),
                    marker='o',
                    color='green',
                    s=100,
                    label='Ground Truth')
        plt.title(f"Sample {sample_num}")
        plt.legend()
        plt.show()