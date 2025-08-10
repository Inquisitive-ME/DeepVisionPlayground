from typing import cast

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import patches
from torch.utils.data import DataLoader
from torchvision import transforms

from data.annotations import BackgroundType, ShapeOutline, ShapeType
from data.synthetic_shapes_dataset import ShapeDataset
from models.encoders import EncodeType
from models.simple_center_net import ModelType, SimpleCenterNet

# Define transformation to convert PIL images to tensors.
transform = transforms.ToTensor()

dataset = ShapeDataset(
    num_images=1000,
    image_size=(256, 256),
    num_shapes_range=(1, 1),  # Exactly one shape per image for this demo.
    shape_types=tuple(ShapeType),
    shape_size_range=(20, 128),
    background=BackgroundType.SOLID,
    shape_outline=ShapeOutline.FILL,
    add_noise=False,
    fixed_dataset=False,
    transform=transform,
    rotate_shapes=True,
)

dataloader = DataLoader(
    dataset,
    batch_size=100,
    shuffle=True,
    collate_fn=ShapeDataset.collate_function
)

# Set up device, model, optimizer, and loss function.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model_type = ModelType.center_localization_and_class_id
model = SimpleCenterNet(len(dataset.get_classes()),
                        model_type=model_type,
                        encoder_type=EncodeType.simple).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.7, patience=15
)
mse_loss = torch.nn.MSELoss()
cross_entropy_loss = torch.nn.CrossEntropyLoss()
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, annotations in dataloader:
        # images: tensor of shape [batch_size, 3, 256, 256]
        images = images.to(device)

        centers = []
        object_classes = []
        for ann in annotations:
            center = ann[0]['center']  # Grab the center from the first shape
            centers.append(center)
            object_classes.append(ann[0]["shape"])
        centers_tensor = torch.tensor(centers, dtype=torch.float32).to(device)
        object_classes_tensor = torch.tensor(
            object_classes, dtype=torch.uint8
        ).to(device)
        optimizer.zero_grad()
        model_outputs = model(images)
        if model_type is ModelType.center_localization:
            loss = mse_loss(model_outputs, centers_tensor)
        elif model_type is ModelType.center_localization_and_class_id:
            center_predictions = model_outputs[:, :2]
            class_predictions = model_outputs[:, 2:]
            class_loss = F.cross_entropy(
                class_predictions, object_classes_tensor
            )
            loss = (mse_loss(center_predictions, centers_tensor) + class_loss)
        else:
            assert False, "Unsupported Model Type"
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataset)
    prev_lr = optimizer.param_groups[0]["lr"]
    scheduler.step(epoch_loss)
    new_lr = optimizer.param_groups[0]["lr"]
    if new_lr != prev_lr:
        print(
            f"Learning rate reduced from {prev_lr:.6f} to {new_lr:.6f} "
            f"at epoch {epoch + 1}"
        )
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
        # Assume there's one shape per image.
        gt_center = ann[0].center

        image = cast(torch.Tensor, image).to(device)
        model_outputs = model(image.unsqueeze(0))
        pred_center = model_outputs.cpu().numpy()[0, :2]
        if model_type is ModelType.center_localization_and_class_id:
            pred_shape_class = ShapeType(int(torch.argmax(model_outputs[0, 2:], dim=0).cpu().numpy()))
        # Convert the image tensor back to a NumPy array.
        image_np = image.detach().cpu().permute(1, 2, 0).numpy()

        # Use Matplotlib to display the image along with its annotations.
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image_np)

        ax.scatter(*dataset.convert_center_to_image_coordinates(pred_center),
                   marker='x',
                   color='red',
                   s=100,
                   label='Predicted')
        ax.scatter(*dataset.convert_center_to_image_coordinates(gt_center),
                   marker='o',
                   color='green',
                   s=100,
                   label='Ground Truth')
        ax.text(ann[0].bbox[0], ann[0].bbox[1] - 5,
                ann[0].shape.name,
                color='yellow',
                fontsize=8,
                backgroundcolor='black')

        if model_type is ModelType.center_localization_and_class_id:
            ax.text(ann[0].bbox[0], ann[0].bbox[1] + 10,
                    pred_shape_class.name,
                    color='red',
                    fontsize=8,
                    backgroundcolor='black')

        bbox = ann[0].bbox
        # Draw the bounding box.
        rect = patches.Rectangle((bbox[0], bbox[1]),
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1],
                                 linewidth=2,
                                 edgecolor='red',
                                 facecolor='none')
        ax.add_patch(rect)
        plt.title(f"Sample {sample_num}")
        plt.legend()
        plt.show()
