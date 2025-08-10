import os
from typing import cast

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from matplotlib import patches
from torch.utils.data import DataLoader
from torchvision import transforms

from data.annotations import BackgroundType, ShapeOutline, ShapeType
from data.synthetic_shapes_dataset import ShapeDataset
from models.multiple_center_predictor import CenterPredictor, ModelType
from utils.losses import CenterPredictionLoss

# Define transformation to convert PIL images to tensors.
transform = transforms.ToTensor()

dataset = ShapeDataset(
    num_images=1000,
    image_size=(256, 256),
    num_shapes_range=(0, 5),
    shape_types=tuple(ShapeType),
    shape_size_range=(20, 90),
    background=BackgroundType.SOLID,
    shape_outline=ShapeOutline.FILL,
    add_noise=False,
    fixed_dataset=False,
    transform=transform,
    rotate_shapes=False,
    max_overlap=0.6,
)

dataloader = DataLoader(dataset, batch_size=100, shuffle=True, collate_fn=ShapeDataset.collate_function)

# Set up device, model, optimizer, and loss function.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model_type = ModelType.center_localization_and_class_id
num_classes = len(dataset.get_classes())
save_model_path = "center_predictor.pth"
model = CenterPredictor(num_classes=num_classes,
                        model_type=model_type).to(device)
if os.path.exists(save_model_path):
    print("loading model: ", save_model_path)
    checkpoint = torch.load(save_model_path)
    model_state = model.state_dict()
    filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_state and model_state[k].shape == v.shape}
    model.load_state_dict(filtered_checkpoint, strict=False)

optimizer = optim.Adam(model.parameters(), lr=1e-6)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=15)
mse_loss = torch.nn.MSELoss()
cross_entropy_loss = torch.nn.CrossEntropyLoss(ignore_index=num_classes)
num_epochs = 100
center_prediction_loss = CenterPredictionLoss(model_type=model_type)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, annotations in dataloader:
        # images: tensor of shape [batch_size, 3, 256, 256]
        images = images.to(device)

        object_classes_per_image = []
        centers_per_image = []
        for ann in annotations:
            centers = []  # Centers for this specific image
            object_classes = []
            for obj in ann:  # Loop through all objects in the annotation
                centers.append(obj['center'])  # Collect all centers for this image
                object_classes.append(obj["shape"])

            centers_tensor = torch.tensor(centers, dtype=torch.float32).to(device)
            centers_per_image.append(centers_tensor)

            object_classes_tensor = torch.tensor(object_classes, dtype=torch.uint8).to(device)
            # **Padding: Ensure each tensor is max_objects long**
            num_objects = len(object_classes)
            max_objects = 10
            if num_objects < max_objects:
                # Pad object classes with padding_value
                pad_classes = torch.full((max_objects - num_objects,), num_classes, dtype=torch.long, device=device)
                object_classes_tensor = torch.cat((object_classes_tensor, pad_classes), dim=0)

            object_classes_per_image.append(object_classes_tensor)
        object_classes_tensor = torch.stack(object_classes_per_image)

        optimizer.zero_grad()
        model_outputs = model(images)
        center_prediction_outputs = model_outputs[:, :, :3]
        loss = center_prediction_loss(model_outputs, centers_per_image, object_classes_per_image)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataset)
    prev_lr = optimizer.param_groups[0]["lr"]
    scheduler.step(epoch_loss)
    new_lr = optimizer.param_groups[0]["lr"]
    if new_lr != prev_lr:
        print(f"Learning rate reduced from {prev_lr:.8f} to {new_lr:.8f} at epoch {epoch + 1}")
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

print("Training complete.")
torch.save(model.state_dict(), save_model_path)  # Save weights
print("saved model to: ", save_model_path)
# Set the model to evaluation mode.
model.eval()

# Visualize predictions for a few samples from the dataset.
num_samples_to_show = 5
with torch.no_grad():
    data_iter = iter(dataset)
    for i in range(200):
        image, annotations = next(data_iter)
    for sample_num in range(num_samples_to_show):
        image, annotations = next(data_iter)

        image = cast("torch.Tensor", image).to(device)
        model_outputs = model(image.unsqueeze(0))
        model_outputs = model.filter_predictions(model_outputs, confidence_threshold=0.5)[0]

        # Convert the image tensor back to a NumPy array.
        image_np = image.detach().cpu().permute(1, 2, 0).numpy()

        # Use Matplotlib to display the image along with its annotations.
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image_np)

        label: str | None = 'Predicted'
        for mout in model_outputs:
            pred = mout.cpu().numpy()
            pred_center = pred[:2]
            conf = pred[2]
            if model_type is ModelType.center_localization_and_class_id:
                pred_shape_class = ShapeType(int(torch.argmax(mout[3:], dim=0).cpu().numpy()))
            pred_center_image_location = dataset.convert_center_to_image_coordinates(pred_center)
            ax.scatter(*pred_center_image_location,
                       marker='x',
                       color='red',
                       s=100,
                       label=label)
            if label:
                label = None
            ax.text(*dataset.convert_center_to_image_coordinates(pred_center),
                    conf)
            if model_type is ModelType.center_localization_and_class_id:
                ax.text(pred_center_image_location[0], pred_center_image_location[1] + 10,
                        pred_shape_class.name,
                        color='red',
                        fontsize=8,
                        backgroundcolor='black')

        label = 'Ground Truth'
        for ann in annotations:
            ax.scatter(*dataset.convert_center_to_image_coordinates(ann.center),
                       marker='o',
                       color='green',
                       s=100,
                       label=label)
            if label:
                label = None
            ax.text(ann.bbox.x_min, ann.bbox.y_min - 5,
                    ann.shape.name,
                    color='yellow',
                    fontsize=8,
                    backgroundcolor='black')

            bbox = ann.bbox
            # Draw the bounding box.
            rect = patches.Rectangle((bbox.x_min, bbox.y_min),
                                     bbox.x_max - bbox.x_min,
                                     bbox.y_max - bbox.y_min,
                                     linewidth=2,
                                     edgecolor='red',
                                     facecolor='none')
            ax.add_patch(rect)

        plt.title(f"Sample {sample_num}")
        plt.legend()
        plt.show()
