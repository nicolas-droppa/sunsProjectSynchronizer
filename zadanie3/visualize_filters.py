import torch
import numpy as np
import matplotlib.pyplot as plt
from model import SimpleCNNRegressor

def visualize_filters(layer, title="Filters"):
    weights = layer.weight.data.cpu().numpy()

    out_channels, in_channels, H, W = weights.shape

    print(f"{title}: shape = {weights.shape}")

    cols = 8
    rows = int(np.ceil(out_channels / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(12, 6))
    fig.suptitle(title)

    axs = axs.flatten()

    for i in range(out_channels):
        f = weights[i]  # shape (in_channels, H, W)

        if in_channels == 3:
            # RGB filter -> transposition to (H, W, 3)
            f_rgb = np.transpose(f, (1, 2, 0))
        else:
            # grey filter (conv1 with grayscale) -> show one channel
            f_rgb = f[0]

        # Normalization
        f_rgb = (f_rgb - f_rgb.min()) / (f_rgb.max() - f_rgb.min() + 1e-5)

        axs[i].imshow(f_rgb)
        axs[i].axis("off")

    for j in range(out_channels, len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_all():
    model = SimpleCNNRegressor()
    model.eval()
    visualize_filters(model.conv1[0], "Conv1 Filters")
    visualize_filters(model.conv2[0], "Conv2 Filters")
    visualize_filters(model.conv3[0], "Conv3 Filters")
