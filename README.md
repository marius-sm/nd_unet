# 123d-unet
Flexible 1D, 2D, and 3D U-Nets in PyTorch

```python
from nd_unet import UNet2d # or UNet1d or UNet3d

unet = UNet2d(
    in_channels=3,            # Mandatory. Number of input channels
    out_channels=5,           # Mandatory. Number of output channels (or classes)
    num_stages=4,             # Optional, default is 4. Number of stages
    initial_num_channels=32,  # Optional, default is 32. Number of channels if the first stage, doubled in each subsequent stage
    norm=None,                # Optional, default is None. Type of normalization. Can be None (no normalization), 'bn' (batch norm), 'gnx' (group norm where x is optionally the number of groups), 'in' (instance norm), 'ln' (layer norm)
    non_lin='relu',           # Optional, default is 'relu'. Type of activation function. Can be None, 'relu', 'leaky_relu', 'gelu', 'elu'
    kernel_size=3             # Optional, default is 3. Kernel size for the convolutions
)
```
