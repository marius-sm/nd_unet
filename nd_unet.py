import torch
import torch.nn as nn
import torch.nn.functional as F
        
def get_norm(name, num_channels, dim=None):
    if name == 'bn':
        assert dim is not None, 'Please specify the dim argument (1, 2 or 3D)'
        if dim == 1:
            norm = nn.BatchNorm1d(num_channels)
        if dim == 2:
            norm = nn.BatchNorm2d(num_channels)
        if dim == 3:
            norm = nn.BatchNorm3d(num_channels)
        return norm
    elif 'gn' in name:
        num_groups = name[2:]
        if num_groups == '': num_groups = 8
        num_groups = int(num_groups)
        return nn.GroupNorm(num_groups, num_channels)
    elif name == 'in':
        return nn.GroupNorm(num_channels, num_channels)
    elif name == 'ln':
        return nn.GroupNorm(1, num_channels)
    else:
        raise ValueError(f"Normalization '{name}' not recognized. Possible values are None (no normalization), 'bn' (batch norm), 'gnx' (group norm where x is optionally the number of groups), 'in' (instance norm), 'ln' (layer norm)")

def get_non_lin(name):
    if name == 'relu':
        return nn.ReLU()
    if name == 'leaky_relu':
        return nn.LeakyReLU()
    if name == 'gelu':
        return nn.GELU()
    if name == 'elu':
        return nn.ELU()
    else:
        raise ValueError(f"Activation {name} not recognized. Possible values are 'relu', 'leaky_relu', 'gelu', 'elu'")

def get_conv(dim, *args, **kwargs):
    if dim == 1:
        return nn.Conv1d(*args, **kwargs)
    if dim == 2:
        return nn.Conv2d(*args, **kwargs)
    if dim == 3:
        return nn.Conv3d(*args, **kwargs)

def get_conv_block(dim, in_channels, out_channels, norm, non_lin, kernel_size=3, bias=True, padding='same', padding_mode='zeros'):
    if padding == 'same':
        padding = kernel_size//2
    layers = [get_conv(dim, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=bias, padding_mode=padding_mode)]
    if norm is not None:
        layers.append(get_norm(norm, num_channels=out_channels, dim=dim))
    if non_lin is not None:
        layers.append(get_non_lin(non_lin))  
    return nn.Sequential(*layers)

class UNetEncoder(nn.Module):
    def __init__(self,
        dim,
        in_channels,
        num_stages,
        initial_num_channels,
        norm=None,
        non_lin='relu',
        kernel_size=3,
        pooling='max',
        bias=True,
        padding='same',
        padding_mode='zeros',
        stride_sequence=None,
        skip_connections=True
    ):
        super().__init__()

        assert pooling in ['avg', 'max'], f"pooling can be 'avg' or 'max'"
        self.skip_connections = skip_connections

        if dim == 1:
            if pooling == 'avg':
                pooling_class = nn.AvgPool1d
            else:
                pooling_class = nn.MaxPool1d
        if dim == 2:
            if pooling == 'avg':
                pooling_class = nn.AvgPool2d
            else:
                pooling_class = nn.MaxPool2d
        if dim == 3:
            if pooling == 'avg':
                pooling_class = nn.AvgPool3d
            else:
                pooling_class = nn.MaxPool3d
                
        if stride_sequence is None:
            stride_sequence = [2] * (num_stages - 1) 

        self.module_list = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()
        
        for i in range(num_stages):
            block_1_in_channels = in_channels if i == 0 else (2**i)*initial_num_channels
            block_1_out_channels = (2**i)*initial_num_channels
            block_2_in_channels = block_1_out_channels
            block_2_out_channels = (2**(i+1))*initial_num_channels
            m = nn.Sequential(
                get_conv_block(
                    dim=dim,
                    in_channels=block_1_in_channels,
                    out_channels=block_1_out_channels,
                    kernel_size=kernel_size,
                    norm=norm,
                    non_lin=non_lin,
                    bias=bias,
                    padding=padding,
                    padding_mode=padding_mode
                ),
                get_conv_block(
                    dim=dim,
                    in_channels=block_2_in_channels,
                    out_channels=block_2_out_channels,
                    kernel_size=kernel_size,
                    norm=norm,
                    non_lin=non_lin,
                    bias=bias,
                    padding=padding,
                    padding_mode=padding_mode
                )
            )
            self.module_list.append(m)
            if i < num_stages - 1:
                self.pooling_layers.append(pooling_class(3, stride_sequence[i], padding=1))
            
    def forward(self, x, print_shapes=False):
        
        acts = []
        for i, (m, p) in enumerate(zip(self.module_list[:-1], self.pooling_layers)):
            x = m(x)
            if self.skip_connections:
                acts.append(x)
            else:
                acts.append(x[:, 0:0, ...]) # this tensor will only be used by the decoder to infer the shape to which to resize
            x = p(x)
            if print_shapes:
                print(f'Shape after encoder block {i}: {x.shape}')
        x = self.module_list[-1](x)
        if print_shapes:
            print(f'Shape after encoder block {i+1}: {x.shape}')

        return x, acts

class UNetDecoder(nn.Module):
    def __init__(self,
        dim,
        out_channels,
        num_stages,
        initial_num_channels,
        norm=None,
        non_lin='relu',
        kernel_size=3,
        bias=True,
        padding='same',
        padding_mode='zeros',
        skip_connections=False
    ):
        super().__init__()
        
        self.module_list = nn.ModuleList()
        
        for i in range(num_stages-1):
            if skip_connections:
                block_in_channels = (2**(i+1) + (2**(i+2)))*initial_num_channels
            else:
                block_in_channels = (2**(i+2))*initial_num_channels
            block_out_channels = (2**(i+1))*initial_num_channels
            m = nn.Sequential(
                get_conv_block(
                    dim=dim,
                    in_channels=block_in_channels,
                    out_channels=block_out_channels,
                    kernel_size=kernel_size,
                    norm=norm,
                    non_lin=non_lin,
                    bias=bias,
                    padding=padding,
                    padding_mode=padding_mode
                ),
                get_conv_block(
                    dim=dim,
                    in_channels=block_out_channels,
                    out_channels=block_out_channels,
                    kernel_size=kernel_size,
                    norm=norm,
                    non_lin=non_lin,
                    bias=bias,
                    padding=padding,
                    padding_mode=padding_mode
                )
            )
            self.module_list.append(m)

        self.final_conv = get_conv(dim, 2*initial_num_channels, out_channels, 1, bias=bias, padding=0, padding_mode=padding_mode)
            
    def forward(self, x, acts, print_shapes=False):
        
        interpolation = 'linear'
        if x.dim() == 4:
            interpolation = 'bilinear'
        if x.dim() == 5:
            interpolation = 'trilinear'

        for i, (y, m) in enumerate(zip(reversed(acts), reversed(self.module_list))):
            x = F.interpolate(x, y.shape[2:], mode=interpolation, align_corners=True)
            x = m(torch.cat([y, x], 1))
            if print_shapes:
                print(f'Shape after decoder block {i}: {x.shape}')
            
        x = self.final_conv(x)
            
        return x
        
class UNet(nn.Module):
    def __init__(self,
        dim,
        in_channels,
        out_channels,
        num_stages,
        initial_num_channels,
        norm=None,
        non_lin='relu',
        kernel_size=3,
        pooling='max',
        bias=True,
        padding='same',
        padding_mode='zeros',
        stride_sequence=None,
        skip_connections=True
    ):
        super().__init__()
        
        self.encoder = UNetEncoder(
            dim=dim,
            in_channels=in_channels,
            num_stages=num_stages,
            initial_num_channels=initial_num_channels,
            norm=norm,
            non_lin=non_lin,
            kernel_size=kernel_size,
            pooling=pooling,
            bias=bias,
            padding=padding,
            padding_mode=padding_mode,
            stride_sequence=stride_sequence,
            skip_connections=skip_connections
        )
        self.decoder = UNetDecoder(
            dim=dim,
            out_channels=out_channels,
            num_stages=num_stages,
            initial_num_channels=initial_num_channels,
            norm=norm,
            non_lin=non_lin,
            kernel_size=kernel_size,
            bias=bias,
            padding=padding,
            padding_mode=padding_mode,
            skip_connections=skip_connections
        )
            
    def forward(self, x, print_shapes=False):

        if print_shapes:
            print('Input shape:', x.shape)
        
        x, acts = self.encoder(x, print_shapes=print_shapes)
        x = self.decoder(x, acts, print_shapes=print_shapes)

        if print_shapes:
            print('Output shape:', x.shape)
            
        return x

def UNet1d(*args, **kwargs):
    return UNet(1, *args, **kwargs)

def UNet2d(*args, **kwargs):
    return UNet(2, *args, **kwargs)

def UNet3d(*args, **kwargs):
    return UNet(3, *args, **kwargs)

    
    
if __name__ == '__main__':
    
    unet = UNet3d(
        in_channels=1,
        out_channels=1,
        num_stages=4,
        initial_num_channels=8,
        norm='bn',
        non_lin='relu',
        kernel_size=3,
        stride_sequence=[1, 1, 2],
        skip_connections=False
    )

    print(unet(torch.zeros(5, 1, 32, 64, 16), print_shapes=True).shape)

    unet = UNet2d(
        in_channels=1,
        out_channels=1,
        num_stages=4,
        initial_num_channels=8,
        norm='bn',
        non_lin='relu',
        kernel_size=3
    )

    print(unet(torch.zeros(5, 1, 32, 64)).shape)

    unet = UNet1d(
        in_channels=1,
        out_channels=1,
        num_stages=4,
        initial_num_channels=8,
        norm='bn',
        non_lin='relu',
        kernel_size=3
    )

    print(unet(torch.zeros(5, 1, 32)).shape)
    print(unet)
            
