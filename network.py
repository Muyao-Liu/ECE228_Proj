import torch
import torch.nn as nn
import torch.nn.functional as F


# kernel decomposition
class DecomposedConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride=1, padding=0, 
                 use_bias=True, scale_factor=1):
        super(DecomposedConv2d, self).__init__()
        
        self.decomposed_conv = nn.Sequential(
            nn.Conv2d(in_channels=dim_in, out_channels=dim_in * scale_factor, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=dim_in, bias=use_bias),
            nn.InstanceNorm2d(dim_in * scale_factor),
            nn.Conv2d(in_channels=dim_in * scale_factor, out_channels=dim_out,
                      kernel_size=1, stride=1, bias=use_bias),
        )

    def forward(self, x):
        return self.decomposed_conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        
        self.residual_conv = nn.Sequential(
            DecomposedConv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            DecomposedConv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim_out, affine=True)
        )

    def forward(self, x):
        return x + self.residual_conv(x)

class generator(nn.Module):
    def __init__(self, dim_in = 3, dim_out = 3, hidden_dim = 64, residual_num = 8):
        super(generator, self).__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(dim_in, hidden_dim, kernel_size = 7, stride = 1, padding = 3),
            nn.InstanceNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.down_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size = 3, stride = 2, padding = 1),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size = 3, stride = 1, padding = 1),
            nn.InstanceNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size = 3, stride = 2, padding = 1),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 4, kernel_size = 3, stride = 1, padding = 1),
            nn.InstanceNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True)
        )

        layers = []
        # Bottleneck
        for i in range(residual_num):
            layers.append(ResidualBlock(dim_in=hidden_dim * 4, dim_out=hidden_dim * 4))

        self.residual = nn.Sequential(*layers)


        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 3,2,1,1),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size = 3, stride = 1, padding = 1),
            nn.InstanceNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 3,2,1,1),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size = 3, stride = 1, padding = 1),
            nn.InstanceNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, dim_out, kernel_size = 7, stride = 1, padding = 3),
            nn.Tanh()
        )

    def forward(self, x):
        in_down = self.in_conv(x)
        out_down = self.down_conv(in_down)
        out_res = self.residual(out_down)
        out_up = self.up_conv(out_res)
        output = self.out_conv(out_up)
        return output



class discriminator(nn.Module):
    def __init__(self, dim_in = 3, dim_out = 3, hidden_dim = 32):
        super(discriminator, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(dim_in, hidden_dim, kernel_size = 3, stride = 1, padding = 1), 
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size = 3, stride = 2, padding = 1), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size = 3, stride = 1, padding = 1), 
            nn.InstanceNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, kernel_size = 3, stride = 2, padding = 1), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim * 8, hidden_dim * 8, kernel_size = 3, stride = 1, padding = 1), 
            nn.InstanceNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim * 8, hidden_dim * 8, kernel_size = 3, stride = 1, padding = 1), 
            nn.InstanceNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim * 8, hidden_dim * 8, kernel_size = 3, stride = 1, padding = 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.convs(x)
    
        return output

