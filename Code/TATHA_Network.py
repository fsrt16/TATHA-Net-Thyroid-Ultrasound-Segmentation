import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models

# T-block with dilated and regular convolutions
class TBlock(nn.Module):
    """
    T-Block performs standard and dilated convolutions, followed by batch normalization and ReLU activation.
    The block is designed to capture both local and expanded receptive fields through its convolutional paths.
    """
    def __init__(self, in_channels, out_channels, dilation_rate=2):
        super(TBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilation_rate, padding=dilation_rate)
        self.bn = nn.BatchNorm2d(out_channels * 2)  # Concatenated output has doubled channels
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass for the T-Block. Applies both normal and dilated convolutions, concatenates the results,
        normalizes, and applies ReLU activation.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        
        Returns:
            torch.Tensor: Output tensor after convolution and attention.
        """
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(x)
        concat = torch.cat([conv1_out, conv2_out], dim=1)  # Concatenate along the channel dimension
        out = self.bn(concat)
        out = self.relu(out)
        return out

# Channel Attention Module (CAM)
class ChannelAttention(nn.Module):
    """
    Implements channel-wise attention to help the network focus on the most informative channels.
    The module applies both average and max pooling, followed by dense layers to generate attention weights.
    """
    def __init__(self, in_channels, ratio=8):
        super(ChannelAttention, self).__init__()
        self.shared_dense_1 = nn.Linear(in_channels, in_channels // ratio)
        self.shared_dense_2 = nn.Linear(in_channels // ratio, in_channels)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        """
        Forward pass for channel attention. Computes attention weights based on average and max pooled features.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
        
        Returns:
            torch.Tensor: Output tensor after applying attention.
        """
        avg_pool = self.global_avg_pool(x)
        max_pool = self.global_max_pool(x)
        avg_pool = avg_pool.view(avg_pool.size(0), -1)  # Flatten
        max_pool = max_pool.view(max_pool.size(0), -1)  # Flatten

        avg_out = self.shared_dense_2(F.relu(self.shared_dense_1(avg_pool)))
        max_out = self.shared_dense_2(F.relu(self.shared_dense_1(max_pool)))
        
        attention = torch.sigmoid(avg_out + max_out).view(-1, x.size(1), 1, 1)
        out = x * attention
        return out

# Decoder Block with Skip Connections
class DecoderBlock(nn.Module):
    """
    Decoder block for up-sampling the feature maps and incorporating skip connections from the encoder.
    The block uses transposed convolutions followed by T-blocks for feature refinement.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.concat_conv = TBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        """
        Forward pass for the Decoder block. Upsamples the input, concatenates skip connections, and applies T-Block.
        
        Args:
            x (torch.Tensor): Input tensor from the bottleneck (batch_size, in_channels, height, width)
            skip (torch.Tensor): Skip connection tensor from the encoder (batch_size, skip_channels, height, width)
        
        Returns:
            torch.Tensor: Output tensor after upsampling and applying the T-block.
        """
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)  # Concatenate along the channel dimension
        x = self.concat_conv(x)
        return x

# T-Net Model with Backbone Support
class TNet(nn.Module):
    """
    T-Net is a flexible architecture with an encoder-decoder structure. It includes optional attention mechanisms,
    and support for different backbones (VGG, ResNet, etc.) for feature extraction.
    """
    def __init__(self, input_channels, num_classes, backbone=None, use_attention=True, ablation_options=None):
        """
        Initializes the T-Net model with various customizable options such as backbone selection, attention,
        and ablation configurations.
        
        Args:
            input_channels (int): Number of input channels (e.g., 1 for grayscale, 3 for RGB).
            num_classes (int): Number of output classes.
            backbone (str, optional): Backbone architecture to use ('vgg', 'resnet', etc.). Defaults to None.
            use_attention (bool, optional): Whether to use channel attention. Defaults to True.
            ablation_options (dict, optional): Options to control which components are included (e.g., attention, decoder).
        """
        super(TNet, self).__init__()

        self.backbone = backbone
        self.use_attention = use_attention
        self.ablation_options = ablation_options if ablation_options else {}

        # Backbone selection
        if self.backbone == 'vgg':
            self.vgg = nn.Sequential(*list(torchvision.models.vgg19(pretrained=True).features.children())[:16])
        elif self.backbone == 'resnet':
            self.resnet = models.resnet34(pretrained=True)
            self.resnet.fc = nn.Identity()  # Remove the final fully connected layer

        # Encoder
        self.encoder_block1 = TBlock(input_channels, 64)
        self.encoder_block2 = TBlock(64, 128)
        self.encoder_block3 = TBlock(128, 256)
        self.encoder_block4 = TBlock(256, 512)

        # Bottleneck
        self.bottleneck = TBlock(512, 1024)

        # Attention (optional)
        if self.use_attention:
            self.attention = ChannelAttention(1024)

        # Decoder
        self.decoder_block1 = DecoderBlock(1024, 512, 512)
        self.decoder_block2 = DecoderBlock(512, 256, 256)
        self.decoder_block3 = DecoderBlock(256, 128, 128)
        self.decoder_block4 = DecoderBlock(128, 64, 64)

        # Output Layer
        self.output = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass for the T-Net model. Encodes the input, applies bottleneck processing, decodes with skip connections,
        and outputs the final segmentation result.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width)
        
        Returns:
            torch.Tensor: Output segmentation map of shape (batch_size, num_classes, height, width)
        """
        # Backbone feature extraction (if any)
        if self.backbone:
            if self.backbone == 'vgg':
                x = self.vgg(x)
            elif self.backbone == 'resnet':
                x = self.resnet(x)

        # Encoder path
        e1 = self.encoder_block1(x)
        e2 = self.encoder_block2(F.max_pool2d(e1, 2))
        e3 = self.encoder_block3(F.max_pool2d(e2, 2))
        e4 = self.encoder_block4(F.max_pool2d(e3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(e4, 2))

        # Attention (optional)
        if self.use_attention:
            bottleneck = self.attention(bottleneck)

        # Decoder path
        d1 = self.decoder_block1(bottleneck, e4)
        d2 = self.decoder_block2(d1, e3)
        d3 = self.decoder_block3(d2, e2)
        d4 = self.decoder_block4(d3, e1)

        # Output layer
        out = self.output(d4)
        return out

# Example usage
if __name__ == "__main__":
    input_channels = 1  # Grayscale images
    num_classes = 1  # Binary segmentation (e.g., foreground/background)
    
    # Create an ablation configuration dict
    ablation_config = {'use_attention': True, 'decoder_depth': 4}  # Example of controlling components

    # Instantiate the model
    model = TNet(input_channels, num_classes, backbone='vgg', use_attention=True, ablation_options=ablation_config) 

    # Print model summary
    print(model)

    # Example input tensor (batch_size, channels, height, width)
    input_tensor = torch.randn(1, input_channels, 256, 256)

    # Forward pass
    output = model(input_tensor)
    print("Output shape:", output.shape)
