import torch
import torch.nn as nn
import torch.nn.functional as F

class CBRBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        super(CBRBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(dilation * (kernel_size - 1)) // 2, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.pool(x).squeeze(2)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.unsqueeze(2)
        x = x * out
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, layer_n, kernel, dilation, use_se=True):
        super(ResBlock, self).__init__()
        self.cbr1 = CBRBlock(in_channels, layer_n, kernel, stride=1, dilation=dilation)
        self.cbr2 = CBRBlock(layer_n, layer_n, kernel, stride=1, dilation=dilation)
        self.use_se = use_se
        if use_se:
            self.se_block = SEBlock(layer_n)

    def forward(self, x):
        x_residual = x
        x = self.cbr1(x)
        x = self.cbr2(x)
        if self.use_se:
            x = self.se_block(x)
        x = x + x_residual
        return x

class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        layer_n = 64
        kernel_size = 7
        depth = 3
        num_classes = 2

        self.avg_pool_1 = nn.AvgPool1d(4)
        self.avg_pool_2 = nn.AvgPool1d(16)

        self.dropout = nn.Dropout(0.3)#0.3

        self.col_index = [cfg.all_features[i] for i in cfg.num_features]
        self.col_minute = [cfg.all_features[i] for i in cfg.col_minute]
        # Encoder
        self.cbr_input = CBRBlock(len(self.col_index), layer_n, kernel_size, stride=1, dilation=1)
        self.resblocks = nn.ModuleList()
        for _ in range(depth):
            self.resblocks.append(ResBlock(layer_n, layer_n, kernel_size, dilation=1))
        self.out_0 = self.resblocks[-1]  # Output from the last residual block in the encoder

        self.cbr1 = CBRBlock(layer_n, layer_n * 2, kernel_size, stride=4, dilation=1)
        self.resblocks1 = nn.ModuleList()
        for _ in range(depth):
            self.resblocks1.append(ResBlock(layer_n * 2, layer_n * 2, kernel_size, dilation=1))
        self.out_1 = self.resblocks1[-1]  # Output from the last residual block in the encoder

        self.cbr2 = CBRBlock(layer_n * 2 + len(self.col_index), layer_n * 3, kernel_size, stride=4, dilation=1)
        self.resblocks2 = nn.ModuleList()
        for _ in range(depth):
            self.resblocks2.append(ResBlock(layer_n * 3, layer_n * 3, kernel_size, dilation=1))
        self.out_2 = self.resblocks2[-1]  # Output from the last residual block in the encoder

        self.cbr3 = CBRBlock(layer_n * 3 + len(self.col_index), layer_n * 4, kernel_size, stride=4, dilation=1)
        self.resblocks3 = nn.ModuleList()
        for _ in range(depth):
            self.resblocks3.append(ResBlock(layer_n * 4, layer_n * 4, kernel_size, dilation=1))

        # Decoder
        self.up_sample1 = nn.Upsample(scale_factor=4, mode='linear', align_corners=False)
        self.up_sample2 = nn.Upsample(scale_factor=4, mode='linear', align_corners=False)
        self.up_sample3 = nn.Upsample(scale_factor=4, mode='linear', align_corners=False)

        self.decoder_cbr2 = CBRBlock(448, layer_n * 3, kernel_size, stride=1, dilation=1)
        self.decoder_cbr1 = CBRBlock(320, layer_n * 2, kernel_size, stride=1, dilation=1)
        self.decoder_cbr0 = CBRBlock(192, layer_n, kernel_size, stride=1, dilation=1)

        self.classifier = nn.Conv1d(layer_n, num_classes, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x,y,z,train=True):
        x = x[:, :, self.col_index]
        x_init = x.permute(0, 2, 1)
        # Encoder
        x = self.cbr_input(x_init)
        for block in self.resblocks:
            x = block(x)
            x = self.dropout(x)
        out_0 = x

        x = self.cbr1(x)
        for block in self.resblocks1:
            x = block(x)
        out_1 = x


        x = torch.cat((x, self.avg_pool_1(x_init)), 1)
        x = self.cbr2(x)
        for block in self.resblocks2:
            x = block(x)
            x = self.dropout(x)
        out_2 = x

        x = torch.cat((x, self.avg_pool_2(x_init)), 1)
        x = self.cbr3(x)
        for block in self.resblocks3:
            x = block(x)
            x = self.dropout(x)

        # Decoder
        x = self.up_sample1(x)
        x = torch.cat((x, out_2), 1)
        x = self.decoder_cbr2(x)
        x = self.dropout(x)

        x = self.up_sample2(x)
        x = torch.cat((x, out_1), 1)
        x = self.decoder_cbr1(x)
        x = self.dropout(x)

        x = self.up_sample3(x)
        x = torch.cat((x, out_0), 1)
        x = self.decoder_cbr0(x)
        x = self.dropout(x)

        # Classifier
        x = self.classifier(x)
        x = x.permute(0, 2, 1)
        return x