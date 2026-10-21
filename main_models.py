class ArConv(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.k = kernel_size
        self.weight = nn.Parameter(
            torch.randn(channels, 1, kernel_size)
        )

    def forward(self, x):
        p = self.k // 2

        # horizontal: 1 x k
        wx = self.weight.unsqueeze(2)
        x = F.conv2d(
            x, wx,
            padding=(0, p),
            groups=x.shape[1]
        )

        # vertical: k x 1
        wy = self.weight.unsqueeze(3)
        x = F.conv2d(
            x, wy,
            padding=(p, 0),
            groups=x.shape[1]
        )

        return x

class InvertedResidualArConvRaw(nn.Module):
    def __init__(self,arconvKernel, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidualArConvRaw, self).__init__()
        self.stride = stride
        hidden_dim = in_channels * expand_ratio
        self.use_residual = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # Pointwise convolution (expand)
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        # Depthwise convolution
        if stride!=1:
            layers.extend([
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        else:
            layers.extend([
                ArConv(hidden_dim, arconvKernel),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        # Pointwise convolution (project)
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

class AlModelV2ArConvRaw(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0,arconvKernel=5):
        super(AlModelV2ArConvRaw, self).__init__()
        # Configuration for inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # First layer (Conv2D + BatchNorm + ReLU6)
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.features = [nn.Sequential(
            nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )]

        # Building inverted residual blocks
        for t, c, n, s in self.cfgs:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidualArConvRaw(arconvKernel,input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # Building last layer (Conv2D + BatchNorm + ReLU6)
        self.features.append(nn.Sequential(
            nn.Conv2d(input_channel, self.last_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.last_channel),
            nn.ReLU6(inplace=True)
        ))

        self.features = nn.Sequential(*self.features)

        # Classification layer
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes)
        )

    def forward(self, x):
        h = self.features(x)
        x = h.mean([2, 3])  # Global Average Pooling
        h = torch.amax(h, dim=[2, 3])  # Global Max Pooling
        x = self.classifier(x)
        return x,h

class InputWeightsv5(nn.Module):
    def __init__(self,weightPerInput=16,kernelSize=5,imgC=3,inputSize=224):
        super(InputWeightsv5, self).__init__()
        self.weightPerInput = weightPerInput
        self.kernelSize = kernelSize
        self.imgC=imgC
        self.inputSize=inputSize
        
        # Input shape: (N, C, H, W)
        self.expansion=[]
        self.expansion = [nn.Sequential(
            nn.Conv2d(self.imgC, self.weightPerInput, kernel_size=3, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(self.weightPerInput),
            nn.ReLU(inplace=True)
        )]
        steps = int(torch.floor(torch.log2(torch.tensor(self.inputSize / (self.kernelSize+1)))).item())
        for i in range(2,steps):
            self.expansion.append(nn.Sequential(
            nn.Conv2d(weightPerInput, self.weightPerInput, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.weightPerInput),
            nn.ReLU6(inplace=True),
            # nn.Conv3d(weightPerInput, self.weightPerInput, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm3d(self.weightPerInput),
            # nn.ReLU6(inplace=True)
            ))
        self.expansion.append(nn.Sequential(
            nn.Conv2d(self.weightPerInput, self.imgC, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.imgC),
            nn.ReLU6(inplace=True)
        ))
        self.features = []
        # self.features = [nn.Sequential(
        #     nn.Conv3d(1, self.weightPerInput, kernel_size=(imgC,3,3), stride=(1,2,2), padding=1,bias=False),
        #     nn.BatchNorm3d(self.weightPerInput),
        #     nn.ReLU(inplace=True)
        # )]
        if kernelSize==7:
            self.features.append(nn.Conv3d(1, self.weightPerInput, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1), bias=False))
        if kernelSize==5:
            self.features.append(nn.Conv3d(1, self.weightPerInput, kernel_size=(3,5,5), stride=(1,1,1), padding=(1,1,1), bias=False))
        if kernelSize==3:
            self.features.append(nn.Conv3d(1, self.weightPerInput, kernel_size=(3,4,4), stride=(1,2,2), padding=1, bias=False))
            
        self.expansion = nn.Sequential(*self.expansion)
        self.features = nn.Sequential(*self.features)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax2d()


    def forward(self, x):
        x = self.expansion(x)
        x = x.unsqueeze(1)
        x = self.features(x)
        x = self.sigmoid(x)
        # x = self.tanh(x)
        # x = x.view(x.shape[0], 32, 3, 5, 5)
        return x

class AlModelV2ArConvRawVarWeight(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0,firstKernel=3,arconvKernel=7):
        super(AlModelV2ArConvRawVarWeight, self).__init__()
        # Configuration for inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        
        self.firstPad=firstKernel//2


        # First layer (Conv2D + BatchNorm + ReLU6)
        input_channel = int(16 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        # self.first_layer = nn.Conv2d(3, input_channel, kernel_size=7, stride=2, padding=3, bias=False)
        # for param in self.first_layer.parameters():
        #     param.requires_grad = False
        self.features = [nn.Sequential(
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True)
        )]

        # Building inverted residual blocks
        for t, c, n, s in self.cfgs:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidualArConvRaw(arconvKernel,input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # Building last layer (Conv2D + BatchNorm + ReLU6)
        self.features.append(nn.Sequential(
            nn.Conv2d(input_channel, self.last_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.last_channel),
            nn.ReLU(inplace=True)
        ))

        self.features = nn.Sequential(*self.features)

        # Classification layer
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes,bias=True) # vase tta bayad false beshe
        )
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x,weights):
    # def forward(self, x):
        
        # with torch.no_grad():  # Avoid gradient tracking for weight assignment
        #     self.first_layer.weight.copy_(weights)
        
        # self.first_layer.weight = nn.Parameter(weights)
        # x = self.first_layer(x)
        N=weights.size(0)
        Hw=weights.size(3)
        Hi=x.size(2)
        weight_reshaped = weights.view(N * 16, 3, Hw, Hw)
        input_reshaped = x.view(1, N * 3, Hi, Hi)
        # print(input_reshaped.shape, weights.shape)
        x = F.conv2d(input_reshaped, weight_reshaped, stride=2, padding=self.firstPad,groups=N)
        x = x.view(N,16,112,112)
        # return x
        h = self.features(x)
        x = h.mean([2, 3])  # Global Average Pooling
        h = torch.amax(h, dim=[2, 3])  # Global Max Pooling
        x = self.classifier(x)
        return x,h

class AlMainModelV2ArConvRawVarWeight(nn.Module):
    def __init__(self, num_classes=1):
        super(AlMainModelV2ArConvRawVarWeight, self).__init__()
        self.model = AlModelV2ArConvRawVarWeight(num_classes=num_classes,firstKernel=5)
        self.modelW = InputWeightsv5(weightPerInput=16,kernelSize=5,inputSize=224)
        # self.bestKernel = randomKWeight
        self.last_weights = None

    def forward(self, x):
        weight = self.modelW(x)
        self.last_weights = weight
        # weight = weight.view(32, 3, 5, 5)
        # with torch.no_grad():  # Ensure no gradients for CPU operation
            # weight=self.bestKernel(weight,k=32)
        # weight = weight.clone().detach().requires_grad_(True)
        # weight=self.bestKernel(weight,k=32)
        # weight=torch.squeeze(weight, dim=0)
        return self.model(x,weight)
