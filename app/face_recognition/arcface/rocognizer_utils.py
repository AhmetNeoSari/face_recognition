import cv2
import numpy as np
from skimage import transform as trans
import torch
import torch.nn.functional as F
from torch import nn

# Define a standard set of destination landmarks for ArcFace alignment
arcface_dst = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def estimate_norm(lmk, image_size=112, mode="arcface"):
    """
    Estimate the transformation matrix for aligning facial landmarks.

    Args:
        lmk (numpy.ndarray): 2D array of shape (5, 2) representing facial landmarks.
        image_size (int): Desired output image size.
        mode (str): Alignment mode, currently only "arcface" is supported.

    Returns:
        numpy.ndarray: Transformation matrix (2x3) for aligning facial landmarks.
    """
    # Check input conditions
    assert lmk.shape == (5, 2)
    assert image_size % 112 == 0 or image_size % 128 == 0

    # Adjust ratio and x-coordinate difference based on image size
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio

    # Scale and shift the destination landmarks
    dst = arcface_dst * ratio
    dst[:, 0] += diff_x

    # Estimate the similarity transformation
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]

    return M


def norm_crop(img, landmark, image_size=112, mode="arcface"):
    """
    Normalize and crop a facial image based on provided landmarks.

    Args:
        img (numpy.ndarray): Input facial image.
        landmark (numpy.ndarray): 2D array of shape (5, 2) representing facial landmarks.
        image_size (int): Desired output image size.
        mode (str): Alignment mode, currently only "arcface" is supported.

    Returns:
        numpy.ndarray: Normalized and cropped facial image.
    """
    # Estimate the transformation matrix
    M = estimate_norm(landmark, image_size, mode)

    # Apply the affine transformation to the image
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)

    return warped


def read_features(feature_path):
    try:
        data = np.load(feature_path + ".npz", allow_pickle=True)
        images_name = data["images_name"]
        images_emb = data["images_emb"]

        return images_name, images_emb
    except:
        return None


def compare_encodings(encoding, encodings):
    if encodings.size == 0:
        raise ValueError("No embeddings to compare with.")
    sims = np.dot(encodings, encoding.T)
    pare_index = np.argmax(sims)
    score = sims[pare_index]
    return score, pare_index



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
    ):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(
            inplanes,
            eps=1e-05,
        )
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(
            planes,
            eps=1e-05,
        )
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(
            planes,
            eps=1e-05,
        )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class IResNet(nn.Module):
    fc_scale = 7 * 7

    def __init__(
        self,
        block,
        layers,
        dropout=0,
        num_features=512,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        fp16=False,
    ):
        super(IResNet, self).__init__()
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.bn2 = nn.BatchNorm2d(
            512 * block.expansion,
            eps=1e-05,
        )
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(
                    planes * block.expansion,
                    eps=1e-05,
                ),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        x = self.fc(x.float() if self.fp16 else x)
        x = self.features(x)
        x = F.normalize(x, dim=1)
        return x


def _iresnet(arch, block, layers, pretrained, progress, **kwargs):
    model = IResNet(block, layers, **kwargs)
    if pretrained:
        raise ValueError()
    return model


def iresnet18(pretrained=False, progress=True, **kwargs):
    return _iresnet("iresnet18", IBasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def iresnet34(pretrained=False, progress=True, **kwargs):
    return _iresnet("iresnet34", IBasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def iresnet50(pretrained=False, progress=True, **kwargs):
    return _iresnet("iresnet50", IBasicBlock, [3, 4, 14, 3], pretrained, progress, **kwargs)


def iresnet100(pretrained=False, progress=True, **kwargs):
    return _iresnet("iresnet100", IBasicBlock, [3, 13, 30, 3], pretrained, progress, **kwargs)


def iresnet200(pretrained=False, progress=True, **kwargs):
    return _iresnet("iresnet200", IBasicBlock, [6, 26, 60, 6], pretrained, progress, **kwargs)


def iresnet_inference(model_name, path, device="cuda"):
    if model_name == "r18":
        model = iresnet18()
    elif model_name == "r34":
        model = iresnet34()
    elif model_name == "r50":
        model = iresnet50()
    elif model_name == "r100":
        model = iresnet100()
    else:
        raise ValueError()

    weight = torch.load(path, map_location=device, weights_only=True)

    model.load_state_dict(weight)
    model.to(device)

    return model.eval()
