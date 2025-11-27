import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dehazeformer import BasicLayer, RLN
from models.unet_model import UNet


from models.layers import *
class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res-1)]
        layers.append(ResBlock(out_channel, out_channel, filter=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res-1)]
        layers.append(ResBlock(channel, channel, filter=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
class EBlock1(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock1, self).__init__()

        self.layers = Unet_fs(out_channel, out_channel, num_res)
    def forward(self, x):
        return self.layers(x)


class DBlock1(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock1, self).__init__()

        self.layers = Unet_fs(channel, channel, num_res)
    def forward(self, x):
        return self.layers(x)

class SCM(nn.Module):
    def __init__(self, out_plane,in_channel=3):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True)
        )


    def forward(self, x):
        x = self.main(x)
        return x


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel*2, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))



class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, bilateral_grid, guidemap):
        device = bilateral_grid.get_device()

        N, _, H, W = guidemap.shape
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
        if device >= 0:
            hg = hg.to(device)
            wg = wg.to(device)
        hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H - 1)  # norm to [0,1] NxHxWx1
        wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W - 1)  # norm to [0,1] NxHxWx1
        hg, wg = hg * 2 - 1, wg * 2 - 1
        guidemap = guidemap.permute(0, 2, 3, 1).contiguous()
        guidemap_guide = torch.cat([wg, hg, guidemap], dim=3).unsqueeze(1)  # Nx1xHxWx3
        # print("g",guidemap_guide.shape)
        # print("b",bilateral_grid.shape)
        coeff = F.grid_sample(bilateral_grid, guidemap_guide, align_corners=True)
        return coeff.squeeze(2)


class ApplyCoeffs(nn.Module):
    def __init__(self):
        super(ApplyCoeffs, self).__init__()
        self.degree = 3

    def forward(self, coeff, full_res_input):  # 12 3
        R = torch.sum(full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True) + coeff[:, 3:4, :, :]
        G = torch.sum(full_res_input * coeff[:, 4:7, :, :], dim=1, keepdim=True) + coeff[:, 7:8, :, :]
        B = torch.sum(full_res_input * coeff[:, 8:11, :, :], dim=1, keepdim=True) + coeff[:, 11:12, :, :]
        result = torch.cat([R, G, B], dim=1)

        return result


class Pyramid(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.pool = F.avg_pool2d
        self.upsample = F.interpolate
        self.conv1010 = nn.Conv2d(in_channel, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(in_channel, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(in_channel, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(in_channel, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.p = nn.PReLU()

    def forward(self, x_output):
        shape_out = x_output.data.size()[2:4]
        x101 = self.pool(x_output, 32)  # 2,3,8,8
        # print("x101", x101.shape)
        x102 = self.pool(x_output, 16)
        x103 = self.pool(x_output, 8)
        x104 = self.pool(x_output, 4)
        # upsample
        x1010 = self.upsample(self.p(self.conv1010(x101)), size=shape_out)
        x1020 = self.upsample(self.p(self.conv1020(x102)), size=shape_out)
        x1030 = self.upsample(self.p(self.conv1030(x103)), size=shape_out)
        x1040 = self.upsample(self.p(self.conv1040(x104)), size=shape_out)
        x_output = torch.cat((x1010, x1020, x1030, x1040, x_output), 1)
        return x_output
class LPATRNet(nn.Module):
    def __init__(self, num_res=4):
        super(LPATRNet, self).__init__()
        base_channel = 32
        self.Encoder = nn.ModuleList([
            EBlock1(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock1(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

        self.SCM = SCM(32, in_channel=39)
        self.SCM_2 = SCM(base_channel * 2,in_channel=39)
        self.SCM_4 = SCM(base_channel * 4, in_channel=39)

        self.guide_r = nn.Sequential(
            Pyramid(in_channel=1),
            nn.Conv2d(5, 3, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(3, 3, 1, 1),
            nn.PReLU(),
        )
        self.guide_g = nn.Sequential(
            Pyramid(in_channel=1),
            nn.Conv2d(5, 3, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(3, 3, 1, 1),
            nn.PReLU(),
        )
        self.guide_b = nn.Sequential(
            Pyramid(in_channel=1),
            nn.Conv2d(5, 3, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(3, 3, 1, 1),
            nn.PReLU(),
        )
        self.slice = Slice()
        self.apply_coeffs = ApplyCoeffs()

        self.u_net = UNet(n_channels=3, o_channels=3)
        self.R_down = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.PReLU(),
            nn.Conv2d(32, 32, 1, 1),
            nn.PReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.PReLU(),
            nn.Conv2d(64, 64, 1, 1),
            nn.PReLU(),
        )
        self.G_down = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.PReLU(),
            nn.Conv2d(32, 32, 1, 1),
            nn.PReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.PReLU(),
            nn.Conv2d(64, 64, 1, 1),
            nn.PReLU(),
        )
        self.B_down = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.PReLU(),
            nn.Conv2d(32, 32, 1, 1),
            nn.PReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.PReLU(),
            nn.Conv2d(64, 64, 1, 1),
            nn.PReLU(),
        )
        self.R_coeff = nn.Sequential(
            nn.Conv3d(1, 3, 3, 2, 1),
            nn.PReLU(),
            nn.Conv3d(3, 3, 1, 1, 0),
            nn.PReLU(),
            nn.Conv3d(3, 6, 3, 2, 1),
            nn.PReLU(),
            nn.Conv3d(6, 6, 1, 1, 0),
            nn.PReLU(),
            nn.Conv3d(6, 12, 3, 2, 1),
            nn.PReLU(),
            nn.Conv3d(12, 12, 1, 1, 0),
            nn.PReLU(),
        )
        self.G_coeff = nn.Sequential(
            nn.Conv3d(1, 3, 3, 2, 1),
            nn.PReLU(),
            nn.Conv3d(3, 3, 1, 1, 0),
            nn.PReLU(),
            nn.Conv3d(3, 6, 3, 2, 1),
            nn.PReLU(),
            nn.Conv3d(6, 6, 1, 1, 0),
            nn.PReLU(),
            nn.Conv3d(6, 12, 3, 2, 1),
            nn.PReLU(),
            nn.Conv3d(12, 12, 1, 1, 0),
            nn.PReLU(),
        )
        self.B_coeff = nn.Sequential(
            nn.Conv3d(1, 3, 3, 2, 1),
            nn.PReLU(),
            nn.Conv3d(3, 3, 1, 1, 0),
            nn.PReLU(),
            nn.Conv3d(3, 6, 3, 2, 1),
            nn.PReLU(),
            nn.Conv3d(6, 6, 1, 1, 0),
            nn.PReLU(),
            nn.Conv3d(6, 12, 3, 2, 1),
            nn.PReLU(),
            nn.Conv3d(12, 12, 1, 1, 0),
            nn.PReLU(),
        )
        self.u_net_mini = UNet(n_channels=3, o_channels=3)
        self.p = nn.PReLU()
        self.R_Pyramid = Pyramid(in_channel=9)
        self.G_Pyramid = Pyramid(in_channel=9)
        self.B_Pyramid = Pyramid(in_channel=9)
        self.R_x_u_Pyramid=nn.Sequential(
            Pyramid(1),
            nn.Conv2d(5,3,1,1),
            nn.PReLU()
        )
        self.G_x_u_Pyramid=nn.Sequential(
            Pyramid(1),
            nn.Conv2d(5,3,1,1),
            nn.PReLU()
        )
        self.B_x_u_Pyramid=nn.Sequential(
            Pyramid(1),
            nn.Conv2d(5,3,1,1),
            nn.PReLU()
        )
        self.guidance=nn.ModuleList([
            nn.Conv2d(1,1,1,1),
            nn.Conv2d(1, 1, 1, 1),
            nn.Conv2d(1, 1, 1, 1),
            nn.Conv2d(1, 1, 1, 1),
            nn.Conv2d(1, 1, 1, 1),
            nn.Conv2d(1, 1, 1, 1),
            nn.Conv2d(1, 1, 1, 1),
            nn.Conv2d(1, 1, 1, 1),
            nn.Conv2d(1, 1, 1, 1),
        ])

        self.Trans_layer1 = BasicLayer(network_depth=4, dim=128, depth=4,
                                 num_heads=2, mlp_ratio=2.,
                                 norm_layer=RLN, window_size=8,
                                 attn_ratio=1/4, attn_loc='last', conv_type='DWConv')
        self.Trans_layer2 = BasicLayer(network_depth=4, dim=256, depth=4,
                                 num_heads=4, mlp_ratio=4.,
                                 norm_layer=RLN, window_size=8,
                                 attn_ratio=1/2, attn_loc='last', conv_type='DWConv')
        self.Trans_layer3 = BasicLayer(network_depth=4, dim=64, depth=4,
                                 num_heads=8, mlp_ratio=2.,
                                 norm_layer=RLN, window_size=8,
                                 attn_ratio=3/4, attn_loc='last', conv_type='DWConv')
        self.T_ConvsOut = nn.ModuleList(
            [
                BasicConv(128, 64, kernel_size=1, relu=False, stride=1),
                BasicConv(256, 128, kernel_size=1, relu=False, stride=1),
                BasicConv(64, 32, kernel_size=1, relu=False, stride=1),
            ]
        )
    def forward(self, x):
        x_r = x[:, 0:1, :, :]
        x_g = x[:, 1:2, :, :]
        x_b = x[:, 2:3, :, :]
        # guidance map
        guidanceOutput_r = self.guide_r(x_r)
        guidanceOutput_g = self.guide_g(x_g)
        guidanceOutput_b = self.guide_b(x_b)
        guidanceL_r = self.p(self.guidance[0](guidanceOutput_r[:, 0:1, :, :]))
        guidanceM_r = self.p(self.guidance[1](guidanceOutput_r[:, 1:2, :, :]))
        guidanceH_r = self.p(self.guidance[2](guidanceOutput_r[:, 2:3, :, :]))
        guidanceL_g = self.p(self.guidance[3](guidanceOutput_g[:, 0:1, :, :]))
        guidanceM_g = self.p(self.guidance[4](guidanceOutput_g[:, 1:2, :, :]))
        guidanceH_g = self.p(self.guidance[5](guidanceOutput_g[:, 2:3, :, :]))
        guidanceL_b = self.p(self.guidance[6](guidanceOutput_b[:, 0:1, :, :]))
        guidanceM_b = self.p(self.guidance[7](guidanceOutput_b[:, 1:2, :, :]))
        guidanceH_b = self.p(self.guidance[8](guidanceOutput_b[:, 2:3, :, :]))
        # Model Enhancing through Trilinearly Interpolation
        xx = F.interpolate(x, (256, 256), mode='bicubic', align_corners=True)
        # Content Enhancing
        xx = self.u_net(xx)
        xx_r = self.R_down(xx[:, 0:1, :, :]).unsqueeze(1)
        xx_g = self.G_down(xx[:, 1:2, :, :]).unsqueeze(1)
        xx_b = self.B_down(xx[:, 2:3, :, :]).unsqueeze(1)
        # Piecewise Regression Model Acquisition
        coeff_r = self.R_coeff(xx_r)
        coeff_g = self.G_coeff(xx_g)
        coeff_b = self.B_coeff(xx_b)
        slice_coeffsL_r = self.slice(coeff_r, guidanceL_r)
        slice_coeffsM_r = self.slice(coeff_r, guidanceM_r)
        slice_coeffsH_r = self.slice(coeff_r, guidanceH_r)
        slice_coeffsL_g = self.slice(coeff_g, guidanceL_g)
        slice_coeffsM_g = self.slice(coeff_g, guidanceM_g)
        slice_coeffsH_g = self.slice(coeff_g, guidanceH_g)
        slice_coeffsL_b = self.slice(coeff_b, guidanceL_b)
        slice_coeffsM_b = self.slice(coeff_b, guidanceM_b)
        slice_coeffsH_b = self.slice(coeff_b, guidanceH_b)
        # General Enhancing
        x_u = self.u_net_mini(x)
        x_u_r = self.R_x_u_Pyramid(x_u[:, 0:1, :, :])
        x_u_g = self.G_x_u_Pyramid(x_u[:, 1:2, :, :])
        x_u_b = self.B_x_u_Pyramid(x_u[:, 2:3, :, :])
        # affine transformation
        outputL_r = self.apply_coeffs(slice_coeffsL_r, x_u_r)
        outputM_r = self.apply_coeffs(slice_coeffsM_r, x_u_r)
        outputH_r = self.apply_coeffs(slice_coeffsH_r, x_u_r)
        outputL_g = self.apply_coeffs(slice_coeffsL_g, x_u_g)
        outputM_g = self.apply_coeffs(slice_coeffsM_g, x_u_g)
        outputH_g = self.apply_coeffs(slice_coeffsH_g, x_u_g)
        outputL_b = self.apply_coeffs(slice_coeffsL_b, x_u_b)
        outputM_b = self.apply_coeffs(slice_coeffsM_b, x_u_b)
        outputH_b = self.apply_coeffs(slice_coeffsH_b, x_u_b)
        r = torch.cat((outputL_r, outputM_r, outputH_r), dim=1)
        g = torch.cat((outputL_g, outputM_g, outputH_g), dim=1)
        b = torch.cat((outputL_b, outputM_b, outputH_b), dim=1)
        r_P = self.R_Pyramid(r)
        g_P = self.G_Pyramid(g)
        b_P = self.B_Pyramid(b)
        grid_feature=torch.cat((r_P, g_P, b_P), dim=1)
        grid_feature_2=F.interpolate(grid_feature, scale_factor=0.5)
        grid_feature_4=F.interpolate(grid_feature_2, scale_factor=0.5)
        # Feature Normalization
        grid_2=self.SCM_2(grid_feature_2)
        grid_4 = self.SCM_4(grid_feature_4)

        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        # Feature Normalization
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()
        # 256
        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)
        # 128
        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        zz=torch.cat((z,grid_2),dim=1)
        zz = self.Trans_layer1(zz)
        z=self.T_ConvsOut[0](zz)
        res2 = self.Encoder[1](z)
        # 64
        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        zz = torch.cat((z, grid_4), dim=1)
        zz = self.Trans_layer2(zz)
        z=self.T_ConvsOut[1](zz)
        z = self.Encoder[2](z)
        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        # 128
        z = self.feat_extract[3](z)
        outputs.append(z_+x_4)
        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        # 256
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)
        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        zz=torch.cat((z,self.SCM(grid_feature)),dim=1)
        zz = self.Trans_layer3(zz)
        z = self.T_ConvsOut[2](zz)
        z = self.feat_extract[5](z)
        outputs.append(z+x)
        return outputs


def build_net():
    model = LPATRNet()
    return model


if __name__ == '__main__':
    from ptflops import get_model_complexity_info

    model = build_net().eval()
    with torch.cuda.device(0):
        flops, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True, print_per_layer_stat=True)
        print('Flops:  ' + flops)
        print('Params: ' + params)
    pass
