
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable
from functools import partial
import segmentation_models_pytorch as smp

nonlinearity = partial(F.relu, inplace=True)



class DACblock137(nn.Module):
    def __init__(self, channel):
        super(DACblock137, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=7, padding=7)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),  # 归一化处理
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),  #归一化处理
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class ASPP(nn.Module):
    def __init__(self, in_channel, depth):
        super(ASPP, self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)

        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)

        atrous_block6 = self.atrous_block6(x)

        atrous_block12 = self.atrous_block12(x)

        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net
class N_Unet(nn.Module):
    def __init__(self, in_ch, out_ch, encoder_weights='imagenet'):
        super(N_Unet, self).__init__()

        self.conv1 = DoubleConv(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.pool1_m = nn.MaxPool2d(2)
        # self.conv1_m = nn.Conv2d(in_ch,64,3, padding=1)

        self.conv2 = DoubleConv(40, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConv(32, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.pool2_m = nn.MaxPool2d(2)
        # self.conv2_m = nn.Conv2d(in_ch,128,3, padding=1)


        self.pool3_m = nn.MaxPool2d(2)
        self.conv3_m = nn.Conv2d(in_ch,256,3, padding=1)
        self.conv4 = DoubleConv(48, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(136, 512)
        self.DAC = DACblock137(512)
        self.asspp_cov2 = DoubleConv(512, 512)
        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = DoubleConv(512, 256)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = DoubleConv(256, 128)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = DoubleConv(128, 64)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9 = DoubleConv(64, 32)
        self.conv10 = nn.Conv2d(32, out_ch, 1)
        self.conv11 = nn.Conv2d(256,2,3,padding=1)
        self.conv12 = nn.Conv2d(128,2,3,padding=1)
        self.conv13 = nn.Conv2d(64,2,3,padding=1)
        self.conv14 = nn.Conv2d(32,2,3,padding=1)

        # self.gf = FastGuidedFilter_attention(r=2, eps=1e-2)
        # attention blocks
        self.attentionblock1 = GridAttentionBlock(in_channels=32)
        self.attentionblock2 = GridAttentionBlock(in_channels=64)
        self.attentionblock3 = GridAttentionBlock(in_channels=128)
        self.attentionblock4 = GridAttentionBlock(in_channels=256)
        ###the modified idea
        self.p_csv = nn.MaxPool2d(16)
        self.conv_csv_x = nn.Conv2d(1, 512, kernel_size=1, padding=0, stride=1)
        self.cov_feature = nn.Conv2d(1024, 512, kernel_size=1, padding=0, stride=1)


        self.conv_att_x1 = nn.Conv2d(in_ch, 32, kernel_size=1, padding=0, stride=1)
        self.conv_att_x2 = nn.Conv2d(in_ch, 64, kernel_size=1, padding=0, stride=1)
        self.conv_att_x3 = nn.Conv2d(in_ch, 128, kernel_size=1, padding=0, stride=1)
        self.conv_att_x4 = nn.Conv2d(in_ch, 256, kernel_size=1, padding=0, stride=1)

        self.conv_att_1 = nn.Conv2d(1, 32, kernel_size=1, padding=0, stride=1)
        self.conv_att_2 = nn.Conv2d(1, 64, kernel_size=1, padding=0, stride=1)
        self.conv_att_3 = nn.Conv2d(1, 128, kernel_size=1, padding=0, stride=1)
        self.conv_att_4 = nn.Conv2d(1, 256, kernel_size=1, padding=0, stride=1)
        ## resnet module
        self.residual_layer1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, 1, 1, dilation=2, bias=False),
            nn.BatchNorm2d(32),
        )
        self.residual_layer2 = nn.Sequential(
            nn.Conv2d(in_ch, 64, 1, 1, dilation=2, bias=False),
            nn.BatchNorm2d(64),
        )
        self.residual_layer3 = nn.Sequential(
            nn.Conv2d(in_ch, 128, 1, 1, dilation=2, bias=False),
            nn.BatchNorm2d(128),
        )
        self.residual_layer4 = nn.Sequential(
            nn.Conv2d(in_ch, 256, 1, 1, dilation=2, bias=False),
            nn.BatchNorm2d(256),
        )

        self.aspp1 = ASPP(in_ch, 32)
        self.aspp2 = ASPP(40, 64)
        self.aspp3 = ASPP(32, 128)
        self.aspp4 = ASPP(48, 256)
        self.aspp5 = ASPP(136, 512)
        # self.asspp_cov2 = DoubleConv(1024, 512)
        ## efficient_net_encoders

        # eff = smp.encoders.get_encoder(name='efficientnet-b0', in_channels=3, depth=4, weights=None)
        self.eff = smp.encoders.get_encoder(name='efficientnet-b3', in_channels=3, depth=5, weights=encoder_weights)

    def forward(self, x):


        # res1 = self.residual_layer1(x)
        eff_model = self.eff(x)
        e0 = eff_model[0]  # [2,3,512,512]
        e1 = eff_model[1]  # [2,40,256,256]
        e2 = eff_model[2]  # [2,32,128,128]
        e3 = eff_model[3]  # [2,48,64,64]
        e4 = eff_model[4]  # [2,136,32,32]

        # layer1
        # aspp1 = self.aspp1(x)  # (3,32)
        # c0 = torch.cat([aspp1,e0],dim=1) #(35)
        c1 = self.aspp1(e0)    #(3,32)
        # p1 = self.pool1(c1)  #(32，32）
        att_c1 = self.conv_att_x1(x) #（3，32）
        att_1 = self.attentionblock1(att_c1,c1) #（64）


        # layer2
        p_m_1 = self.pool1_m(x) #(3,3)
        # aspp2 = self.aspp2(p_m_1)   #（3，64）
        # c20 = torch.cat([aspp2,e1],dim=1) #(64,64+40=104)
        c21 = self.aspp2(e1) # (40,64)
        # # attention block
        att_c2 = self.conv_att_x2(p_m_1)
        att_2 = self.attentionblock2(att_c2, c21)#（64+64=128）


        # layer3
        p_m_2 = self.pool2_m(p_m_1) #(3,3)
        # aspp3 = self.aspp3(p_m_2) #(3,128)
        # c30= torch.cat([aspp3,e2],dim=1) #(128,128+32=160)
        # merge_m2 = torch.cat([c_m_2,p2],dim=1) #[4,192,100,100]
        c31 = self.aspp3(e2) #(32,128)
        # p3 = self.pool3(c3) #[4,128,50,50]
        # attention block
        att_c3 = self.conv_att_x3(p_m_2)
        att_3 = self.attentionblock3(att_c3,c31) #(128+128=256)


        # layer4
        p_m_3 = self.pool3_m(p_m_2)
        # aspp4 = self.aspp4(p_m_3) #(3,256)
        # # c_m_3 = self.conv3_m(p_m_3) #[4,256,50,50]
        # # merge_m3 = torch.cat([aspp4,p3],dim=1) #[4,384,50,50]
        # c40 = torch.cat([aspp4,e3],dim=1) #(256+48=304)
        c41 = self.aspp4(e3) #(48,256)
        # Attention
        att_c4 = self.conv_att_x4(p_m_3)
        att_4 = self.attentionblock4(att_c4, c41)#256+256=512
        #print(p4.shape)



        c50 = e4    #( 136)
        c51 = self.aspp5(c50)
        ## dac layer
        DAC = self.DAC(c51)

        # c12 = self.asspp_cov2(DAC)



        up_6 = self.up6(DAC) #[4,256,50,50]
        att_4 = self.conv_att_4(att_4)
        merge6 = torch.cat([up_6, att_4], dim=1) #[4,512,50,50]
        c6 = self.conv6(merge6) #[4,256,50,50]

        up_7 = self.up7(c6) #[4,128,100,100]
        att_3 = self.conv_att_3(att_3)
        merge7 = torch.cat([up_7, att_3], dim=1) #[4,256,100,100]
        c7 = self.conv7(merge7) #[4,128,100,100]

        up_8 = self.up8(c7) #[4,64,200,200]
        att_2 = self.conv_att_2(att_2)
        merge8 = torch.cat([up_8, att_2], dim=1) #[4,128,200,200]
        c8 = self.conv8(merge8) #[4,64,200,200]

        up_9 = self.up9(c8) #[4,32,400,400]
        att_1 = self.conv_att_1(att_1)
        merge9 = torch.cat([up_9, att_1], dim=1) #[4,64,400,400]
        c9 = self.conv9(merge9) #[4,32,400,400]
        c10 = self.conv10(c9)
        out = c10

        return out


nonlinearity = partial(F.relu, inplace=True)

class GridAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(GridAttentionBlock, self).__init__()

        self.inter_channels = in_channels
        self.in_channels = in_channels
        self.gating_channels = in_channels

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1)

        self.phi = nn.Conv2d(in_channels=self.gating_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv2d(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode='bilinear')
        f = F.relu(theta_x + phi_g, inplace=True)

        sigm_psi_f = F.sigmoid(self.psi(f))

        return sigm_psi_f

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class resnet34_unet(nn.Module):
    def __init__(self, num_classes=1, num_channels=3,pretrained=True):
        super(resnet34_unet, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=pretrained)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return nn.Sigmoid()(out)

if __name__ == '__main__':
    from thop import profile

    a = torch.rand((2, 3, 512, 512))
    model = N_Unet(3, 2)
    # model = H_Net_Efficient(2,3)

    print(model(a))
    # 计算FLOPs
    flops, params = profile(model, inputs=(a,))
    print(f"单生成器FLOPs: {flops / 1e9:.2f} GFLOPs")

    print(model(a))
