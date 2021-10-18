import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

##############################################################################
# Classes
##############################################################################

class VI_Pose_Encoder(nn.Module):

    def __init__(self, input_nc=3, ngf=64, pose_num_capsules=50, pose_dim=3):
        super(VI_Pose_Encoder, self).__init__()

        self.ngf = ngf
        self.pose_num_capsules= pose_num_capsules
        self.pose_dim = pose_dim
        self.dconv_down1 = self.double_conv_dwn(input_nc, ngf)
        self.dconv_down2 = self.double_conv_dwn(ngf, ngf * 2)
        self.dconv_down3 = self.double_conv_dwn(ngf * 2, ngf * 4)
        self.dconv_down4 = self.double_conv_last_dwn(ngf * 4, ngf * 8)
        self.maxpool = nn.MaxPool2d(2)

        self.lp1 = nn.Linear((ngf * 8) * 16 * 16, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.lp2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.lp3 = nn.Linear(512, self.pose_dim * self.pose_num_capsules)
        self.bn3 = nn.BatchNorm1d(self.pose_dim * self.pose_num_capsules)

        self.relu = nn.ReLU(inplace=True)
        self.silu = nn.SiLU(inplace=True)
        self.drp =nn.Dropout(p=0.3)

    def double_conv_dwn(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def double_conv_last_dwn(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):

        x = self.dconv_down1(input)
        x = self.maxpool(x)

        x = self.dconv_down2(x)
        x = self.maxpool(x)

        x = self.dconv_down3(x)
        x = self.maxpool(x)

        x = self.dconv_down4(x).view(-1, (self.ngf * 8) * 16 * 16)

        l = self.relu(self.silu(self.lp1(x)))
        l = self.relu(self.silu(self.lp2(l)))
        l = (self.lp3(l))

        return l

class View_Encoder(nn.Module):

    def __init__(self, input_nc=3, ngf=64, num_parameters=6):
        super(View_Encoder, self).__init__()

        self.ngf = ngf
        self.dconv_view_down1 = self.double_conv_view_dwn(input_nc, ngf * 2)
        self.dconv_view_down2 = self.double_conv_view_dwn(ngf * 2, ngf * 2)
        self.view_maxpool = nn.MaxPool2d(7)
        self.lp1 = nn.Linear((ngf * 2) * 3 * 3, 256)
        self.lp2 = nn.Linear(256, num_parameters)

        self.relu = nn.ReLU(inplace=True)
        self.silu = nn.SiLU(inplace=True)
        self.drp =nn.Dropout(p=0.3)


    def double_conv_view_dwn(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 5, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):

        x = self.dconv_view_down1(input)

        x = self.dconv_view_down2(x)
        x = self.view_maxpool(x).view(-1, (self.ngf * 2) * 3 * 3)
        l = self.drp(self.relu(self.lp1(x)))
        l = (self.lp2(l))
        return l[:, 0:3], l[:, 3:6]

class Decoder(nn.Module):

    def __init__(self, input_nc, output_nc, ngf):
        super(Decoder, self).__init__()

        self.ngf =ngf
        self.dconv_up4 = self.double_conv_dwn(ngf * 8, ngf * 4)
        self.dconv_up3 = self.double_conv_up(ngf * 4, ngf * 2)
        self.dconv_up2 = self.double_conv_up(ngf * 2, ngf)
        self.dconv_up1 = self.double_conv_up(ngf, output_nc)

        self.maxpool = nn.MaxPool2d(2)
        self.lp1 = nn.Linear(input_nc, (ngf * 8) * 16 * 16)

        self.relu = nn.ReLU(inplace=True)
        self.drp =nn.Dropout(p=0.3)
        self.silu = nn.SiLU(inplace=True)
        self.tanh = nn.Tanh()

    def double_conv_dwn(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def double_conv_up(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding= 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, vs_pose):

        x = self.drp(self.relu(self.lp1(vs_pose))).view(-1, (self.ngf * 8), 16, 16)
        x = self.dconv_up4(x)
        x = self.dconv_up3(x)
        x = self.dconv_up2(x)
        x = self.dconv_up1(x)
        x = self.tanh(x)

        return x

class VI_Dense_Encoder_Decoder(nn.Module):

    def __init__(self, input_nc=3, output_nc=3, ngf=64, pose_num=50, pose_dim=3, view_num_parameters=6):
        super(VI_Dense_Encoder_Decoder, self).__init__()

        self.pose_num = pose_num
        self.pose_dim = pose_dim

        self.pose_encoder = VI_Pose_Encoder(input_nc, ngf, self.pose_num, self.pose_dim)
        self.view_encoder = View_Encoder(input_nc, ngf, view_num_parameters)
        self.decoder = Decoder(self.pose_num * self.pose_dim, output_nc, ngf)

    def view_transform(self, pose, rot, trans):

        cos_r, sin_r = rot.cos(), rot.sin()
        zeros = Variable(rot.data.new(rot.size()[:1] + (1,)).zero_())
        ones = Variable(rot.data.new(rot.size()[:1] + (1,)).fill_(1))

        r1 = torch.stack((ones, zeros, zeros), dim=-1)
        rx2 = torch.stack((zeros, cos_r[:, 0:1], -sin_r[:, 0:1]), dim=-1)
        rx3 = torch.stack((zeros, sin_r[:, 0:1], cos_r[:, 0:1]), dim=-1)
        rx = torch.cat((r1, rx2, rx3), dim=1)

        ry1 = torch.stack((cos_r[:, 2:3], zeros, sin_r[:, 2:3]), dim=-1)
        r2 = torch.stack((zeros, ones, zeros), dim=-1)
        ry3 = torch.stack((-sin_r[:, 2:3], zeros, cos_r[:, 2:3]), dim=-1)
        ry = torch.cat((ry1, r2, ry3), dim=1)

        rz1 = torch.stack((cos_r[:, 1:2], -sin_r[:, 1:2], zeros), dim=-1)
        r3 = torch.stack((zeros, zeros, ones), dim=-1)
        rz2 = torch.stack((sin_r[:, 1:2], cos_r[:, 1:2], zeros), dim=-1)
        rz = torch.cat((rz1, rz2, r3), dim=1)

        rt1 = torch.stack((ones, zeros, zeros, trans[:, 0:1]), dim=-1)
        rt2 = torch.stack((zeros, ones, zeros, trans[:, 1:2]), dim=-1)
        rt3 = torch.stack((zeros, zeros, ones, trans[:, 2:3]), dim=-1)
        trans = torch.cat((rt1, rt2, rt3), dim=1)

        final_rot = torch.matmul(torch.matmul(rz, ry), rx)
        rot_pose = torch.matmul(final_rot, torch.transpose(pose, 2, 1))

        rot_pose2 = Variable(rot.data.new(pose.size(0), self.pose_dim + 1, self.pose_num_capsules).fill_(1))
        rot_pose2[:, 0:3, :] = rot_pose[:, 0:3, :]

        view_specific_pose = torch.matmul(trans, rot_pose2)
        view_specific_pose = view_specific_pose.view(-1, view_specific_pose.size(2), view_specific_pose.size(1))

        return view_specific_pose

    def forward(self, a, b, a_aug, b_aug, c, a2, b2):

        a_pose = self.pose_encoder(a).view(-1, self.pose_num_capsules, self.pose_dim)
        b_pose = self.pose_encoder(b).view(-1, self.pose_num_capsules, self.pose_dim)

        # a2_pose = self.pose_encoder(a2).view(-1, self.pose_num_capsules, self.pose_dim)
        # b2_pose = self.pose_encoder(b2).view(-1, self.pose_num_capsules, self.pose_dim)
        a2_v_p_rot, a2_v_p_trans = self.view_encoder(a2)
        b2_v_p_rot, b2_v_p_trans = self.view_encoder(b2)

        a_pose_aug = self.pose_encoder(a_aug).view(-1, self.pose_num_capsules, self.pose_dim)
        b_pose_aug = self.pose_encoder(b_aug).view(-1, self.pose_num_capsules, self.pose_dim)

        a_v_p_rot, a_v_p_trans = self.view_encoder(a)
        b_v_p_rot, b_v_p_trans = self.view_encoder(b)

        a_v_p_aug_rot, a_v_p_aug_trans = self.view_encoder(a_aug)
        b_v_p_aug_rot, b_v_p_aug_trans = self.view_encoder(b_aug)

        r_f = np.random.randint(1, 4)
        if r_f == 2:
            a_rot = a2_v_p_rot
            b_rot = b2_v_p_rot
        elif r_f == 3:
            a_rot = a_v_p_rot
            b_rot = b_v_p_aug_rot
        else:
            a_rot = a_v_p_aug_rot
            b_rot = b_v_p_rot

        a_vs_pose = self.view_transform(a_pose, a_rot, a_v_p_trans)
        b_vs_pose = self.view_transform(b_pose, b_rot, b_v_p_trans)

        a_vs_pose_t = self.view_transform(b_pose, a_rot, a_v_p_trans)
        b_vs_pose_t = self.view_transform(a_pose, b_rot, b_v_p_trans)

        a_vs_pose_aug = self.view_transform(a_pose_aug, a_rot, a_v_p_aug_trans)
        b_vs_pose_aug = self.view_transform(b_pose_aug, b_rot, b_v_p_aug_trans)

        f_a = self.decoder(a_vs_pose.view(-1, self.pose_num_capsules * self.pose_dim))
        f_b = self.decoder(b_vs_pose.view(-1, self.pose_num_capsules * self.pose_dim))
        #
        f_a_t = self.decoder(a_vs_pose_t.view(-1, self.pose_num_capsules * self.pose_dim))
        f_b_t = self.decoder(b_vs_pose_t.view(-1, self.pose_num_capsules * self.pose_dim))

        f_a_aug = self.decoder(a_vs_pose_aug.view(-1, self.pose_num_capsules * self.pose_dim))
        f_b_aug = self.decoder(b_vs_pose_aug.view(-1, self.pose_num_capsules * self.pose_dim))

        with torch.no_grad():
            c_pose = self.pose_encoder(c).view(-1, self.pose_num_capsules, self.pose_dim)
            a2_pose = self.pose_encoder(a2).view(-1, self.pose_num_capsules, self.pose_dim)
            b2_pose = self.pose_encoder(b2).view(-1, self.pose_num_capsules, self.pose_dim)

            c_v_p_rot, c_v_p_trans = self.view_encoder(c)
            #b2_v_p_rot, b2_v_p_trans = self.view_encoder(b2)

            c_vs_pose = self.view_transform(c_pose, c_v_p_rot, c_v_p_trans)
            b2_vs_pose = self.view_transform(b2_pose, b2_v_p_rot, b2_v_p_trans)

            f_c = self.decoder(c_vs_pose.view(-1, self.pose_num_capsules * self.pose_dim))

        return f_a, f_b, f_a_aug, f_b_aug, f_a_t, f_b_t, f_c, a_vs_pose, b_vs_pose, a_vs_pose_aug, b_vs_pose_aug, c_vs_pose, b2_vs_pose, a_pose, b_pose, c_pose, a2_pose, b2_pose
        #return f_a, f_b, f_a_aug, f_b_aug, f_c, a_vs_pose, b_vs_pose, a_vs_pose_aug, b_vs_pose_aug, c_vs_pose, b2_vs_pose, a_pose, b_pose, c_pose, a2_pose, b2_pose

class VI_GRU_Classifier(nn.Module):
    def __init__(self, latent_dim, hidden_size, gru_layers, bidirectional, n_class):
        super(VI_GRU_Classifier, self).__init__()

        self.latent_dim = latent_dim
        self.n_class = n_class
        self.gru_layers = gru_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.relu = nn.ReLU(inplace=True)

        if self.bidirectional is True:
            self.num_direction = 2
        else:
            self.num_direction = 1

        self.pose_encoder= VI_Pose_Encoder()
        self.lp4 = nn.Linear(70 * 3, 70 * 3)

        self.GRU = nn.GRU(latent_dim, hidden_size=hidden_size, num_layers=gru_layers, batch_first=True, bidirectional=bidirectional)

        #self.output_layer = nn.Linear(2 * hidden_size if bidirectional == True else hidden_size, 256)
        self.output_layer2 = nn.Linear(2 * hidden_size if bidirectional == True else hidden_size, n_class)
        self.drp =nn.Dropout(p=0.2)

    def forward(self, input):
        device = input.device
        [batch_size, timesteps, channel_x, h_x, w_x] = input.size()
        gru_input = Variable(torch.zeros(batch_size, timesteps, self.latent_dim)).to(device)
        output = Variable(torch.zeros(batch_size, timesteps, self.n_class)).to(device)
        for i in range(timesteps):
            gru_input[:, i, :] = self.relu(self.lp4(self.pose_encoder(input[:, i, :, :, :])))
        gru_output, _ = self.GRU(gru_input)
        for i in range(timesteps):
            output[:, i, :] = self.linear_forward(gru_output[:, i, :])

        mean_output = output[:, 0, :]
        for i in range(output.size(1)-1):
            mean_output += output[:, i + 1, :]
        mean_output = output[:, 0, :] / output.size(1)
        return mean_output

    def linear_forward(self, input):
        #l = self.drp(self.relu(self.output_layer(input)))
        l = self.output_layer2(input)
        return l

class VI_GRU_Self_Attention_Classifier(nn.Module):
    def __init__(self, latent_dim, hidden_size, gru_layers, bidirectional, n_class):
        super(VI_GRU_Self_Attention_Classifier, self).__init__()

        self.latent_dim = latent_dim
        self.n_class = n_class
        self.gru_layers = gru_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.relu = nn.ReLU(inplace=True)

        if self.bidirectional is True:
            self.num_direction = 2
        else:
            self.num_direction = 1

        self.pose_encoder= VI_Pose_Encoder()
        self.lp4 = nn.Linear(latent_dim, latent_dim)
        self.attenstion = SelfAttention(attention_size=(2 * hidden_size if bidirectional == True else hidden_size), batch_first=True, layers=2, dropout=.0, non_linearity="tanh")
        self.GRU = nn.GRU(latent_dim, hidden_size=hidden_size, num_layers=gru_layers, batch_first=True, bidirectional=bidirectional)
        self.output_layer = nn.Linear(2 * hidden_size if bidirectional == True else hidden_size, n_class)
        #self.output_layer2 = nn.Linear(256, n_class)
        self.drp = nn.Dropout(p=0.2)

    def forward(self, input):
        device = input.device
        [batch_size, timesteps, channel_x, h_x, w_x] = input.size()
        gru_input = Variable(torch.zeros(batch_size, timesteps, self.latent_dim)).to(device)

        for i in range(timesteps):
            gru_input[:, i, :] = self.relu(self.lp4(self.pose_encoder(input[:, i, :, :, :])))

        gru_output, _ = self.GRU(gru_input)
        y, attentions = self.attenstion(gru_output)
        final_output = self.linear_forward(y)

        return final_output

    def linear_forward(self, input):
        #l = self.drp(self.relu(self.output_layer1(input)))
        l = self.output_layer(input)
        return l

class VI_GRU_Self_Attention_Classifier2(nn.Module):
    def __init__(self, latent_dim = 70 * 3, hidden_size = 512, gru_layers = 2, bidirectional = True, feature_pool = 'attention', n_class=60):
        super(VI_GRU_Self_Attention_Classifier2, self).__init__()

        self.latent_dim = latent_dim
        self.n_class = n_class
        self.gru_layers = gru_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.feature_pool = feature_pool
        self.relu = nn.ReLU(inplace=True)

        if self.bidirectional is True:
            self.num_direction = 2
        else:
            self.num_direction = 1

        if self.feature_pool == 'concat':
            self.num_hidden = 2
        else:
            self.num_hidden = 1

        self.pose_encoder= VI_Pose_Encoder()
        #self.lp4 = nn.Linear(latent_dim, latent_dim)
        self.attenstion = SelfAttention(attention_size=(2 * hidden_size if bidirectional == True else hidden_size), batch_first=True, layers=2, dropout=0.0, non_linearity="tanh")
        self.GRU = nn.GRU(latent_dim, hidden_size=hidden_size, num_layers=gru_layers, batch_first=True, bidirectional=bidirectional)

        self.output_layer = nn.Linear(self.num_direction * self.num_hidden * self.hidden_size, self.n_class)
        #self.output_layer2 = nn.Linear(256, n_class)
        self.drp = nn.Dropout(p=0.2)

    def forward(self, input):
        device = input.device
        [batch_size, timesteps, channel_x, h_x, w_x] = input.size()
        gru_input = Variable(torch.zeros(batch_size, timesteps, self.latent_dim)).to(device)

        for i in range(timesteps):
            gru_input[:, i, :] = self.pose_encoder(input[:, i, :, :, :])

        gru_output, _ = self.GRU(gru_input)

        if self.feature_pool == "concat":

            outputs = F.relu(gru_output)
            avg_pool = F.adaptive_avg_pool1d(outputs.permute(0, 2, 1), 1).view(batch_size, -1)
            max_pool = F.adaptive_max_pool1d(outputs.permute(0, 2, 1), 1).view(batch_size, -1)
            final_output = self.linear_forward(torch.cat([avg_pool, max_pool], dim=1))

        elif self.feature_pool == "attention":
            y, attentions = self.attenstion(gru_output)
            final_output = self.linear_forward(y)

        else:
            gru_output = gru_output.permute(1, 0, 2)
            final_output = self.linear_forward(gru_output[-1])

        return final_output

    def linear_forward(self, input):
        #l = self.drp(self.relu(self.output_layer1(input)))
        l = self.output_layer(input)
        return l

class VI_GRU_Self_Attention_Classifier3(nn.Module):
    def __init__(self, latent_dim = 70 * 3, hidden_size=512, gru_layers=2, bidirectional=True, n_class=60):
        super(VI_GRU_Self_Attention_Classifier3, self).__init__()

        self.latent_dim = latent_dim
        self.n_class = n_class
        self.gru_layers = gru_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.relu = nn.ReLU(inplace=True)

        if self.bidirectional is True:
            self.num_direction = 2
        else:
            self.num_direction = 1


        self.pose_encoder= VI_Pose_Encoder(input_nc=3, ngf=64, pose_num_capsules=int(self.latent_dim/3), pose_dim=3)
        # self.pose_encoder= VI_Pose_Encoder()
        self.lp4 = nn.Linear(latent_dim, latent_dim)
        self.attenstion = SelfAttention(attention_size=(2 * hidden_size if bidirectional == True else hidden_size), batch_first=True, layers=2, dropout=0.0, non_linearity="tanh")
        self.GRU = nn.GRU(latent_dim, hidden_size=hidden_size, num_layers=gru_layers, batch_first=True, bidirectional=bidirectional)

        self.output_layer = nn.Linear(self.num_direction * self.hidden_size, self.n_class)
        #self.output_layer2 = nn.Linear(256, n_class)
        self.drp = nn.Dropout(p=0.2)

    def forward(self, input):
        device = input.device
        [batch_size, timesteps, _, _, _] = input.size()

        gru_input = Variable(torch.zeros(batch_size, timesteps, self.latent_dim)).to(device)
        output = Variable(torch.zeros(batch_size, timesteps, self.n_class)).to(device)

        for i in range(timesteps):
            gru_input[:, i, :] = self.relu(self.lp4(self.pose_encoder(input[:, i, :, :, :])))

        gru_output, _ = self.GRU(gru_input)
        gru_output2 = gru_output

        y, attentions = self.attenstion(gru_output)
        for i in range(timesteps):
            output[:, i, :] = self.linear_forward(gru_output2[:, i, :])
        weighted_output = torch.mul(output, attentions.unsqueeze(-1).expand_as(output))
        final_output = weighted_output.sum(1).squeeze()
        return final_output

    def linear_forward(self, input):
        l = self.output_layer(input)
        return l


