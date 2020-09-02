import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.models as models
import backbone
import time


class DecodeBlock(nn.Module):
    def __init__(self, in_channels_nums, out_channels_nums, kernel_size=4, stride=2, use_bias=False,
                 previous_kernel_size=4, need_activate=True):
        super(DecodeBlock, self).__init__()
        self.need_activate = need_activate
        self.deconv = nn.ConvTranspose2d(in_channels=in_channels_nums, out_channels=out_channels_nums,
                                         kernel_size=kernel_size, stride=stride,
                                         bias=use_bias, padding=1, output_padding=0)
        self.bn = nn.BatchNorm2d(num_features=out_channels_nums)
        self.relu = nn.ReLU(inplace=True)
        if self.need_activate:
            self.fus_relu = nn.ReLU(inplace=True)
            self.fus_bn = nn.BatchNorm2d(num_features=out_channels_nums)

    def forward(self, pre_feature, x):
        x = self.relu(self.bn(self.deconv(x)))
        out = x + pre_feature
        if self.need_activate:
            out = self.fus_relu(self.fus_bn(out))
        return out

    # def _initialize_weights(self, layers):
    #     for m in layers:
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)


class LaneNet(nn.Module):
    def __init__(
            self,
            embed_dim=4,
            delta_v=0.5,
            delta_d=3.0,
            scale_lane_line=1.0,
            scale_var=1.0,
            scale_dist=1.0,
            pretrained=False,
            **kwargs
    ):
        super(LaneNet, self).__init__()
        self.pretrained = pretrained
        self.embed_dim = embed_dim + 1 # background
        self.delta_v = delta_v
        self.delta_d = delta_d

        self.net_init()

        self.scale_seg = scale_lane_line
        self.scale_var = scale_var
        self.scale_dist = scale_dist
        self.scale_reg = 0
        self.seg_loss = nn.CrossEntropyLoss()

    def net_init(self):
        # ----------------- process backbone -----------------
        # -----------------vgg16 bn---------------------------
        self.backbone = backbone.VGG16Encode(pretrained=self.pretrained, batch_norm=True)
        # ----------------- additional conv -----------------
        # ----------------- embedding -----------------True

        self.embedding_fuse4 = DecodeBlock(in_channels_nums=512, out_channels_nums=512)
        self.embedding_fuse3 = DecodeBlock(in_channels_nums=512, out_channels_nums=256)
        self.embedding_fuse2 = DecodeBlock(in_channels_nums=256, out_channels_nums=128)
        self.embedding_fuse1 = DecodeBlock(in_channels_nums=128, out_channels_nums=64)
        self.embedding = nn.Sequential(
            nn.Conv2d(64, self.embed_dim, 1)
        )
        # ----------------- binary segmentation -----------------

        self.binary_seg_fuse4 = DecodeBlock(in_channels_nums=512, out_channels_nums=512)
        self.binary_seg_fuse3 = DecodeBlock(in_channels_nums=512, out_channels_nums=256)
        self.binary_seg_fuse2 = DecodeBlock(in_channels_nums=256, out_channels_nums=128)
        self.binary_seg_fuse1 = DecodeBlock(in_channels_nums=128, out_channels_nums=64)
        self.binary_seg = nn.Sequential(
            nn.Conv2d(64, 2, 1)
        )

    def forward(self, img, segLabel=None):
        f1, f2, f3, f4, f_seg, f_emb = self.backbone(img)

        embedding = self.embedding_fuse4(f4, f_emb)
        embedding = self.embedding_fuse3(f3, embedding)
        embedding = self.embedding_fuse2(f2, embedding)
        embedding = self.embedding_fuse1(f1, embedding)
        embedding = self.embedding(embedding)

        binary_seg = self.binary_seg_fuse4(f4, f_seg)
        binary_seg = self.binary_seg_fuse3(f3, binary_seg)
        binary_seg = self.binary_seg_fuse2(f2, binary_seg)
        binary_seg = self.binary_seg_fuse1(f1, binary_seg)
        binary_seg = self.binary_seg(binary_seg)

        if self.train():
            if segLabel is not None:
                var_loss, dist_loss = self.discriminative_loss(embedding, segLabel)
                seg_loss = self.seg_loss(binary_seg, torch.gt(segLabel, 0).type(torch.long))
            else:
                var_loss = torch.tensor(0, dtype=img.dtype, device=img.device)
                dist_loss = torch.tensor(0, dtype=img.dtype, device=img.device)
                seg_loss = torch.tensor(0, dtype=img.dtype, device=img.device)

            loss = seg_loss * self.scale_seg + var_loss * self.scale_var + dist_loss * self.scale_dist

            output = {
                "embedding": embedding,
                "binary_seg": binary_seg,
                "loss_seg": seg_loss,
                "loss_var": var_loss,
                "loss_dist": dist_loss,
                "loss": loss
            }
        else:
            output = {
                "embedding": embedding,
                "binary_seg": binary_seg
            }
        return output

    def discriminative_loss(self, embedding, seg_gt):
        batch_size = embedding.shape[0]

        var_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
        dist_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)

        for b in range(batch_size):
            embedding_b = embedding[b] # (embed_dim, H, W)
            seg_gt_b = seg_gt[b]

            labels = torch.unique(seg_gt_b)
            labels = labels[labels != 0]
            num_lanes = len(labels)
            if num_lanes==0:
                # please refer to issue here: https://github.com/harryhan618/LaneNet/issues/12
                _nonsense = embedding.sum()
                _zero = torch.zeros_like(_nonsense)
                var_loss = var_loss + _nonsense * _zero
                dist_loss = dist_loss + _nonsense * _zero
                # reg_loss = reg_loss + _nonsense * _zero
                continue

            centroid_mean = []
            for lane_idx in labels:
                seg_mask_i = (seg_gt_b == lane_idx)
                if not seg_mask_i.any():
                    continue
                embedding_i = embedding_b[:, seg_mask_i]

                mean_i = torch.mean(embedding_i, dim=1)
                centroid_mean.append(mean_i)

                # ---------- var_loss -------------
                var_loss = var_loss + \
                           torch.mean(F.relu(torch.norm(embedding_i-mean_i.reshape(self.embed_dim, 1), dim=0)
                                                        - self.delta_v)**2) / num_lanes
            centroid_mean = torch.stack(centroid_mean)  # (n_lane, embed_dim)

            if num_lanes > 1:
                centroid_mean1 = centroid_mean.reshape(-1, 1, self.embed_dim)
                centroid_mean2 = centroid_mean.reshape(1, -1, self.embed_dim)
                dist = torch.norm(centroid_mean1-centroid_mean2, dim=2)  # shape (num_lanes, num_lanes)
                dist = dist + torch.eye(num_lanes, dtype=dist.dtype, device=dist.device) * self.delta_d
                # diagonal elements are 0, now mask above delta_d

                # divided by two for double calculated loss above, for implementation convenience
                dist_loss = dist_loss + torch.sum(F.relu(-dist + self.delta_d)**2) / (num_lanes * (num_lanes-1)) / 2

            # reg_loss is not used in original paper
            # reg_loss = reg_loss + torch.mean(torch.norm(centroid_mean, dim=1))

        var_loss = var_loss / batch_size
        dist_loss = dist_loss / batch_size
        return var_loss, dist_loss

    def set_test(self):
        self.pretrained = False

    def set_train(self):
        self.pretrained = True


if __name__ == "__main__":
    net = LaneNet(pretrained=True)
    data = torch.ones([1, 3, 512, 256])
    out = net(data)
    print(out.size())
