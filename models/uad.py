import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ViTill(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
            fuse_layer_decoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
            remove_class_token=False,
            encoder_require_grad_layer=[],
            adaptive_context_decouple=True
    ) -> None:
        super(ViTill, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer
        self.adaptive_context_decouple = adaptive_context_decouple

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0

        if self.adaptive_context_decouple:
            self.context_alpha = nn.ParameterList([
                nn.Parameter(torch.tensor(0.5))
                for _ in range(len(self.fuse_layer_encoder))
            ])

    def forward(self, x):
        x = self.encoder.prepare_tokens(x)
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                if i in self.encoder_require_grad_layer:
                    x = blk(x)
                else:
                    with torch.no_grad():
                        x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)
        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]

        x = self.fuse_feature(en_list)
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        if not self.remove_class_token:  # class tokens have not been removed above
            de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]

        if self.adaptive_context_decouple:
            en_new = []
            for idx, e in enumerate(en):
                cls_token = e[:, :1, :]
                spatial_feat = e[:, 1 + self.encoder.num_register_tokens:, :]
                spatial_mean = spatial_feat.mean(dim=1, keepdim=True)

                # 动态混合CLS与空间均值
                alpha = torch.sigmoid(self.context_alpha[idx])
                mixed_context = alpha * cls_token + (1 - alpha) * spatial_mean

                e_decoupled = spatial_feat - mixed_context
                e_normed = F.layer_norm(e_decoupled, normalized_shape=(e_decoupled.shape[-1],), eps=1e-8)
                en_new.append(e_normed)
            en = en_new
        else:
            en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]
        return en, de

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)
