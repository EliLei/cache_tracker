
from lib.models.stark.backbone import build_backbone
from lib.models.cache_tracker.transformer import build_transformer
from lib.models.stark.head import build_box_head, MLP
from torch import nn
from lib.utils.misc import NestedTensor
from einops import rearrange
import torch
import torch.nn.functional as F

from ...utils.box_ops import box_xyxy_to_cxcywh
from .membank import MemoryBank

class CACHET(nn.Module):
    """ This is the base class for Transformer Tracking """
    def __init__(self, backbone, transformer, box_head, num_queries,
                 aux_loss=False, head_type="CORNER", cls_head=None, use_uncertain_embed=True):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super(CACHET, self).__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.box_head = box_head
        self.num_queries = num_queries
        hidden_dim = transformer.d_model
        self.bottleneck = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)  # the bottleneck layer
        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)
        self.cls_head = cls_head

        self.uncertain_embed = use_uncertain_embed
        if use_uncertain_embed:
            self.search_uncertain_embed = nn.Parameter(torch.zeros(400, 1, hidden_dim))
            self.gt_uncertain_embed = nn.Parameter(torch.zeros(64, 1, hidden_dim))
            self.template_uncertain_embed = nn.Parameter(torch.zeros(64, 1, hidden_dim))

    def apply_uncertain_embed(self, seq_dict):
        if self.uncertain_embed:
            ret = seq_dict.copy()
            pos = ret['pos']
            n_template = (pos.shape[0]-400-64)//64
            uncertain_embed = torch.cat([self.gt_uncertain_embed]+
                                        [self.template_uncertain_embed]*n_template+
                                        [self.search_uncertain_embed],
                                        dim=0)
            ret['pos'] = pos+uncertain_embed
            return ret
        else:
            return seq_dict

    def forward(self, img=None, seq_dict=None, mode="backbone", run_box_head=False, run_cls_head=False):
        if mode == "backbone":
            return self.forward_backbone(img)
        elif mode == "transformer":
            seq_dict = self.apply_uncertain_embed(seq_dict)
            return self.forward_transformer(seq_dict, run_box_head=run_box_head, run_cls_head=run_cls_head)
        else:
            raise ValueError

    def forward_backbone(self, input: NestedTensor):
        """The input type is NestedTensor, which consists of:
               - tensor: batched images, of shape [batch_size x 3 x H x W]
               - mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        assert isinstance(input, NestedTensor)
        # Forward the backbone
        output_back, pos = self.backbone(input)  # features & masks, position embedding for the search
        # Adjust the shapes
        return self.adjust(output_back, pos)

    def forward_transformer(self, seq_dict, run_box_head=False, run_cls_head=False):
        if self.aux_loss:
            raise ValueError("Deep supervision is not supported.")
        # Forward the transformer encoder and decoder
        encode_embed = self.transformer(seq_dict["feat"], seq_dict["mask"],
                                                 seq_dict["pos"])
        # Forward the corner head
        out, outputs_coord = self.forward_head(encode_embed, run_box_head=run_box_head, run_cls_head=run_cls_head)
        return out, outputs_coord, encode_embed

    def forward_head(self, encode_embed, run_box_head=False, run_cls_head=False):
        """
        hs: output embeddings (1, B, N, C)
        memory: encoder embeddings (HW1+HW2, B, C)"""
        out_dict = {}
        if run_cls_head:
            # forward the classification head
            enc_opt = encode_embed[-self.feat_len_s:] # (HW, B, C)
            enc_opt = enc_opt.mean(dim=0) # (B, C)
            out_dict.update({'pred_logits': self.cls_head(enc_opt)})
        if run_box_head:
            # forward the box prediction head
            out_dict_box, outputs_coord = self.forward_box_head(encode_embed)
            # merge results
            out_dict.update(out_dict_box)
            return out_dict, outputs_coord
        else:
            return out_dict, None

    def forward_box_head(self, encode_embed):
        """
        hs: output embeddings (1, B, N, C)
        memory: encoder embeddings (HW1+HW2, B, C)"""
        if self.head_type == "CORNER":
            # adjust shape
            enc_opt = encode_embed[-self.feat_len_s:] # encoder output for the search region (HW, B, C)
            #dec_opt = hs.squeeze(0).transpose(1, 2)  # (B, C, N)
            #att = torch.matmul(enc_opt, dec_opt)  # (B, HW, N)
            #opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
            #bs, Nq, C, HW = opt.size()
            #opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
            enc_opt = rearrange(enc_opt, '(h w) b c -> b c h w', h=self.feat_sz_s, w=self.feat_sz_s)
            # run the corner head
            outputs_coord = box_xyxy_to_cxcywh(self.box_head(enc_opt))
            outputs_coord_new = outputs_coord.view(-1, 1, 4)
            out = {'pred_boxes': outputs_coord_new}
            return out, outputs_coord_new
        # elif self.head_type == "MLP":
        #     # Forward the class and box head
        #     outputs_coord = self.box_head(hs).sigmoid()
        #     out = {'pred_boxes': outputs_coord[-1]}
        #     if self.aux_loss:
        #         out['aux_outputs'] = self._set_aux_loss(outputs_coord)
        #     return out, outputs_coord

    def adjust(self, output_back: list, pos_embed: list):
        """
        """
        src_feat, mask = output_back[-1].decompose()
        assert mask is not None
        # reduce channel
        feat = self.bottleneck(src_feat)  # (B, C, H, W)
        # adjust shapes
        feat_vec = feat.flatten(2).permute(2, 0, 1)  # HWxBxC
        pos_embed_vec = pos_embed[-1].flatten(2).permute(2, 0, 1)  # HWxBxC
        mask_vec = mask.flatten(1)  # BxHW
        return {"feat": feat_vec, "mask": mask_vec, "pos": pos_embed_vec}

    @torch.jit.unused
    def _set_aux_loss(self, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_boxes': b}
                for b in outputs_coord[:-1]]

def build_cachet(cfg):
    backbone = build_backbone(cfg)  # backbone and positional encoding are built together
    #backbone = build_backbone_NOPE(cfg)
    transformer = build_transformer(cfg)
    box_head = build_box_head(cfg)
    cls_head = MLP(cfg.MODEL.HIDDEN_DIM, cfg.MODEL.HIDDEN_DIM, 1, cfg.MODEL.NLAYER_HEAD)
    model = CACHET(
        backbone,
        transformer,
        box_head,
        num_queries=cfg.MODEL.NUM_OBJECT_QUERIES,
        aux_loss=cfg.TRAIN.DEEP_SUPERVISION,
        head_type=cfg.MODEL.HEAD_TYPE,
        cls_head=cls_head,
        use_uncertain_embed = getattr(cfg.MODEL,'UNCERTAINTY',False)
    )

    return model
