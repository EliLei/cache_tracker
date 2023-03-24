from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search


class STARKSActor(BaseActor):
    """ Actor for training the STARK-S and STARK-ST(Stage1)"""
    def __init__(self, net, objective, loss_weight, settings):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        create_calibration=False
        if create_calibration:
            return self.save_calibration(data)
        out_dict = self.forward_pass(data, run_box_head=True, run_cls_head=False)

        # process the groundtruth
        gt_bboxes = data['search_anno']  # (Ns, batch, 4) (x1,y1,w,h)

        # compute losses
        loss, status = self.compute_losses(out_dict, gt_bboxes[0])

        return loss, status

    def forward_pass(self, data, run_box_head, run_cls_head):
        feat_dict_list = []
        # process the templates
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1, *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            feat_dict_list.append(self.net(img=NestedTensor(template_img_i, template_att_i), mode='backbone'))

        # process the search regions (t-th frame)
        search_img = data['search_images'].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        search_att = data['search_att'].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)
        feat_dict_list.append(self.net(img=NestedTensor(search_img, search_att), mode='backbone'))

        # run the transformer and compute losses
        seq_dict = merge_template_search(feat_dict_list)
        out_dict, _, _ = self.net(seq_dict=seq_dict, mode="transformer", run_box_head=run_box_head, run_cls_head=run_cls_head)
        # out_dict: (B, N, C), outputs_coord: (1, B, N, C), target_query: (1, B, N, C)

        return out_dict

    def save_calibration(self, data):

        from lib.train.admin.environment import env_settings
        import os
        env = env_settings()
        calibration_dir = os.path.join(env.workspace_dir, 'calibration', 'starklightning')
        self.savemask = True
        z_dir = os.path.join(calibration_dir, 'z')
        x_dir = os.path.join(calibration_dir, 'x')
        if self.savemask:
            zmask_dir = os.path.join(calibration_dir, 'zmask')
            xmask_dir = os.path.join(calibration_dir, 'xmask')

        os.makedirs(z_dir, exist_ok=True)
        os.makedirs(x_dir, exist_ok=True)
        if self.savemask:
            os.makedirs(zmask_dir, exist_ok=True)
            os.makedirs(xmask_dir, exist_ok=True)

        index_z = len(os.listdir(z_dir))
        index_x = len(os.listdir(x_dir))
        if self.savemask:
            index_zmask = len(os.listdir(zmask_dir))
            index_xmask = len(os.listdir(xmask_dir))

        path_z = os.path.join(z_dir, f'{index_z:06}.data')
        path_x = os.path.join(x_dir, f'{index_x:06}.data')
        if self.savemask:
            path_zmask = os.path.join(zmask_dir, f'{index_zmask:06}.data')
            path_xmask = os.path.join(xmask_dir, f'{index_xmask:06}.data')

        torch.save(data['template_images'][0].detach().cpu(), path_z)
        torch.save(data['search_images'][0].detach().cpu(), path_x)
        if self.savemask:
            torch.save(data['template_att'][0].detach().cpu(), path_zmask)
            torch.save(data['search_att'][0].detach().cpu(), path_xmask)
            mask1 = data['template_att'][0].sum().item()
            mask2 = data['search_att'][0].sum().item()
            #print(mask1, mask2)

        device = data['template_images'].device
        loss = torch.tensor([0.0], requires_grad=True, device=device)
        stats = {'Loss/total': loss.item()}
        return loss,stats


    def compute_losses(self, pred_dict, gt_bbox, return_status=True):
        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss
