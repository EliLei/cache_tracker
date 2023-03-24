from lib.models.cache_tracker.arc import ARC
from lib.models.cache_tracker.cache_tracker import build_cachet
from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.train.data.processing_utils import sample_target, target_image_crop
from copy import deepcopy
# for debug
import cv2
import os
from lib.utils.merge import merge_template_search
from lib.test.tracker.stark_utils import Preprocessor
from lib.utils.box_ops import clip_box
import torch.nn.functional as F
import lib.train.data.transforms as tfm
import numpy as np



class CACHET(BaseTracker):
    def __init__(self, params, dataset_name):
        super(CACHET, self).__init__(params)
        network = build_cachet(params.cfg)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        # for debug
        self.debug = False
        self.frame_id = 0
        if self.debug:
            self.save_dir = "debug"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        # template update
        self.z_dict1 = {}
        self.z_dict_list = []
        # Set the update interval
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
        else:
            self.update_intervals = self.cfg.DATA.MAX_SAMPLE_INTERVAL
        print("Update interval is: ", self.update_intervals)
        self.num_extra_template = len(self.update_intervals)

        self.arc = ARC(3)
        self.window = self.hanning_window(10, 10)
        self.window = self.window[1:-1,1:-1]
        self.window = self.window.reshape(1,64,1).float().cuda()


        self.missing = False
        self.missing_count = 0
        self.last_state = self.state
        pass

    def hanning_window(self, height, width):
        win_col = np.hanning(width)
        win_row = np.hanning(height)
        mask_col, mask_row = np.meshgrid(win_col, win_row)
        win = mask_col * mask_row
        win = torch.from_numpy(win)
        return win

    def get_similar_template(self, bbfeat):
        templates = self.arc.getall()
        if len(templates)==0:
            return None

        t_labels = []
        t_feat = []
        t_mask = []
        for l,v in templates.items():
            t_labels.append(l)
            t_feat.append(v['feat'])
            t_mask.append(v['mask'])
        t_feat = torch.stack(t_feat, dim=0)
        t_feat = t_feat.reshape(t_feat.shape[0],64,-1)
        t_mask = torch.stack(t_mask, dim=0)
        t_mask = t_mask.reshape(t_mask.shape[0],64,-1)


        feat = bbfeat['feat']
        feat = feat.reshape(1,64,-1)*self.window
        mask = bbfeat['mask']
        mask = mask.reshape(1,64,-1)*self.window

        t_feat = (t_feat * (t_mask.logical_not()) * (mask.logical_not())).reshape(t_mask.shape[0], -1) # (N, dim)
        t_feat = F.normalize(t_feat, dim=-1)
        feat = (feat * (t_mask.logical_not()) * (mask.logical_not())).reshape(t_mask.shape[0], -1) # (N, dim)
        feat = F.normalize(feat, dim=-1)

        similarity = torch.einsum('ij,ij->i', t_feat, feat)
        v,i = torch.max(similarity, dim=0)
        #if v>threshold:
        if v > 0.8:
            return t_labels[i.item()]

        return None





    def initialize(self, image, info: dict):
        self.frame_id = 0
        # initialize z_dict_list
        self.z_dict_list = [] # 'feat' (64,1,256)
        # get the 1st template
        # tfs = [
        #     #tfm.Transform(tfm.RandomHorizontalFlip(probability=1.)),
        #     #tfm.Transform(tfm.RandomBlur(sigma=0.5,probability=1.)),
        #     tfm.Transform(tfm.RandomAffine(p_flip=0.0, max_rotation=10.0,
        #                                    max_shear=0.0, max_ar_factor=0.0,
        #                                    max_scale=0.0, pad_amount=0,
        #                                    border_mode='replicate')),
        #     tfm.Transform(tfm.RandomAffine(p_flip=0.0, max_rotation=10.0,
        #                                    max_shear=0.0, max_ar_factor=0.0,
        #                                    max_scale=0.0, pad_amount=0,
        #                                    border_mode='replicate')),
        #     tfm.Transform(tfm.RandomAffine(p_flip=0.0, max_rotation=10.0,
        #                                    max_shear=0.0, max_ar_factor=0.0,
        #                                    max_scale=0.0, pad_amount=0,
        #                                    border_mode='replicate')),
        # ]
        # bbox_tensor = torch.tensor(info['init_bbox'],dtype=torch.float)
        # with torch.no_grad():
        #     for i in range(len(tfs)):
        #         tf = tfs[i]
        #         tf_img , tf_anno = tf(image=[image], bbox=[bbox_tensor])
        #         tf_img=tf_img[0]
        #         tf_anno=tf_anno[0].tolist()
        #         tf_patch, _, tf_amask = sample_target(tf_img, tf_anno, self.params.template_factor,
        #                                                       output_sz=self.params.template_size)
        #         template = self.preprocessor.process(tf_patch, tf_amask)
        #         z = self.network.forward_backbone(template)
        #
        #         self.arc.update(f'template{i}',f'template{i}',z)




        z_patch_arr1, _, z_amask_arr1 = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                      output_sz=self.params.template_size)
        template1 = self.preprocessor.process(z_patch_arr1, z_amask_arr1)
        with torch.no_grad():
            self.z_dict1 = self.network.forward_backbone(template1)
        # get the complete z_dict_list
        self.arc.update(f'init1', f'init1', self.z_dict1)
        self.arc.update(f'init2', f'init2', self.z_dict1)
        self.arc.update(f'init3', f'init3', self.z_dict1)



        # save states
        self.state = info['init_bbox']
        self.saveimg(image, gt=True)

        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1

        #if self.frame_id==160:
        #    print('haha')
        # get the t-th search region
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        #from matplotlib import pyplot as plt;plt.ion();plt.figure('search');plt.imshow(x_patch_arr);
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)



        with torch.no_grad():
            x_dict = self.network.forward_backbone(search)
            # merge the template and the search
            feat_dict_list = [self.z_dict1]+list(self.arc.get().values()) + [x_dict]
            seq_dict = merge_template_search(feat_dict_list)
            # run the transformer
            out_dict, _, _ = self.network.forward_transformer(seq_dict=seq_dict, run_box_head=True, run_cls_head=True)
        # get the final result
        pred_boxes = out_dict['pred_boxes'].view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]

        # get confidence score (whether the search region is reliable)
        conf_score = out_dict["pred_logits"].view(-1).sigmoid().item()
        #print('conf_score',conf_score)

        # get the final box result
        #if conf_score>0.1:

        if True:
            self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
            self.last_state = self.state
            self.missing = False
            self.missing_count = 0

        if self.debug:
            self.saveimg(image)
        # else:
        #     self.missing = True
        #     self.missing_count+=1
        #     exponent = 10 if self.missing_count>15 else self.missing_count-5
        #     if exponent<0:
        #         exponent = 0
        #
        #     self.state = [self.last_state[0]-self.last_state[2]*(1.07**exponent-1)/2,
        #                   self.last_state[1]-self.last_state[3]*(1.07**exponent-1)/2,
        #                   self.last_state[2]*(1.07**exponent),
        #                   self.last_state[3]*(1.07**exponent)]
        # update template

        if self.frame_id % 3 == 0:
            if False:
                print("frame_id",self.frame_id)
                print('T1', list(self.arc.T1.keys()))
                print('T2', list(self.arc.T2.keys()))
                print('B1', list(self.arc.B1.keys()))
                print('B2', list(self.arc.B2.keys()))
                print()
            if conf_score > 0.5:
                z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                            output_sz=self.params.template_size)  # (x1, y1, w, h)
                template_t = self.preprocessor.process(z_patch_arr, z_amask_arr)
                with torch.no_grad():
                    z_dict_t = self.network.forward_backbone(template_t)
                key = self.get_similar_template(z_dict_t)
                new_key = f'f{self.frame_id}'
                self.arc.update(key,new_key,z_dict_t)   # the 1st element of z_dict_list is template from the 1st frame


        # for debug

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save,
                    "conf_score": conf_score}
        else:
            return {"target_bbox": self.state,
                    "conf_score": conf_score}

    def saveimg(self,image,gt=False):
        if not self.debug:
            return
        x1, y1, w, h = self.state
        image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        startx = 10
        starty = 20
        p = 30
        fontsize = 0.8
        thickness = 2
        tcolor = (0, 242, 255)
        cv2.putText(image_BGR, f'Frame {self.frame_id:04d}', (startx, starty), cv2.FONT_HERSHEY_SIMPLEX, fontsize,
                    color=(37, 28, 237), thickness=thickness)

        if gt:
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 255, 0), thickness=2)
        else:
            cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)

            cv2.putText(image_BGR, r'Template Manager:',
                        (startx, starty + 1 * p), cv2.FONT_HERSHEY_SIMPLEX, fontsize, color=tcolor, thickness=thickness)
            cv2.putText(image_BGR, f'T1:{"-".join([k.split(r"/")[0] for k in list(self.arc.T1.keys())])}',
                        (startx, starty + 2 * p), cv2.FONT_HERSHEY_SIMPLEX, fontsize, color=tcolor, thickness=thickness)
            cv2.putText(image_BGR, f'B1:{"-".join([k.split(r"/")[0] for k in list(self.arc.B1.keys())])}',
                        (startx, starty + 3 * p), cv2.FONT_HERSHEY_SIMPLEX, fontsize, color=tcolor, thickness=thickness)
            cv2.putText(image_BGR, f'T2:{"-".join([k.split(r"/")[0] for k in list(self.arc.T2.keys())])}',
                        (startx, starty + 4 * p), cv2.FONT_HERSHEY_SIMPLEX, fontsize, color=tcolor, thickness=thickness)
            cv2.putText(image_BGR, f'B2:{"-".join([k.split(r"/")[0] for k in list(self.arc.B2.keys())])}',
                        (startx, starty + 5 * p), cv2.FONT_HERSHEY_SIMPLEX, fontsize, color=tcolor, thickness=thickness)

        save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
        cv2.imwrite(save_path, image_BGR)

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return CACHET
