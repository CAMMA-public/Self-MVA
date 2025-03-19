from torch.utils.data import Dataset
from collections import defaultdict
import os
import numpy as np
import cv2
import random
from torchvision import transforms as T
import torch
import json
from utils import bbox_iou, generate_neg_bboxes


class Loader(Dataset):
    def __init__(self, config, mode='train', inference=False):
        self.img_size = config.IMG_SIZE
        self.sampling_range = config.DATASET.SAMPLING_RANGE
        self.mode = mode
        self.inference = inference
        if self.mode == 'train':
            self.dataset = config.DATASET.TRAIN_DATASET
        else:
            self.dataset = config.DATASET.TEST_DATASET
        self.train_mode = config.TRAIN.TRAIN_MODE
        self.crop_box = config.CROP_BOX
        self.zoom_out_ratio = config.ZOOM_OUT_RATIO
        self.fix_view_id = config.FIX_VIEW_ID
        self.edge_association = config.TRAIN.EDGE_ASSOCIATION
        self.pseudo_neg = config.TRAIN.PSEUDO_NEG

        self.suffix = '.jpg'
        self.prompt_mode = config.DATASET.PROMPT_MODE
        self.crop_img = False
        if self.prompt_mode == 'box':
            if config.TRAIN.REID or config.TEST.REID:
                self.crop_img = True

        if self.dataset == 'wildtrack':
            self.dataset_dir = 'data/Wildtrack/sequence1'
            intrinsics = np.array([
                [1743.4478759765625, 0.0, 934.5202026367188, 0.0, 1735.1566162109375, 444.3987731933594, 0.0, 0.0, 1.0], 
                [1707.266845703125, 0.0, 978.1306762695312, 0.0, 1719.0408935546875, 417.01922607421875, 0.0, 0.0, 1.0], 
                [1738.7144775390625, 0.0, 906.56689453125, 0.0, 1752.8876953125, 462.0346374511719, 0.0, 0.0, 1.0], 
                [1725.2772216796875, 0.0, 995.0142211914062, 0.0, 1720.581787109375, 520.4190063476562, 0.0, 0.0, 1.0], 
                [1708.6573486328125, 0.0, 936.0921630859375, 0.0, 1737.1904296875, 465.18243408203125, 0.0, 0.0, 1.0], 
                [1742.977783203125, 0.0, 1001.0738525390625, 0.0, 1746.0140380859375, 362.4325866699219, 0.0, 0.0, 1.0], 
                [1732.4674072265625, 0.0, 931.2559204101562, 0.0, 1757.58203125, 459.43389892578125, 0.0, 0.0, 1.0]
            ])
            self.intrinsics = np.reshape(intrinsics, (len(intrinsics), 3, 3))
            rvecs = np.array([
                [1.759099006652832, 0.46710100769996643, -0.331699013710022], 
                [0.6167870163917542, -2.14595890045166, 1.6577140092849731], 
                [0.5511789917945862, 2.229501962661743, -1.7721869945526123], 
                [1.6647210121154785, 0.9668620228767395, -0.6937940120697021], 
                [1.2132920026779175, -1.4771349430084229, 1.2775369882583618], 
                [1.6907379627227783, -0.3968360126018524, 0.355197012424469], 
                [1.6439390182495117, 1.126188039779663, -0.7273139953613281]
            ])
            tvecs = np.array([
                [-525.8941650390625, 45.40763473510742, 986.7235107421875], 
                [1195.231201171875, -336.5144958496094, 2040.53955078125], 
                [55.07157897949219, -213.2444610595703, 1992.845703125], 
                [42.36193084716797, -45.360652923583984, 1106.8572998046875], 
                [836.6625366210938, 85.86837005615234, 600.2880859375], 
                [-338.5532531738281, 62.87659454345703, 1044.094482421875], 
                [-648.9456787109375, -57.225215911865234, 1052.767578125]
            ])
            extrinsics = []
            for i in range(len(rvecs)):
                R, _ = cv2.Rodrigues(rvecs[i])
                extrinsic = np.eye(4)
                extrinsic[0:3, 0:3] = R
                extrinsic[0:3, 3] = tvecs[i]
                extrinsics.append(extrinsic)
                
            self.extrinsics = np.stack(extrinsics, axis=0)
        elif self.dataset == 'soldiers':
            self.dataset_dir = 'data/soldiers'
            cam_path = os.path.join(self.dataset_dir, 'calibration.json')
            with open(cam_path) as f:    
                calibration = json.load(f)
            cam_order = ['3_2', '3_4', '6_1', '6_3', '6_4', '6_5']
            intrinsics = []
            extrinsics = []
            for cam_name in cam_order:
                K = np.array(calibration[cam_name]['K'])
                R = np.array(calibration[cam_name]['R'])
                t = np.array(calibration[cam_name]['t'])
                extrinsic = np.eye(4)
                extrinsic[0:3, 0:3] = R
                extrinsic[0:3, 3] = t
                intrinsics.append(K)
                extrinsics.append(extrinsic)
            self.intrinsics = np.stack(intrinsics, axis=0)
            self.extrinsics = np.stack(extrinsics, axis=0)
        elif self.dataset == 'mvor':
            self.dataset_dir = 'data/MVOR/sequence1'
            intrinsics = [[[538.597 ,   0.    , 315.8367],
                        [  0.    , 538.2393, 241.9166],
                        [  0.    ,   0.    ,   1.    ]], 
                        [[534.8386,   0.    , 321.2326],
                        [  0.    , 534.4008, 243.3514],
                        [  0.    ,   0.    ,   1.    ]], 
                        [[541.4062,   0.    , 323.9545],
                        [  0.    , 540.5641, 238.6629],
                        [  0.    ,   0.    ,   1.    ]]]
            self.intrinsics = np.array(intrinsics)
            extrinsics = [[[-1.49590753e-02, -9.94444053e-01, -1.04198135e-01,  1.00000000e+03],
                        [-5.70907471e-01,  9.40478136e-02, -8.15609998e-01, -7.76041561e-14],
                        [ 8.20878119e-01,  4.72867226e-02, -5.69142407e-01,  2.40000000e+03],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]], 
                        [[-9.55929983e-02,  9.94401044e-01, -4.50393425e-02, -1.04817989e+03],
                        [ 6.35584010e-01,  2.61519595e-02, -7.71588647e-01, -5.22565353e+01],
                        [-7.66090689e-01, -1.02384758e-01, -6.34525348e-01,  2.61148881e+03],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]], 
                        [[-9.49290188e-01, -3.12213969e-01, -3.70212865e-02,  3.42436893e+02],
                        [-1.75067808e-01,  6.22722923e-01, -7.62605680e-01, -9.01827240e+02],
                        [ 2.61150150e-01, -7.17452854e-01, -6.45803376e-01,  3.73129648e+03],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]]
            self.extrinsics = np.array(extrinsics)
        else:
            raise ValueError('Dataset ' + self.dataset + ' is not implemented! ')
        self.image_dir = os.path.join(self.dataset_dir, 'output', 'frames')

        if self.mode == 'train':
            if config.TRAIN.GT_BOX:
                self.anno_path = os.path.join(self.dataset_dir, 'output', 'gt_train.json')
            else:
                self.anno_path = os.path.join(self.dataset_dir, 'output', 'detections_train.json')
        elif self.mode == 'valid':
            if self.dataset == 'soldiers':
                self.anno_path = os.path.join(self.dataset_dir, 'output', 'annotated_box_test.json')
            else:
                self.anno_path = os.path.join(self.dataset_dir, 'output', 'gt_eval.json')
        elif self.mode == 'test':
            if self.dataset == 'soldiers':
                self.anno_path = os.path.join(self.dataset_dir, 'output', 'annotated_box_test.json')
            else:
                self.anno_path = os.path.join(self.dataset_dir, 'output', 'gt_test.json')

        self.view_ls = config.DATASET.VIEW_IDS
        self.anno_dict, self.img_dict = self.gen_anno_path_dict()

        self.heights = []
        self.widths = []
        for cam_id in range(len(self.view_ls)):
            demo_img = cv2.imread(self.img_dict[self.view_ls[cam_id]][0])
            height, width, _ = demo_img.shape
            self.heights.append(height)
            self.widths.append(width)
    
    def get_width_height(self):
        return self.widths, self.heights
    
    def get_cam_param(self):
        return self.intrinsics, self.extrinsics

    def gen_anno_path_dict(self):
        anno_dict = {}
        path_dict = defaultdict(list)

        with open(self.anno_path, 'r') as fp:
            frames = json.load(fp)
        frames_id = list(map(int, frames.keys())) # [0, 1, ..., frame_num - 1]

        for f_id in frames_id:
            for view in self.view_ls:
                img_name = str(f_id) + '_' + str(view) + self.suffix
                img_path = os.path.join(self.image_dir, img_name)
                path_dict[view].append(img_path)

        for i, f_id in enumerate(frames_id):
            nodes = frames[str(f_id)]
            for node in nodes:
                if self.prompt_mode == 'box':
                    x, y, w, h, tid, cid = node
                    if cid not in anno_dict:
                        anno_dict[cid] = defaultdict(list)
                    anno_dict[cid][f_id].append([f_id, tid, x, y, x+w, y+h])
                elif self.prompt_mode == 'point':
                    x, y, tid, cid = node
                    if cid not in anno_dict:
                        anno_dict[cid] = defaultdict(list)
                    anno_dict[cid][f_id].append([f_id, tid, x, y])

        return anno_dict, path_dict

    def read_anno(self, path: str):
        img_name = path.rsplit('/', 1)[1]
        f_id, view = img_name.rsplit('.', 1)[0].split('_')
        c_id = int(view)
        annos = self.anno_dict[c_id][int(f_id)]
        bbox_dict = {}
        for idx, anno in enumerate(annos):
            bbox = anno[2:]
            bbox = [int((float(i))) for i in bbox]
            if anno[1] == -1:
                dict_key = idx
            else:
                dict_key = anno[1]
            bbox_dict[dict_key] = bbox
        return bbox_dict
    
    def load_image(self, frame_img, img_size=(224, 224), tensor=True):
        img = cv2.imread(frame_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if tensor:
            img = T.ToTensor()(img)  # (C, H, W), float
        else:
            img = torch.from_numpy(img)
            img = torch.permute(img, (2, 0, 1))  # (C, H, W), uint8
        img = T.Resize(img_size)(img)
        # img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        return img
    
    def get_bboxes(self, bbox_dict, c_id):
        label_ls = []
        bbox_ls = []
        for key in bbox_dict:
            bbox = bbox_dict[key]
            if self.prompt_mode == 'box':
                bbox[0] /= self.widths[c_id]
                bbox[1] /= self.heights[c_id]
                bbox[2] /= self.widths[c_id]
                bbox[3] /= self.heights[c_id]
            elif self.prompt_mode == 'point':
                bbox[0] /= self.widths[c_id]
                bbox[1] /= self.heights[c_id]
            # bbox = [0 if i < 0 else i for i in bbox]
            # bbox = [1 if i > 1 else i for i in bbox]
            bbox = torch.tensor(bbox, dtype=torch.float32)
            label_ls.append(key)
            bbox_ls.append(bbox)
        if len(bbox_ls):
            bbox_ls = torch.stack(bbox_ls)
        return label_ls, bbox_ls

    
    def get_view_num(self):
        return len(self.view_ls)

    def __len__(self):
        # return self.len
        return min([len(self.img_dict[i]) for i in self.view_ls])

    def __getitem__(self, item):
        if self.inference:
            return self.get_association_item(item)
        if self.fix_view_id == -1:
            anchor_view_id, sample_view_id = random.sample(range(0, len(self.view_ls)), 2)
        else:
            anchor_view_id = self.fix_view_id
            sample_view_id = random.sample(range(0, len(self.view_ls) - 1), 1)[0]
            if sample_view_id >= anchor_view_id:
                sample_view_id += 1

        if self.crop_box:
            return self.get_crop_item(item, anchor_view_id, sample_view_id)

        anchor_img_path = self.img_dict[self.view_ls[anchor_view_id]][item]
        anchor_anno = self.read_anno(anchor_img_path)
        _, anchor_bboxes = self.get_bboxes(anchor_anno, anchor_view_id)

        pos_img_path = self.img_dict[self.view_ls[sample_view_id]][item]
        pos_anno = self.read_anno(pos_img_path)
        _, pos_bboxes = self.get_bboxes(pos_anno, sample_view_id)

        anchor_img = self.load_image(anchor_img_path, self.img_size)
        positive_img = self.load_image(pos_img_path, self.img_size)

        anchor_info = {
            'img': anchor_img, 
            'bboxes': anchor_bboxes, 
            'cam_id': anchor_view_id
        }
        pos_info = {
            'img': positive_img, 
            'bboxes': pos_bboxes, 
            'cam_id': sample_view_id
        }

        if self.mode == 'train' and self.train_mode == 'triplet':
            sample_min, sample_max = self.sampling_range
            interval = sample_min + int(random.random() * (sample_max - sample_min))
            previous_sample = False
            if random.random() > 0.5:
                previous_sample = True
            if previous_sample:
                if item - interval >= 0:
                    new_item = item - interval
                else:
                    new_item = item + interval
            else:
                if item + interval < len(self.img_dict[self.view_ls[sample_view_id]]):
                    new_item = item + interval
                else:
                    new_item = item - interval
            
            negative_img = self.load_image(self.img_dict[self.view_ls[sample_view_id]][new_item], self.img_size)

            neg_anno = self.read_anno(self.img_dict[self.view_ls[sample_view_id]][new_item])
            _, neg_bboxes = self.get_bboxes(neg_anno, sample_view_id)

            loaded_imgs = torch.stack([anchor_img, positive_img, negative_img], axis=0)

            neg_info = {
                'img': negative_img, 
                'bboxes': neg_bboxes, 
                'cam_id': sample_view_id
            }

            return anchor_info, pos_info, neg_info
        else:
            return anchor_info, pos_info
    

    def load_cropped_bboxes(self, img_path, bbox_dict, c_id, image_key=None, img_size=(224, 224), tensor=True, bbox_jitter=False):
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if tensor:
            img = T.ToTensor()(img)  # (C, H, W), float
        else:
            img = torch.from_numpy(img)
            img = torch.permute(img, (2, 0, 1))  # (C, H, W), uint8
        
        if self.crop_img:
            enlarge_ratio = (self.zoom_out_ratio - 1) * 0.5
            cropped_imgs = []
        label_ls = []
        bbox_ls = []
        for key in bbox_dict:
            bbox = bbox_dict[key].copy()

            if self.crop_img:
                h = bbox[3] - bbox[1]
                w = bbox[2] - bbox[0]
                new_bbox_y1 = int(max(bbox[1] - h * enlarge_ratio, 0))
                new_bbox_y2 = int(min(bbox[3] + h * enlarge_ratio, height))
                new_bbox_x1 = int(max(bbox[0] - w * enlarge_ratio, 0))
                new_bbox_x2 = int(min(bbox[2] + w * enlarge_ratio, width))
                cropped_img = img[:, new_bbox_y1:new_bbox_y2, new_bbox_x1:new_bbox_x2]
                cropped_img = T.Resize(img_size)(cropped_img)

                cropped_imgs.append(cropped_img)

            label_ls.append(key)
            if self.prompt_mode == 'box':
                bbox[0] /= self.widths[c_id]
                bbox[1] /= self.heights[c_id]
                bbox[2] /= self.widths[c_id]
                bbox[3] /= self.heights[c_id]
            elif self.prompt_mode == 'point':
                bbox[0] /= self.widths[c_id]
                bbox[1] /= self.heights[c_id]
            # bbox = [0 if i < 0 else i for i in bbox]
            # bbox = [1 if i > 1 else i for i in bbox]
            bbox = torch.tensor(bbox, dtype=torch.float32)
            bbox_ls.append(bbox)
        
        if len(bbox_ls):
            bbox_ls = torch.stack(bbox_ls)
            if bbox_jitter:
                bbox_ls, _ = generate_neg_bboxes(bbox_ls, in_boundry=True)
        
        info = {
            'img': T.Resize(img_size)(img), 
            'labels': label_ls, 
            'bboxes': bbox_ls, 
            'width': self.widths[c_id], 
            'height': self.heights[c_id], 
        }
        if self.crop_img:
            info['cropped_imgs'] = torch.stack(cropped_imgs) if len(cropped_imgs) else []
        return info
    
    def load_edges(self, bbox_ls):
        bbox_num = len(bbox_ls)
        if bbox_num == 0:
            return []
        pts1 = bbox_ls.repeat_interleave(len(bbox_ls), 0)
        pts2 = bbox_ls.repeat(len(bbox_ls), 1)
        if self.prompt_mode == 'box':
            w1 = pts1[:, 2] - pts1[:, 0]
            h1 = pts1[:, 3] - pts1[:, 1]
            ground_x1 = (pts1[:, 0] + pts1[:, 2]) * 0.5
            ground_y1 = pts1[:, 3]

            w2 = pts2[:, 2] - pts2[:, 0]
            h2 = pts2[:, 3] - pts2[:, 1]
            ground_x2 = (pts2[:, 0] + pts2[:, 2]) * 0.5
            ground_y2 = pts2[:, 3]

            new_ground_x = (ground_x1 + ground_x2) * 0.5
            new_ground_y = (ground_y1 + ground_y2) * 0.5
            new_w = (w1 + w2) * 0.5
            new_h = (h1 + h2) * 0.5

            new_x1 = new_ground_x - new_w * 0.5
            new_y1 = new_ground_y - new_h
            new_x2 = new_ground_x + new_w * 0.5
            new_y2 = new_ground_y

            center_prompts = torch.stack([new_x1, new_y1, new_x2, new_y2], dim=1).reshape(bbox_num, bbox_num, 4)
        elif self.prompt_mode == 'point':
            ground_x1 = pts1[:, 0]
            ground_y1 = pts1[:, 1]

            ground_x2 = pts2[:, 0]
            ground_y2 = pts2[:, 1]

            new_ground_x = (ground_x1 + ground_x2) * 0.5
            new_ground_y = (ground_y1 + ground_y2) * 0.5

            center_prompts = torch.stack([new_ground_x, new_ground_y], dim=1).reshape(bbox_num, bbox_num, 2)
        return center_prompts

    def get_crop_item(self, item, anchor_view_id, sample_view_id):
        anchor_img_path = self.img_dict[self.view_ls[anchor_view_id]][item]
        anchor_anno = self.read_anno(anchor_img_path)
        anchor_key = anchor_img_path.rsplit('/', 1)[1].rsplit('.', 1)[0]
        anchor_info = self.load_cropped_bboxes(self.img_dict[self.view_ls[anchor_view_id]][item], anchor_anno, anchor_view_id, anchor_key)

        pos_img_path = self.img_dict[self.view_ls[sample_view_id]][item]
        pos_anno = self.read_anno(pos_img_path)
        pos_key = pos_img_path.rsplit('/', 1)[1].rsplit('.', 1)[0]
        pos_info = self.load_cropped_bboxes(self.img_dict[self.view_ls[sample_view_id]][item], pos_anno, sample_view_id, pos_key)

        anchor_info['cam_id'] = anchor_view_id
        anchor_info['image_path'] = anchor_img_path
        pos_info['cam_id'] = sample_view_id
        pos_info['image_path'] = pos_img_path

        if self.mode == 'train' and self.train_mode == 'triplet':
            sample_min, sample_max = self.sampling_range
            interval = sample_min + int(random.random() * (sample_max - sample_min))
            previous_sample = False
            if random.random() > 0.5:
                previous_sample = True
            if previous_sample:
                if item - interval >= 0:
                    new_item = item - interval
                else:
                    new_item = item + interval
            else:
                if item + interval < len(self.img_dict[self.view_ls[sample_view_id]]):
                    new_item = item + interval
                else:
                    new_item = item - interval
            
            neg_img_path = self.img_dict[self.view_ls[sample_view_id]][new_item]
            neg_anno = self.read_anno(neg_img_path)
            neg_key = neg_img_path.rsplit('/', 1)[1].rsplit('.', 1)[0]
            if self.pseudo_neg:
                neg_info = self.load_cropped_bboxes(self.img_dict[self.view_ls[sample_view_id]][item], pos_anno, sample_view_id, pos_key, bbox_jitter=self.pseudo_neg)
                # neg_info = self.load_cropped_bboxes(self.img_dict[self.view_ls[sample_view_id]][new_item], neg_anno, sample_view_id, neg_key, bbox_jitter=self.pseudo_neg)
            else:
                neg_info = self.load_cropped_bboxes(self.img_dict[self.view_ls[sample_view_id]][new_item], neg_anno, sample_view_id, neg_key)

            neg_info['cam_id'] = sample_view_id
            neg_info['image_path'] = neg_img_path
            if self.mode == 'train' and self.edge_association:
                anchor_center_bboxes = self.load_edges(anchor_info['bboxes'])
                anchor_info['center_bboxes'] = anchor_center_bboxes
                pos_center_bboxes = self.load_edges(pos_info['bboxes'])
                pos_info['center_bboxes'] = pos_center_bboxes
                neg_center_bboxes = self.load_edges(neg_info['bboxes'])
                neg_info['center_bboxes'] = neg_center_bboxes
            return anchor_info, pos_info, neg_info
        else:
            if self.mode == 'train' and self.edge_association:
                anchor_center_bboxes = self.load_edges(anchor_info['bboxes'])
                anchor_info['center_bboxes'] = anchor_center_bboxes
                pos_center_bboxes = self.load_edges(pos_info['bboxes'])
                pos_info['center_bboxes'] = pos_center_bboxes
            return anchor_info, pos_info
    

    def load_image_bboxes(self, frame_img, bbox_dict, c_id, img_size=(224, 224)):
        img = self.load_image(frame_img, img_size)
        label_ls = []
        bbox_ls = []
        for key in bbox_dict:
            bbox = bbox_dict[key]
            if self.prompt_mode == 'box':
                bbox[0] /= self.widths[c_id]
                bbox[1] /= self.heights[c_id]
                bbox[2] /= self.widths[c_id]
                bbox[3] /= self.heights[c_id]
            elif self.prompt_mode == 'point':
                bbox[0] /= self.widths[c_id]
                bbox[1] /= self.heights[c_id]
            # bbox = [0 if i < 0 else i for i in bbox]
            # bbox = [1 if i > 1 else i for i in bbox]
            bbox = torch.tensor(bbox, dtype=torch.float32)
            label_ls.append(key)
            bbox_ls.append(bbox)

        info = {
            'img': img, 
            'labels': label_ls, 
            'bboxes': torch.stack(bbox_ls) if len(bbox_ls) else [], 
        }
        return info

    
    def get_association_item(self, item):
        infos = []
        for cam_id, view in enumerate(self.view_ls):
            frame_img = self.img_dict[view][item]
            anno = self.read_anno(frame_img)
            image_key = frame_img.rsplit('/', 1)[1].rsplit('.', 1)[0]
            if self.crop_box:
                info = self.load_cropped_bboxes(frame_img, anno, cam_id, image_key)
            else:
                info = self.load_image_bboxes(frame_img, anno, cam_id)

            info['cam_id'] = cam_id
            info['image_path'] = frame_img
            info['image_key'] = image_key
            infos.append(info)

        return infos
