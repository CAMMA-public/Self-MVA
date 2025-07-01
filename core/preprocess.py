import os
import json
import random
import pandas as pd
import csv
import cv2
import tqdm
from tqdm import trange
import argparse
import numpy as np

DATASET_NAME = ''

class BasePreprocess:

    def __init__(self,
                 dataset_name: str,
                 base_dir: str,
                 eval_ratio: float,
                 test_ratio: float,
                 valid_frames_range=None,
                 output_dir=None,
                 image_format='jpg',
                 random_seed=202204):
        self.dataset_name = dataset_name
        self.base_dir = base_dir
        self.dataset_path = os.path.join(base_dir, dataset_name)
        self.eval_ratio = eval_ratio
        self.test_ratio = test_ratio
        self.valid_frames_range = valid_frames_range
        self.image_format = image_format
        if output_dir is None:
            self.output_path = os.path.join(self.dataset_path, 'output')
        else:
            self.output_path = output_dir
        self.frames_output_path = os.path.join(self.output_path, 'frames')
        self.frames = {}

        random.seed(random_seed)

    def process(self):
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.frames_output_path, exist_ok=True)

        if DATASET_NAME == 'MVOR':
            self.prepare_data()
        self.load_annotations()
        self.filter_frames()
        if DATASET_NAME == 'Wildtrack':
            self.load_frames()
        self.train_test_split()

    def load_annotations(self):
        """
        Parse annotation files that across all cameras to obtain frames(self).
        Within data format:
            [frame_id]: (top_left_x, top_left_y, width, height, track_id, camera_id)
        """
        raise NotImplementedError

    def filter_frames(self):
        # Filter out frames that the person only appear in one camera view.
        invalid_frames_id = []
        for frame_id, frame in self.frames.items():
            cams = set([sample[-1] for sample in frame])
            if len(cams) <= 1:
                invalid_frames_id.append(frame_id)

        for frame_id in invalid_frames_id:
            self.frames.pop(frame_id)

    def select_frames(self, frames_id: list):
        ret = {}
        for frame_id in frames_id:
            ret[frame_id] = self.frames[frame_id]
        return ret

    def save_json(self, obj, name):
        with open(os.path.join(self.output_path, f'{name}.json'), 'w') as fp:
            json.dump(obj, fp)

    def train_test_split(self):
        frames_id = list(sorted(self.frames.keys()))
        # random.shuffle(frames_id)
        n = len(frames_id)
        n_test = int(self.test_ratio * n)
        n_eval = int(self.eval_ratio * (n - n_test))
        test_frames_id = frames_id[-n_test:]
        rest_frames_id = frames_id[:-n_test]
        eval_frames_id = rest_frames_id[-n_eval:]
        train_frames_id = rest_frames_id[:-n_eval]

        train_frames = self.select_frames(train_frames_id)
        eval_frames = self.select_frames(eval_frames_id)
        test_frames = self.select_frames(test_frames_id)

        sorted_train_frames = dict(sorted(train_frames.items()))
        sorted_eval_frames = dict(sorted(eval_frames.items()))
        sorted_test_frames = dict(sorted(test_frames.items()))

        self.save_json(sorted_train_frames, 'gt_train')
        self.save_json(sorted_eval_frames, 'gt_eval')
        self.save_json(sorted_test_frames, 'gt_test')


class WildtrackPreprocess(BasePreprocess):

    def load_annotations(self):
        """
        Parse annotation files that across all cameras to obtain frames(self).
        Within data format:
            [frame_id]: (top_left_x, top_left_y, width, height, track_id, camera_id)
        """
        for frame_id in range(self.valid_frames_range[0], self.valid_frames_range[1]):
            anno_file = f'{self.base_dir}/{self.dataset_name}/src/annotations_positions/0000{frame_id*5:04d}.json'
            f = open(anno_file)
            data = json.load(f)
            f.close()
            for raw in data:
                track_id = raw['personID']
                for cam_id in range(len(raw['views'])):
                    if raw['views'][cam_id]['xmin'] == -1:
                        continue
                    x_min = max(raw['views'][cam_id]['xmin'], 0) # prevent negative value
                    y_min = max(raw['views'][cam_id]['ymin'], 0) # prevent negative value
                    width = raw['views'][cam_id]['xmax'] - x_min
                    height = raw['views'][cam_id]['ymax'] - y_min
                    self.frames.setdefault(frame_id, []).append(
                        (x_min, y_min, width, height, track_id, cam_id)
                    )

    def load_frames(self):
        # Copy from original dataset
        # 0000~1995 -> 0~400
        if not os.path.exists(self.frames_output_path):
            os.mkdir(self.frames_output_path)
        for cam_id in range(7):
            path = os.path.join(self.base_dir, f'{self.dataset_name}/src', 'Image_subsets', f'C{cam_id+1}')
            for frame_id in range(self.valid_frames_range[0], self.valid_frames_range[1]):
                img = cv2.imread(os.path.join(path, f'0000{frame_id*5:04d}.png'))
                cv2.imwrite(os.path.join(self.frames_output_path,
                                            f'{frame_id}_{cam_id}.{self.image_format}'), img)


class MVORPreprocess(BasePreprocess):

    def create_index(self, camma_mvor_gt):
        """
        get the 2D and 3D annotations for each image from the coco style annotations
        :param camma_mvor_gt: ground truth dict
        :return: 2D and 3D annotations dictionary; key=image_id, value = 2D or 3D annotations
        """
        im_ids_2d = [p['id'] for p in camma_mvor_gt['images']]
        im_ids_3d = [p['id'] for p in camma_mvor_gt['multiview_images']]
        mv3d_paths = {p["id"]:p["images"] for p in camma_mvor_gt['multiview_images']}
        imid_to_path = {p["id"]:p["file_name"] for p in camma_mvor_gt['images']}

        anns_2d = {str(key): [] for key in im_ids_2d}
        anns_3d = {str(key): [] for key in im_ids_3d}
        print('creating index for 2D annotations')
        for ann in camma_mvor_gt['annotations']:
            anns_2d[str(ann['image_id'])].append({
                'keypoints': ann['keypoints'],
                'bbox': ann['bbox'],
                'person_id': ann['person_id'],
                'person_role': ann['person_role'],
                'only_bbox': ann['only_bbox'],
                'id': ann['id']
            })
        print('done')
        print('creating index for 3D annotations')
        for ann3d in camma_mvor_gt['annotations3D']:
            anns_3d[str(ann3d['image_ids'])].append({
                'id': ann3d['id'],
                'person_id': ann3d['person_id'],
                'ref_camera': ann3d['ref_camera'],
                'keypoints3D': ann3d['keypoints3D']
            })
        print('Index creation done')

        # anno3d is the 3d coordinates in the 1st view
        # tr_mat @ anno3d = world_3d
        # inv(tr_mat0) @ anno3d = cam0_3d
        # Therefore: inv(tr_mat0) @ inv(tr_mat) @ world_3d = cam0_3d
        # extrinsics_cam0 = inv(tr_mat0) @ inv(tr_mat)
        tr_mat = np.array(camma_mvor_gt['cameras_info']['camParams']['firstCamToRoomRef']).reshape((4, 4))
        tr_mat0 = np.array(camma_mvor_gt['cameras_info']['camParams']['extrinsics'][0]).reshape((4, 4))
        tr_mat1 = np.array(camma_mvor_gt['cameras_info']['camParams']['extrinsics'][1]).reshape((4, 4))
        tr_mat2 = np.array(camma_mvor_gt['cameras_info']['camParams']['extrinsics'][2]).reshape((4, 4))
        extrinsics_cam0 = np.linalg.inv(tr_mat0) @ np.linalg.inv(tr_mat)
        extrinsics_cam1 = np.linalg.inv(tr_mat1) @ np.linalg.inv(tr_mat)
        extrinsics_cam2 = np.linalg.inv(tr_mat2) @ np.linalg.inv(tr_mat)

        intrinsics_list = []
        for intrinsic_info in camma_mvor_gt['cameras_info']['camParams']['intrinsics']:
            intrinsics = np.zeros((3, 3), dtype=float)
            focal = intrinsic_info['focallength']
            pp = intrinsic_info['principalpoint']
            intrinsics[0][0] = focal[0]
            intrinsics[1][1] = focal[1]
            intrinsics[0][2] = pp[0]
            intrinsics[1][2] = pp[1]
            intrinsics[2][2] = 1.
            intrinsics_list.append(intrinsics)
        
        homography = []
        H_cam0 = np.delete(intrinsics_list[0] @ extrinsics_cam0[:3], 2, axis=1)
        H_cam1 = np.delete(intrinsics_list[1] @ extrinsics_cam1[:3], 2, axis=1)
        H_cam2 = np.delete(intrinsics_list[2] @ extrinsics_cam2[:3], 2, axis=1)
        homography = [H_cam0.tolist(), H_cam1.tolist(), H_cam2.tolist()]

        return anns_2d, anns_3d, mv3d_paths, imid_to_path, homography

    def prepare_data(self):
        GT_ANNO_PATH = os.path.join(self.base_dir, 'camma_mvor_2018_v2.json')
        camma_mvor_gt = json.load(open(GT_ANNO_PATH))
        anno_2d, anno_3d, mv_paths, imid_to_path, homography = self.create_index(camma_mvor_gt)
        self.generate_data(anno_2d, anno_3d, mv_paths, imid_to_path)
    
    def generate_data(self, anno_2d, anno_3d, mv_paths, imid_to_path):
        width = 640
        height = 480

        save_json_dir = os.path.join(self.dataset_path, 'tracking_annotations')
        save_img_dir = os.path.join(self.dataset_path, 'output', 'frames')
        os.makedirs(save_json_dir, exist_ok=True)
        os.makedirs(save_img_dir, exist_ok=True)

        frame_idx = 0
        for key, value in mv_paths.items():
            anno = {}
            for img_info in value:
                img_path = os.path.join(self.base_dir, 'camma_mvor_dataset', img_info['file_name'])
                cam_id = img_info['cam_id']
                save_img_name = str(frame_idx) + '_' + str(cam_id - 1) + '.jpg'
                save_img_path = os.path.join(save_img_dir, save_img_name)
                img = cv2.imread(img_path)
                cv2.imwrite(save_img_path, img)

                img_id = img_info['id']
                img_annos = anno_2d[str(img_id)]
                for img_anno in img_annos:
                    bbox = img_anno['bbox'] # [x1, y1, w, h]
                    person_id = img_anno['person_id']
                    if person_id < 0: # skip the ones that don't have 3d keypoints
                        continue
                    if person_id not in anno:
                        anno[person_id] = {}
                        anno[person_id]['personID'] = person_id
                        anno[person_id]['views'] = []
                    view_dict = {}
                    view_dict['viewNum'] = cam_id - 1
                    view_dict['xmax'] = min(bbox[0] + bbox[2], width)
                    view_dict['xmin'] = max(bbox[0], 0)
                    view_dict['ymax'] = min(bbox[1] + bbox[3], height)
                    view_dict['ymin'] = max(bbox[1], 0)
                    anno[person_id]['views'].append(view_dict)
            anno_keys = list(anno.keys())
            save_anno = []
            for anno_key in anno_keys:
                save_anno.append(anno[anno_key])
        
            json_anno = json.dumps(save_anno, indent=4)
            save_json_path = os.path.join(save_json_dir, f'{frame_idx:06d}.json')
            with open(save_json_path, "w") as outfile:
                outfile.write(json_anno)

            frame_idx += 1

    def load_annotations(self):
        """
        Parse annotation files that across all cameras to obtain frames(self).
        Within data format:
            [frame_id]: (top_left_x, top_left_y, width, height, track_id, camera_id)
        """
        for frame_id in range(self.valid_frames_range[0], self.valid_frames_range[1]):
            anno_file = f'{self.dataset_path}/tracking_annotations/{frame_id:06d}.json'
            f = open(anno_file)
            data = json.load(f)
            f.close()
            for raw in data:
                track_id = raw['personID']
                for i in range(len(raw['views'])):
                    if raw['views'][i]['xmin'] == -1:
                        continue
                    x_min = max(raw['views'][i]['xmin'], 0) # prevent negative value
                    y_min = max(raw['views'][i]['ymin'], 0) # prevent negative value
                    width = raw['views'][i]['xmax'] - x_min
                    height = raw['views'][i]['ymax'] - y_min
                    cam_id = raw['views'][i]["viewNum"]
                    if width == 0 or height == 0:
                        print((x_min, y_min, width, height, track_id, cam_id))
                    self.frames.setdefault(frame_id, []).append(
                        (x_min, y_min, width, height, track_id, cam_id)
                    )


def preprocess(dataset_dir, dataset_name, PreProcess, valid_frames_range, output_dir=None):
    dataset = PreProcess(
        dataset_name,
        dataset_dir,
        0.2,
        0.1,
        valid_frames_range,
        output_dir=output_dir
    )
    dataset.process()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Wildtrack", help="pre-process which dataset", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args.dataset

if __name__ == '__main__':
    DATASET_NAME = parse_args()
    if DATASET_NAME not in ['Wildtrack', 'MVOR']:
        print('Please enter valid dataset.')
    else:
        print(f'Pre-processing {DATASET_NAME}...')

        if DATASET_NAME == 'Wildtrack':
            preprocess(dataset_dir=f'./data/{DATASET_NAME}', dataset_name='sequence1', valid_frames_range=[0,400], PreProcess=WildtrackPreprocess)
        elif DATASET_NAME == 'MVOR':
            preprocess(dataset_dir=f'./data/{DATASET_NAME}', dataset_name='sequence1', valid_frames_range=[0,732], PreProcess=MVORPreprocess)
        else:
            raise ValueError(f'Dataset {DATASET_NAME} is not implemented! ')
