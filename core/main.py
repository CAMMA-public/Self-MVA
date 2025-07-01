import os
from config import config, update_config
os.environ["CUDA_VISIBLE_DEVICES"] = config.TRAIN_GPUS
import torch
from torch.utils.data import DataLoader
from dataset import Loader
from model import Multi_View_Predictor_prompt
import logging
import time
from utils import *
from scipy.optimize import linear_sum_assignment
from losses import *
import numpy as np
from sklearn.metrics import average_precision_score
import argparse
import pprint
from torchreid.utils import FeatureExtractor

logger = logging.getLogger(__name__)


def train_p3de(config, epoch, model, dataloader_train, optimizer, device, fully_supervised=False, thresh=0.0, reid_model=None):
    # alpha: the weight of appearance distance
    alpha = config.DATASET.ALPHA

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    epoch_loss = 0
    for step_i, data in enumerate(dataloader_train):
        data_time.update(time.time() - end)

        optimizer.zero_grad()

        # fully supervised training for ablation study
        if fully_supervised:
            anchor_info = data[0]
            pos_info = data[1]
            anchor_labels = anchor_info['labels']
            anchor_bboxes = anchor_info['bboxes']
            pos_labels = pos_info['labels']
            pos_bboxes = pos_info['bboxes']
            # skip the frames with no detections
            if len(anchor_bboxes) == 0 or len(pos_bboxes) == 0:
                continue
            anchor_prompts = anchor_bboxes.squeeze(0)
            pos_prompts = pos_bboxes.squeeze(0)
            anchor_prompts = anchor_prompts.to(device)
            pos_prompts = pos_prompts.to(device)

            labels1 = torch.cat(anchor_labels, dim=0)
            labels1 = labels1.to(device)
            labels2 = torch.cat(pos_labels, dim=0)
            labels2 = labels2.to(device)

            anchor_view_id = anchor_info['cam_id']
            sample_view_id = pos_info['cam_id']

            anchor_h = anchor_info['height'][0].item()
            anchor_w = anchor_info['width'][0].item()
            pos_h = pos_info['height'][0].item()
            pos_w = pos_info['width'][0].item()

            anchor_out = model.module.encode_decode(anchor_view_id, (anchor_h, anchor_w), anchor_prompts)
            pos_out = model.module.encode_decode(sample_view_id, (pos_h, pos_w), pos_prompts)

            loss = fully_supervised_triplet_loss(anchor_out, pos_out, labels1, labels2, anchor_prompts, pos_prompts, anchor_view_id, sample_view_id)
            if loss == -1:
                continue

        else:
            anchor_info, pos_info, neg_info = data
            anchor_bboxes = anchor_info['bboxes']
            pos_bboxes = pos_info['bboxes']
            neg_bboxes = neg_info['bboxes']
            # skip the frames with no detections
            if len(anchor_bboxes) == 0 or len(pos_bboxes) == 0 or len(neg_bboxes) == 0:
                continue
            
            # edge association preparation
            use_edge_loss = False
            if 'center_bboxes' in anchor_info:
                anchor_center_bboxes = anchor_info['center_bboxes']
                pos_center_bboxes = pos_info['center_bboxes']
                neg_center_bboxes = neg_info['center_bboxes']
                if len(anchor_center_bboxes) and len(pos_center_bboxes) and len(neg_center_bboxes):
                    anchor_center_prompts = anchor_center_bboxes.squeeze(0)
                    pos_center_prompts = pos_center_bboxes.squeeze(0)
                    neg_center_prompts = neg_center_bboxes.squeeze(0)
                    anchor_center_prompts = anchor_center_prompts.to(device)
                    pos_center_prompts = pos_center_prompts.to(device)
                    neg_center_prompts = neg_center_prompts.to(device)
                    use_edge_loss = True

            anchor_prompts = anchor_bboxes.squeeze(0)
            pos_prompts = pos_bboxes.squeeze(0)
            neg_prompts = neg_bboxes.squeeze(0)
            anchor_prompts = anchor_prompts.to(device)
            pos_prompts = pos_prompts.to(device)
            neg_prompts = neg_prompts.to(device)

            if 'cropped_imgs' in anchor_info:
                anchor_imgs = anchor_info['cropped_imgs']
                pos_imgs = pos_info['cropped_imgs']
                neg_imgs = neg_info['cropped_imgs']
                anchor_imgs = anchor_imgs.to(device)
                pos_imgs = pos_imgs.to(device)
                neg_imgs = neg_imgs.to(device)
                anchor_imgs_input = anchor_imgs.squeeze(0)
                pos_imgs_input = pos_imgs.squeeze(0)
                neg_imgs_input = neg_imgs.squeeze(0)
            
            anchor_view_id = anchor_info['cam_id']
            sample_view_id = pos_info['cam_id']

            anchor_h = anchor_info['height'][0].item()
            anchor_w = anchor_info['width'][0].item()
            pos_h = pos_info['height'][0].item()
            pos_w = pos_info['width'][0].item()
            neg_h = neg_info['height'][0].item()
            neg_w = neg_info['width'][0].item()

            anchor_fea, anchor_preds = model.module.encode_decode(anchor_view_id, (anchor_h, anchor_w), anchor_prompts)
            pos_fea, pos_preds = model.module.encode_decode(sample_view_id, (pos_h, pos_w), pos_prompts)
            neg_fea, neg_preds = model.module.encode_decode(sample_view_id, (neg_h, neg_w), neg_prompts)
                
            # extract person Re-ID feature using trained OSNet
            if reid_model is not None:
                with torch.no_grad():
                    anchor_reid = reid_model(anchor_imgs_input)
                    pos_reid = reid_model(pos_imgs_input)
                    neg_reid = reid_model(neg_imgs_input)
                pos_dis, pos_matches = get_matching_dis(anchor_fea, pos_fea, reid_fea1=anchor_reid, reid_fea2=pos_reid, thresh=thresh, alpha=alpha)
                neg_dis, neg_matches = get_matching_dis(anchor_fea, neg_fea, reid_fea1=anchor_reid, reid_fea2=neg_reid, thresh=thresh, alpha=alpha)
            else:
                # pass
                pos_dis, pos_matches = get_matching_dis(anchor_fea, pos_fea, thresh=thresh)
                neg_dis, neg_matches = get_matching_dis(anchor_fea, neg_fea, thresh=thresh)

            M = 1.0
            # node-based triplet loss
            triplet_loss = torch.nn.functional.relu(pos_dis - neg_dis + M).mean()
            # re-projection constraint
            anchor_loss1 = get_dis(anchor_preds[:, anchor_view_id[0]], anchor_prompts, mode='l1') 
            pos_loss2 = get_dis(pos_preds[:, sample_view_id[0]], pos_prompts, mode='l1')
            loss = (anchor_loss1 + pos_loss2) + triplet_loss

            # edge-based triplet loss
            if use_edge_loss:
                # given obtained node matches, find the edge association
                pts1, pts2 = get_center_prompts(anchor_center_prompts, pos_center_prompts, pos_matches[0], pos_matches[1])
                pts3, pts4 = get_center_prompts(anchor_center_prompts, neg_center_prompts, neg_matches[0], neg_matches[1])
                center_triplet_loss = 0
                if len(pts1) and len(pts2) and len(pts3) and len(pts4):
                    pts1_fea, pts1_preds = model.module.encode_decode(anchor_view_id, (anchor_h, anchor_w), pts1)
                    pts2_fea, pts2_preds = model.module.encode_decode(sample_view_id, (pos_h, pos_w), pts2)
                    center_pos_dis = torch.sqrt((pts1_fea - pts2_fea).pow(2).sum(-1)).mean(-1)

                    pts3_fea, pts3_preds = model.module.encode_decode(anchor_view_id, (anchor_h, anchor_w), pts3)
                    pts4_fea, pts4_preds = model.module.encode_decode(sample_view_id, (neg_h, neg_w), pts4)
                    center_neg_dis = torch.sqrt((pts3_fea - pts4_fea).pow(2).sum(-1)).mean(-1)

                    center_triplet_loss = torch.nn.functional.relu(center_pos_dis - center_neg_dis + M).mean()

                    loss = loss + center_triplet_loss
        
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        losses.update(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()

        if step_i % config.PRINT_FREQ == 0:
            gpu_memory_usage = torch.cuda.memory_allocated(0)
            msg = (
                "Epoch: [{0}][{1}/{2}]\t"
                "Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t"
                "Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t"
                "Loss: {loss.val:.6f} ({loss.avg:.6f})\t"
                "Memory {memory:.1f}".format(
                    epoch,
                    step_i,
                    len(dataloader_train),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    memory=gpu_memory_usage,
                )
            )
            logger.info(msg)

    return epoch_loss / len(dataloader_train)


def evaluate_p3de(config, model, dataloader_test, device, fully_supervised=False, thresh=0.0, fix_view_id=-1, reid_model=None, vis=False):
    alpha = config.DATASET.ALPHA
    
    correct_sum = 0
    precision_total_sum = 0
    recall_total_sum = 0
    scores = []
    labels = []
    IPAA_correct_sum = 0
    IPAA_total_sum = 0
    IPAA_80_sum = 0
    IPAA_90_sum = 0
    IPAA_100_sum = 0
    IPAA_img_sum = 0

    if vis:
        # creating a visualization folder to save images
        this_dir = Path(os.path.dirname(__file__))
        vis_output_dir = (this_dir / '..' / config.OUTPUT_DIR / 'test_vis').resolve()
        if not vis_output_dir.exists():
            print('=> creating {}'.format(vis_output_dir))
            vis_output_dir.mkdir()
    
    for step_i, data in enumerate(dataloader_test):
        view_num = len(data)
        feas = []
        if reid_model is not None:
            reid_feas = []
        for i in range(view_num):
            info = data[i]
            bboxes = info['bboxes']
            if len(bboxes) == 0:
                feas.append([])
                if reid_model is not None:
                    reid_feas.append([])
                continue
            prompts = bboxes.squeeze(0)
            prompts = prompts.to(device)
            if 'cropped_imgs' in info:
                imgs = info['cropped_imgs']
                imgs = imgs.to(device)
                imgs_input = imgs.squeeze(0)
            view_id = info['cam_id']
            
            h = info['height'][0].item()
            w = info['width'][0].item()
            fea, preds = model.module.encode_decode(view_id, (h, w), prompts)

            feas.append(fea)
            if reid_model is not None:
                reid_feature = reid_model(imgs_input)
                reid_feas.append(reid_feature)
        # fix an anchor view for association in case that we have one view that captures the whole scene
        if fix_view_id == -1:
            cam_pairs = []
            for i in range(view_num - 1):
                for j in range(i + 1, view_num):
                    cam_pairs.append((i, j))
        else:
            cam_pairs = []
            for j in range(view_num):
                if j == fix_view_id:
                    continue
                cam_pairs.append((fix_view_id, j))

        for cam_pair in cam_pairs:
            view1, view2 = cam_pair
            anchor_fea = feas[view1]
            pos_fea = feas[view2]
            if len(anchor_fea) == 0 or len(pos_fea) == 0:
                continue
            anchor_info = data[view1]
            pos_info = data[view2]
            anchor_labels = anchor_info['labels']
            pos_labels = pos_info['labels']

            labels1 = torch.cat(anchor_labels, dim=0)
            labels1 = labels1.to(device)
            labels2 = torch.cat(pos_labels, dim=0)
            labels2 = labels2.to(device)

            if reid_model is not None:
                anchor_reid_fea = reid_feas[view1]
                pos_reid_fea = reid_feas[view2]
                result = cross_view_matching_evaluation(anchor_fea, pos_fea, labels1, labels2, reid_fea1=anchor_reid_fea, reid_fea2=pos_reid_fea, mode='l2', thresh=thresh, alpha=alpha)
            else:
                result = cross_view_matching_evaluation(anchor_fea, pos_fea, labels1, labels2, mode='l2', thresh=thresh)

            correct_sum += result['correct']
            precision_total_sum += result['precision_total']
            recall_total_sum += result['recall_total']
            scores.append(result['scores'])
            labels.append(result['labels'])
            IPAA_correct_sum += result['IPAA_correct']
            IPAA_total_sum += result['IPAA_total']
            ratio = result['IPAA_correct'] / result['IPAA_total']
            IPAA_100_sum += (ratio == 1.)
            IPAA_90_sum += (ratio >= 0.9)
            IPAA_80_sum += (ratio >= 0.8)
            IPAA_img_sum += 1

            if vis:
                anchor_img_path = anchor_info['image_path'][0]
                pos_img_path = pos_info['image_path'][0]
                true_or_false = result['true_or_false']
                matches_x = result['matches_x']
                matches_y = result['matches_y']
                # matches_scores = result['matches_scores']
                anchor_bboxes = anchor_info['bboxes']
                pos_bboxes = pos_info['bboxes']
                anchor_bboxes = anchor_bboxes.squeeze(0)
                pos_bboxes = pos_bboxes.squeeze(0)

                anchor_name = anchor_img_path.rsplit('/', 1)[1].rsplit('.', 1)[0]
                save_path = os.path.join(str(vis_output_dir), anchor_name + '_' + str(view1) + str(view2))
                # visualize(anchor_img_path, save_path, true_or_false, matches_x, anchor_bboxes, scores=matches_scores)
                # visualize(anchor_img_path, save_path, true_or_false, matches_x, anchor_bboxes)
                visualize_highlights(anchor_img_path, save_path, true_or_false, matches_x, anchor_bboxes)

                pos_name = pos_img_path.rsplit('/', 1)[1].rsplit('.', 1)[0]
                save_path = os.path.join(str(vis_output_dir), pos_name + '_' + str(view1) + str(view2))
                # visualize(pos_img_path, save_path, true_or_false, matches_y, pos_bboxes, scores=matches_scores)
                # visualize(pos_img_path, save_path, true_or_false, matches_y, pos_bboxes)
                visualize_highlights(pos_img_path, save_path, true_or_false, matches_y, pos_bboxes)
    
    if precision_total_sum:
        precision = correct_sum / precision_total_sum
    else:
        precision = 0
    recall = correct_sum / recall_total_sum

    labels = np.concatenate(labels)
    scores = np.concatenate(scores)
    ap = average_precision_score(labels, scores)
    fpr_95 = FPR_95(labels, scores)
    match_acc = IPAA_correct_sum / IPAA_total_sum
    IPAA_100 = IPAA_100_sum / IPAA_img_sum
    IPAA_90 = IPAA_90_sum / IPAA_img_sum
    IPAA_80 = IPAA_80_sum / IPAA_img_sum

    msg = (
        "Average Precision: {0:.4f}\t"
        "FPR-95: {1:.4f}\t"
        "Precision: {2:.4f}\t"
        "Recall: {3:.4f}\t"
        "Accuracy: {4:.4f}\t"
        "IPAA 100: {5:.4f}\t"
        "IPAA 90: {6:.4f}\t"
        "IPAA 80: {7:.4f}".format(
            ap,
            fpr_95,
            precision,
            recall,
            match_acc,
            IPAA_100,
            IPAA_90,
            IPAA_80
        )
    )
    logger.info(msg)


def parse_args():
    parser = argparse.ArgumentParser(description="Train keypoints network")
    parser.add_argument("--cfg", help="experiment configure file name", required=True, type=str)
    parser.add_argument("--test", help="test mode", action='store_true', default=False)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    return args


def train():
    gpus = [int(i) for i in config.GPUS.split(",")]
    device = torch.device('cuda')
    fully_supervised = config.TRAIN.FULLY_SUPERVISED
    thresh = config.TRAIN.THRESH
    batch_size = config.TRAIN.BATCH_SIZE
    fix_view_id = config.FIX_VIEW_ID
    reid = config.TRAIN.REID

    dataset_train = Loader(config=config, mode='train')
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, num_workers=6, pin_memory=True, shuffle=True)

    dataset_valid = Loader(config=config, mode='valid', inference=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, num_workers=6, pin_memory=True, shuffle=False)

    view_num = dataset_train.get_view_num()

    model = Multi_View_Predictor_prompt(view_num, mode=config.DATASET.PROMPT_MODE)
    
    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    
    reid_model = None
    if reid:
        reid_model = FeatureExtractor(
            model_name='osnet_ain_x1_0',
            model_path='weights/osnet_ain_ms_d_c.pth.tar',
            device='cuda'
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAIN.LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    start_epoch = config.TRAIN.BEGIN_EPOCH
    end_epoch = config.TRAIN.END_EPOCH
    last_epoch = -1
    # resume the training
    if config.TRAIN.RESUME:
        checkpoint_path = config.TRAIN.CKP_PATH
        if len(checkpoint_path):
            ckp = torch.load(checkpoint_path)
            model.module.load_state_dict(ckp['model'])
        else:
            checkpoint_path = os.path.join(final_output_dir, 'checkpoint.pth.tar')
            ckp = torch.load(checkpoint_path)
            model.module.load_state_dict(ckp['model'])
            optimizer.load_state_dict(ckp['optimizer'])
            last_epoch = ckp['epoch']
            start_epoch = last_epoch

    max_loss = 1e8

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR, last_epoch=last_epoch)

    for epoch_i in range(start_epoch, end_epoch):
        logger.info("learning rate for this epoch {}".format(lr_scheduler.get_last_lr()))
        model.train()
        epoch_loss = train_p3de(config, epoch_i, model, dataloader_train, optimizer, device, fully_supervised, thresh, reid_model)
        model.eval()
        lr_scheduler.step()
        logger.info('Epoch: {}'.format(epoch_i))
        logger.info('Epoch loss: {}'.format(epoch_loss))
        is_best = False
        if epoch_loss < max_loss:
            is_best = True
            with torch.no_grad():
                evaluate_p3de(config, model, dataloader_valid, device, fully_supervised, thresh, fix_view_id, reid_model)
            max_loss = epoch_loss

        save_checkpoint(
            {
                "epoch": epoch_i,
                'loss': epoch_loss,
                'optimizer': optimizer.state_dict(),
                'model': model.module.state_dict()
            },
            is_best,
            final_output_dir,
        )
        logger.info("=> saving checkpoint to {} (Best: {})".format(final_output_dir, is_best))


def test():
    gpus = [int(i) for i in config.GPUS.split(",")]
    device = torch.device('cuda')
    fully_supervised = config.TRAIN.FULLY_SUPERVISED
    thresh = config.TEST.THRESH
    batch_size = config.TRAIN.BATCH_SIZE
    fix_view_id = config.FIX_VIEW_ID
    reid = config.TEST.REID
    vis = config.TEST.VIS

    dataset_test = Loader(config=config, mode='test', inference=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, num_workers=6, pin_memory=True, shuffle=False)

    view_num = dataset_test.get_view_num()

    model = Multi_View_Predictor_prompt(view_num, mode=config.DATASET.PROMPT_MODE)
    
    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    checkpoint_path = config.TEST.CKP_PATH
    ckp = torch.load(checkpoint_path)
    model.module.load_state_dict(ckp['model'])

    reid_model = None
    if reid:
        reid_model = FeatureExtractor(
            model_name='osnet_ain_x1_0',
            model_path='weights/osnet_ain_ms_d_c.pth.tar',
            device='cuda'
        )

    model.eval()
    with torch.no_grad():
        evaluate_p3de(config, model, dataloader_test, device, fully_supervised, thresh, fix_view_id, reid_model, vis)


if __name__ == '__main__':
    args = parse_args()
    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, "train")
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    if args.test:
        test()
    else:
        train()