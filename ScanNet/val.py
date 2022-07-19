import os
import sys
import math
import argparse
from datetime import datetime
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(TASK_DIR)
sys.path.append(ROOT_DIR)

import pccnn_lib
import models
from dataset import ScanNetDataSet

current_milli_time = lambda: time.time() * 1000.0

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Validation ScanNet')
    parser.add_argument('--model_path', default='model.pth',  help='Path to the saved model (default: model.pth)')
    args = parser.parse_args()
        
    # Validation config
    rooms_per_batch = 2
    inner_radius = 1.0
    outer_radius = 2.0
    num_epochs = 10
    num_val_steps_per_epoch = 8000//rooms_per_batch

    # Load model params.
    model_config_dict = torch.load(args.model_path)

    # Create the model.
    if model_config_dict['conv_type']  == "mcconv":
        layer_factory = pccnn_lib.pc.layers.MCConvFactory(16, False, 1, 1.0)
    elif model_config_dict['conv_type']  == "kpconv":
        layer_factory = pccnn_lib.pc.layers.KPConvFactory(16, False, 1.0)
    elif model_config_dict['conv_type'] == "kpconvn":
        layer_factory = pccnn_lib.pc.layers.KPConvNFactory(16, False, 1.0)
    elif model_config_dict['conv_type']  == "pointconv":
        layer_factory = pccnn_lib.pc.layers.PointConvFactory(16, False, 1.0, model_config_dict['use_batch_norm'])
    elif model_config_dict['conv_type']  == "sphconv":
        layer_factory = pccnn_lib.pc.layers.SPHConvFactory(1, 5, 3, False, 1.0)
    elif model_config_dict['conv_type']  == "pccnn":
        layer_factory = pccnn_lib.pc.layers.PCCNNFactory(16, False, 1.0)

    if model_config_dict['model_type'] == "resnetb":
        model = models.SegmentationResNetModel(model_config_dict, layer_factory, p_bottleneck = True)
    elif model_config_dict['model_type'] == "resnet":
        model = models.SegmentationResNetModel(model_config_dict, layer_factory)
    elif model_config_dict['model_type'] == "unet":
        model = models.SegmentationUNetModel(model_config_dict, layer_factory)
    model.cuda()
    model.load_state_dict(model_config_dict['state_dict'])

    # Load example data.
    val_dataset = ScanNetDataSet(
        p_dataset = 1, 
        p_use_color = True, 
        p_sample_points = 0,
        p_inner_radius = inner_radius, 
        p_oute_radius = outer_radius)

    cell_size = model_config_dict['cell_size']
    badnwidth_np = np.array([cell_size*0.5 for i in range(3)], dtype=np.float32)

    # Create the accumulated logits.
    labels_categories = val_dataset.get_labels()
    num_categories = len(labels_categories)
    accum_logits = []
    for cur_room in range(val_dataset.get_num_rooms()):
        accum_logits.append(
            np.zeros((val_dataset.get_num_pts_room(cur_room), num_categories), dtype=np.float32))
        accum_logits[-1].fill(0.0)

    # Evalaute the model.
    print()
    print("##################### Validation")
    for cur_epoch in range(num_epochs):

        epoch_start_time = current_milli_time()

        # Evaluate the model.
        model = model.eval()
        with torch.no_grad():
            for i in range(num_val_steps_per_epoch):

                # Get the batch.
                outer_pc, outer_pt_colors, inner_pc, inner_pt_labels, \
                    inner_pt_acc_mask, room_ids, pt_ids = \
                        val_dataset.get_next_batch(
                            p_num_rooms = rooms_per_batch, 
                            p_augment=True, 
                            p_agument_noise = False, 
                            p_agument_mirror = True, 
                            p_augment_scaling = True, 
                            p_augment_rot = True, 
                            p_rot_angle = None, 
                            p_update_pot = True, 
                            p_get_ids = True, 
                            p_room_id = None)
                inner_pt_labels.requires_grad = False
                outer_pt_colors.requires_grad = True
                outer_pc.compute_pdf(badnwidth_np)

                # Accumulate the logits.
                batch_ids = inner_pc.batch_ids_.cpu().numpy()
                out_features = model(outer_pc, inner_pc, outer_pt_colors).cpu().numpy()
                for cur_room_iter, cur_room in enumerate(room_ids):
                    batch_mask = batch_ids == cur_room_iter
                    accum_logits[cur_room][pt_ids[cur_room_iter]] += out_features[batch_mask]

                if i % 5 == 0:
                    print("\rV ({:4d}/{:4d})".format(
                        i, num_val_steps_per_epoch),
                        end = "")
                    sys.stdout.flush()
            
        print("\r##### EPOCH {:4d})".format(cur_epoch))
        print()
        print()

        # Compute metrics.
        accum_intersection = np.array([0.0 for i in range(num_categories)])
        accum_union = np.array([0.0 for i in range(num_categories)])
        accum_gt = np.array([0.0 for i in range(num_categories)])
        for cur_room in range(val_dataset.get_num_rooms()):

            cur_labels = val_dataset.get_room_labels(cur_room)
            pred = np.argmax(accum_logits[cur_room], axis=-1)

            mask_valid = cur_labels > 0
            cur_labels = cur_labels[mask_valid]
            pred = pred[mask_valid]

            mask_eq = pred == cur_labels

            num_labels = np.bincount(cur_labels, minlength=num_categories)
            num_pred = np.bincount(pred, minlength=num_categories)
            num_equal = np.bincount(cur_labels[mask_eq], minlength=num_categories)

            accum_gt += num_labels
            accum_union += num_labels
            accum_union += num_pred
            accum_intersection += num_equal
            accum_union -= num_equal

        mean_iou = 0.0
        mean_acc = 0.0
        accum_gt = accum_gt.astype(np.float32)
        accum_union = accum_union.astype(np.float32)
        accum_intersection = accum_intersection.astype(np.float32)
        class_acc = accum_intersection[1:]/np.maximum(accum_gt[1:], 1)
        class_iou = accum_intersection[1:]/np.maximum(accum_union[1:], 1)
        for cat_iter, cur_cat in enumerate(labels_categories[1:]):
            print("Category %15s: IoU  %.4f | Acc %.4f" %(
                cur_cat, class_iou[cat_iter]*100.0, class_acc[cat_iter]*100.0))
        print()

        mean_acc = np.mean(class_acc)*100.0
        mean_iou = np.mean(class_iou)*100.0

        epoch_end_time = current_milli_time()
        print("Epoch time: {:.4f} sec".format((epoch_end_time - epoch_start_time)/1000.0))
        print("IoU:  {:.2f}".format(mean_iou))
        print("Acc:  {:.2f}".format(mean_acc))
        print()
        print()

    for cur_room in range(val_dataset.get_num_rooms()):
        room_labels = np.argmax(accum_logits[cur_room], -1)
        room_name = val_dataset.rooms_[cur_room]
        np.save("./predictions/"+room_name, room_labels)