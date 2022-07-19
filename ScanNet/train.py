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

    parser = argparse.ArgumentParser(description='Train ScanNet')
    parser.add_argument('--conv_type', default='mcconv', help='Type of convolution used (default: mcconv)')
    parser.add_argument('--model_type', default='unet', help='Type of model used (default: unet)')
    parser.add_argument('--no_bn', action='store_true', help='Use batch norm? (default: False)')
    parser.add_argument('--no_const_var', action='store_true', 
        help='Not us constant variance weight init (default: False)')
    args = parser.parse_args()

    #rooms_per_batch = 1
    #inner_radius = 2.0
    #outer_radius = 4.0
    #scene_min_pts = 30000
    #accum_grad_steps = 4
    #cell_size = 0.03

    rooms_per_batch = 4
    inner_radius = 1.0
    outer_radius = 2.0
    scene_min_pts = 0
    accum_grad_steps = 1
    cell_size = 0.03

    # Load example data.
    train_dataset = ScanNetDataSet(
        p_dataset = 0, 
        p_use_color = True, 
        p_inner_radius = inner_radius, 
        p_oute_radius = outer_radius,
        p_sample_points = 0,
        p_rand_seed = 25,
        p_min_pts=scene_min_pts)
    val_dataset = ScanNetDataSet(
        p_dataset = 1, 
        p_use_color = True, 
        p_inner_radius = inner_radius, 
        p_oute_radius = outer_radius,
        p_sample_points = 0,
        p_rand_seed = 25,
        p_min_pts=scene_min_pts)


    # Training config
    pooling_method = 'poisson_disk'
    scale_conv_radius = 3.0

    num_epochs = 500
    num_train_steps_per_epoch = 750
    num_val_steps_per_epoch = 100
    num_weight_init_steps = 30
    learning_rate = 0.005
    weight_decay = 0.0001
    #const_var_value = train_dataset.compute_variance_color()
    const_var_value =  1.0
    max_neighbors = 16
    clip_grads = 0.0

    
    cur_date = datetime.now()
    dt_string = cur_date.strftime("__%d_%m_%Y__%H_%M_%S")
    if not args.no_bn:
        params_string = "__1_"
    else:
        params_string = "__0_"
    if not args.no_const_var:
        params_string += "__1"
    else:
        params_string += "__0"
    log_folder_path = "runs/"+args.conv_type+dt_string+params_string
    os.mkdir(log_folder_path)
    os.mkdir(log_folder_path+"/saved_models")
    os.mkdir(log_folder_path+"/tensorboard")

    # Create the model.
    if args.conv_type == "mcconv":
        layer_factory = pccnn_lib.pc.layers.MCConvFactory(16, not args.no_const_var, 1, const_var_value)
    elif args.conv_type == "kpconv":
        layer_factory = pccnn_lib.pc.layers.KPConvFactory(16, not args.no_const_var, const_var_value)
    elif args.conv_type == "kpconvn":
        layer_factory = pccnn_lib.pc.layers.KPConvNFactory(16, not args.no_const_var, const_var_value)
    elif args.conv_type == "pointconv":
        layer_factory = pccnn_lib.pc.layers.PointConvFactory(16, not args.no_const_var, const_var_value, not args.no_bn)
    elif args.conv_type == "sphconv":
        layer_factory = pccnn_lib.pc.layers.SPHConvFactory(1, 5, 3, not args.no_const_var, const_var_value)
    elif args.conv_type == "pccnn":
        layer_factory = pccnn_lib.pc.layers.PCCNNFactory(16, not args.no_const_var, const_var_value)

    if args.model_type == "resnetb":
        model_config_dict = {
            'model_type': args.model_type,
            'conv_type': args.conv_type,
            'num_dims' : 3,
            'num_in_features' : 3,
            'num_out_features' : 21,
            'num_levels' : 5,
            'num_blocks_enc' : [2, 2, 2, 2, 2],
            'num_blocks_dec' : [1, 1, 1, 1],
            'pooling_radii' : np.array([1.0, 2.0, 4.0, 8.0, 16.0])*cell_size,#[0.02, 0.04, 0.08, 0.16, 0.32],
            'conv_radii' : np.array([1.0, 2.0, 4.0, 8.0, 16.0])*cell_size*scale_conv_radius,#[0.06, 0.12, 0.24, 0.48, 0.96],
            'feature_sizes_enc': [64, 128, 256, 512, 512],
            'feature_sizes_dec': [256, 256, 256, 512],
            'first_pooling_features': 32,
            'last_mlp': 64,
            'first_pooling_radius': 0.05,
            'last_upsampling_radius': 0.05,
            'use_batch_norm': not args.no_bn,
            'act_funct': 'leaky_relu',
            'drop_out': 0.2,
            'pdf_bandwidth': 0.5,
            'cell_size': cell_size,
            'pooling_method' : pooling_method,
            'max_neighbors': max_neighbors}
        model = models.SegmentationResNetModel(model_config_dict, layer_factory, p_bottleneck = True)
    elif args.model_type == "resnet":
        model_config_dict = {
            'model_type': args.model_type,
            'conv_type': args.conv_type,
            'num_dims' : 3,
            'num_in_features' : 3,
            'num_out_features' : 21,
            'num_levels' : 4,
            'num_blocks_enc' : [2, 2, 2, 2],
            'num_blocks_dec' : [2, 2, 2],
            'pooling_radii' : np.array([1.0, 2.0, 4.0, 8.0])*cell_size,#[0.02, 0.04, 0.08, 0.16, 0.32],
            'conv_radii' : np.array([1.0, 2.0, 4.0, 8.0])*cell_size*scale_conv_radius,#[0.06, 0.12, 0.24, 0.48, 0.96],
            'feature_sizes_enc': [64, 128, 256, 512],
            'feature_sizes_dec': [64, 128, 256],
            'first_pooling_features': 32,
            'last_mlp': 0,
            'first_pooling_radius': cell_size,
            'last_upsampling_radius': cell_size*scale_conv_radius,
            'use_batch_norm': not args.no_bn,
            'act_funct': 'leaky_relu',
            'drop_out': 0.2,
            'pdf_bandwidth': 0.5,
            'cell_size': cell_size,
            'pooling_method' : pooling_method,
            'max_neighbors': max_neighbors}
        model = models.SegmentationResNetModel(model_config_dict, layer_factory)
    elif args.model_type == "unet":
        model_config_dict = {
            'model_type': args.model_type,
            'conv_type': args.conv_type,
            'num_dims' : 3,
            'num_in_features' : 3,
            'num_out_features' : 21,
            'num_levels' : 4,
            'num_blocks_enc' : [2, 2, 2, 2],
            'num_blocks_dec' : [2, 2, 2],
            'pooling_radii' : np.array([1.0, 2.0, 4.0, 8.0])*cell_size,#[0.02, 0.04, 0.08, 0.16, 0.32],
            'conv_radii' : np.array([1.0, 2.0, 4.0, 8.0])*cell_size*scale_conv_radius,#[0.06, 0.12, 0.24, 0.48, 0.96],
            'feature_sizes': [64, 128, 256, 512],
            'first_pooling_features': 64,
            'first_pooling_radius': cell_size,
            'last_upsampling_radius': cell_size*scale_conv_radius,
            'use_batch_norm': not args.no_bn,
            'act_funct': 'leaky_relu',
            'drop_out': 0.2,
            'pdf_bandwidth': 0.5,
            'cell_size': cell_size,
            'pooling_method' : pooling_method,
            'max_neighbors': max_neighbors}
        model = models.SegmentationUNetModel(model_config_dict, layer_factory)
    model.cuda()


    badnwidth_np = np.array([cell_size*0.5 for i in range(3)], dtype=np.float32)

    # Get the parameters with and without weight decay.
    params_no_wd = []
    params_wd = []
    for name, cur_param in model.named_parameters():
        if "conv_weights_" in name:
            params_wd.append(cur_param)
        else:
            params_no_wd.append(cur_param)

    print(len(params_wd))
    print(len(params_no_wd))

    # Optimizer.
    device = torch.device("cuda:0")
    weight_tensor = torch.as_tensor(train_dataset.weights_, device=device)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.SGD(
        [{  'params': params_no_wd, 
            'weight_decay': 0.}, 
         {  'params': params_wd, 
            'weight_decay': weight_decay}], 
        lr=learning_rate,
        momentum=0.9)

    lr_muls = []
    for i in range(num_epochs+1):
        if i < 300:
            lr_muls.append(1.0)
        elif i < 400:
            lr_muls.append(0.25)
        else:
            lr_muls.append(0.0625)
    lambda_lr = lambda epoch: lr_muls[epoch]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

    # Tensorboard.
    writer = SummaryWriter(log_folder_path+"/tensorboard")

    # Initialized the weights with the data statistics.
    if not args.no_const_var:

        rooms_per_batch_init = 2
        start_init_time = current_milli_time()
        print()
        print("##################### Initializing weights")
        model = model.train()
        optimizer.zero_grad()
        for cur_layer in range(model.get_num_convs()):
            model.set_init_warmup_state(cur_layer, True)
            for i in range(num_weight_init_steps):

                # Get the batch.
                outer_pc, outer_pt_colors, inner_pc, inner_pt_labels, \
                    inner_pt_acc_mask = train_dataset.get_next_batch(
                        p_num_rooms = rooms_per_batch_init, 
                        p_augment=True, 
                        p_agument_noise = False, 
                        p_agument_mirror = True, 
                        p_augment_scaling = True, 
                        p_augment_rot = True, 
                        p_rot_angle = None, 
                        p_update_pot = True, 
                        p_get_ids = False, 
                        p_room_id = None)
                inner_pt_labels.requires_grad = False
                outer_pt_colors.requires_grad = True
                outer_pc.compute_pdf(badnwidth_np)
                
                out_features = model(outer_pc, inner_pc, outer_pt_colors)

            print("\rT ({:4d}/{:4d})".format(cur_layer, model.get_num_convs()), end ="")
            sys.stdout.flush()
            model.set_init_warmup_state(cur_layer, False)

        layer_factory.set_init_warmup_state_convs(False)
        train_dataset.restart_potentials()
        end_init_time = current_milli_time()
        print()
        print("Time weight init:", end_init_time-start_init_time)
        
    print()
    print("Variance")
    for cur_layer in range(model.get_num_convs()):
        print(model.conv_list_[cur_layer].accum_weight_var_/model.conv_list_[cur_layer].feat_input_size_)
    print()

    # Training the model.
    mean_pts_out = 0
    mean_pts_in = 0
    labels_categories = train_dataset.get_labels()
    num_categories = len(labels_categories)
    best_iou = 0.0
    best_acc = 0.0
    best_loss = 1000.0
    model_counter = 0
    test_pc_queue = []
    print()
    print()
    print("##################### Training")
    for cur_epoch in range(num_epochs):

        epoch_start_time = current_milli_time()

        print()
        print("###### Epoc", cur_epoch)

        accum_loss = 0.0
        accum_acc = 0.0
        accum_iou = 0.0
        accum_intersection = np.array([0.0 for i in range(num_categories)])
        accum_union = np.array([0.0 for i in range(num_categories)])
        accum_gt = np.array([0.0 for i in range(num_categories)])
        model = model.train()
        optimizer.zero_grad()
        for i in range(num_train_steps_per_epoch):

            start_time = current_milli_time()

            # Get the batch.
            outer_pc, outer_pt_colors, inner_pc, inner_pt_labels, \
                inner_pt_acc_mask = train_dataset.get_next_batch(
                    p_num_rooms = rooms_per_batch, 
                    p_augment=True, 
                    p_agument_noise = False, 
                    p_agument_mirror = True, 
                    p_augment_scaling = True, 
                    p_augment_rot = True, 
                    p_rot_angle = None, 
                    p_update_pot = True, 
                    p_get_ids = False, 
                    p_room_id = None)
            inner_pt_labels.requires_grad = False
            outer_pt_colors.requires_grad = True
            outer_pc.compute_pdf(badnwidth_np)

            end_time = current_milli_time()
            data_time = end_time - start_time

            start_time = current_milli_time()
            
            out_features = model(outer_pc, inner_pc, outer_pt_colors)
            loss = criterion(out_features, inner_pt_labels)
            cur_weights = weight_tensor[inner_pt_labels]
            norm_loss = loss*cur_weights
            per_batch_loss = torch.mean(inner_pc.global_pooling(norm_loss.reshape((-1, 1))))
            norm_loss = per_batch_loss/float(accum_grad_steps)
            norm_loss.backward()

            ####################### TEST
            #print()
            #var_list = []
            #for cur_room in range(rooms_per_batch):
            #    mask_pts = inner_pc.batch_ids_ == cur_room
            #    var_list.append(torch.var(out_features[mask_pts]).item())
            #print(var_list)
            #for cur_room, cur_var in enumerate(var_list):
            #    
            #    bad_room = False
            #    for cur_room_o, cur_var_o in enumerate(var_list):
            #        if cur_room_o != cur_room:
            #            bad_room = bad_room or cur_var/cur_var_o > 100.0
            #    if bad_room and cur_var > 100.0:
            #        for cur_room_o, cur_var_o in enumerate(var_list):
            #            mask_pts = outer_pc.batch_ids_ == cur_room_o
            #            cur_pts = outer_pc.pts_[mask_pts].cpu().numpy()
            #            cur_colors = outer_pt_colors[mask_pts].detach().cpu().numpy()
            #            if cur_room_o == cur_room:
            #                file_name = "room_"+str(cur_room_o)+"_bad.txt"
            #            else:
            #                file_name = "room_"+str(cur_room_o)+".txt"
            #            with open(file_name, 'w') as my_file:
            #                for cur_pt_iter in range(cur_pts.shape[0]):
            #                    my_file.write(
            #                        str(cur_pts[cur_pt_iter, 0])+","+
            #                        str(cur_pts[cur_pt_iter, 1])+","+
            #                        str(cur_pts[cur_pt_iter, 2])+","+
            #                        str(int(cur_colors[cur_pt_iter, 0]*255.0))+","+
            #                        str(int(cur_colors[cur_pt_iter, 1]*255.0))+","+
            #                        str(int(cur_colors[cur_pt_iter, 2]*255.0))+"\n")
            #            mask_pts = inner_pc.batch_ids_ == cur_room_o
            #            cur_pts = inner_pc.pts_[mask_pts].cpu().numpy()
            #            cur_feats = torch.mean(torch.square(out_features[mask_pts]), dim=-1).detach().cpu().numpy()
            #            with open("room_"+str(cur_room_o)+"_var.txt", 'w') as my_file:
            #                for cur_pt_iter in range(cur_pts.shape[0]):
            #                    cur_var_feat = cur_feats[cur_pt_iter]/var_list[cur_room_o]
            #                    if cur_var_feat > 1.0:
            #                        cur_var_feat = 1.0
            #                    cur_var_feat = int(cur_var_feat*255.0)
            #                    my_file.write(
            #                        str(cur_pts[cur_pt_iter, 0])+","+
            #                        str(cur_pts[cur_pt_iter, 1])+","+
            #                        str(cur_pts[cur_pt_iter, 2])+","+
            #                        str(cur_var_feat)+","+
            #                        str(cur_var_feat)+","+
            #                        str(cur_var_feat)+"\n")
            #        exit()
            #print(torch.var(out_features).item())
            #print()
            ####################### TEST

            if i == 0:
                for name, cur_param in model.named_parameters():
                    if "conv_weights_" in name or "axis_" in name:
                        writer.add_scalar(name+'_grad_mean', torch.mean(cur_param.grad), cur_epoch)
                        writer.add_scalar(name+'_grad_var', torch.var(cur_param.grad), cur_epoch)
                        writer.add_scalar(name+'_param_mean', torch.mean(cur_param), cur_epoch)
                        writer.add_scalar(name+'_param_var', torch.var(cur_param), cur_epoch)
            
            if i% accum_grad_steps == 0:
                if clip_grads > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grads)
                optimizer.step()
                optimizer.zero_grad()

            pred = torch.argmax(out_features, 1)
            mask_eq = pred == inner_pt_labels

            num_labels = np.bincount(inner_pt_labels.cpu().numpy(), 
                minlength=num_categories).astype(np.float32)
            num_pred = np.bincount(pred.cpu().numpy(), 
                minlength=num_categories).astype(np.float32)
            num_equal = np.bincount(inner_pt_labels[mask_eq].cpu().numpy(), 
                minlength=num_categories).astype(np.float32)
            union = num_labels + num_pred - num_equal
            accum_gt += num_labels
            accum_union += num_labels + num_pred - num_equal
            accum_intersection += num_equal
            
            class_acc = accum_intersection[1:]/np.maximum(accum_gt[1:], 1)
            class_iou = accum_intersection[1:]/np.maximum(accum_union[1:], 1)
            
            accum_acc = np.mean(class_acc)*100.0
            accum_iou = np.mean(class_iou)*100.0

            end_time = current_milli_time()
            learning_step_time = end_time - start_time

            accum_loss += (per_batch_loss.item() - accum_loss)/(i+1)

            ####################### TEST
            cur_pts = outer_pc.pts_.cpu().numpy()
            cur_in_pts = inner_pc.pts_.cpu().numpy()
            test_pc_queue.append((cur_pts.copy(), cur_in_pts.copy(), per_batch_loss.item()))
            if len(test_pc_queue) > 50:
                test_pc_queue.pop(0)

            if math.isnan(per_batch_loss.item()):
                with open("aux_log.txt", "w") as file_object:
                    for cur_elem_iter, cur_elem in enumerate(test_pc_queue):
                        np.savetxt("room_"+str(cur_elem_iter)+"_o.txt", cur_elem[0])
                        np.savetxt("room_"+str(cur_elem_iter)+"_i.txt", cur_elem[1])
                        file_object.write(str(cur_elem_iter)+":"+str(cur_elem[2])+"\n")
                sys.exit()
            ####################### TEST

            if i%5 == 0:
                print("\rT ({:4d}/{:4d}): {:.4f} {:.2f} {:.2f} ({:.2f}, {:.2f}) ({:d}, {:d})      ".format(
                    i, num_train_steps_per_epoch, accum_loss, accum_acc, accum_iou, data_time, 
                    learning_step_time, outer_pc.pts_.shape[0], inner_pc.pts_.shape[0]),
                    end ="")
                sys.stdout.flush()

        # Evaluate the model.
        accum_intersection = np.array([0.0 for i in range(num_categories)])
        accum_union = np.array([0.0 for i in range(num_categories)])
        accum_gt = np.array([0.0 for i in range(num_categories)])
        test_accum_loss = 0.0
        model = model.eval()
        with torch.no_grad():
            for i in range(num_val_steps_per_epoch):
                outer_pc, outer_pt_colors, inner_pc, inner_pt_labels, \
                    inner_pt_acc_mask = val_dataset.get_next_batch(
                        p_num_rooms = rooms_per_batch, 
                        p_augment=False, 
                        p_agument_noise = False, 
                        p_agument_mirror = False, 
                        p_augment_scaling = False, 
                        p_augment_rot = False, 
                        p_rot_angle = None, 
                        p_update_pot = True, 
                        p_get_ids = False, 
                        p_room_id = None)
                inner_pt_labels.requires_grad = False
                outer_pt_colors.requires_grad = True
                outer_pc.compute_pdf(badnwidth_np)

                start_time = current_milli_time()
                out_features = model(outer_pc, inner_pc, outer_pt_colors)
                loss = criterion(out_features, inner_pt_labels)
                cur_weights = weight_tensor[inner_pt_labels]
                norm_loss = loss*cur_weights
                per_batch_loss = torch.mean(inner_pc.global_pooling(norm_loss.reshape((-1, 1))))

                pred = torch.argmax(out_features, 1)
                mask_valid = inner_pt_acc_mask > 0.0
                inner_pt_labels = inner_pt_labels[mask_valid]
                pred = pred[mask_valid]

                mask_eq = pred == inner_pt_labels

                num_labels = np.bincount(inner_pt_labels.cpu().numpy(), minlength=num_categories)
                num_pred = np.bincount(pred.cpu().numpy(), minlength=num_categories)
                num_equal = np.bincount(inner_pt_labels[mask_eq].cpu().numpy(), minlength=num_categories)

                accum_gt += num_labels
                accum_union += num_labels
                accum_union += num_pred
                accum_intersection += num_equal
                accum_union -= num_equal

                end_time = current_milli_time()
                val_time = end_time - start_time

                test_accum_loss += (per_batch_loss.item() - test_accum_loss)/(i+1)

                if i % 5 == 0:
                    print("\rV ({:4d}/{:4d}): {:.4f} ({:.2f})                                   ".format(
                        i, num_val_steps_per_epoch, test_accum_loss, val_time),
                        end = "")
                    sys.stdout.flush()
            
        print()
        print()

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
        print("")

        mean_acc = np.mean(class_acc)*100.0
        mean_iou = np.mean(class_iou)*100.0

        epoch_end_time = current_milli_time()
        print("Epoch time: {:.4f} sec".format((epoch_end_time - epoch_start_time)/1000.0))
        print("Train: {:.2f} / {:.2f} / {:.4f}".format(accum_iou, accum_acc, accum_loss))
        print("Val:  {:.2f} [{:.2f}] / {:.2f} [{:.2f}] / {:.4f} [{:.2f}]".format(
            mean_iou, best_iou, mean_acc, best_acc, test_accum_loss, best_loss))

        writer.add_scalar('train_loss', accum_loss, cur_epoch)
        writer.add_scalar('train_acc', accum_acc, cur_epoch)
        writer.add_scalar('train_iou', accum_iou, cur_epoch)
        writer.add_scalar('val_loss', test_accum_loss, cur_epoch)
        writer.add_scalar('val_acc', mean_acc, cur_epoch)
        writer.add_scalar('val_iou', mean_iou, cur_epoch)

        model_config_dict['state_dict'] = model.state_dict()
        torch.save(model_config_dict, log_folder_path+"/saved_models/model{:d}.pth".format(model_counter))
        model_counter = (model_counter+1)%5

        scheduler.step()

        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(model_config_dict, 
                log_folder_path+"/saved_models/model_best_iou.pth")
        if mean_acc > best_acc:
            best_acc = mean_acc
            torch.save(model_config_dict, 
                log_folder_path+"/saved_models/model_best_acc.pth")
        if test_accum_loss < best_loss:
            best_loss = test_accum_loss
            torch.save(model_config_dict, 
                log_folder_path+"/saved_models/model_best_loss.pth")