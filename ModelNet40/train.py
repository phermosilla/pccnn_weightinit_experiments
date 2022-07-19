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
from dataset import ModelNetDataSet

current_milli_time = lambda: time.time() * 1000.0

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train ModelNet')
    parser.add_argument('--conv_type', default='mcconv', help='Type of convolution used (default: mcconv)')
    parser.add_argument('--no_bn', action='store_true', help='Use batch norm? (default: False)')
    parser.add_argument('--no_const_var', action='store_true', 
        help='Not us constant variance weight init (default: False)')
    args = parser.parse_args()

    # Load example data.
    train_dataset = ModelNetDataSet(
        pTrain = True, 
        pNumPts=2048, 
        pPath="./data", 
        pPermute = True)
    print("Num models loaded:", train_dataset.get_num_models())
    val_dataset = ModelNetDataSet(
        pTrain = False, 
        pNumPts=2048, 
        pPath="./data", 
        pPermute = False)
    print("Num models loaded:", val_dataset.get_num_models())
        
    # Training config
    pooling_method = 'poisson_disk'
    num_weight_init_steps = 100
    max_neighbors = 16
    num_epochs = 250
    accum_grad_steps = 1
    learning_rate = 0.005
    weight_decay = 0.0001
    const_var_value = train_dataset.compute_variance_features()
    clip_grads = 0.0
    batch_size = 16
    cur_cell_size = 0.05
    conv_radius_scale = 3.0

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
        up_axis = 1
        layer_factory = pccnn_lib.pc.layers.MCConvFactory(16, not args.no_const_var, 1, const_var_value)
    elif args.conv_type == "kpconv":
        up_axis = 1
        layer_factory = pccnn_lib.pc.layers.KPConvFactory(16, not args.no_const_var, const_var_value)
    elif args.conv_type == "kpconvn":
        up_axis = 1
        layer_factory = pccnn_lib.pc.layers.KPConvNFactory(16, not args.no_const_var, const_var_value)
    elif args.conv_type == "pointconv":
        up_axis = 1
        layer_factory = pccnn_lib.pc.layers.PointConvFactory(16, not args.no_const_var, const_var_value, not args.no_bn)
    elif args.conv_type == "sphconv":
        up_axis = 2
        layer_factory = pccnn_lib.pc.layers.SPHConvFactory(1, 5, 3, not args.no_const_var, const_var_value)
    elif args.conv_type == "pccnn":
        up_axis = 1
        layer_factory = pccnn_lib.pc.layers.PCCNNFactory(16, not args.no_const_var, const_var_value)

    # Create the model.
    model_config_dict = {
        'conv_type': args.conv_type,
        'num_dims' : 3,
        'num_in_features' : 3,
        'num_out_features' : 40,
        'num_levels' : 3,
        'num_blocks' : [2, 2, 2],
        'pooling_radii' : np.array([1.0, 2.0, 4.0])*cur_cell_size,
        'conv_radii' : np.array([1.0, 2.0, 4.0])*cur_cell_size*conv_radius_scale,
        'feature_sizes': [64, 256, 1024],
        'use_batch_norm': not args.no_bn,
        'act_funct': 'leaky_relu',
        'drop_out': 0.2,
        'pdf_bandwidth': 0.2,
        'fc_num_feat': 1024,
        'fc_drop_out': 0.5,
        'batch_size': batch_size,
        'pooling_method' : pooling_method,
        'max_neighbors': max_neighbors
    }

    model = models.ClassificationModel(model_config_dict, layer_factory)
    model.cuda()

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
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        [{  'params': params_no_wd, 
            'weight_decay': 0.}, 
         {  'params': params_wd, 
            'weight_decay': weight_decay}], 
        lr=learning_rate,
        momentum=0.9)

    lr_muls = []
    for i in range(num_epochs+1):
        if i < 175:
            lr_muls.append(1.0)
        elif i < 225:
            lr_muls.append(0.1)
        else:
            lr_muls.append(0.01)
    lambda_lr = lambda epoch: lr_muls[epoch]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
        
    # Tensorboard.
    writer = SummaryWriter(log_folder_path+"/tensorboard")

    # Initialized the weights with the data statistics.
    if not args.no_const_var:

        start_init_time = current_milli_time()
        print()
        print("##################### Initializing weights")
        model = model.train()
        train_dataset.start_epoch()
        optimizer.zero_grad()
        for cur_layer in range(model.get_num_convs()):
            model.set_init_warmup_state(cur_layer, True)
            for i in range(num_weight_init_steps):

                # Get the batch.
                pts, batch_ids, labels = train_dataset.get_next_batch(
                    batch_size, 
                    pAugment = True, 
                    pAgumentNoise = False, 
                    pAugmentScaling = True, 
                    pAugmentRot = False,
                    pUpAxis = up_axis)
                features = torch.clone(pts[:,3:6])
                pts = torch.clone(pts[:,0:3])
                
                point_cloud = pccnn_lib.pc.Pointcloud(pts, batch_ids, device=device)
                features.requires_grad = True

                out_features = model(point_cloud, features)

            print("\rT ({:4d}/{:4d})".format(cur_layer, model.get_num_convs()), end ="")
            sys.stdout.flush()
            model.set_init_warmup_state(cur_layer, False)

        layer_factory.set_init_warmup_state_convs(False)

        end_init_time = current_milli_time()
        print()
        print("Time weight init:", end_init_time-start_init_time)
        print("Variance")
        for cur_layer in range(model.get_num_convs()):
            print(model.conv_list_[cur_layer].accum_weight_var_/model.conv_list_[cur_layer].feat_input_size_)
        print()

    # Training the model.
    model_counter = 0
    print()
    print("##################### Training")
    for cur_epoch in range(num_epochs):

        epoch_start_time = current_milli_time()

        print()
        print("###### Epoc", cur_epoch)

        train_dataset.start_epoch()
        val_dataset.start_epoch()

        accum_loss = 0.0
        accum_acc = 0.0
        model = model.train()
        optimizer.zero_grad()
        num_train_steps_per_epoch = train_dataset.get_num_models()//batch_size
        for i in range(num_train_steps_per_epoch):

            start_time = current_milli_time()

            # Get the batch.
            pts, batch_ids, labels = train_dataset.get_next_batch(
                    batch_size, 
                    pAugment = True, 
                    pAgumentNoise = False, 
                    pAugmentScaling = True, 
                    pAugmentRot = False,
                    pUpAxis = up_axis)
            features = torch.clone(pts[:,3:6])
            pts = torch.clone(pts[:,0:3])
            
            point_cloud = pccnn_lib.pc.Pointcloud(pts, batch_ids, device=device)
            features.requires_grad = True

            end_time = current_milli_time()
            data_time = end_time - start_time

            start_time = current_milli_time()
            out_features = model(point_cloud, features)
            loss = criterion(out_features, labels)
            norm_loss = loss/float(accum_grad_steps)
            norm_loss.backward()

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
            mask = pred == labels
            acc = torch.mean(mask.float())*100.0
            end_time = current_milli_time()
            learning_step_time = end_time - start_time

            accum_loss += (loss.item() - accum_loss)/(i+1)
            accum_acc += (acc.item() - accum_acc)/(i+1)

            if i%5 == 0:
                print("\rT ({:4d}/{:4d}): {:.4f} {:.2f} ({:.2f}, {:.2f})       ".format(
                    i, num_train_steps_per_epoch, accum_loss, accum_acc, data_time, 
                    learning_step_time), end ="")
                sys.stdout.flush()

        # Evaluate the model.
        num_val_steps_per_epoch = val_dataset.get_num_models()//batch_size
        test_accum_loss = 0.0
        test_accum_acc = 0.0
        model = model.eval()
        with torch.no_grad():
            for i in range(num_val_steps_per_epoch):
                pts, batch_ids, labels = val_dataset.get_next_batch(
                    batch_size, 
                    pAugment = True, 
                    pAgumentNoise = False, 
                    pAugmentScaling = True, 
                    pAugmentRot = False,
                    pUpAxis = up_axis)

                features = torch.clone(pts[:,3:6])
                pts = torch.clone(pts[:,0:3])
            
                point_cloud = pccnn_lib.pc.Pointcloud(pts, batch_ids, device=device)
                features.requires_grad = True

                start_time = current_milli_time()
                out_features = model(point_cloud, features)
                loss = criterion(out_features, labels)

                pred = torch.argmax(out_features, 1)
                mask_eq = pred == labels
                acc = torch.mean(mask_eq.float())*100.0

                end_time = current_milli_time()
                val_time = end_time - start_time

                test_accum_loss += (loss.item() - test_accum_loss)/(i+1)
                test_accum_acc += (acc.item() - test_accum_acc)/(i+1)

                if i % 5 == 0:
                    print("\rV ({:4d}/{:4d}): {:.4f} {:.2f} ({:.2f})                            ".format(
                        i, num_val_steps_per_epoch, test_accum_loss, test_accum_acc, val_time),
                        end = "")
                    sys.stdout.flush()
            
        print()
        print()


        epoch_end_time = current_milli_time()
        print("Epoch time: {:.4f} sec".format((epoch_end_time - epoch_start_time)/1000.0))
        print("Train: {:.2f} / {:.4f}".format(accum_acc, accum_loss))
        print("Val:  {:.2f} / {:.2f} ".format(
            test_accum_acc, test_accum_loss))

        writer.add_scalar('train_loss', accum_loss, cur_epoch)
        writer.add_scalar('train_acc', accum_acc, cur_epoch)
        writer.add_scalar('val_loss', test_accum_loss, cur_epoch)
        writer.add_scalar('val_acc', test_accum_acc, cur_epoch)

        model_config_dict['state_dict'] = model.state_dict()
        torch.save(model_config_dict, log_folder_path+"/saved_models/model"+str(model_counter)+".pth")
        model_counter = (model_counter+1)%5

        scheduler.step()
