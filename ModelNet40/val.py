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
    parser.add_argument('--model_path', help='Model path')
    args = parser.parse_args()

    # Load example data.
    val_dataset = ModelNetDataSet(
        pTrain = False, 
        pNumPts=2048, 
        pPath="./data", 
        pPermute = True)
    print("Num models loaded:", val_dataset.get_num_models())

     # Load model params.
    model_config_dict = torch.load(args.model_path)

    # Training params
    num_epochs = 25
    batch_size = 1

    # Create the model.
    if model_config_dict['conv_type'] == "mcconv":
        up_axis = 1
        layer_factory = pccnn_lib.pc.layers.MCConvFactory(16, False, 1, 1.0)
    elif model_config_dict['conv_type'] == "kpconv":
        up_axis = 1
        layer_factory = pccnn_lib.pc.layers.KPConvFactory(16, False, 1.0)
    elif model_config_dict['conv_type'] == "kpconvn":
        up_axis = 1
        layer_factory = pccnn_lib.pc.layers.KPConvNFactory(16, False, 1.0)
    elif model_config_dict['conv_type'] == "pointconv":
        up_axis = 1
        layer_factory = pccnn_lib.pc.layers.PointConvFactory(16, False, 1.0, 
            model_config_dict['use_batch_norm'])
    elif model_config_dict['conv_type'] == "sphconv":
        up_axis = 2
        layer_factory = pccnn_lib.pc.layers.SPHConvFactory(1, 5, 3, False, 1.0)
    elif model_config_dict['conv_type'] == "pccnn":
        up_axis = 1
        layer_factory = pccnn_lib.pc.layers.PCCNNFactory(16, False, 1.0)

    # Create the model.
    model = models.ClassificationModel(model_config_dict, layer_factory)
    model.cuda()
    model.load_state_dict(model_config_dict['state_dict'])

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

    # Training the model.
    model = model.eval()
    print()
    print("##################### Testing")
    accum_logits = np.zeros((val_dataset.get_num_models(), 40), dtype=np.float32)
    accum_labels = np.zeros((val_dataset.get_num_models()), dtype=np.int32)
    for cur_epoch in range(num_epochs):

        val_dataset.start_epoch()

        # Evaluate the model.
        num_val_steps_per_epoch = val_dataset.get_num_models()//batch_size
        test_accum_loss = 0.0
        test_accum_acc = 0.0
        
        with torch.no_grad():
            for i in range(num_val_steps_per_epoch):
                pts, batch_ids, labels, batch_indexs = val_dataset.get_next_batch(
                    batch_size, 
                    pAugment = True, 
                    pAgumentNoise = False, 
                    pAugmentScaling = True, 
                    pAugmentRot = False,
                    pReturnIndexs = True,
                    pUpAxis = up_axis)

                features = torch.clone(pts[:,3:6])
                pts = torch.clone(pts[:,0:3])
            
                point_cloud = pccnn_lib.pc.Pointcloud(pts, batch_ids, device=device)
                features.requires_grad = True

                start_time = current_milli_time()
                out_features = model(point_cloud, features).detach().cpu().numpy()
                labels = labels.cpu().numpy()
                
                for cur_elem in range(batch_size):
                    accum_logits[batch_indexs[cur_elem], :] += out_features[cur_elem,:]
                    accum_labels[batch_indexs[cur_elem]] = labels[cur_elem]

                if i % 5 == 0:
                    print("\rV ({:4d}/{:4d})   ".format(i, num_val_steps_per_epoch), end = "")
                    sys.stdout.flush()
            
        print()
        print()

        # Compute accuracy.
        predictions = np.argmax(accum_logits, -1)
        accuracy = np.mean(predictions == accum_labels)
        print("Accuracy: {:.4f}".format(accuracy))
        print()
