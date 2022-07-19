import numpy as np
import torch
import pccnn_lib

class ClassificationModel(torch.nn.Module, pccnn_lib.pc.models.IModel):

    def __init__(self, p_config_dict, p_conv_factory):
        """Constructor.

        Args:
            p_config_dict (dict): Configuration dictionary.
            p_conv_factory (IConvLayerFactory): Convolution layer
                factory object.
        """

        # Super class init.
        torch.nn.Module.__init__(self)
        pccnn_lib.pc.models.IModel.__init__(self)

        # Get the config parameters.
        self.num_dims_ = p_config_dict['num_dims']
        self.num_in_features_ = p_config_dict['num_in_features']
        self.num_out_features_ = p_config_dict['num_out_features']
        self.num_levels_ = p_config_dict['num_levels']
        self.num_blocks_ = p_config_dict['num_blocks']
        self.pooling_radii_ = p_config_dict['pooling_radii']
        self.conv_radii_ = p_config_dict['conv_radii']
        self.feature_sizes_ = p_config_dict['feature_sizes']
        self.use_batch_nom_ = p_config_dict['use_batch_norm']
        self.act_funct_ = p_config_dict['act_funct']
        self.drop_out_ = p_config_dict['drop_out']
        self.pdf_bandwith_ = p_config_dict['pdf_bandwidth']
        self.fc_layer_features_ = p_config_dict['fc_num_feat']
        self.fc_drop_out_ = p_config_dict['fc_drop_out']
        self.batch_size_ = p_config_dict['batch_size']
        self.pooling_method_ = p_config_dict['pooling_method']
        self.max_neighbors_ = p_config_dict['max_neighbors']
        if 'use_group_norm' in p_config_dict:
            self.use_group_nom_ = p_config_dict['use_group_norm']
            self.num_groups_ = p_config_dict['num_groups_group_norm']
            self.use_batch_nom_ = self.use_batch_nom_ and not self.use_group_nom_
        else:
            self.use_group_nom_ = False

        # Create the activation function.
        if self.act_funct_ == "relu":
            cur_act_funct = torch.nn.ReLU()
        elif self.act_funct_ == "leaky_relu":
            cur_act_funct = torch.nn.LeakyReLU(0.2)
        else:
            cur_act_funct = None

        # Create the network.
        self.blocks_ = torch.nn.ModuleList()

        # Function to create batch norm + activation function + drop out. 
        def bn_act_do(
            num_features, 
            drop_out = self.drop_out_, 
            p_batch_norm = self.use_batch_nom_):
            layer_list = []
            if p_batch_norm:
                layer_list.append(torch.nn.BatchNorm1d(
                    num_features, momentum=0.2))
            layer_list.append(cur_act_funct)
            if drop_out > 0.0:
                layer_list.append(torch.nn.Dropout(drop_out))
            return torch.nn.Sequential(*layer_list)
        
        # Encoder.
        # For each level.
        print()
        print("############ Classification Model")
        print()
        print("######## Encoder")
        prev_num_features = self.num_in_features_
        for cur_level in range(self.num_levels_):
            
            print()
            print("### Level", cur_level)
            cur_num_blocks = self.num_blocks_[cur_level]
            cur_num_features = self.feature_sizes_[cur_level]
            cur_radius = self.conv_radii_[cur_level]
            cur_pooling_radius = self.pooling_radii_[cur_level]
            
            # Create the pooling layer.
            print("Level (", cur_pooling_radius, ")")

            # For each block.
            for cur_block in range(cur_num_blocks):
                print("Block (", cur_radius, prev_num_features, cur_num_features, ")")
                
                # Create the block.
                if cur_block > 0 or cur_level > 0:
                    if self.use_group_nom_:
                        self.blocks_.append(
                            pccnn_lib.layers.GroupNormalization(
                                prev_num_features, self.num_groups_))
                    self.blocks_.append(bn_act_do(prev_num_features))
                cur_conv = p_conv_factory.create_conv_layer(
                    self.num_dims_, 
                    prev_num_features, 
                    cur_num_features)
                self.blocks_.append(cur_conv)
                self.add_conv(cur_conv)
                prev_num_features = cur_num_features
                

        # Last linear layer
        self.global_num_feats_ = prev_num_features
        if self.use_group_nom_:
            self.blocks_.append(torch.nn.GroupNorm(self.num_groups_, prev_num_features))
        self.blocks_.append(bn_act_do(prev_num_features, self.fc_drop_out_))
        self.blocks_.append(torch.nn.Linear(prev_num_features, self.fc_layer_features_))
        if self.use_group_nom_:
            self.blocks_.append(torch.nn.GroupNorm(self.num_groups_, prev_num_features))
        self.blocks_.append(bn_act_do(self.fc_layer_features_, self.fc_drop_out_))
        self.blocks_.append(torch.nn.Linear(self.fc_layer_features_, self.num_out_features_))
        

    def forward(self, 
        p_pc_in,
        p_in_features,
        p_return_layers = False):
        """Forward pass.

        Args:
            p_pc_in (Pointcloud): Input point cloud.
            p_in_features (tensor nxfi): Input features.
            p_return_layers (bool): Boolean that indicates if the layers are
                returned.
        Returns:
            tensor mxfo: Output features.
        """
        badnwidth_np = np.array([self.pdf_bandwith_ for i in range(self.num_dims_)], 
            dtype=np.float32)

        if p_return_layers:
            p_in_features.retain_grad()
            layer_list = [p_in_features]

        # Encoder.
        # For each level.
        cur_block_iter = 0
        cur_pc = p_pc_in
        cur_features = p_in_features
        neighborhoods = []

        #print(cur_features)
        for cur_level in range(self.num_levels_):
            
            cur_num_blocks = self.num_blocks_[cur_level]
            cur_radius = self.conv_radii_[cur_level]
            cur_pooling_radius = self.pooling_radii_[cur_level]

            torch_cur_radius = torch.as_tensor(
                np.array([cur_radius for i in range(self.num_dims_)], 
                dtype=np.float32), 
                device=p_pc_in.pts_.device)
            torch_cur_pool_radius = torch.as_tensor(
                np.array([cur_pooling_radius for i in range(self.num_dims_)], 
                dtype=np.float32), 
                device=p_pc_in.pts_.device)
            
            # Create the pooling layer.
            pt_pooling = pccnn_lib.pc.PointPooling(cur_pc, 
                torch_cur_pool_radius, self.pooling_method_)
            pool_pts = pt_pooling.pool_tensor(cur_pc.pts_)
            pool_batch_ids = pt_pooling.pool_tensor(cur_pc.batch_ids_)
            pool_pc = pccnn_lib.pc.Pointcloud(pool_pts, pool_batch_ids)
            pool_pc.compute_pdf(badnwidth_np*cur_radius)
            cur_features = pt_pooling.pool_tensor(cur_features)
            cur_pc = pool_pc

            # For each block.
            cur_neighborhood = None
            for cur_block_level in range(cur_num_blocks):

                # Evaluate the block.
                if cur_block_level > 0 or cur_level > 0:
                    if self.use_group_nom_:
                        cur_block = self.blocks_[cur_block_iter]
                        cur_features = cur_block(cur_features, cur_pc)
                        cur_block_iter += 1
                    cur_block = self.blocks_[cur_block_iter]
                    cur_features = cur_block(cur_features)
                    cur_block_iter += 1
                cur_block = self.blocks_[cur_block_iter]
                cur_features = cur_block(
                    p_pc_in = cur_pc,
                    p_pc_out = cur_pc,
                    p_in_features = cur_features,
                    p_radius = torch_cur_radius,
                    p_neighborhood = cur_neighborhood,
                    p_max_neighs = self.max_neighbors_)
                cur_block_iter += 1

                if cur_block_level == 0:
                    cur_neighborhood = cur_features[1]
                    cur_features = cur_features[0]   
                    neighborhoods.append(cur_neighborhood) 
                  
                if p_return_layers:
                    cur_features.retain_grad()
                    layer_list.append(cur_features)          

        # Global mean.
        batch_size = torch.max(cur_pc.batch_ids_)+1
        batch_id_indexs = cur_pc.batch_ids_.to(torch.int64)
        mean_features = torch.zeros((batch_size, self.global_num_feats_), 
            device=cur_features.device).to(torch.float32)
        num_pts = torch.zeros((batch_size), device=cur_features.device).to(torch.float32)
        aux_ones = torch.ones_like(batch_id_indexs).to(torch.float32)
        
        num_pts.index_add_(0, batch_id_indexs, aux_ones)
        mean_features.index_add_(0, batch_id_indexs, cur_features)
        cur_features = mean_features/torch.reshape(num_pts, (-1, 1))

        # Last fully connected.
        if self.use_group_nom_:
            cur_block = self.blocks_[cur_block_iter]
            cur_features = cur_block(cur_features)
            cur_block_iter += 1
        cur_block = self.blocks_[cur_block_iter]
        cur_features = cur_block(cur_features)
        cur_block_iter += 1
        cur_block = self.blocks_[cur_block_iter]
        cur_features = cur_block(cur_features)
        cur_block_iter += 1
        if self.use_group_nom_:
            cur_block = self.blocks_[cur_block_iter]
            cur_features = cur_block(cur_features)
            cur_block_iter += 1
        cur_block = self.blocks_[cur_block_iter]
        cur_features = cur_block(cur_features)
        cur_block_iter += 1
        cur_block = self.blocks_[cur_block_iter]
        cur_features = cur_block(cur_features)
        cur_block_iter += 1

        if p_return_layers:
            return cur_features, layer_list
        else:
            return cur_features