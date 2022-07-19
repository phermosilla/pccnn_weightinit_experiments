import numpy as np
import torch
import pccnn_lib

class SegmentationUNetModel(torch.nn.Module, pccnn_lib.pc.models.IModel):

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
        self.num_blocks_enc_ = p_config_dict['num_blocks_enc']
        self.num_blocks_dec_ = p_config_dict['num_blocks_dec']
        self.pooling_radii_ = p_config_dict['pooling_radii']
        self.conv_radii_ = p_config_dict['conv_radii']
        self.feature_sizes_ = p_config_dict['feature_sizes']
        self.first_pooling_features_ = p_config_dict['first_pooling_features']
        self.first_pooling_radius_ = p_config_dict['first_pooling_radius']
        self.last_upsampling_radius_ = p_config_dict['last_upsampling_radius']
        self.use_batch_nom_ = p_config_dict['use_batch_norm']
        self.act_funct_ = p_config_dict['act_funct']
        self.drop_out_ = p_config_dict['drop_out']
        self.pdf_bandwith_ = p_config_dict['pdf_bandwidth']
        self.pooling_method_ = p_config_dict['pooling_method']
        self.max_neighbors_ = p_config_dict['max_neighbors']

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
        def bn_act_do(num_features, drop_out = self.drop_out_,
            p_batch_norm = self.use_batch_nom_):
            layer_list = []
            if p_batch_norm:
                layer_list.append(torch.nn.BatchNorm1d(num_features))
            else:
                layer_list.append(pccnn_lib.pc.Bias(num_features))
            layer_list.append(cur_act_funct)
            if drop_out > 0.0:
                layer_list.append(torch.nn.Dropout(drop_out))
            return torch.nn.Sequential(*layer_list)

        # Encoder.
        print()
        print("############ Segmentation Model")
        print()
        print("######## Encoder")

        # First pooling.
        print("### First Pooling (", self.first_pooling_radius_*3.0, 
            self.num_in_features_, self.first_pooling_features_, ")")
        cur_conv = p_conv_factory.create_conv_layer(
            self.num_dims_, 
            self.num_in_features_, 
            self.first_pooling_features_)
        self.blocks_.append(cur_conv)
        self.add_conv(cur_conv)
        
        # For each level.
        prev_num_features = self.first_pooling_features_
        for cur_level in range(self.num_levels_):
            
            print()
            print("### Level", cur_level)
            cur_num_blocks = self.num_blocks_enc_[cur_level]
            cur_num_features = self.feature_sizes_[cur_level]
            cur_radius = self.conv_radii_[cur_level]

            # For each block.
            self.blocks_.append(
                torch.nn.Sequential(
                    bn_act_do(prev_num_features),
                    torch.nn.Linear(
                        prev_num_features, cur_num_features)))
            
            # For each block.
            for cur_block in range(cur_num_blocks):
                print("Block (", cur_radius, cur_num_features, ")")
                # Create the block.
                self.blocks_.append(bn_act_do(cur_num_features))
                cur_conv = p_conv_factory.create_conv_layer(
                        self.num_dims_, 
                        cur_num_features, 
                        cur_num_features)
                self.blocks_.append(cur_conv)
                self.add_conv(cur_conv)
            prev_num_features = cur_num_features
                 

        # Decoder.
        print()
        print("######## Decoder")
        for cur_level in range(self.num_levels_-2, -1, -1):
            
            print()
            print("### Level", cur_level)
            cur_num_blocks = self.num_blocks_dec_[cur_level]
            cur_num_features = self.feature_sizes_[cur_level]
            cur_radius = self.conv_radii_[cur_level]
            
            # Create the upsampling layer.             
            print("Upsampling (", cur_radius, 
                prev_num_features, cur_num_features, ")")    
            self.blocks_.append(bn_act_do(prev_num_features))
            cur_conv = p_conv_factory.create_conv_layer(
                self.num_dims_, 
                prev_num_features, 
                cur_num_features)
            self.blocks_.append(cur_conv)
            self.add_conv(cur_conv)

            # For each block.
            self.blocks_.append(
                torch.nn.Sequential(
                    bn_act_do(cur_num_features*2),
                    torch.nn.Linear(
                        cur_num_features*2, cur_num_features)))

            for cur_block in range(cur_num_blocks):
                print("Block (", cur_radius, cur_num_features, ")")
                # Create the block.
                self.blocks_.append(bn_act_do(cur_num_features))
                cur_conv = p_conv_factory.create_conv_layer(
                        self.num_dims_, 
                        cur_num_features, 
                        cur_num_features)
                self.blocks_.append(cur_conv)
                self.add_conv(cur_conv)
            prev_num_features = cur_num_features
        

        # Last upsampling.
        print()
        print("### Last Upsampling (", self.last_upsampling_radius_, 
            prev_num_features, self.num_out_features_, ")")
        self.blocks_.append(bn_act_do(prev_num_features))
        cur_conv = p_conv_factory.create_conv_layer(
            self.num_dims_, 
            prev_num_features, 
            self.num_out_features_)
        self.blocks_.append(cur_conv)
        self.add_conv(cur_conv)
        self.last_bias_ = torch.nn.Parameter(
            torch.empty(1, self.num_out_features_))
        self.last_bias_.data.fill_(0.0)
        

    def forward(self, 
        p_pc_in,
        p_pc_out,
        p_in_features,
        p_return_layers = False):
        """Forward pass.

        Args:
            p_pc_in (Pointcloud): Input point cloud.
            p_pc_out (Pointcloud): Output point cloud.
            p_in_features (tensor nxfi): Input features.
            p_return_layers (bool): Boolean that indicates if the layers are
                returned.
        Returns:
            tensor mxfo: Output features.
        """

        if p_return_layers:
            p_in_features.retain_grad()
            layer_list = [p_in_features]

        # Encoder.
        # For each level.
        cur_block_iter = 0
        cur_pc = p_pc_in
        cur_features = p_in_features
        point_cloud_list = []
        skip_links = []
        neighborhoods = []
        for cur_level in range(self.num_levels_):
            
            cur_num_blocks = self.num_blocks_enc_[cur_level]
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
            pool_batch_ids = pt_pooling.pool_tensor(cur_pc.batch_ids_, True)
            pool_pc = pccnn_lib.pc.Pointcloud(pool_pts, pool_batch_ids, cur_pc.dim_manifold_)
            pool_pc.compute_pdf(self.pdf_bandwith_*torch_cur_pool_radius)
            point_cloud_list.append(pool_pc)
    
            # Pool features.
            if cur_level > 0:
               cur_features = pt_pooling.pool_tensor(cur_features)
            else:
                torch_first_pooling_radius = torch.as_tensor(
                    np.array([self.first_pooling_radius_*1.5 for i in range(self.num_dims_)], 
                    dtype=np.float32), 
                    device=p_pc_in.pts_.device)
                cur_block = self.blocks_[cur_block_iter]
                cur_block_iter += 1
                cur_features, _ = cur_block(
                    p_pc_in = cur_pc,
                    p_pc_out = pool_pc,
                    p_in_features = cur_features,
                    p_radius = torch_first_pooling_radius,
                    p_max_neighs = self.max_neighbors_)
                if p_return_layers:
                    cur_features.retain_grad()
                    layer_list.append(cur_features)

            # Feature transform.
            cur_block = self.blocks_[cur_block_iter]
            cur_features = cur_block(cur_features)
            cur_block_iter += 1

            # For each block.
            cur_pc = pool_pc
            cur_neighborhood = None
            for cur_block_level in range(cur_num_blocks):

                # Evaluate the block.
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
                if cur_block_level == 0:
                    cur_neighborhood = cur_features[1]
                    cur_features = cur_features[0]   
                    neighborhoods.append(cur_neighborhood)  
                if p_return_layers:
                    cur_features.retain_grad()
                    layer_list.append(cur_features)          
                    
                cur_block_iter += 1
            
            skip_links.append(cur_features)

        # Decoder
        for cur_level in range(self.num_levels_-2, -1, -1):

            cur_num_blocks = self.num_blocks_dec_[cur_level]
            cur_radius = self.conv_radii_[cur_level]

            torch_cur_radius = torch.as_tensor(
                np.array([cur_radius for i in range(self.num_dims_)], 
                dtype=np.float32), 
                device=p_pc_in.pts_.device)
            
            # Upsampling layer.
            cur_block = self.blocks_[cur_block_iter]
            cur_features = cur_block(cur_features)
            cur_block_iter += 1
            
            upsample_pc = point_cloud_list[cur_level]
            cur_block = self.blocks_[cur_block_iter]
            cur_features, _ = cur_block(
                p_pc_in = cur_pc,
                p_pc_out = upsample_pc,
                p_in_features = cur_features,
                p_radius = torch_cur_radius,
                p_max_neighs = self.max_neighbors_)
            if p_return_layers:
                cur_features.retain_grad()
                layer_list.append(cur_features)
            cur_pc = upsample_pc
            cur_block_iter += 1

            # Skip links.
            skip_features = skip_links[cur_level]
            cur_features = torch.cat((cur_features, skip_features), -1)

            cur_block = self.blocks_[cur_block_iter]
            cur_features = cur_block(cur_features)
            cur_block_iter += 1

            # For each block.
            init_block_feats = cur_features
            cur_neighborhood = neighborhoods[cur_level]
            for cur_block_level in range(cur_num_blocks):
                
                # Execute the block.
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
                if p_return_layers:
                    cur_features.retain_grad()
                    layer_list.append(cur_features)
                cur_block_iter += 1


        # Last upsampling.
        cur_block = self.blocks_[cur_block_iter]
        cur_features = cur_block(cur_features)
        cur_block_iter += 1
        cur_block = self.blocks_[cur_block_iter]
        torch_last_upsampling_radius = torch.as_tensor(
            np.array([self.last_upsampling_radius_ for i in range(self.num_dims_)], 
            dtype=np.float32), 
            device=p_pc_in.pts_.device)
        cur_features, _ = cur_block(
            p_pc_in = cur_pc,
            p_pc_out = p_pc_out,
            p_in_features = cur_features,
            p_radius = torch_last_upsampling_radius,
            p_max_neighs = self.max_neighbors_)
        cur_features = cur_features + self.last_bias_
        
        if p_return_layers:
            return cur_features, layer_list
        else:
            return cur_features