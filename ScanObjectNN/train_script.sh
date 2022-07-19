python train.py --conv_type kpconvn --no_bn                     # Variance-aware weight init
python train.py --conv_type kpconvn --no_const_var              # Batch normalization
python train.py --conv_type kpconvn --use_gn --no_const_var     # Group normalization
python train.py --conv_type kpconvn --no_bn --no_const_var      # Standard weight init