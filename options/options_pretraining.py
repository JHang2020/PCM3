# graph based model arguments
agcn_model_arguments = {
   "num_class": 128,
   "num_point": 25,
   "num_person": 2,
   'graph_args': {
     'labeling_mode': 'spatial'}
}

#image based model arguments
hcn_model_arguments = {
   "in_channel":3,
   "out_channel":64,
   "window_size":64,
   "num_joint":25,
   "num_person":2,
   "num_class":128
 }

#Sequence based model arguments
bi_gru_model_arguments = {
   "en_input_size":150,
   "en_hidden_size":1024,
   "en_num_layers":3,
   "num_class":128
 }

input_frame = 64

class  opts_ntu_60_cross_view_hiclr():

  def __init__(self):

   self.agcn_model_args = agcn_model_arguments

   self.hcn_model_args = hcn_model_arguments

   self.bi_gru_model_args = bi_gru_model_arguments
   # feeder
   self.train_feeder_args = {
     'data_path': '/mnt/netdisk/linlilang/SkeletonContrast/data/NTU-RGB-D-60-AGCN/xview/train_data_joint.npy',
     'num_frame_path': '/mnt/netdisk/linlilang/SkeletonContrast/data/NTU-RGB-D-60-AGCN/xview/train_num_frame.npy',
     'l_ratio': [0.1,1],
     'input_size': input_frame
   }
class  opts_ntu_60_cross_view_transformer():

  def __init__(self):

   self.agcn_model_args = agcn_model_arguments

   self.hcn_model_args = hcn_model_arguments

   self.bi_gru_model_args = bi_gru_model_arguments
   # feeder
   self.train_feeder_args = {
      'data_path': '/mnt/netdisk/linlilang/SkeleMixCLR-main/data/Base/NTU60_frame32/xview/train_position.npy',
      'label_path': '/mnt/netdisk/linlilang/CrosSCLR/data/NTU-RGB-D/xview/train_label.pkl',
      'shear_amplitude': 0.5,
      'temperal_padding_ratio': 6,
      'mmap': True
   }

class  opts_ntu_60_cross_view():

  def __init__(self):

   self.agcn_model_args = agcn_model_arguments

   self.hcn_model_args = hcn_model_arguments

   self.bi_gru_model_args = bi_gru_model_arguments
   
   # feeder
   self.train_feeder_args = {
     'data_path': '/mnt/netdisk/linlilang/SkeletonContrast/data/NTU-RGB-D-60-AGCN/xview/train_data_joint.npy',
     'num_frame_path': '/mnt/netdisk/linlilang/SkeletonContrast/data/NTU-RGB-D-60-AGCN/xview/train_num_frame.npy',
     'l_ratio': [0.1,1],
     'input_size': input_frame
   }

class  opts_ntu_60_cross_subject():

  def __init__(self):

   self.agcn_model_args = agcn_model_arguments

   self.hcn_model_args = hcn_model_arguments

   self.bi_gru_model_args = bi_gru_model_arguments
   
   # feeder
   self.train_feeder_args = {
     'data_path': '/mnt/netdisk/linlilang/SkeletonContrast/data/NTU-RGB-D-60-AGCN/xsub/train_data_joint.npy',
     'num_frame_path': '/mnt/netdisk/linlilang/SkeletonContrast/data/NTU-RGB-D-60-AGCN/xsub/train_num_frame.npy',
     'l_ratio': [0.1,1],
     'input_size': input_frame
   }

class  opts_ntu_120_cross_subject():

  def __init__(self):

   self.agcn_model_args = agcn_model_arguments

   self.hcn_model_args = hcn_model_arguments

   self.bi_gru_model_args = bi_gru_model_arguments
   
   # feeder
   self.train_feeder_args = {
     'data_path': '/mnt/netdisk/zhangjh/data/CMD/NTU120/xsub/train_data_joint.npy',
     'num_frame_path': '/mnt/netdisk/zhangjh/data/CMD/NTU120/xsub/train_num_frame.npy',
     'l_ratio': [0.1,1],
     'input_size': input_frame
   }

class  opts_ntu_120_cross_setup():

  def __init__(self):

   self.agcn_model_args = agcn_model_arguments

   self.hcn_model_args = hcn_model_arguments

   self.bi_gru_model_args = bi_gru_model_arguments
   
   # feeder
   self.train_feeder_args = {
     'data_path': '/mnt/netdisk/zhangjh/data/CMD/NTU120/xset/train_data_joint.npy',
     'num_frame_path': '/mnt/netdisk/zhangjh/data/CMD/NTU120/xset/train_num_frame.npy',
     'l_ratio': [0.1,1],
     'input_size': input_frame
   }


# PKU-MMD
class  opts_pku_v2_cross_view():

  def __init__(self):

   self.agcn_model_args = agcn_model_arguments

   self.hcn_model_args = hcn_model_arguments

   self.bi_gru_model_args = bi_gru_model_arguments
   
   # feeder
   self.train_feeder_args = {
     'data_path': './data/PKU-MMD-v2-AGCN/xview/train_data_joint.npy',
     'num_frame_path': './data/PKU-MMD-v2-AGCN/xview/train_num_frame.npy',
     'l_ratio': [0.1,1],
     'input_size': input_frame
   }

  
class  opts_pku_v2_cross_subject():

  def __init__(self):

   self.agcn_model_args = agcn_model_arguments

   self.hcn_model_args = hcn_model_arguments

   self.bi_gru_model_args = bi_gru_model_arguments
   
   # feeder
   self.train_feeder_args = {
     'data_path': '/mnt/netdisk/zhangjh/data/CMD/PKUMMD2/xsub/train_data_joint.npy',
     'num_frame_path': '/mnt/netdisk/zhangjh/data/CMD/PKUMMD2/xsub/train_num_frame.npy',
     'l_ratio': [0.1,1],
     'input_size': input_frame
   }
  
class  opts_pku_v1_cross_subject():

  def __init__(self):

   self.agcn_model_args = agcn_model_arguments

   self.hcn_model_args = hcn_model_arguments

   self.bi_gru_model_args = bi_gru_model_arguments
   
   # feeder
   self.train_feeder_args = {
     'data_path': '/mnt/netdisk/zhangjh/data/CMD/PKUMMD1/xsub/train_data_joint.npy',
     'num_frame_path': '/mnt/netdisk/zhangjh/data/CMD/PKUMMD1/xsub/train_num_frame.npy',
     'l_ratio': [0.1,1],
     'input_size': input_frame
   }