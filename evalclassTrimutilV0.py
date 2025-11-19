import opts
import torch
import opts
from modelclassTriV0 import AUwGCN
from datasetsclassmutil import LOSO_DATASET

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import yaml
import numpy as np
from utils.eval_utilsclassTrimutil import eval_single_epoch, nms_single_epoch, calculate_epoch_metrics, choose_best_epoch
import pandas as pd
def class_single_epoch():

    csv_dir = os.path.join(
        opt['output_dir_name'], 'output_csv'
    )
    predict_file = os.path.join(
        csv_dir, 'proposals_epoch_' + str(epoch).zfill(3) + '.csv'
    )
    # no proposals generated for this epoch
    if not os.path.exists(predict_file):
        return

    nms_dir = os.path.join(
        opt['output_dir_name'], 'nms_csv'
    )
    if not os.path.exists(nms_dir):
        os.makedirs(nms_dir)

    nms_file = os.path.join(
        nms_dir, 'final_proposals_epoch_' + str(epoch).zfill(3) + '.csv'
    )

    df = pd.read_csv(predict_file)
    df = df.groupby(['video_name', "type_idx"], group_keys=False).apply(
        lambda x: nms(x, opt)).reset_index(drop=True)

    if os.path.exists(nms_file):
        os.remove(nms_file)
    df.to_csv(nms_file, index=False)

if __name__ == '__main__':
    import opts
    args = opts.parse_args()
    
    # load config & params.
    with open("./config.yaml", encoding="UTF-8") as f:
        yaml_config = yaml.safe_load(f)
        if args.dataset is not None:
            dataset = args.dataset
        else:
            dataset = yaml_config['dataset']
        opt = yaml_config[dataset]
        opt['dataset'] = dataset
        
    subject = args.subject
    
    # update opt. according to args.
    opt['output_dir_name'] = os.path.join(args.output, subject) # ./debug/casme_016
    opt['model_save_root'] = os.path.join(opt['output_dir_name'], 'models')  # ./debug/casme_016/models/
    opt['subject'] = subject
    
    # define dataset & loader
    dataset = LOSO_DATASET(opt, args.input,'test', subject)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=opt['batch_size'], 
                                             shuffle=False,
                                             num_workers=2,
                                             pin_memory=True, 
                                             drop_last=False)
    
    # define and load model
    device = opt['device'] if torch.cuda.is_available() else 'cpu'
    #-------------
    # num_features_stream1 = 2  # x, y 坐标
    # num_features_stream2 = 2
    # hidden_channels = 64
    # file = np.load("/data/DataSets/cas(me)^2/feature_segmentsubject_lmandop/test/casme_015/casme_015_0101_0000.npz")
    # edge_index = torch.from_numpy(np.load("/code/CodeTest/wzl/AUW-GCN/assets/cas(me)^2.npy"))
    # model = AUwGCN(opt, num_features_stream1, num_features_stream2, hidden_channels, edge_index)
    model = AUwGCN(opt)
    model = model.to(device)
    
    # evaluate each ckpt's model and generate proposals
    # after generating proposals, NMS to reduce overlapped proposals
    epoch_begin = opt['epoch_begin']
    for epoch in range(opt['epochs']):

        if epoch >= epoch_begin:
            with torch.no_grad():

                weight_file = os.path.join(
                    opt["model_save_root"],
                    "checkpoint_epoch_" + str(epoch).zfill(3) + ".pth.tar")
                checkpoint = torch.load(weight_file,
                                        map_location=torch.device("cpu"))
                model.load_state_dict(checkpoint['state_dict'])
                eval_single_epoch(opt, model, dataloader, epoch, device)
                nms_single_epoch(opt, epoch)

   # calculate metrics of all the epochs
    calculate_epoch_metrics(opt)

    
    # choose the best epoch according to criterion
    choose_best_epoch(opt, criterion='analyse')
