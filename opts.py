import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training script for XXX')
    
    # parser.add_argument("--dataset", type=str, default='samm',
    #                     choices=['cas(me)^2', 'samm'])
    # parser.add_argument("--output", type=str, default='/code/CodeTest/wzl/AUW-GCN/output/sammclass/',#'/code/CodeTest/wzl/AUW-GCN/output/casmeallsub_flowa/',#"/code/CodeTest/wzl/AUW-GCN/output/newclassmaloss0/",
    #                     help='dir for saving logs, models, etc.')
    # parser.add_argument("--subject", type=str, default='samm_007',
    #                     help='Leave out subject for evaluation')
    # #'/data/DataSets/cas(me)^2/feature_segment_fblabel/'
    # parser.add_argument("--input", type=str, default='/data/DataSets/cas(me)^2/landmarkmat_2model_samm/',#'/data/DataSets/cas(me)^2/feature_segmentsubjectflowa/',#/data/DataSets/cas(me)^2/feature_segment_fblabel/',
    #                     help='dir for inputdata')

    parser.add_argument("--dataset", type=str, default='samm',
                        choices=['cas(me)2','cas(me)^2', 'samm','all'])
    parser.add_argument("--output", type=str, default='/code/CodeTest/wzl/RRSN-MEA-0324/output/samm0729/',
                        # '/code/CodeTest/wzl/AUW-GCN/output/casmeallsub_flowa/',#"/code/CodeTest/wzl/AUW-GCN/output/newclassmaloss0/",
                        help='dir for saving logs, models, etc.')
    parser.add_argument("--subject", type=str, default='samm_016',
                        help='Leave out subject for evaluation')
    # '/data/DataSets/cas(me)^2/feature_segment_fblabel/'
    parser.add_argument("--input", type=str, default='/data/DataSets/cas(me)^2/feature_segment_sammlabel272/',
                # '/data/DataSets/cas(me)^2/feature_segmentsubjectflowa/',#/data/DataSets/cas(me)^2/feature_segment_fblabel/',
                help='dir for inputdata')
    # parser.add_argument("--input", type=str, default='/data/DataSets/CASME2pro/feature_segment/',
    #                     # '/data/DataSets/cas(me)^2/feature_segmentsubjectflowa/',#/data/DataSets/cas(me)^2/feature_segment_fblabel/',
    #                     help='dir for inputdata')

    # parser.add_argument("--dataset", type=str, default='cas(me)^2',
    #                     choices=['cas(me)^2', 'samm'])
    # parser.add_argument("--output", type=str, default='/code/CodeTest/wzl/AUW-GCN/output/modelI3d/',
    #                     # '/code/CodeTest/wzl/AUW-GCN/output/casmeallsub_flowa/',#"/code/CodeTest/wzl/AUW-GCN/output/newclassmaloss0/",
    #                     help='dir for saving logs, models, etc.')
    # parser.add_argument("--subject", type=str, default='casme_015',
    #                     help='Leave out subject for evaluation')
    # # '/data/DataSets/cas(me)^2/feature_segment_fblabel/'
    # parser.add_argument("--input", type=str, default='/data/DataSets/cas(me)^2/roi_feats_I3d/',
    #                     # '/data/DataSets/cas(me)^2/feature_segmentsubjectflowa/',#/data/DataSets/cas(me)^2/feature_segment_fblabel/',
    #                     help='dir for inputdata')

    args = parser.parse_args()
    # parser.add_argument("--dataset", type=str, default='samm',
    #                     choices=['cas(me)^2', 'samm'])
    # parser.add_argument("--output", type=str, default='/code/CodeTest/wzl/AUW-GCN/output/landmarkmat_samm_notri/',
    #                     # '/code/CodeTest/wzl/AUW-GCN/output/casmeallsub_flowa/',#"/code/CodeTest/wzl/AUW-GCN/output/newclassmaloss0/",
    #                     help='dir for saving logs, models, etc.')
    # parser.add_argument("--subject", type=str, default='samm_006',
    #                     help='Leave out subject for evaluation')
    # # '/data/DataSets/cas(me)^2/feature_segment_fblabel/'
    # parser.add_argument("--input", type=str, default='/data/DataSets/cas(me)^2/landmarkmat_withlabel_samm/',
    #                     # '/data/DataSets/cas(me)^2/feature_segmentsubjectflowa/',#/data/DataSets/cas(me)^2/feature_segment_fblabel/',
    #                     help='dir for inputdata')
    #
    # args = parser.parse_args()
    return args