import glob, os
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
def confusionMatrix(gt, pred, show=False):
    TN, FP, FN, TP = confusion_matrix(gt, pred).ravel()
    f1_score = (2 * TP) / (2 * TP + FP + FN)
    num_samples = len([x for x in gt if x == 1])
    average_recall = TP / num_samples
    return f1_score, average_recall
if __name__ == '__main__':
    import opts
    args = opts.parse_args()
    
    def _cal_metrics(tp, n, m):
        recall = float(tp) / m if m > 0 else 0
        precision = float(tp) / n if n > 0 else 0
        f1_score = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0
        
    
        print('Precision = ', round(precision, 4))
        print('Recall = ', round(recall, 4))
        print('F1-Score = ', round(f1_score, 4)) 
        
        return recall, precision, f1_score
    
    res_list = glob.glob(
        os.path.join(args.output, '*/best_res.csv')
    )
    df_list = []

    for res_file in res_list:
        df = pd.read_csv(res_file)
        df_list.append(df)



    full_df = pd.concat(df_list, ignore_index=True)
    
    full_df = list(full_df.sum(axis=0).values)
    # micro_tp, micro_n, micro_m, macro_tp, macro_n, macro_m, all_tp, all_n, all_m= full_df
    #
    #
    # print(f'Micro result: TP:{micro_tp}, FP:{micro_n - micro_tp}, FN:{micro_m - micro_tp}')
    # mic_rec, mic_pr, mic_f1 = _cal_metrics(micro_tp, micro_n, micro_m)
    #
    # print(f'Macro result: TP:{macro_tp}, FP:{macro_n - macro_tp}, FN:{macro_m - macro_tp}')
    # mac_rec, mac_pr, mac_f1 = _cal_metrics(macro_tp, macro_n, macro_m)
    #
    # print(f'Total result: TP:{all_tp}, FP:{all_n - all_tp}, FN:{all_m - all_tp}')
    # all_rec, all_pr, all_f1 = _cal_metrics(all_tp, all_n, all_m)


    micro_tp, micro_n, micro_m, macro_tp, macro_n, macro_m, all_tp, all_n, all_m, \
    cmicro_tp, cmicro_n, cmicro_m, cmacro_tp, cmacro_n, cmacro_m, call_tp, call_n, call_m  ,po_tp, po_fp, po_fn,ne_tp, ne_fp, ne_fn,sur_tp, sur_fp, sur_fn,uf1= full_df

    print(f'Micro result: TP:{micro_tp}, FP:{micro_n - micro_tp}, FN:{micro_m - micro_tp}')
    mic_rec, mic_pr, mic_f1 = _cal_metrics(micro_tp, micro_n, micro_m)

    print(f'Macro result: TP:{macro_tp}, FP:{macro_n - macro_tp}, FN:{macro_m - macro_tp}')
    mac_rec, mac_pr, mac_f1 = _cal_metrics(macro_tp, macro_n, macro_m)

    print(f'Total result: TP:{all_tp}, FP:{all_n - all_tp}, FN:{all_m - all_tp}')
    all_rec, all_pr, all_f1 = _cal_metrics(all_tp, all_n, all_m)

    print(f'Micro recognition: TP:{cmicro_tp}, FP:{cmicro_n - cmicro_tp}, FN:{cmicro_m - cmicro_tp}')
    mic_rec, mic_pr, mic_f1 = _cal_metrics(cmicro_tp, cmicro_n, cmicro_m)

    print(f'Po result: TP:{po_tp}, FP:{po_fp}, FN:{po_fn}')
    po_rec, po_pr, po_f1 = _cal_metrics(po_tp, po_tp+po_fp, po_tp+po_fn)
    print(f'Ne result: TP:{ne_tp}, FP:{ne_fp}, FN:{ne_fn}')
    ne_rec, ne_pr, ne_f1 = _cal_metrics(ne_tp, ne_tp + ne_fp, ne_tp + ne_fn)
    print(f'Sur result: TP:{sur_tp}, FP:{sur_fp}, FN:{sur_fn}')
    sur_rec, sur_pr, sur_f1 = _cal_metrics(sur_tp, sur_tp + sur_fp, sur_tp + sur_fn)
    # print((po_f1+ne_f1+sur_f1)/3)
    upr=(po_pr+ne_pr+sur_pr)/3
    urec = (po_rec + ne_rec + sur_rec) / 3
    recf1=2 * upr * urec / (upr + urec) if (upr + urec) > 0 else 0
    print(f'recognition f1:{recf1}')
