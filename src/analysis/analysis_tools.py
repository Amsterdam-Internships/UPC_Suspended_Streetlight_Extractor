from fnmatch import fnmatch
import numpy as np

from ..labels import Labels
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.cluster import DBSCAN


def conf_matrix(true, pred):
    pred[pred>15] = Labels.ARMATUUR
    pred[pred==14] = Labels.TRAM_CABLE
    return confusion_matrix(true,pred, labels=[0,1,2,3,11,12,13,15])

def get_objectwise_stats(points, pred_labels, true_labels):

    pred_mask = pred_labels >= Labels.ARMATUUR
    true_mask = true_labels == Labels.ARMATUUR

    pred_clustering = DBSCAN(eps=1, min_samples=1).fit(points[pred_mask])
    true_clustering = DBSCAN(eps=1, min_samples=1).fit(points[true_mask])

    tp = 0
    fn = len(set(true_clustering.labels_))
    fp = len(set(pred_clustering.labels_))

    true_cluster_masks = []
    for true_cl in set(true_clustering.labels_):
        true_cluster_masks.append(true_clustering.labels_ == true_cl)

    for pred_cl in set(pred_clustering.labels_):
        pred_mask = pred_clustering.labels_ == pred_cl
        for i in range(len(true_cluster_masks)):
            if np.sum(pred_mask[true_cluster_masks[i]]) > 0:
                tp += 1
                fn -= 1
                fp -= 1
                del true_cluster_masks[i]
                break

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)

    return recall, precision, (tp,fp,fn)

def get_label_stats(labels):
    """Returns a string describing statistics based on labels."""
    N = len(labels)
    labels, counts = np.unique(labels, return_counts=True)
    stats = f'Total: {N:25} points\n'
    for label, cnt in zip(labels, counts):
        name = Labels.get_str(label)
        perc = (float(cnt) / N) * 100
        stats += f'Class {label:2}, {name:14} ' +\
                 f'{cnt:7} points ({perc:4.1f} %)\n'
    return stats


def get_cable_stats_m1(labels, true_mask):
    """Returns a string describing statistics based on labels."""
    N = len(labels)
    labels_, counts = np.unique(labels, return_counts=True)
    N_0 = np.sum(labels == 0)
    stats = f'Reduced: {N:10} --> {N_0:10} points\n'

    report = {
        'start_pts':N,
        'end_pts':N_0,
        'reduce_per':round((N-N_0)/N,2),
        'cable_pts': np.sum(true_mask)}

    for label, cnt in zip(labels_, counts):
        name = Labels.get_str(label)
        perc = (float(cnt) / N) * 100
        N_cable = np.sum(true_mask & (labels == label))
        recall = N_cable / np.sum(true_mask)
        stats += f'Class {label:2}, {name:14} ' +\
                 f'{cnt:7} points ({perc:4.1f} %) ' +\
                 f'/ {N_cable:7} cable-points ({recall:4.3f} %)\n'
        if label != 0:
            report['FN_'+name] = N_cable

    report['f1_score'] = classification_report((~true_mask).astype(int), labels, labels=[0,1,2], target_names=['unclassified', 'ground', 'building'], output_dict=True, zero_division=1)['unclassified']['f1-score']

    return stats, report

def get_cable_stats_m2(labels, true_labels):
    """Returns a string describing statistics based on labels."""

    label_mask = labels == 11
    true_mask = (true_labels < 15) & (true_labels > 10)

    report = {
        'recall': recall_score(true_mask, label_mask),
        'precision': precision_score(true_mask, label_mask),
        'f1_score': f1_score(true_mask, label_mask)
    }
    
    return report

def get_cable_stats_m3(labels, true_labels):
    """Returns a string describing statistics based on labels."""

    label_mask = (labels > 10) & (labels < 15)
    true_mask = (true_labels > 10) & (true_labels < 15)

    report = {
        'recall': recall_score(true_mask, label_mask),
        'precision': precision_score(true_mask, label_mask),
        'f1_score': f1_score(true_mask, label_mask),
        'TP': np.sum(true_mask & label_mask),
        'FP': np.sum(~true_mask & label_mask),
        'FN': np.sum(true_mask & ~label_mask)
    }
    
    return report

def get_cable_stats_m4(labels, true_labels):
    """Returns a string describing statistics based on labels."""

    label_mask = labels >= Labels.ARMATUUR
    true_mask = true_labels >= Labels.ARMATUUR

    report = {
        'recall': recall_score(true_mask, label_mask, zero_division=0),
        'precision': precision_score(true_mask, label_mask, zero_division=0),
        'f1_score': f1_score(true_mask, label_mask, zero_division=0)
    }

    return report

def get_cable_stats(labels, true_labels):

    mask = labels == 12
    true_mask = true_labels > 10

    recall = np.sum(mask & true_mask)/np.sum(true_mask)
    precision = np.sum(mask & true_mask) / np.sum(mask)
    f1_score = 2/(1/recall + 1/precision)

    f2_score = 5 / (4/precision + 1/recall)

    report = {
        'start_pts': len(labels),
        'unmasked_pts': np.sum(mask),
        'reduce_per': round((len(labels)-np.sum(mask)) / len(labels), 3),
        'recall': recall,
        'precision': precision,
        'f1_score': f1_score,
        'f2_score': f2_score,
        'ground_points (%)': round(np.sum(labels==1) / len(labels), 3),
        'ground_points (FN)': np.sum((labels==1) & true_mask),
        'bld_points (%)': round(np.sum(labels==2) / len(labels), 3),
        'bld_points (FN)': np.sum((labels==2) & true_mask),
        'sky_points (%)': round(np.sum(labels==3) / len(labels), 3),
        'sky_points (FN)': np.sum((labels==3) & true_mask)
    }

    return report

    





    
    
