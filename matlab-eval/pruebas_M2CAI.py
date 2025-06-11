import numpy as np
from scipy.ndimage import label
import glob
from scipy.io import loadmat
import os

def evaluate(gtLabelID, predLabelID, fps):
    """
    Evalúa el desempeño de un modelo de reconocimiento de fases quirúrgicas.

    Parámetros:
        gtLabelID (np.ndarray): Arreglo 1D con los labels verdaderos (ground truth).
        predLabelID (np.ndarray): Arreglo 1D con los labels predichos.
        fps (int): Número de frames por segundo del video.

    Retorna:
        res (list): Jaccard index por fase (con relaxed boundary), NaN si no aparece en GT.
        prec (list): Precisión por fase (con relaxed boundary).
        rec (list): Recall por fase (con relaxed boundary).
        acc (float): Exactitud total del video (con relaxed boundary).
    """

    gtLabelID += 1
    predLabelID += 1
    
    oriT = 10 * fps  # Duración de la ventana relajada (10 segundos en frames)

    res = []
    prec = []
    rec = []
    diff = predLabelID - gtLabelID  # Diferencia entre predicción y GT
    #diff = diff[0] # remove innecesary dimension
    updatedDiff = np.zeros(diff.shape)  # Se irá modificando con reglas de relajación

    nPhases = 8

    for iPhase in range(1, nPhases + 1):
        # Encontrar regiones conectadas donde la fase actual está presente en el GT
        mask_gt = (gtLabelID == iPhase).astype(int)
        labeled_gt, num_objects = label(mask_gt)

        # Iteramos sobre cada una de esas regiones conectadas
        for conn_id in range(1, num_objects + 1):
            #Encuentro la primer region de la fase en el Gt
            indices = np.where(labeled_gt == conn_id)[-1]
            startIdx = indices.min()
            endIdx = indices.max()

            curDiff = diff[startIdx:endIdx + 1].copy()

            # Determinar tamaño de ventana relajada (no mayor que la duración de la fase)
            t = min(oriT, len(curDiff))

            # Aplicar reglas de transición relajada, específicas por fase
            if iPhase in [5, 6]:
                #Working
                mask_late = curDiff[:t] == -1
                curDiff[np.where(mask_late)[0]] = 0 # late transition

                last_t = curDiff[-t:]  # últimos t elementos
                mask_early = (last_t == 1) | (last_t == 2)
                curDiff[:t][mask_early] = 0 # early transition
            
            elif iPhase in [7, 8]:
                # Working
                first_t = curDiff[:t]  # primeros t elementos
                mask_last = (first_t == -1) | (first_t == -2)
                curDiff[:t][mask_last] = 0 # late transition

                last_t = curDiff[-t:]  # primeros t elementos
                mask_early = (last_t == 1) | (last_t == 2)
                curDiff[:t][mask_early] = 0 # late transition
            
            else:
                #Working
                mask_first_t = curDiff[:t] == -1
                curDiff[np.where(mask_first_t)[0]] = 0 # late transition
                curDiff[:t][curDiff[-t:] == 1] = 0 # early transition
            

            updatedDiff[startIdx:endIdx + 1] = curDiff

            # Cálculo de Jaccard, Precisión y Recall por fase
    for iPhase in range(1, nPhases + 1):
        mask_gt = (gtLabelID == iPhase).astype(int)
        mask_pred = (predLabelID == iPhase).astype(int)

        # Regiones conectadas para la fase actual
        _, gt_objects = label(mask_gt)
        _, pred_objects = label(mask_pred)

        if gt_objects == 0:
            # Si no hay presencia de la fase en GT, se asigna NaN
            res.append(np.nan)
            prec.append(np.nan)
            rec.append(np.nan)
            continue

        union_idx = np.where((gtLabelID == iPhase) | (predLabelID == iPhase))[-1]
        tp = np.sum(updatedDiff[union_idx] == 0)  # true positives relajados

        # Jaccard Index
        jaccard = tp / len(union_idx) * 100
        res.append(jaccard)
        
        # Precision
        pred_sum = np.sum(predLabelID == iPhase)
        prec.append(tp * 100 / pred_sum)

        # Recall
        gt_sum = np.sum(gtLabelID == iPhase)
        rec.append(tp * 100 / gt_sum)

    # Exactitud total
    acc = np.sum(updatedDiff == 0) / len(gtLabelID) * 100

    return acc, prec, rec, res


# Inicialización
num_videos = 196 - 183 + 1
num_phases = 8
fps = 1  # o el que uses

jaccard = np.zeros((num_videos, num_phases))
prec = np.zeros((num_videos, num_phases))
rec = np.zeros((num_videos, num_phases))
acc = np.zeros(num_videos)



for i in range(183, 197):
    try:
        # Cargar archivos .mat
        pred_path = f'Predictions_dummy/M2CAI/2025-06-11-03-46-07/video_{i}_preds.mat'
        annot_path = f'Annotations_dummy/M2CAI/2025-06-11-03-46-07/video_{i}_annots.mat'
        
        pred_data = loadmat(pred_path)['Preds'].squeeze()
        gt_data = loadmat(annot_path)['Annots'].squeeze()

        # Evaluar un video
        a, p, r, j = evaluate(gt_data, pred_data, fps)

        idx = i - 183
        jaccard[idx, :] = j
        prec[idx, :] = p
        rec[idx, :] = r
        acc[idx] = a

        print(f"Video {i} procesado correctamente.")
    except Exception as e:
        print(f"Error procesando el video {i}: {e}")

breakpoint()

# Limitar valores máximos a 100
jaccard = np.clip(jaccard, None, 100)
prec = np.clip(prec, None, 100)
rec = np.clip(rec, None, 100)

mean_jacc_per_phase = np.nanmean(jaccard, axis=0)
mean_prec_per_phase = np.nanmean(prec, axis=0)
mean_rec_per_phase = np.nanmean(rec, axis=0)


mean_jaccard = np.mean(mean_jacc_per_phase)
mean_precision = np.mean(mean_prec_per_phase)
mean_recall = np.mean(mean_rec_per_phase)
mean_accuracy = np.mean(acc)