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

    nPhases = 7
    
    # Aplicar relaxed boundary para cada fase en el GT
    for iPhase in range(1, nPhases + 1):
        # Encontrar regiones conectadas donde la fase actual está presente en el GT
        mask_gt = (gtLabelID == iPhase).astype(int)
        labeled_gt, num_objects = label(mask_gt)
        
        # Iteramos sobre cada una de esas regiones conectadas
        for conn_id in range(1, num_objects + 1):
            # Encuentro la primer region de la fase en el Gt
            indices = np.where(labeled_gt == conn_id)[-1]
            startIdx = indices.min()
            endIdx = indices.max()

            curDiff = diff[startIdx:endIdx + 1].copy()

            # Determinar tamaño de ventana relajada (no mayor que la duración de la fase)
            t = min(oriT, len(curDiff))

            # Aplicar reglas de transición relajada, específicas por fase
            if iPhase in [4, 5]:
                # Working
                mask_late = curDiff[:t] == -1
                curDiff[np.where(mask_late)[0]] = 0 # late transition

                last_t = curDiff[-t:]  # últimos t elementos
                mask_early = (last_t == 1) | (last_t == 2)
                curDiff[:t][mask_early] = 0 # early transition


            elif iPhase in [6, 7]:
                # Working
                first_t = curDiff[:t]  # primeros t elementos
                mask_last = (first_t == -1) | (first_t == -2)
                curDiff[:t][mask_last] = 0 # late transition

                last_t = curDiff[-t:]  # primeros t elementos
                mask_early = (last_t == 1) | (last_t == 2)
                curDiff[:t][mask_early] = 0 # late transition

            else:
                # Working
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
    
    # Pasamos de metricas por fase a metrica total

    # Limitar valores máximos a 100
    jaccard = np.clip(res, None, 100)
    prec = np.clip(prec, None, 100)
    rec = np.clip(rec, None, 100)

    # --- Métricas de Jaccard ---
    mean_jaccard = np.nanmean(jaccard)

    # --- Métricas de Precisión ---
    mean_precision = np.nanmean(prec)


    # --- Métricas de Recall ---
    mean_recall = np.nanmean(rec)

    return acc, mean_precision, mean_recall, mean_jaccard


def evaluate_v2(gtLabelID, predLabelID, fps):
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

    nPhases = 7
    
    # Aplicar relaxed boundary para cada fase en el GT
    for iPhase in range(1, nPhases + 1):
        # Encontrar regiones conectadas donde la fase actual está presente en el GT
        mask_gt = (gtLabelID == iPhase).astype(int)
        labeled_gt, num_objects = label(mask_gt)
        
        # Iteramos sobre cada una de esas regiones conectadas
        for conn_id in range(1, num_objects + 1):
            # Encuentro la primer region de la fase en el Gt
            indices = np.where(labeled_gt == conn_id)[-1]
            startIdx = indices.min()
            endIdx = indices.max()

            curDiff = diff[startIdx:endIdx + 1].copy()

            # Determinar tamaño de ventana relajada (no mayor que la duración de la fase)
            t = min(oriT, len(curDiff))

            # Aplicar reglas de transición relajada, específicas por fase
            if iPhase in [4, 5]:
                # Working
                mask_late = curDiff[:t] == -1
                curDiff[np.where(mask_late)[0]] = 0 # late transition

                last_t = curDiff[-t:]  # últimos t elementos
                mask_early = (last_t == 1) | (last_t == 2)
                curDiff[:t][mask_early] = 0 # early transition


            elif iPhase in [6, 7]:
                # Working
                first_t = curDiff[:t]  # primeros t elementos
                mask_last = (first_t == -1) | (first_t == -2)
                curDiff[:t][mask_last] = 0 # late transition

                last_t = curDiff[-t:]  # primeros t elementos
                mask_early = (last_t == 1) | (last_t == 2)
                curDiff[:t][mask_early] = 0 # late transition

            else:
                # Working
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





"""
# Supongamos que el video tiene 30 frames y hay 7 fases (del 1 al 7)
num_frames = 100
fps = 1  # frames por segundo (simularemos 4 segundos de tolerancia)

# Creamos ground truth (gtLabelID) con 7 fases distintas en bloques
gtLabelID = np.zeros(num_frames, dtype=int)
gtLabelID[0:14]   = 1   # Fase 1
gtLabelID[14:28]  = 2   # Fase 2
gtLabelID[28:42]  = 3   # Fase 3
gtLabelID[42:57]  = 4   # Fase 4
gtLabelID[57:71]  = 5   # Fase 5
gtLabelID[71:85]  = 6   # Fase 6
gtLabelID[85:100] = 7   # Fase 7

# Creamos predicciones con algunos errores intencionales
predLabelID = np.zeros(num_frames, dtype=int)
predLabelID[0:14]   = 1   # Correcto
predLabelID[14:28]  = 3   # Error: predice fase 3 en vez de 2
predLabelID[28:42]  = 3   # Correcto
predLabelID[42:57]  = 4   # Correcto
predLabelID[57:71]  = 6   # Error: predice fase 6 en vez de 5
predLabelID[71:85]  = 6   # Correcto
predLabelID[85:100] = 1   # Error: predice fase 1 en vez de 7

acc, mean_precision, mean_recall, mean_jaccard = evaluate(gtLabelID, predLabelID, fps)


# --- Mostrar promedios generales ---
print(f"Mean relaxed accuracy:  {acc:5.2f}")
print(f"Mean relaxed precision: {mean_precision:5.2f}")
print(f"Mean relaxed recall:    {mean_recall:5.2f}")
print(f"Mean relaxed jaccard:   {mean_jaccard:5.2f}")"""


"""
# Prueba con predicciones guardadas en archivos .mat de Cholec80
pred_file = loadmat(os.path.join('Predictions/Cholec80/2025-06-09-21-00-23/video_92_preds.mat'))['Preds']
annots_file = loadmat(os.path.join('Annotations/Cholec80/2025-06-09-21-00-23/video_92_annot.mat'))['Annots']

# Manual accuracy confirmed
manual_acc = sum(pred_file[0] == annots_file[0]) / pred_file.shape[-1]
print(f'Manual Accuracy: {manual_acc}')
acc, mean_precision, mean_recall, mean_jaccard = evaluate(annots_file, pred_file, fps=1)"""


# Inicialización
num_videos = 101 - 62 + 1
num_phases = 7
fps = 1  # o el que uses

jaccard = np.zeros((num_videos, num_phases))
prec = np.zeros((num_videos, num_phases))
rec = np.zeros((num_videos, num_phases))
acc = np.zeros(num_videos)



for i in range(62, 102):
    try:
        # Cargar archivos .mat
        pred_path = f'Predictions/Cholec80/2025-06-10-19-46-37/video_{i}_preds.mat'
        annot_path = f'Annotations/Cholec80/2025-06-10-19-46-37/video_{i}_annot.mat'
        
        pred_data = loadmat(pred_path)['Preds'].squeeze()
        gt_data = loadmat(annot_path)['Annots'].squeeze()

        # Evaluar un video
        a, p, r, j = evaluate_v2(gt_data, pred_data, fps)

        idx = i - 62
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

