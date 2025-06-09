import numpy as np
from scipy.ndimage import label
import pandas as pd


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
    oriT = 10 * fps  # Duración de la ventana relajada (10 segundos en frames)

    res = []
    prec = []
    rec = []
    diff = predLabelID - gtLabelID  # Diferencia entre predicción y GT
    updatedDiff = np.copy(diff)  # Se irá modificando con reglas de relajación

    nPhases = 7
    
    # Aplicar relaxed boundary para cada fase en el GT
    for iPhase in range(1, nPhases + 1):
        # Encontrar regiones conectadas donde la fase actual está presente
        mask_gt = (gtLabelID == iPhase).astype(int)
        labeled_gt, num_objects = label(mask_gt)
        #breakpoint()
        for conn_id in range(1, num_objects + 1):
            indices = np.where(labeled_gt == conn_id)[0]
            startIdx = indices.min()
            endIdx = indices.max()

            curDiff = diff[startIdx:endIdx + 1].copy()

            # Determinar tamaño de ventana relajada (no mayor que la duración de la fase)
            t = min(oriT, len(curDiff))

            # Aplicar reglas de transición relajada, específicas por fase
            if iPhase in [4, 5]:
                head = curDiff[:t]
                tail = curDiff[-t:]
                head[head == -1] = 0
                tail[np.isin(tail, [1, 2])] = 0
                curDiff[:t] = head
                curDiff[-t:] = tail

            elif iPhase in [6, 7]:
                head = curDiff[:t]
                tail = curDiff[-t:]
                head[np.isin(head, [-1, -2])] = 0
                tail[np.isin(tail, [1, 2])] = 0
                curDiff[:t] = head
                curDiff[-t:] = tail

            else:
                head = curDiff[:t]
                tail = curDiff[-t:]
                head[head == -1] = 0
                tail[tail == 1] = 0
                curDiff[:t] = head
                curDiff[-t:] = tail

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

        union_idx = np.where((gtLabelID == iPhase) | (predLabelID == iPhase))[0]
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

    return res, prec, rec, acc


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

jaccard, prec, rec, acc = evaluate(gtLabelID, predLabelID, fps)

# Supón que estas variables ya están definidas:
# jaccard: numpy array de shape (num_fases, num_videos)
# prec: numpy array de shape (num_fases, num_videos)
# rec: numpy array de shape (num_fases, num_videos)
# acc: numpy array de shape (num_videos,)
# phases: lista de strings con nombres de las fases

acc_per_video = acc.copy()

# Limitar valores máximos a 100
jaccard = np.clip(jaccard, None, 100)
prec = np.clip(prec, None, 100)
rec = np.clip(rec, None, 100)

# --- Métricas de Jaccard ---
mean_jacc_per_phase = np.nanmean(jaccard)
mean_jacc = mean_jacc_per_phase
std_jacc = mean_jacc_per_phase.std()

meanjaccphase = mean_jacc_per_phase
stdjaccphase = jaccard.std()

# --- Métricas de Precisión ---
mean_prec = np.nanmean(prec)
std_prec = prec.std()


# --- Métricas de Recall ---
mean_rec = np.nanmean(rec)
std_rec = rec.std()

# --- Exactitud ---
mean_acc = np.nanmean(acc)
std_acc = acc.std()



# --- Mostrar promedios generales ---
print(f"Mean accuracy:  {mean_acc:5.2f} ± {std_acc:5.2f}")
print(f"Mean jaccard:   {mean_jacc:5.2f} ± {std_jacc:5.2f}")
print(f"Mean precision: {mean_prec:5.2f} ± {std_prec:5.2f}")
print(f"Mean recall:    {mean_rec:5.2f} ± {std_rec:5.2f}")
