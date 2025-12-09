import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Définition du chemin du fichier
# Note : On utilise r"..." pour que Python gère correctement les antislashes Windows
file_path = r"C:\Users\prich\Documents\Lynred\im_ir\im_ir_00000001.tif"

# 2. Chargement de l'image
# IMPORTANT : Le flag 'cv2.IMREAD_UNCHANGED' ou -1 est crucial.
# Sans lui, OpenCV risque de convertir l'image brute 16 bits en 8 bits, 
# ce qui écraserait la dynamique nécessaire pour la NUC et le BPR.
img_raw = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

# 3. Vérification du chargement
if img_raw is None:
    print(f"Erreur : Impossible de charger l'image à l'emplacement : {file_path}")
    print("Vérifiez que le chemin est correct et que le fichier existe.")
else:
    print("Image chargée avec succès !")
    print(f"Dimensions de l'image : {img_raw.shape}")
    print(f"Type de données : {img_raw.dtype}") 
    # Vous devriez voir 'uint16' ici, confirmant que c'est bien l'image brute.


    # Statistiques simples pour comprendre la dynamique du capteur
    print(f"Valeur min : {np.min(img_raw)}")
    print(f"Valeur max : {np.max(img_raw)}")
    print(f"Moyenne   : {np.mean(img_raw):.2f}")

def tone_mapping(img_raw, percentile_low=1, percentile_high=99):
    """
    Applique un tone mapping linéaire simple.
    
    Paramètres:
    -----------
    img_raw : ndarray
        Image brute en uint16
    percentile_low : float
        Percentile inférieur pour le clipping (0-100)
    percentile_high : float
        Percentile supérieur pour le clipping (0-100)
    
    Retourne:
    ---------
    img_mapped : ndarray
        Image mappée en uint8 (0-255)
    """
    # Clipping sur les percentiles
    vmin = np.percentile(img_raw, percentile_low)
    vmax = np.percentile(img_raw, percentile_high)
    img_clipped = np.clip(img_raw, vmin, vmax)
    
    # Normalisation et conversion en uint8
    img_output = ((img_clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    
    return img_output




#Traitement NUC avec correction d'offset

# Charger TOUTES les images du dossier bb1
bb1_folder = r"C:\Users\prich\Documents\Lynred\bb1"
offset_images = []

for filename in sorted(os.listdir(bb1_folder)):
    if filename.endswith('.tif'):
        img_path = os.path.join(bb1_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            offset_images.append(img)

print(f"Nombre d'images de référence chargées : {len(offset_images)}")

def compute_offset_map(ref_images):
    """
    Calcule la carte d'offset à partir d'une série d'images de référence (corps noir).
    
    Args:
        ref_images (list ou np.array): Liste d'images (2D) ou tableau 3D (N, H, W)
                                       d'une scène uniforme.
    
    Returns:
        offset_map (np.array): La matrice des offsets à soustraire (type float64).
    """
    # 1. Moyennage temporel pour réduire le bruit (recommandé slide 30)
    # On transforme la liste en tableau numpy si nécessaire
    stack = np.array(ref_images, dtype=np.float64)
    
    # Moyenne pixel par pixel sur l'axe temporel (axis 0)
    # Images_T1 dans la formule
    avg_ref_img = np.mean(stack, axis=0)
    
    # 2. Calcul de la moyenne spatiale de l'image de référence
    # <Images_T1> dans la formule
    global_mean = np.mean(avg_ref_img)
    
    # 3. Calcul de la carte d'offset
    # Formule : Offset = Images_T1 - <Images_T1>
    offset_map = avg_ref_img - global_mean
    
    return offset_map

def apply_offset_correction(raw_img, offset_map):
    """
    Applique la correction d'offset sur une image brute.
    
    Args:
        raw_img (np.array): Image brute (uint14/16).
        offset_map (np.array): Carte d'offset calculée précédemment.
        
    Returns:
        corrected_img (np.array): Image corrigée (float64).
    """
    # Conversion en float pour éviter les erreurs de signe (underflow)
    img_float = raw_img.astype(np.float64)
    
    # Application de la formule : pixel_corrige = pixel_brut - offset
    corrected_img = img_float - offset_map
    
    return corrected_img

# Maintenant calcul de l'offset map avec plusieurs images
offset_map = compute_offset_map(offset_images)
corrected_img = apply_offset_correction(img_raw, offset_map)





# Charger les images du corps noir CHAUD (bb2 = T2)
bb2_folder = r"C:\Users\prich\Documents\Lynred\bb2"
bb2_images = []

for filename in sorted(os.listdir(bb2_folder)):
    if filename.endswith('.tif'):
        img_path = os.path.join(bb2_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            bb2_images.append(img)

def compute_gain_map(ref_images_hot, offset_map):
    """
    Calcule la carte de gain pour NUC 2 points.
    
    Args:
        ref_images_hot: Images du corps noir à T2 (plus chaud que T1)
        offset_map: Carte d'offset calculée avec bb1 (T1)
    
    Returns:
        gain_map: Matrice des gains
    """
    # Moyenne des images chaudes
    stack_hot = np.array(ref_images_hot, dtype=np.float64)
    avg_ref_hot = np.mean(stack_hot, axis=0)
    
    # Correction d'offset sur l'image chaude
    img_hot_corrected = avg_ref_hot - offset_map
    
    # Moyenne cible
    target_mean = np.mean(img_hot_corrected)
    
    # Carte de gain
    epsilon = 1e-6
    gain_map = target_mean / (img_hot_corrected + epsilon)

    
    return gain_map

# Calcul du gain avec bb2 (température chaude)
gain_map = compute_gain_map(bb2_images, offset_map)


def apply_gain(gain_map, img_raw):
 
    corrected_img = img_raw * gain_map
    return corrected_img


image_offset = apply_offset_correction(img_raw, offset_map)
image_offset_tone = tone_mapping(image_offset)

img_gain_offset = apply_gain(gain_map, image_offset)

img_gain_offset = tone_mapping(img_gain_offset)


fig, axes = plt.subplots(1, 2, figsize=(15, 6))




#réduction du bruit temporel

def temporal_noise_reduction(current_img, accumulator, alpha=0.2):
    """
    Applique un filtre temporel récursif pour réduire le bruit.
    
    Args:
        current_img (np.array): L'image brute de la frame actuelle.
        accumulator (np.array): L'image filtrée de la frame précédente (mémoire).
                                Si None, initialise l'accumulateur avec l'image actuelle.
        alpha (float): Facteur de lissage (0.0 à 1.0). 
                       0.1 = fort lissage (bcp d'inertie).
                       0.8 = faible lissage (réactif).
                       
    Returns:
        new_accumulator (np.array): La nouvelle image filtrée (à stocker pour la prochaine boucle).
    """
    # Conversion en float pour la précision du calcul
    current_float = current_img.astype(np.float64)
    
    # Initialisation : si c'est la première image, on n'a pas d'historique
    if accumulator is None:
        return current_float
    
    # Application de la formule : Moyenne glissante
    # Out = (alpha * Input) + ((1 - alpha) * Prev_Out)
    new_accumulator = (alpha * current_float) + ((1.0 - alpha) * accumulator)
    
    return new_accumulator

# Dossier contenant les images
im_ir_folder = r"C:\Users\prich\Documents\Lynred\im_ir"
list_images = sorted([f for f in os.listdir(im_ir_folder) if f.endswith('.tif')])

# Initialisation de l'accumulateur
accumulator = None

print(f"Traitement de {len(list_images)} images...")

for filename in list_images:
    img_path = os.path.join(im_ir_folder, filename)
    frame = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    
    # Vérification que l'image est bien chargée
    if frame is None:
        print(f"Erreur de chargement : {filename}")
        continue
    
    # Pipeline complet de traitement
    frame_corr = apply_offset_correction(frame, offset_map)
    
    # Application du gain AVANT le filtrage temporel (recommandé)
    frame_nuc = apply_gain(gain_map, frame_corr)
    
    # Utilisation cohérente de 'accumulator'
    accumulator = temporal_noise_reduction(frame_nuc, accumulator, alpha=0.3)
    
    # Affichage
    cv2.imshow("IR Filtré", tone_mapping(accumulator))
    
    # Attendre 10ms (ou appuyer sur 'q' pour quitter)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("Traitement terminé !")