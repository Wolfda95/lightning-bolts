# 3D Nifti to 2D jpeg File für Medical Segmentation Decathlon

import glob # Pade einlesen
import pydicom as dicom # Dicom Einlesen
import nibabel as nib # Nifti Einlesen
import cv2       # Numpy to jepg/png
#from tqdm import tqdm
#import scipy.ndimage # rezize: zoom

import os
import numpy as np
import torch

# ------------------- CT: Scale pixel intensity --------------------------------------------
# von https://gist.github.com/lebedov/e81bd36f66ea1ab60a1ce890b07a6229
# abdomen: {'wl': 60, 'ww': 400} || angio: {'wl': 300, 'ww': 600} || bone: {'wl': 300, 'ww': 1500} || brain: {'wl': 40, 'ww': 80} || chest: {'wl': 40, 'ww': 400} || lungs: {'wl': -400, 'ww': 1500}
def win_scale(data, wl, ww, dtype, out_range):
    data_new = np.empty(data.shape, dtype=np.double)
    data_new.fill(out_range[1] - 1)

    data_new[data <= (wl - ww / 2.0)] = out_range[0]
    data_new[(data > (wl - ww / 2.0)) & (data <= (wl + ww / 2.0))] = \
        ((data[(data > (wl - ww / 2.0)) & (data <= (wl + ww / 2.0))] - (wl - 0.5)) / (ww - 1.0) + 0.5) * (
                out_range[1] - out_range[0]) + out_range[0]
    data_new[data > (wl + ww / 2.0)] = out_range[1] - 1

    return data_new.astype(dtype)

# ------------------------------------image normalization (for png/jepg)------------------------------------------
def interval_mapping(image, from_min, from_max, to_min, to_max):
    # map values from [from_min, from_max] to [to_min, to_max]
    # image: input array
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / (float(from_range)+0.00001), dtype=float)
    return to_min + (scaled * to_range)

# =============================================================================
# Save
# =============================================================================

def save(image, save_path1, save_path2, patient_name,n, wl, ww):

    # --------------------------Image----------------------------------------------------
    # Nfti -> Numpy
    patient_pixels = nib.load(image)
    patient_pixels = patient_pixels.get_fdata()
    patient_pixels = patient_pixels.transpose(2, 1, 0)

    # CT Skalierung
    patient_pixels = win_scale(patient_pixels, wl, ww, type(patient_pixels), [patient_pixels.min(), patient_pixels.max()])  # Numpy Array Korrigiert

    # Resize [48,800,800]
    #patient_pixels = scipy.ndimage.zoom(patient_pixels, (min(1, (48 / patient_pixels.shape[0])), (800 / patient_pixels.shape[1]), (800 / patient_pixels.shape[2])),mode="nearest", grid_mode=True)

    for i in range(patient_pixels.shape[0]):
        # 3D -> 2D
        img = patient_pixels[i, :, :]

        # Normalization for jpeg/png
        img = interval_mapping(img, img.min(), img.max(), 0, 255)
        img = img.astype(np.uint8)
        img.astype(np.uint8)

        # Save
        path = save_path1 + "/" + str(patient_name) + "_" + str(n) + "_" + str(i) + ".jpeg"
        cv2.imwrite(path, img)
        path = save_path2 + "/" + str(patient_name) + "_" + str(n) + "_" + str(i) + ".jpeg"
        cv2.imwrite(path, img)

    print (i)



# ============================================================would =================
# Main
# =============================================================================
def main():

    # Daten: Images: DICOM Files in einem Ordner


    # ToDo: Pfade wo die Daten gespeichert sind:
    data_path = "/home/wolfda/Clinic_Data/Challenge/CT_PreTrain/PET_CT_Tuebingen/nifti(alle)/FDG-PET-CT-Lesions"

    # ToDo: Pfad wo die PyThorch Files gespeichert werden sollen
    save_path1 = "/home/wolfda/Clinic_Data/Challenge/CT_PreTrain/PET_CT_Tuebingen/AutoPET_2D"
    save_path2 = "/home/wolfda/Clinic_Data/Challenge/CT_PreTrain/PreTrain_Gesamt/Data/LIDC-MSD-CQ500-AutoPET/LIDC-MSD-CQ500-AutoPET_Data"


    #body_part = "abdomen": wl = 60, ww = 400
    #body_part == "angio": wl = 300, ww = 600
    #body_part == "bone": wl = 300, ww = 150
    #body_part == "brain": wl = 40, ww = 80
    #body_part == "chest": wl = 40, ww = 400
    #body_part == "lungs": wl = -400, ww = 1500
    wl = 40
    ww = 400



    # ToDo: Je nach Ordnerstruktur anpassen:
    n=0
    Anzahl = 0
    Ordner = sorted(glob.glob(data_path + "/*"))  # Liste: Alle Pfade aus dem Ordner FDG-PET-CT-Lesions (Alle Patienten)
    for fileA in Ordner:  # durchläuft alle Pfade im Ordner FDG-PET-CT-Lesions (Alle Patienten)
        Patient_Name = fileA.split("/")[-1]  # Name des Patienten

        print("Name: " + Patient_Name)

        Ordner2 = sorted(glob.glob(fileA + "/*")) # Liste: Unterordner
        for fileB in Ordner2: # durchläuft alle Pfade
            print("Unterordner: " + fileB.split("/")[-1])
            n = n+1

            Ordner3 = sorted(glob.glob(fileB + "/*"))  # Liste: verschiedene Modalitäten (CT, PET,...)
            for fileC in Ordner3: # durchläuft alle verschiedene Modalitäten (CT, PET,...)
                file_name = fileC.split("/")[-1]  # Name des Files (CT, PET,...)
                if file_name == "CT.nii.gz":
                    print(file_name)
                    save(fileC, save_path1, save_path2, Patient_Name,n, wl, ww)
                    Anzahl += 1
                    print("Anzahl: ", Anzahl)



if __name__ == '__main__':
    main()