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
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)

# =============================================================================
# Save
# =============================================================================

def save(image, save_path, task_name, patient_name, i, wl, ww):

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
        path = save_path + "/" + str(patient_name) + "_" + str(i) + ".jpeg"
        cv2.imwrite(path, img)

    print(i)

# =============================================================================
# Main
# =============================================================================
def main():

    # Daten: Images: DICOM Files in einem Ordner


    # ToDo: Pfade wo die Daten gespeichert sind:
    data_path = "/home/wolfda/Clinic_Data/Challenge/CT_PreTrain/Medical_Dcathlon/CT/Scheisse"

    # ToDo: Pfad wo die PyThorch Files gespeichert werden sollen
    save_path = "/home/wolfda/Clinic_Data/Challenge/CT_PreTrain/Medical_Dcathlon/CT/Jpeg_Data" # Mus vorher angelegt werden!!!!!!!


    #body_part = "abdomen": wl = 60, ww = 400
    #body_part == "angio": wl = 300, ww = 600
    #body_part == "bone": wl = 300, ww = 150
    #body_part == "brain": wl = 40, ww = 80
    #body_part == "chest": wl = 40, ww = 400
    #body_part == "lungs": wl = -400, ww = 1500

    i = 0

    # ToDo: Je nach Ordnerstruktur anpassen:
    Ordner = sorted(glob.glob(data_path + "/*"))  # Liste: Alle Pfade aus dem Ordner Nifti_Data (Ordner Tasks)
    for fileA in Ordner:  # durchläuft alle Pfade im Ordner Nifti_Data (alle Tasks)
        task_name = fileA.split("/")[-1]  # Name des Taks (Task03-Liver,...)

        print("Task: " + task_name)

        if task_name == "Task03_Liver":
            wl = 60
            ww = 400
            print("liver")
        if task_name == "Task06_Lung":
            wl = -400
            ww = 1500
            print("lung")
        if task_name == "Task07_Pancreas":
            wl = 60
            ww = 400
            print("pancreas")
        if task_name == "Task08_HepaticVessel":
            wl = 40
            ww = 400
            print("HepaticVessel")
        if task_name == "Task09_Spleen":
            wl = 60
            ww = 400
            print("Spleen")
        if task_name == "Task10_Colon":
            wl = 60
            ww = 400
            print("Colon")
        else:
            print("No Task")

        Ordner2 = sorted(glob.glob(fileA + "/*")) # Liste: Alle Pfade aus einem Task Ordner (Train, Test, Label, Jason)
        for fileB in Ordner2: # durchläuft alle Pfade (Train, Test, Label, Jason)
            file_name = fileB.split("/")[-1]  # Name ses Files (Train, Test, Label, Jason)
            if file_name == "imagesTr":

                Ordner3 = sorted(glob.glob(fileB + "/*"))  # Liste: Alle Patienten
                for fileC in Ordner3: # durchläuft alle Patienten
                    patient_name = fileC.split("/")[-1]  # File name of patient

                    # (Pfad der Serie (DICOM Files), Patientenname, Schicht Nummer, WL, WW)
                    #fileC = "/home/wolfda/Clinic_Data/Challenge/CT_PreTrain/Medical_Dcathlon/CT/Nifti_Data/Task07_Pancreas/imagesTr/pancreas_410.nii.gz"
                    save(fileC, save_path, task_name, patient_name, i, wl, ww)

                    i = i+1


if __name__ == '__main__':
    main()