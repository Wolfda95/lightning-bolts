import torch
import numpy as np
import glob # Pade einlesen
import cv2  # Numpy to jepg/png

# ------------------------------------image normalization (for png/jepg)------------------------------------------
def interval_mapping(image, from_min, from_max, to_min, to_max):
    # map values from [from_min, from_max] to [to_min, to_max]
    # image: input array
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=float)
    return to_min + (scaled * to_range)


# =============================================================================
# Main
# =============================================================================
def main():


    # ToDo: Pfade wo die Daten gespeichert sind:
    data_path = "/home/wolfda/Clinic_Data/Challenge/CT_PreTrain/CT_Brain_Cq500_Qureai_Dome/CQ500"

    # ToDo: Pfad wo die PyThorch Files gespeichert werden sollen
    save_path_1 =  "/home/wolfda/Clinic_Data/Challenge/CT_PreTrain/CT_Brain_Cq500_Qureai_Dome/CQ500_jpeg"  # Nur CQ500_jpeg Daten
    save_path_2 = "/home/wolfda/Clinic_Data/Challenge/CT_PreTrain/PreTrain_Gesamt/Data/LIDC-MSD-CQ500/"  # LIDC + MSD + CQ500 Daten

    # ToDo: Je nach Ordnerstruktur anpassen:
    Ordner = sorted(glob.glob(data_path + "/*"))  # Liste: Alle Pfade aus dem Ordner CQ500 (Ordner mit allen Torch Files)
    for fileA in Ordner:  # durchläuft alle Pfade im Ordner CQ500 (alle Torch files
        patient_name = fileA.split("/")[-1].split(".")[0]  # Name des Patienten
        print(patient_name)

        # Load Torch File
        tensor = torch.load(fileA)
        # Extract Torch Array Image
        img = tensor["vol"]
        # Torch to Numpy
        img_np = img.numpy()

        # Save 2D
        for i in range(img_np.shape[0]):
            # 3D -> 2D
            img_2D = img_np[i, :, :]

            # Normalization for jpeg/png
            # Normalization for jpeg/png
            img_2D = interval_mapping(img_2D, img_2D.min(), img_2D.max(), 0, 255)
            img_2D = img_2D.astype(np.uint8)
            img_2D.astype(np.uint8)

            # Save
            path1 = save_path_1 + "/" + str(patient_name) + "_" + str(i) + ".jpeg"
            path2 = save_path_2 + "/" + str(patient_name) + "_" + str(i) + ".jpeg"
            cv2.imwrite(path1, img_2D)
            cv2.imwrite(path2, img_2D)





if __name__ == '__main__':
    main()