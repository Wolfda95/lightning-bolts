# Alle Schichten die keine Knochenbrüche haben

import nibabel as nib #NIFTI Images (.nii)
import pandas as pd  # csv einlesen
import cv2       # Numpy to jepg/png
import glob # Pade einlesen
import numpy as np

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

# ------------------- Main --------------------------------------------
def main():
    # Todo: Daten Pfade wählen
    path_seg_gesamt = "//home/wolfda/Clinic_Data/Challenge/CT_PreTrain/Downstream/ribfrac-train-labels-1/Part1"
    path_img_gesamt = "//home/wolfda/Clinic_Data/Challenge/CT_PreTrain/Downstream/ribfrac-train-images-1/Part1"


    # Todo: Pfade zum Speichern wählen: (voher erstellen)
    save_path_Diff_Ges = "/home/wolfda/Clinic_Data/Challenge/CT_PreTrain/Downstream/2D/DiffChan_Brueche_Gesamt/Ohne"
    save_path_Diff_Klasse = "/home/wolfda/Clinic_Data/Challenge/CT_PreTrain/Downstream/2D/DiffChan_Je_Klasse/Ohne"
    #save_path_Same_Ges = "/home/wolfda/Clinic_Data/Challenge/CT_PreTrain/Downstream/2D/SameChan_Brueche_Gesamt/Ohne"
    #save_path_Same_Klasse = "/home/wolfda/Clinic_Data/Challenge/CT_PreTrain/Downstream/2D/SameChan_Je_Klasse/Ohne"

    # Todo: wählen für 3 Channel:
    # True: [DiffChen] Nimmt das mittlere Bild des Bruches + das davor und das dahinter
    # False: [SameChen] Nur das mittlere Bild des Bruches drei mal hintereinander
    three_differnt_channel = True

    file_img = sorted(glob.glob(path_img_gesamt + "/*"))
    # -------------------------------------- per image -----------------------------------------------------
    for path_img in file_img:

        #path_img = "//home/wolfda/Clinic_Data/Challenge/CT_PreTrain/Downstream/ribfrac-train-images-1/Part1/RibFrac2-image.nii.gz"

        # dazugehörige Segmentation laden
        path_seg = path_seg_gesamt + "/" + path_img.split("/")[-1].split("-")[0] + "-label.nii.gz"

        # name image
        name = path_img.split("/")[-1]  # File name of patient
        name = name.split("-")[0]  # File name of patient without extension
        print(name)

        # Image + Seg laden
        seg = nib.load(path_seg)
        img = nib.load(path_img)
        seg = seg.get_fdata()  # Numpy Array (x,y, Anzahl Schichten)
        img = img.get_fdata()  # Numpy Array (x,y, Anzahl Schichten)
        seg = seg.transpose(2, 0, 1) # (Anzahl Schichten,x,y)
        img = img.transpose(2, 0, 1) # (Anzahl Schichten,x,y)

        # img: scalierung
        #body_part == "bone": wl = 300, ww = 150
        #wl= 300
        #ww= 150
        #img = win_scale(img, wl, ww, type(img), [img.min(), img.max()])

        no_class = 0 # Anzahl der Bilder ohne Klasse
        is_class = 0 # Anzahl der Bilder mit Klasse

        print(np.unique(seg))

        for i in range(50, img.shape[0]-50): # durchläuft Schichten: start: schicht 50, ende Schchiten länge-50
            # 3D -> 2D
            img_cut = img[i, :, :]
            seg_cut = seg[i, :, :]

            # Hat Bild eine Segmentierung
            #print(np.unique(seg_cut))
            if len(np.unique(seg_cut)) < 2:
                no_class += 1
                #print("hier")

                # Nur jede dritte Schicht
                if i%3 == 0:

                    if three_differnt_channel == True:
                        # 3 channel: das Bild, das davor und das dananch
                        image = np.empty([3, 512, 512])
                        image[0, ...] = img[i-1, :, :]
                        image[1, ...] = img_cut
                        image[2, ...] = img[i-2, :, :]
                    else:
                        # 3 mal die Mittlere Schicht nehmen
                        image = np.empty([3, 512, 512])
                        image[0, ...] = img_cut
                        image[1, ...] = img_cut
                        image[2, ...] = img_cut

                    # Normalization for jpeg/png
                    image = interval_mapping(image, image.min(), image.max(), 0, 255)
                    image = image.astype(np.uint8)
                    image.astype(np.uint8)

                    # Save
                    image = image.transpose(2, 1, 0)  # (Anzahl Schichten,x,y)

                    path = save_path_Diff_Ges + "/" + str(name) + "_" + str(i) + "_" + "lable_ohne"  + ".jpeg"
                    cv2.imwrite(path, image)
                    path = save_path_Diff_Klasse + "/" + str(name) + "_" + str(i) + "_" + "lable_ohne"  + ".jpeg"
                    cv2.imwrite(path, image)
                    # path = save_path_Same_Ges + "/" + str(name) + "_" + str(i) + "_" + "lable_ohne"  + ".jpeg"
                    # cv2.imwrite(path, image)
                    # path = save_path_Same_Klasse + "/" + str(name) + "_" + str(i) + "_" + "lable_ohne"  + ".jpeg"
                    # cv2.imwrite(path, image)

            else:
                is_class += 1

        print("no class: ", no_class)
        print("is class: ", is_class)
        print("total: ", img.shape[0])






if __name__ == '__main__':
    main()

