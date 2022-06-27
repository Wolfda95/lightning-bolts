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
    path_img_gesamt = "/home/wolfda/Clinic_Data/Challenge/CT_PreTrain/Challenge_COVID-19-20_v2/Data/CT"
    path_seg_gesamt = "/home/wolfda/Clinic_Data/Challenge/CT_PreTrain/Challenge_COVID-19-20_v2/Data/Seg"

    # Todo: Pfade zum Speichern wählen: (voher erstellen)
    All_save_path_0 = "/home/wolfda/Clinic_Data/Challenge/CT_PreTrain/Challenge_COVID-19-20_v2/Data/2D/Half_AlleSeg/0_noCovid"
    All_save_path_1 = "/home/wolfda/Clinic_Data/Challenge/CT_PreTrain/Challenge_COVID-19-20_v2/Data/2D/Half_AlleSeg/1_Covid"

    file_img = sorted(glob.glob(path_img_gesamt + "/*"))

    # -------------------------------------- per image -----------------------------------------------------
    for path_img in file_img:

        #path_img = "//home/wolfda/Clinic_Data/Challenge/CT_PreTrain/Challenge_COVID-19-20_v2/Data/CT/volume-covid19-A-0072_ct.nii.gz"

        # dazugehörige Segmentation laden
        path_seg = path_seg_gesamt + "/" + path_img.split("/")[-1].split("ct")[0] + "seg.nii.gz"

        # name image
        name = path_img.split("/")[-1]  # File name of patient
        name = name.split("_")[0]  # File name of patient without extension
        print(name)

        # Image + Seg laden
        seg = nib.load(path_seg)
        img = nib.load(path_img)
        seg = seg.get_fdata()  # Numpy Array (x,y, Anzahl Schichten)
        img = img.get_fdata()  # Numpy Array (x,y, Anzahl Schichten)
        seg = seg.transpose(2, 0, 1) # (Anzahl Schichten,x,y)
        img = img.transpose(2, 0, 1) # (Anzahl Schichten,x,y)

        # img: scalierung
        wl = -400
        ww = 1500
        img = win_scale(img, wl, ww, type(img), [img.min(), img.max()])

        # leere Lisen erstelln für 2 Klassen
        class_0, class_1  = ([] for i in range(2)) # Dicts zum speichern der Schichten der einzelnen Klassen

        is_class_0 = 0 # Anzahl der Bilder mit Klasse
        is_class_1 = 0  # Anzahl der Bilder mit Klasse

        print(img.shape[0])

        for i in range(0, img.shape[0]-0): # durchläuft alle Schichten
            # 3D -> 2D
            img_2d = img[i, :, :]
            seg_2d = seg[i, :, :]

            img_2d_half_1 = img_2d[:256,:]
            img_2d_half_2 = img_2d[256:, :]
            seg_2d_half_1 = seg_2d[:256,:]
            seg_2d_half_2 = seg_2d[256:, :]

            # listet alle grauwerte auf die es findet
            classes_img, counts_img = np.unique(seg_2d, return_counts=True)

            # Hat Lungen Infiltrate in gesamten Bild
            if 1 in classes_img:

                ###### Hälfte 1 #############
                # listet alle grauwerte auf die es findet
                classes_half_1, counts_half_1 = np.unique(seg_2d_half_1, return_counts=True)

                # Hat Infiltrate in Häfte 1:
                if 1 in classes_half_1:
                    # Nur wenn es eine großflächige Segmentierung ist (mehr als 4000 Pixel Segmentiert)
                    if counts_half_1[1]  > 1:

                        # 3 mal die Mittlere Schicht nehmen
                        image = np.empty([3, 256, 512])
                        image[0, ...] = img_2d_half_1
                        image[1, ...] = img_2d_half_1
                        image[2, ...] = img_2d_half_1

                        # Normalization for jpeg/png
                        image = interval_mapping(image, image.min(), image.max(), 0, 255)
                        image = image.astype(np.uint8)
                        image.astype(np.uint8)

                        # Save
                        image = image.transpose(2, 1, 0)  # (Anzahl Schichten,x,y)
                        path = All_save_path_1 + "/" + str(name) + "_" + str(i) + "_" + "lable_" + str(1) + ".jpeg"
                        cv2.imwrite(path, image)

                        is_class_0+=1

                # In Gesamten Bild gibt es eine Segmentierung, aber in der Hälfte nicht
                else:
                    # 3 mal die Mittlere Schicht nehmen
                    image = np.empty([3, 256, 512])
                    image[0, ...] = img_2d_half_1
                    image[1, ...] = img_2d_half_1
                    image[2, ...] = img_2d_half_1

                    # Normalization for jpeg/png
                    image = interval_mapping(image, image.min(), image.max(), 0, 255)
                    image = image.astype(np.uint8)
                    image.astype(np.uint8)

                    # Save
                    image = image.transpose(2, 1, 0)  # (Anzahl Schichten,x,y)
                    path = All_save_path_0 + "/" + str(name) + "_" + str(i) + "_" + "lable_" + str(1) + ".jpeg"
                    cv2.imwrite(path, image)

                    is_class_1 += 1

                ####### Hälfte 2 ################
                # listet alle grauwerte auf die es findet
                classes_half_2, counts_half_2 = np.unique(seg_2d_half_2, return_counts=True)

                # Hat Infiltrate in Häfte 1:
                if 1 in classes_half_2:
                    # Nur wenn es eine großflächige Segmentierung ist (mehr als 4000 Pixel Segmentiert)
                    if counts_half_2[1] > 1:
                        # 3 mal die Mittlere Schicht nehmen
                        image = np.empty([3, 256, 512])
                        image[0, ...] = img_2d_half_2
                        image[1, ...] = img_2d_half_2
                        image[2, ...] = img_2d_half_2

                        # Normalization for jpeg/png
                        image = interval_mapping(image, image.min(), image.max(), 0, 255)
                        image = image.astype(np.uint8)
                        image.astype(np.uint8)

                        # Save
                        image = image.transpose(2, 1, 0)  # (Anzahl Schichten,x,y)
                        path = All_save_path_1 + "/" + str(name) + "_" + str(i) + "_" + "lable_" + str(1) + ".jpeg"
                        cv2.imwrite(path, image)

                        is_class_0 += 1

                # In Gesamten Bild gibt es eine Segmentierung, aber in der Hälfte nicht
                else:
                    # 3 mal die Mittlere Schicht nehmen
                    image = np.empty([3, 256, 512])
                    image[0, ...] = img_2d_half_2
                    image[1, ...] = img_2d_half_2
                    image[2, ...] = img_2d_half_2

                    # Normalization for jpeg/png
                    image = interval_mapping(image, image.min(), image.max(), 0, 255)
                    image = image.astype(np.uint8)
                    image.astype(np.uint8)

                    # Save
                    image = image.transpose(2, 1, 0)  # (Anzahl Schichten,x,y)
                    path = All_save_path_0 + "/" + str(name) + "_" + str(i) + "_" + "lable_" + str(1) + ".jpeg"
                    cv2.imwrite(path, image)

                    is_class_1 += 1




        print("is class 1: ", is_class_0)
        print("is class 1: ", is_class_1)



if __name__ == '__main__':
    main()