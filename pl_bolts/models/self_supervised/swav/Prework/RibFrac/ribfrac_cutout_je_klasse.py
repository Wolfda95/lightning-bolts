#Von jedem CT
    #Von jeder Schicht
       # Nur die Schichten wo es Knochenbrüche gibt (Seg nicht nur Schwarz)
	        #Ein Bild für jede Klasse: 1, 2, 3, 4, -1
	            # Die Mittlere Schicht von dem Knochenbruch ENTWEDER: plus die davor und die dahinter ODER: 3 mal die Mittlere hintereinader → 3 Channel
	    	                #Nur die Schichen wo es nicht 2 Verschiedene Klassen von Brüchen in einer Schicht gibt


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

# ------------------- Mittelpunkt der Segmentierung finden --------------------------------------------
# Seg value : 1 | Rest: value: 0
def mean_seg(seg):

    # Mitte Z-Richtung
    # print("z__")
    i = 0
    j = np.array([])
    for z in range(seg.shape[0]):
        if str(np.unique(seg[z, ...])) == "[0. 1.]":
            j = np.append(j, z)
            i += 1  # Zählt Schichten wo es Segmentierungen gibt
    x_mean = np.around(j[-1] - ((i - 1) / 2))  # Letzte Schicht mit Seg - ((Anzahl Schichten mit Seg-1)/2) + abrunden
    # print("Ahnzahl Schichten mit Seg: ", i)
    #print("Schichten mit Seg: ", j, len(j))
    # print("Mittlere Schicht: ", z_mean)


    # Mitte y-Richtung
    # print("y__")
    i = 0
    j = np.array([])
    for z in range(seg.shape[1]):
        if str(np.unique(seg[..., z])) == "[0. 1.]":
            j = np.append(j, z)
            i += 1  # Zählt Schichten wo es Segmentierungen gibt
    y_mean = np.around(j[-1] - ((i - 1) / 2))  # Letzte Schicht mit Seg - ((Anzahl Schichten mit Seg-1)/2) + abrunden
    # print("Ahnzahl Schichten mit Seg: ", i)
    #print("y ", len(j))
    # print("Mittlere Schicht: ", y_mean)


    mean = np.array([x_mean, y_mean])
    # print(mean)

    return mean

# ------------------- Main --------------------------------------------
def main():
    # Todo: Daten Pfade wählen
    path_seg_gesamt = "/home/wolfda/Clinic_Data/Challenge/CT_PreTrain/Downstream/Data/Val/ribfrac-val-labels"
    path_img_gesamt = "/home/wolfda/Clinic_Data/Challenge/CT_PreTrain/Downstream/Data/Val/ribfrac-val-images"
    path_table = "/home/wolfda/Clinic_Data/Challenge/CT_PreTrain/Downstream/Data/Val/ribfrac-val-info.csv"

    # Todo: Pfade zum Speichern wählen: (voher erstellen)
    save_path_1 = "/home/wolfda/Clinic_Data/Challenge/CT_PreTrain/Downstream/2D/Cut_DiffChen/lable_1"
    save_path_2 = "/home/wolfda/Clinic_Data/Challenge/CT_PreTrain/Downstream/2D/Cut_DiffChen/lable_2"
    save_path_3 = "/home/wolfda/Clinic_Data/Challenge/CT_PreTrain/Downstream/2D/Cut_DiffChen/lable_3"
    save_path_4 = "/home/wolfda/Clinic_Data/Challenge/CT_PreTrain/Downstream/2D/Cut_DiffChen/lable_4"
    save_path_minus1 = "/home/wolfda/Clinic_Data/Challenge/CT_PreTrain/Downstream/2D/Cut_DiffChen/lable_-1"

    # Todo: wählen für 3 Channel:
    # True: [DiffChen] Nimmt das mittlere Bild des Bruches + das davor und das dahinter
    # False: [SameChen] Nur das mittlere Bild des Bruches drei mal hintereinander
    three_differnt_channel = True

    # ToDo: Tablle laden !!!!!Relevante Zeilen!!!!!
    table = pd.read_csv(path_table)  # Komplette Tabelle einlesen
    table = table.iloc[0:516]  # Nur die relevanten Zeilen (Dann macht es später keine Probleme weil lehre Spalten als Nan angezeigt werden)

    file_img = sorted(glob.glob(path_img_gesamt + "/*"))

    # -------------------------------------- per image -----------------------------------------------------
    for path_img in file_img:

        #path_img = "/home/wolfda/Clinic_Data/Challenge/CT_PreTrain/Downstream/Data/Part1_Part2/images/RibFrac300-image.nii.gz"

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

        # leere Lisen erstelln für jede Potenzielle Klasse
        class_0, class_1, class_2, class_3, class_4, \
        class_5, class_6, class_7, class_8, \
        class_9, class_10, class_11, class_12, \
        class_13, class_14, class_15, class_16, \
        class_17, class_18, class_19, class_20, \
        class_21, class_22, class_23, class_24, \
        class_25, class_26, class_27, class_28, \
        class_29, class_30, class_31, class_32, \
        class_33, class_34, class_35, class_36, \
        class_37, class_38, class_39,  = ([] for i in range(40)) # Dicts zum speichern der Schichten der einzelnen Klassen

        is_class = 0 # Anzahl der Bilder mit Klasse

        print(np.unique(seg))

        for i in range(0, img.shape[0]): # durchläuft alle Schichten
            # 3D -> 2D
            img_cut = img[i, :, :]
            seg_cut = seg[i, :, :]

            # listet alle grauwerte auf die es findet
            classes = np.unique(seg_cut)


            # Schichten des img wo es seg gibt -> Mittelüunkt der Segmentierung finden -> Quadrat (40x40) speichern (eine Liste pro klasse)

            if 1 in classes:

                seg_cut_1 = seg_cut.copy()
                # Durchläuft alle Pixel
                for x in range(seg_cut.shape[0]):
                    for y in range(seg_cut.shape[0]):
                        # Die Pixel mit Value nn zu Value 1 umwandeln
                        if seg_cut[x,y] == 1.0:
                            seg_cut_1[x,y]=1.0
                        # Alle anderen Pixel zu Value 0 umwandeln
                        else:
                            seg_cut_1[x,y] = 0.0
                # Mittelpunkt der Segmenation berechnen
                mean = mean_seg(seg_cut_1).astype(int)
                # Bild cutten [40,40]
                if mean[0] >= 40 and mean[0] <= img_cut.shape[0] - 40 and mean[1] >= 40 and mean[1] <= img_cut.shape[1] - 40 :
                    img_cut_1 = img_cut[mean[0] - 40:mean[0] + 40, mean[1] - 40:mean[1] + 40]
                    # Cut Bild dazufügen
                    class_1.append(img_cut_1)
                    is_class+=1

            if 2 in classes:

                seg_cut_2 = seg_cut.copy() #*
                # Durchläuft alle Pixel
                for x in range(seg_cut.shape[0]):
                    for y in range(seg_cut.shape[0]):
                        # Die Pixel mit Value nn zu Value 1 umwandeln
                        if seg_cut[x, y] == 2.0: #**
                            seg_cut_2[x, y] = 1.0 #*
                        # Alle anderen Pixel zu Value 0 umwandeln
                        else:
                            seg_cut_2[x, y] = 0.0 #*
                # Mittelpunkt der Segmenation berechnen
                mean = mean_seg(seg_cut_2).astype(int) #*
                # Bild cutten [40,40]
                if mean[0] >= 40 and mean[0] <= img_cut.shape[0] - 40 and mean[1] >= 40 and mean[1] <= img_cut.shape[
                    1] - 40:
                    img_cut_2 = img_cut[mean[0] - 40:mean[0] + 40, mean[1] - 40:mean[1] + 40] #***
                    # Cut Bild dazufügen
                    class_2.append(img_cut_2) #***
                    is_class += 1

            if 3 in classes:

                seg_cut_3 = seg_cut.copy()  # *
                # Durchläuft alle Pixel
                for x in range(seg_cut.shape[0]):
                    for y in range(seg_cut.shape[0]):
                        # Die Pixel mit Value nn zu Value 1 umwandeln
                        if seg_cut[x, y] == 3.0:  # **
                            seg_cut_3[x, y] = 1.0  # *
                        # Alle anderen Pixel zu Value 0 umwandeln
                        else:
                            seg_cut_3[x, y] = 0.0  # *
                # Mittelpunkt der Segmenation berechnen
                mean = mean_seg(seg_cut_3).astype(int)  # *
                # Bild cutten [40,40]
                if mean[0] >= 40 and mean[0] <= img_cut.shape[0] - 40 and mean[1] >= 40 and mean[1] <= img_cut.shape[
                    1] - 40:
                    img_cut_3 = img_cut[mean[0] - 40:mean[0] + 40, mean[1] - 40:mean[1] + 40]  # ***
                    # Cut Bild dazufügen
                    class_3.append(img_cut_3)  # ***
                    is_class += 1

            if 4 in classes:

                seg_cut_4 = seg_cut.copy()  # *
                # Durchläuft alle Pixel
                for x in range(seg_cut.shape[0]):
                    for y in range(seg_cut.shape[0]):
                        # Die Pixel mit Value nn zu Value 1 umwandeln
                        if seg_cut[x, y] == 4.0:  # **
                            seg_cut_4[x, y] = 1.0  # *
                        # Alle anderen Pixel zu Value 0 umwandeln
                        else:
                            seg_cut_4[x, y] = 0.0  # *
                # Mittelpunkt der Segmenation berechnen
                mean = mean_seg(seg_cut_4).astype(int)  # *
                # Bild cutten [40,40]
                if mean[0] >= 40 and mean[0] <= img_cut.shape[0] - 40 and mean[1] >= 40 and mean[1] <= img_cut.shape[
                    1] - 40:
                    img_cut_4 = img_cut[mean[0] - 40:mean[0] + 40, mean[1] - 40:mean[1] + 40]  # ***
                    # Cut Bild dazufügen
                    class_4.append(img_cut_4)  # ***
                    is_class += 1

            if 5 in classes:

                seg_cut_5 = seg_cut.copy()  # *
                # Durchläuft alle Pixel
                for x in range(seg_cut.shape[0]):
                    for y in range(seg_cut.shape[0]):
                        # Die Pixel mit Value nn zu Value 1 umwandeln
                        if seg_cut[x, y] == 5.0:  # **
                            seg_cut_5[x, y] = 1.0  # *
                        # Alle anderen Pixel zu Value 0 umwandeln
                        else:
                            seg_cut_5[x, y] = 0.0  # *
                # Mittelpunkt der Segmenation berechnen
                mean = mean_seg(seg_cut_5).astype(int)  # *
                # Bild cutten [40,40]
                if mean[0] >= 40 and mean[0] <= img_cut.shape[0] - 40 and mean[1] >= 40 and mean[1] <= img_cut.shape[
                    1] - 40:
                    img_cut_5 = img_cut[mean[0] - 40:mean[0] + 40, mean[1] - 40:mean[1] + 40]  # ***
                    # Cut Bild dazufügen
                    class_5.append(img_cut_5)  # ***
                    is_class += 1

            if 6 in classes:

                seg_cut_6 = seg_cut.copy()  # *
                # Durchläuft alle Pixel
                for x in range(seg_cut.shape[0]):
                    for y in range(seg_cut.shape[0]):
                        # Die Pixel mit Value nn zu Value 1 umwandeln
                        if seg_cut[x, y] == 6.0:  # **
                            seg_cut_6[x, y] = 1.0  # *
                        # Alle anderen Pixel zu Value 0 umwandeln
                        else:
                            seg_cut_6[x, y] = 0.0  # *
                # Mittelpunkt der Segmenation berechnen
                mean = mean_seg(seg_cut_6).astype(int)  # *
                # Bild cutten [40,40]
                if mean[0] >= 40 and mean[0] <= img_cut.shape[0] - 40 and mean[1] >= 40 and mean[1] <= img_cut.shape[
                    1] - 40:
                    img_cut_6 = img_cut[mean[0] - 40:mean[0] + 40, mean[1] - 40:mean[1] + 40]  # ***
                    # Cut Bild dazufügen
                    class_6.append(img_cut_6)  # ***
                    is_class += 1

            if 7 in classes:

                seg_cut_7 = seg_cut.copy()  # *
                # Durchläuft alle Pixel
                for x in range(seg_cut.shape[0]):
                    for y in range(seg_cut.shape[0]):
                        # Die Pixel mit Value nn zu Value 1 umwandeln
                        if seg_cut[x, y] == 7.0:  # **
                            seg_cut_7[x, y] = 1.0  # *
                        # Alle anderen Pixel zu Value 0 umwandeln
                        else:
                            seg_cut_7[x, y] = 0.0  # *
                # Mittelpunkt der Segmenation berechnen
                mean = mean_seg(seg_cut_7).astype(int)  # *
                # Bild cutten [40,40]
                if mean[0] >= 40 and mean[0] <= img_cut.shape[0] - 40 and mean[1] >= 40 and mean[1] <= img_cut.shape[
                    1] - 40:
                    img_cut_7 = img_cut[mean[0] - 40:mean[0] + 40, mean[1] - 40:mean[1] + 40]  # ***
                    # Cut Bild dazufügen
                    class_7.append(img_cut_7)  # ***
                    is_class += 1

            if 8 in classes:

                seg_cut_8 = seg_cut.copy()  # *
                # Durchläuft alle Pixel
                for x in range(seg_cut.shape[0]):
                    for y in range(seg_cut.shape[0]):
                        # Die Pixel mit Value nn zu Value 1 umwandeln
                        if seg_cut[x, y] == 8.0:  # **
                            seg_cut_8[x, y] = 1.0  # *
                        # Alle anderen Pixel zu Value 0 umwandeln
                        else:
                            seg_cut_8[x, y] = 0.0  # *
                # Mittelpunkt der Segmenation berechnen
                mean = mean_seg(seg_cut_8).astype(int)  # *
                # Bild cutten [40,40]
                if mean[0] >= 40 and mean[0] <= img_cut.shape[0] - 40 and mean[1] >= 40 and mean[1] <= img_cut.shape[
                    1] - 40:
                    img_cut_8 = img_cut[mean[0] - 40:mean[0] + 40, mean[1] - 40:mean[1] + 40]  # ***
                    # Cut Bild dazufügen
                    class_8.append(img_cut_8)  # ***
                    is_class += 1

            if 9 in classes:

                seg_cut_9 = seg_cut.copy()  # *
                # Durchläuft alle Pixel
                for x in range(seg_cut.shape[0]):
                    for y in range(seg_cut.shape[0]):
                        # Die Pixel mit Value nn zu Value 1 umwandeln
                        if seg_cut[x, y] == 9.0:  # **
                            seg_cut_9[x, y] = 1.0  # *
                        # Alle anderen Pixel zu Value 0 umwandeln
                        else:
                            seg_cut_9[x, y] = 0.0  # *
                # Mittelpunkt der Segmenation berechnen
                mean = mean_seg(seg_cut_9).astype(int)  # *
                # Bild cutten [40,40]
                if mean[0] >= 40 and mean[0] <= img_cut.shape[0] - 40 and mean[1] >= 40 and mean[1] <= img_cut.shape[
                    1] - 40:
                    img_cut_9 = img_cut[mean[0] - 40:mean[0] + 40, mean[1] - 40:mean[1] + 40]  # ***
                    # Cut Bild dazufügen
                    class_9.append(img_cut_9)  # ***
                    is_class += 1

            if 10 in classes:

                seg_cut_10 = seg_cut.copy()  # *
                # Durchläuft alle Pixel
                for x in range(seg_cut.shape[0]):
                    for y in range(seg_cut.shape[0]):
                        # Die Pixel mit Value nn zu Value 1 umwandeln
                        if seg_cut[x, y] == 10.0:  # **
                            seg_cut_10[x, y] = 1.0  # *
                        # Alle anderen Pixel zu Value 0 umwandeln
                        else:
                            seg_cut_10[x, y] = 0.0  # *
                # Mittelpunkt der Segmenation berechnen
                mean = mean_seg(seg_cut_10).astype(int)  # *
                # Bild cutten [40,40]
                if mean[0] >= 40 and mean[0] <= img_cut.shape[0] - 40 and mean[1] >= 40 and mean[1] <= img_cut.shape[
                    1] - 40:
                    img_cut_10 = img_cut[mean[0] - 40:mean[0] + 40, mean[1] - 40:mean[1] + 40]  # ***
                    # Cut Bild dazufügen
                    class_10.append(img_cut_10)  # ***
                    is_class += 1

            if 11 in classes:

                seg_cut_11 = seg_cut.copy()  # *
                # Durchläuft alle Pixel
                for x in range(seg_cut.shape[0]):
                    for y in range(seg_cut.shape[0]):
                        # Die Pixel mit Value nn zu Value 1 umwandeln
                        if seg_cut[x, y] == 11.0:  # **
                            seg_cut_11[x, y] = 1.0  # *
                        # Alle anderen Pixel zu Value 0 umwandeln
                        else:
                            seg_cut_11[x, y] = 0.0  # *
                # Mittelpunkt der Segmenation berechnen
                mean = mean_seg(seg_cut_11).astype(int)  # *
                # Bild cutten [40,40]
                if mean[0] >= 40 and mean[0] <= img_cut.shape[0] - 40 and mean[1] >= 40 and mean[1] <= img_cut.shape[
                    1] - 40:
                    img_cut_11 = img_cut[mean[0] - 40:mean[0] + 40, mean[1] - 40:mean[1] + 40]  # ***
                    # Cut Bild dazufügen
                    class_11.append(img_cut_11)  # ***
                    is_class += 1

            if 12 in classes:

                seg_cut_12 = seg_cut.copy()  # *
                # Durchläuft alle Pixel
                for x in range(seg_cut.shape[0]):
                    for y in range(seg_cut.shape[0]):
                        # Die Pixel mit Value nn zu Value 1 umwandeln
                        if seg_cut[x, y] == 12.0:  # **
                            seg_cut_12[x, y] = 1.0  # *
                        # Alle anderen Pixel zu Value 0 umwandeln
                        else:
                            seg_cut_12[x, y] = 0.0  # *
                # Mittelpunkt der Segmenation berechnen
                mean = mean_seg(seg_cut_12).astype(int)  # *
                # Bild cutten [40,40]
                if mean[0] >= 40 and mean[0] <= img_cut.shape[0] - 40 and mean[1] >= 40 and mean[1] <= img_cut.shape[
                    1] - 40:
                    img_cut_12 = img_cut[mean[0] - 40:mean[0] + 40, mean[1] - 40:mean[1] + 40]  # ***
                    # Cut Bild dazufügen
                    class_12.append(img_cut_12)  # ***
                    is_class += 1

            if 13 in classes:

                seg_cut_13 = seg_cut.copy()  # *
                # Durchläuft alle Pixel
                for x in range(seg_cut.shape[0]):
                    for y in range(seg_cut.shape[0]):
                        # Die Pixel mit Value nn zu Value 1 umwandeln
                        if seg_cut[x, y] == 13.0:  # **
                            seg_cut_13[x, y] = 1.0  # *
                        # Alle anderen Pixel zu Value 0 umwandeln
                        else:
                            seg_cut_13[x, y] = 0.0  # *
                # Mittelpunkt der Segmenation berechnen
                mean = mean_seg(seg_cut_13).astype(int)  # *
                # Bild cutten [40,40]
                if mean[0] >= 40 and mean[0] <= img_cut.shape[0] - 40 and mean[1] >= 40 and mean[1] <= img_cut.shape[
                    1] - 40:
                    img_cut_13 = img_cut[mean[0] - 40:mean[0] + 40, mean[1] - 40:mean[1] + 40]  # ***
                    # Cut Bild dazufügen
                    class_13.append(img_cut_13)  # ***
                    is_class += 1

            if 14 in classes:

                seg_cut_14 = seg_cut.copy()  # *
                # Durchläuft alle Pixel
                for x in range(seg_cut.shape[0]):
                    for y in range(seg_cut.shape[0]):
                        # Die Pixel mit Value nn zu Value 1 umwandeln
                        if seg_cut[x, y] == 14.0:  # **
                            seg_cut_14[x, y] = 1.0  # *
                        # Alle anderen Pixel zu Value 0 umwandeln
                        else:
                            seg_cut_14[x, y] = 0.0  # *
                # Mittelpunkt der Segmenation berechnen
                mean = mean_seg(seg_cut_14).astype(int)  # *
                # Bild cutten [40,40]
                if mean[0] >= 40 and mean[0] <= img_cut.shape[0] - 40 and mean[1] >= 40 and mean[1] <= img_cut.shape[
                    1] - 40:
                    img_cut_14 = img_cut[mean[0] - 40:mean[0] + 40, mean[1] - 40:mean[1] + 40]  # ***
                    # Cut Bild dazufügen
                    class_14.append(img_cut_14)  # ***
                    is_class += 1

            if 15 in classes:

                seg_cut_15 = seg_cut.copy()  # *
                # Durchläuft alle Pixel
                for x in range(seg_cut.shape[0]):
                    for y in range(seg_cut.shape[0]):
                        # Die Pixel mit Value nn zu Value 1 umwandeln
                        if seg_cut[x, y] == 15.0:  # **
                            seg_cut_15[x, y] = 1.0  # *
                        # Alle anderen Pixel zu Value 0 umwandeln
                        else:
                            seg_cut_15[x, y] = 0.0  # *
                # Mittelpunkt der Segmenation berechnen
                mean = mean_seg(seg_cut_15).astype(int)  # *
                # Bild cutten [40,40]
                if mean[0] >= 40 and mean[0] <= img_cut.shape[0] - 40 and mean[1] >= 40 and mean[1] <= img_cut.shape[
                    1] - 40:
                    img_cut_15 = img_cut[mean[0] - 40:mean[0] + 40, mean[1] - 40:mean[1] + 40]  # ***
                    # Cut Bild dazufügen
                    class_15.append(img_cut_15)  # ***
                    is_class += 1

            if 16 in classes:

                seg_cut_16 = seg_cut.copy()  # *
                # Durchläuft alle Pixel
                for x in range(seg_cut.shape[0]):
                    for y in range(seg_cut.shape[0]):
                        # Die Pixel mit Value nn zu Value 1 umwandeln
                        if seg_cut[x, y] == 16.0:  # **
                            seg_cut_16[x, y] = 1.0  # *
                        # Alle anderen Pixel zu Value 0 umwandeln
                        else:
                            seg_cut_16[x, y] = 0.0  # *
                # Mittelpunkt der Segmenation berechnen
                mean = mean_seg(seg_cut_16).astype(int)  # *
                # Bild cutten [40,40]
                if mean[0] >= 40 and mean[0] <= img_cut.shape[0] - 40 and mean[1] >= 40 and mean[1] <= img_cut.shape[
                    1] - 40:
                    img_cut_16 = img_cut[mean[0] - 40:mean[0] + 40, mean[1] - 40:mean[1] + 40]  # ***
                    # Cut Bild dazufügen
                    class_16.append(img_cut_16)  # ***
                    is_class += 1

            if 17 in classes:

                seg_cut_17 = seg_cut.copy()  # *
                # Durchläuft alle Pixel
                for x in range(seg_cut.shape[0]):
                    for y in range(seg_cut.shape[0]):
                        # Die Pixel mit Value nn zu Value 1 umwandeln
                        if seg_cut[x, y] == 17.0:  # **
                            seg_cut_17[x, y] = 1.0  # *
                        # Alle anderen Pixel zu Value 0 umwandeln
                        else:
                            seg_cut_17[x, y] = 0.0  # *
                # Mittelpunkt der Segmenation berechnen
                mean = mean_seg(seg_cut_17).astype(int)  # *
                # Bild cutten [40,40]
                if mean[0] >= 40 and mean[0] <= img_cut.shape[0] - 40 and mean[1] >= 40 and mean[1] <= img_cut.shape[
                    1] - 40:
                    img_cut_17 = img_cut[mean[0] - 40:mean[0] + 40, mean[1] - 40:mean[1] + 40]  # ***
                    # Cut Bild dazufügen
                    class_17.append(img_cut_17)  # ***
                    is_class += 1

            if 18 in classes:

                seg_cut_18 = seg_cut.copy()  # *
                # Durchläuft alle Pixel
                for x in range(seg_cut.shape[0]):
                    for y in range(seg_cut.shape[0]):
                        # Die Pixel mit Value nn zu Value 1 umwandeln
                        if seg_cut[x, y] == 18.0:  # **
                            seg_cut_18[x, y] = 1.0  # *
                        # Alle anderen Pixel zu Value 0 umwandeln
                        else:
                            seg_cut_18[x, y] = 0.0  # *
                # Mittelpunkt der Segmenation berechnen
                mean = mean_seg(seg_cut_18).astype(int)  # *
                # Bild cutten [40,40]
                if mean[0] >= 40 and mean[0] <= img_cut.shape[0] - 40 and mean[1] >= 40 and mean[1] <= img_cut.shape[
                    1] - 40:
                    img_cut_18 = img_cut[mean[0] - 40:mean[0] + 40, mean[1] - 40:mean[1] + 40]  # ***
                    # Cut Bild dazufügen
                    class_18.append(img_cut_18)  # ***
                    is_class += 1

            if 19 in classes:

                seg_cut_19 = seg_cut.copy()  # *
                # Durchläuft alle Pixel
                for x in range(seg_cut.shape[0]):
                    for y in range(seg_cut.shape[0]):
                        # Die Pixel mit Value nn zu Value 1 umwandeln
                        if seg_cut[x, y] == 19.0:  # **
                            seg_cut_19[x, y] = 1.0  # *
                        # Alle anderen Pixel zu Value 0 umwandeln
                        else:
                            seg_cut_19[x, y] = 0.0  # *
                # Mittelpunkt der Segmenation berechnen
                mean = mean_seg(seg_cut_19).astype(int)  # *
                # Bild cutten [40,40]
                if mean[0] >= 40 and mean[0] <= img_cut.shape[0] - 40 and mean[1] >= 40 and mean[1] <= img_cut.shape[
                    1] - 40:
                    img_cut_19 = img_cut[mean[0] - 40:mean[0] + 40, mean[1] - 40:mean[1] + 40]  # ***
                    # Cut Bild dazufügen
                    class_19.append(img_cut_19)  # ***
                    is_class += 1

            if 20 in classes:

                seg_cut_20 = seg_cut.copy()  # *
                # Durchläuft alle Pixel
                for x in range(seg_cut.shape[0]):
                    for y in range(seg_cut.shape[0]):
                        # Die Pixel mit Value nn zu Value 1 umwandeln
                        if seg_cut[x, y] == 20.0:  # **
                            seg_cut_20[x, y] = 1.0  # *
                        # Alle anderen Pixel zu Value 0 umwandeln
                        else:
                            seg_cut_20[x, y] = 0.0  # *
                # Mittelpunkt der Segmenation berechnen
                mean = mean_seg(seg_cut_20).astype(int)  # *
                # Bild cutten [40,40]
                if mean[0] >= 40 and mean[0] <= img_cut.shape[0] - 40 and mean[1] >= 40 and mean[1] <= img_cut.shape[
                    1] - 40:
                    img_cut_20 = img_cut[mean[0] - 40:mean[0] + 40, mean[1] - 40:mean[1] + 40]  # ***
                    # Cut Bild dazufügen
                    class_20.append(img_cut_20)  # ***
                    is_class += 1

            if 21 in classes:

                seg_cut_21 = seg_cut.copy()  # *
                # Durchläuft alle Pixel
                for x in range(seg_cut.shape[0]):
                    for y in range(seg_cut.shape[0]):
                        # Die Pixel mit Value nn zu Value 1 umwandeln
                        if seg_cut[x, y] == 21.0:  # **
                            seg_cut_21[x, y] = 1.0  # *
                        # Alle anderen Pixel zu Value 0 umwandeln
                        else:
                            seg_cut_21[x, y] = 0.0  # *
                # Mittelpunkt der Segmenation berechnen
                mean = mean_seg(seg_cut_21).astype(int)  # *
                # Bild cutten [40,40]
                if mean[0] >= 40 and mean[0] <= img_cut.shape[0] - 40 and mean[1] >= 40 and mean[1] <= img_cut.shape[
                    1] - 40:
                    img_cut_21 = img_cut[mean[0] - 40:mean[0] + 40, mean[1] - 40:mean[1] + 40]  # ***
                    # Cut Bild dazufügen
                    class_21.append(img_cut_21)  # ***
                    is_class += 1

            if 22 in classes:

                seg_cut_22 = seg_cut.copy()  # *
                # Durchläuft alle Pixel
                for x in range(seg_cut.shape[0]):
                    for y in range(seg_cut.shape[0]):
                        # Die Pixel mit Value nn zu Value 1 umwandeln
                        if seg_cut[x, y] == 22.0:  # **
                            seg_cut_22[x, y] = 1.0  # *
                        # Alle anderen Pixel zu Value 0 umwandeln
                        else:
                            seg_cut_22[x, y] = 0.0  # *
                # Mittelpunkt der Segmenation berechnen
                mean = mean_seg(seg_cut_22).astype(int)  # *
                # Bild cutten [40,40]
                if mean[0] >= 40 and mean[0] <= img_cut.shape[0] - 40 and mean[1] >= 40 and mean[1] <= img_cut.shape[
                    1] - 40:
                    img_cut_22 = img_cut[mean[0] - 40:mean[0] + 40, mean[1] - 40:mean[1] + 40]  # ***
                    # Cut Bild dazufügen
                    class_22.append(img_cut_22)  # ***
                    is_class += 1

            if 23 in classes:

                seg_cut_23 = seg_cut.copy()  # *
                # Durchläuft alle Pixel
                for x in range(seg_cut.shape[0]):
                    for y in range(seg_cut.shape[0]):
                        # Die Pixel mit Value nn zu Value 1 umwandeln
                        if seg_cut[x, y] == 23.0:  # **
                            seg_cut_23[x, y] = 1.0  # *
                        # Alle anderen Pixel zu Value 0 umwandeln
                        else:
                            seg_cut_23[x, y] = 0.0  # *
                # Mittelpunkt der Segmenation berechnen
                mean = mean_seg(seg_cut_23).astype(int)  # *
                # Bild cutten [40,40]
                if mean[0] >= 40 and mean[0] <= img_cut.shape[0] - 40 and mean[1] >= 40 and mean[1] <= img_cut.shape[
                    1] - 40:
                    img_cut_23 = img_cut[mean[0] - 40:mean[0] + 40, mean[1] - 40:mean[1] + 40]  # ***
                    # Cut Bild dazufügen
                    class_23.append(img_cut_23)  # ***
                    is_class += 1

            if 24 in classes:

                seg_cut_24 = seg_cut.copy()  # *
                # Durchläuft alle Pixel
                for x in range(seg_cut.shape[0]):
                    for y in range(seg_cut.shape[0]):
                        # Die Pixel mit Value nn zu Value 1 umwandeln
                        if seg_cut[x, y] == 24.0:  # **
                            seg_cut_24[x, y] = 1.0  # *
                        # Alle anderen Pixel zu Value 0 umwandeln
                        else:
                            seg_cut_24[x, y] = 0.0  # *
                # Mittelpunkt der Segmenation berechnen
                mean = mean_seg(seg_cut_24).astype(int)  # *
                # Bild cutten [40,40]
                if mean[0] >= 40 and mean[0] <= img_cut.shape[0] - 40 and mean[1] >= 40 and mean[1] <= img_cut.shape[
                    1] - 40:
                    img_cut_24 = img_cut[mean[0] - 40:mean[0] + 40, mean[1] - 40:mean[1] + 40]  # ***
                    # Cut Bild dazufügen
                    class_24.append(img_cut_24)  # ***
                    is_class += 1

            if 25 in classes:

                seg_cut_25 = seg_cut.copy()  # *
                # Durchläuft alle Pixel
                for x in range(seg_cut.shape[0]):
                    for y in range(seg_cut.shape[0]):
                        # Die Pixel mit Value nn zu Value 1 umwandeln
                        if seg_cut[x, y] == 25.0:  # **
                            seg_cut_25[x, y] = 1.0  # *
                        # Alle anderen Pixel zu Value 0 umwandeln
                        else:
                            seg_cut_25[x, y] = 0.0  # *
                # Mittelpunkt der Segmenation berechnen
                mean = mean_seg(seg_cut_25).astype(int)  # *
                # Bild cutten [40,40]
                if mean[0] >= 40 and mean[0] <= img_cut.shape[0] - 40 and mean[1] >= 40 and mean[1] <= img_cut.shape[
                    1] - 40:
                    img_cut_25 = img_cut[mean[0] - 40:mean[0] + 40, mean[1] - 40:mean[1] + 40]  # ***
                    # Cut Bild dazufügen
                    class_25.append(img_cut_25)  # ***
                    is_class += 1

            if 26 in classes:

                seg_cut_26 = seg_cut.copy()  # *
                # Durchläuft alle Pixel
                for x in range(seg_cut.shape[0]):
                    for y in range(seg_cut.shape[0]):
                        # Die Pixel mit Value nn zu Value 1 umwandeln
                        if seg_cut[x, y] == 26.0:  # **
                            seg_cut_26[x, y] = 1.0  # *
                        # Alle anderen Pixel zu Value 0 umwandeln
                        else:
                            seg_cut_26[x, y] = 0.0  # *
                # Mittelpunkt der Segmenation berechnen
                mean = mean_seg(seg_cut_26).astype(int)  # *
                # Bild cutten [40,40]
                if mean[0] >= 40 and mean[0] <= img_cut.shape[0] - 40 and mean[1] >= 40 and mean[1] <= img_cut.shape[
                    1] - 40:
                    img_cut_26 = img_cut[mean[0] - 40:mean[0] + 40, mean[1] - 40:mean[1] + 40]  # ***
                    # Cut Bild dazufügen
                    class_26.append(img_cut_26)  # ***
                    is_class += 1

            if 27 in classes:

                seg_cut_27 = seg_cut.copy()  # *
                # Durchläuft alle Pixel
                for x in range(seg_cut.shape[0]):
                    for y in range(seg_cut.shape[0]):
                        # Die Pixel mit Value nn zu Value 1 umwandeln
                        if seg_cut[x, y] == 27.0:  # **
                            seg_cut_27[x, y] = 1.0  # *
                        # Alle anderen Pixel zu Value 0 umwandeln
                        else:
                            seg_cut_27[x, y] = 0.0  # *
                # Mittelpunkt der Segmenation berechnen
                mean = mean_seg(seg_cut_27).astype(int)  # *
                # Bild cutten [40,40]
                if mean[0] >= 40 and mean[0] <= img_cut.shape[0] - 40 and mean[1] >= 40 and mean[1] <= img_cut.shape[
                    1] - 40:
                    img_cut_27 = img_cut[mean[0] - 40:mean[0] + 40, mean[1] - 40:mean[1] + 40]  # ***
                    # Cut Bild dazufügen
                    class_27.append(img_cut_27)  # ***
                    is_class += 1

            if 28 in classes:

                seg_cut_28 = seg_cut.copy()  # *
                # Durchläuft alle Pixel
                for x in range(seg_cut.shape[0]):
                    for y in range(seg_cut.shape[0]):
                        # Die Pixel mit Value nn zu Value 1 umwandeln
                        if seg_cut[x, y] == 28.0:  # **
                            seg_cut_28[x, y] = 1.0  # *
                        # Alle anderen Pixel zu Value 0 umwandeln
                        else:
                            seg_cut_28[x, y] = 0.0  # *
                # Mittelpunkt der Segmenation berechnen
                mean = mean_seg(seg_cut_28).astype(int)  # *
                # Bild cutten [40,40]
                if mean[0] >= 40 and mean[0] <= img_cut.shape[0] - 40 and mean[1] >= 40 and mean[1] <= img_cut.shape[
                    1] - 40:
                    img_cut_28 = img_cut[mean[0] - 40:mean[0] + 40, mean[1] - 40:mean[1] + 40]  # ***
                    # Cut Bild dazufügen
                    class_28.append(img_cut_28)  # ***
                    is_class += 1

            if 29 in classes:

                seg_cut_29 = seg_cut.copy()  # *
                # Durchläuft alle Pixel
                for x in range(seg_cut.shape[0]):
                    for y in range(seg_cut.shape[0]):
                        # Die Pixel mit Value nn zu Value 1 umwandeln
                        if seg_cut[x, y] == 29.0:  # **
                            seg_cut_29[x, y] = 1.0  # *
                        # Alle anderen Pixel zu Value 0 umwandeln
                        else:
                            seg_cut_29[x, y] = 0.0  # *
                # Mittelpunkt der Segmenation berechnen
                mean = mean_seg(seg_cut_29).astype(int)  # *
                # Bild cutten [40,40]
                if mean[0] >= 40 and mean[0] <= img_cut.shape[0] - 40 and mean[1] >= 40 and mean[1] <= img_cut.shape[
                    1] - 40:
                    img_cut_29 = img_cut[mean[0] - 40:mean[0] + 40, mean[1] - 40:mean[1] + 40]  # ***
                    # Cut Bild dazufügen
                    class_29.append(img_cut_29)  # ***
                    is_class += 1

            if 30 in classes:

                seg_cut_30 = seg_cut.copy()  # *
                # Durchläuft alle Pixel
                for x in range(seg_cut.shape[0]):
                    for y in range(seg_cut.shape[0]):
                        # Die Pixel mit Value nn zu Value 1 umwandeln
                        if seg_cut[x, y] == 30.0:  # **
                            seg_cut_30[x, y] = 1.0  # *
                        # Alle anderen Pixel zu Value 0 umwandeln
                        else:
                            seg_cut_30[x, y] = 0.0  # *
                # Mittelpunkt der Segmenation berechnen
                mean = mean_seg(seg_cut_30).astype(int)  # *
                # Bild cutten [40,40]
                if mean[0] >= 40 and mean[0] <= img_cut.shape[0] - 40 and mean[1] >= 40 and mean[1] <= img_cut.shape[
                    1] - 40:
                    img_cut_30 = img_cut[mean[0] - 40:mean[0] + 40, mean[1] - 40:mean[1] + 40]  # ***
                    # Cut Bild dazufügen
                    class_30.append(img_cut_30)  # ***
                    is_class += 1

            if 31 in classes:

                seg_cut_31 = seg_cut.copy()  # *
                # Durchläuft alle Pixel
                for x in range(seg_cut.shape[0]):
                    for y in range(seg_cut.shape[0]):
                        # Die Pixel mit Value nn zu Value 1 umwandeln
                        if seg_cut[x, y] == 31.0:  # **
                            seg_cut_31[x, y] = 1.0  # *
                        # Alle anderen Pixel zu Value 0 umwandeln
                        else:
                            seg_cut_31[x, y] = 0.0  # *
                # Mittelpunkt der Segmenation berechnen
                mean = mean_seg(seg_cut_31).astype(int)  # *
                # Bild cutten [40,40]
                if mean[0] >= 40 and mean[0] <= img_cut.shape[0] - 40 and mean[1] >= 40 and mean[1] <= img_cut.shape[
                    1] - 40:
                    img_cut_31 = img_cut[mean[0] - 40:mean[0] + 40, mean[1] - 40:mean[1] + 40]  # ***
                    # Cut Bild dazufügen
                    class_31.append(img_cut_31)  # ***
                    is_class += 1

            if 32 in classes:

                seg_cut_32 = seg_cut.copy()  # *
                # Durchläuft alle Pixel
                for x in range(seg_cut.shape[0]):
                    for y in range(seg_cut.shape[0]):
                        # Die Pixel mit Value nn zu Value 1 umwandeln
                        if seg_cut[x, y] == 32.0:  # **
                            seg_cut_32[x, y] = 1.0  # *
                        # Alle anderen Pixel zu Value 0 umwandeln
                        else:
                            seg_cut_32[x, y] = 0.0  # *
                # Mittelpunkt der Segmenation berechnen
                mean = mean_seg(seg_cut_32).astype(int)  # *
                # Bild cutten [40,40]
                if mean[0] >= 40 and mean[0] <= img_cut.shape[0] - 40 and mean[1] >= 40 and mean[1] <= img_cut.shape[
                    1] - 40:
                    img_cut_32 = img_cut[mean[0] - 40:mean[0] + 40, mean[1] - 40:mean[1] + 40]  # ***
                    # Cut Bild dazufügen
                    class_32.append(img_cut_32)  # ***
                    is_class += 1

            if 33 in classes:

                seg_cut_33 = seg_cut.copy()  # *
                # Durchläuft alle Pixel
                for x in range(seg_cut.shape[0]):
                    for y in range(seg_cut.shape[0]):
                        # Die Pixel mit Value nn zu Value 1 umwandeln
                        if seg_cut[x, y] == 33.0:  # **
                            seg_cut_33[x, y] = 1.0  # *
                        # Alle anderen Pixel zu Value 0 umwandeln
                        else:
                            seg_cut_33[x, y] = 0.0  # *
                # Mittelpunkt der Segmenation berechnen
                mean = mean_seg(seg_cut_33).astype(int)  # *
                # Bild cutten [40,40]
                if mean[0] >= 40 and mean[0] <= img_cut.shape[0] - 40 and mean[1] >= 40 and mean[1] <= img_cut.shape[
                    1] - 40:
                    img_cut_33 = img_cut[mean[0] - 40:mean[0] + 40, mean[1] - 40:mean[1] + 40]  # ***
                    # Cut Bild dazufügen
                    class_33.append(img_cut_33)  # ***
                    is_class += 1

            if 34 in classes:

                seg_cut_34 = seg_cut.copy()  # *
                # Durchläuft alle Pixel
                for x in range(seg_cut.shape[0]):
                    for y in range(seg_cut.shape[0]):
                        # Die Pixel mit Value nn zu Value 1 umwandeln
                        if seg_cut[x, y] == 34.0:  # **
                            seg_cut_34[x, y] = 1.0  # *
                        # Alle anderen Pixel zu Value 0 umwandeln
                        else:
                            seg_cut_34[x, y] = 0.0  # *
                # Mittelpunkt der Segmenation berechnen
                mean = mean_seg(seg_cut_34).astype(int)  # *
                # Bild cutten [40,40]
                if mean[0] >= 40 and mean[0] <= img_cut.shape[0] - 40 and mean[1] >= 40 and mean[1] <= img_cut.shape[
                    1] - 40:
                    img_cut_34 = img_cut[mean[0] - 40:mean[0] + 40, mean[1] - 40:mean[1] + 40]  # ***
                    # Cut Bild dazufügen
                    class_34.append(img_cut_34)  # ***
                    is_class += 1

            if 35 in classes:

                seg_cut_35 = seg_cut.copy()  # *
                # Durchläuft alle Pixel
                for x in range(seg_cut.shape[0]):
                    for y in range(seg_cut.shape[0]):
                        # Die Pixel mit Value nn zu Value 1 umwandeln
                        if seg_cut[x, y] == 35.0:  # **
                            seg_cut_35[x, y] = 1.0  # *
                        # Alle anderen Pixel zu Value 0 umwandeln
                        else:
                            seg_cut_35[x, y] = 0.0  # *
                # Mittelpunkt der Segmenation berechnen
                mean = mean_seg(seg_cut_35).astype(int)  # *
                # Bild cutten [40,40]
                if mean[0] >= 40 and mean[0] <= img_cut.shape[0] - 40 and mean[1] >= 40 and mean[1] <= img_cut.shape[
                    1] - 40:
                    img_cut_35 = img_cut[mean[0] - 40:mean[0] + 40, mean[1] - 40:mean[1] + 40]  # ***
                    # Cut Bild dazufügen
                    class_35.append(img_cut_35)  # ***
                    is_class += 1

            if 36 in classes:

                seg_cut_36 = seg_cut.copy()  # *
                # Durchläuft alle Pixel
                for x in range(seg_cut.shape[0]):
                    for y in range(seg_cut.shape[0]):
                        # Die Pixel mit Value nn zu Value 1 umwandeln
                        if seg_cut[x, y] == 36.0:  # **
                            seg_cut_36[x, y] = 1.0  # *
                        # Alle anderen Pixel zu Value 0 umwandeln
                        else:
                            seg_cut_36[x, y] = 0.0  # *
                # Mittelpunkt der Segmenation berechnen
                mean = mean_seg(seg_cut_36).astype(int)  # *
                # Bild cutten [40,40]
                if mean[0] >= 40 and mean[0] <= img_cut.shape[0] - 40 and mean[1] >= 40 and mean[1] <= img_cut.shape[
                    1] - 40:
                    img_cut_36 = img_cut[mean[0] - 40:mean[0] + 40, mean[1] - 40:mean[1] + 40]  # ***
                    # Cut Bild dazufügen
                    class_36.append(img_cut_36)  # ***
                    is_class += 1

            if 37 in classes:

                seg_cut_37 = seg_cut.copy()  # *
                # Durchläuft alle Pixel
                for x in range(seg_cut.shape[0]):
                    for y in range(seg_cut.shape[0]):
                        # Die Pixel mit Value nn zu Value 1 umwandeln
                        if seg_cut[x, y] == 37.0:  # **
                            seg_cut_37[x, y] = 1.0  # *
                        # Alle anderen Pixel zu Value 0 umwandeln
                        else:
                            seg_cut_37[x, y] = 0.0  # *
                # Mittelpunkt der Segmenation berechnen
                mean = mean_seg(seg_cut_37).astype(int)  # *
                # Bild cutten [40,40]
                if mean[0] >= 40 and mean[0] <= img_cut.shape[0] - 40 and mean[1] >= 40 and mean[1] <= img_cut.shape[
                    1] - 40:
                    img_cut_37 = img_cut[mean[0] - 40:mean[0] + 40, mean[1] - 40:mean[1] + 40]  # ***
                    # Cut Bild dazufügen
                    class_37.append(img_cut_37)  # ***
                    is_class += 1

            if 38 in classes:

                seg_cut_38 = seg_cut.copy()  # *
                # Durchläuft alle Pixel
                for x in range(seg_cut.shape[0]):
                    for y in range(seg_cut.shape[0]):
                        # Die Pixel mit Value nn zu Value 1 umwandeln
                        if seg_cut[x, y] == 38.0:  # **
                            seg_cut_38[x, y] = 1.0  # *
                        # Alle anderen Pixel zu Value 0 umwandeln
                        else:
                            seg_cut_38[x, y] = 0.0  # *
                # Mittelpunkt der Segmenation berechnen
                mean = mean_seg(seg_cut_38).astype(int)  # *
                # Bild cutten [40,40]
                if mean[0] >= 40 and mean[0] <= img_cut.shape[0] - 40 and mean[1] >= 40 and mean[1] <= img_cut.shape[
                    1] - 40:
                    img_cut_38 = img_cut[mean[0] - 40:mean[0] + 40, mean[1] - 40:mean[1] + 40]  # ***
                    # Cut Bild dazufügen
                    class_38.append(img_cut_38)  # ***
                    is_class += 1

            if 39 in classes:

                seg_cut_39 = seg_cut.copy()  # *
                # Durchläuft alle Pixel
                for x in range(seg_cut.shape[0]):
                    for y in range(seg_cut.shape[0]):
                        # Die Pixel mit Value nn zu Value 1 umwandeln
                        if seg_cut[x, y] == 39.0:  # **
                            seg_cut_39[x, y] = 1.0  # *
                        # Alle anderen Pixel zu Value 0 umwandeln
                        else:
                            seg_cut_39[x, y] = 0.0  # *
                # Mittelpunkt der Segmenation berechnen
                mean = mean_seg(seg_cut_39).astype(int)  # *
                # Bild cutten [40,40]
                if mean[0] >= 40 and mean[0] <= img_cut.shape[0] - 40 and mean[1] >= 40 and mean[1] <= img_cut.shape[
                    1] - 40:
                    img_cut_39 = img_cut[mean[0] - 40:mean[0] + 40, mean[1] - 40:mean[1] + 40]  # ***
                    # Cut Bild dazufügen
                    class_39.append(img_cut_39)  # ***
                    is_class += 1

        print("is class: ", is_class)
        print("total: ", img.shape[0])

        for i in range(len(np.unique(seg))): # so häufig machen wie es Klassen gibt (so häufig wie es verschiedenen Values in seg gibt)

            # Löscht alles wo es doppelt Klassen gibt
            # wenn mehr als 2 verschiedenne Grauwerte -> es gibt 2 Segmentierungen von unterschiedlichen Klassen in einer Schicht
            # Wenn z.B. in Schicht 40, die drei Klassen [2, 4, 5] sind, müssen die kompketten Listen von class_2, class_4, class_5 gelöscht werden
            # if i in double:
            #     vars()['class_' + str(i)][:] = []

            if len(vars()['class_' + str(i)]) > 3: # checken list empty and len > 3 (damit wir mindestens 3 Channel haben)
                # Mitte bestimmen
                midddle = len(vars()['class_' + str(i)]) / 2 # nimmt den richtigen class dict (class_1, class_2,...)


                if three_differnt_channel == True:
                    # Nur die 3 mittleren Schichten nehmen und in numpy array packen
                    image = np.empty([3, 80, 80])
                    image[0, ...] = vars()['class_' + str(i)][int(midddle) - 1] # nimmt den richtigen class dict (class_1, class_2,...)
                    image[1, ...] = vars()['class_' + str(i)][int(midddle)]     # nimmt den richtigen class dict (class_1, class_2,...)
                    image[2, ...] = vars()['class_' + str(i)][int(midddle) + 1] # nimmt den richtigen class dict (class_1, class_2,...)
                else:
                    # 3 mal die Mittlere Schicht nehmen
                    image = np.empty([3, 80, 80])
                    image[0, ...] = vars()['class_' + str(i)][int(midddle)]  # nimmt den richtigen class dict (class_1, class_2,...)
                    image[1, ...] = vars()['class_' + str(i)][int(midddle)]  # nimmt den richtigen class dict (class_1, class_2,...)
                    image[2, ...] = vars()['class_' + str(i)][int(midddle)]  # nimmt den richtigen class dict (class_1, class_2,...)


                # Normalization for jpeg/png
                image = interval_mapping(image, image.min(), image.max(), 0, 255)
                image = image.astype(np.uint8)
                image.astype(np.uint8)

                # Richtige Klasse aus csv
                table_class = table.loc[(table['public_id'] == name) & (table['label_id'] == i)]
                print(table_class['label_code'])
                lable = table_class['label_code'].item()
                print(lable)

                # Save
                image = image.transpose(2, 1, 0)  # (Anzahl Schichten,x,y)
                if lable == 1:
                    path = save_path_1 + "/" + str(name) + "_" + str(i) + "_" + "lable_" +str(lable) + ".jpeg"
                    cv2.imwrite(path, image)
                if lable == 2:
                    path = save_path_2 + "/" + str(name) + "_" + str(i) + "_" + "lable_" +str(lable) + ".jpeg"
                    cv2.imwrite(path, image)
                if lable == 3:
                    path = save_path_3 + "/" + str(name) + "_" + str(i) + "_" + "lable_" +str(lable) + ".jpeg"
                    cv2.imwrite(path, image)
                if lable == 4:
                    path = save_path_4 + "/" + str(name) + "_" + str(i) + "_" + "lable_" +str(lable) + ".jpeg"
                    cv2.imwrite(path, image)
                if lable == -1:
                    path = save_path_minus1 + "/" + str(name) + "_" + str(i) + "_" + "lable_" +str(lable) + ".jpeg"
                    cv2.imwrite(path, image)





if __name__ == '__main__':
    main()

