#Von jedem CT
    #Von jeder Schicht
       # Nur die Schichten wo es Knochenbrüche gibt (Seg nicht nur Schwarz)
	        #Ein Bild für jede Klasse: 1, 2, 3, 4, -1
	            # Die Mittlere Schicht von dem Knochenbruch ENTWEDER: plus die davor und die dahinter ODER: 3 mal die Mittlere hintereinader → 3 Channel
	    	                # Alle Klassen in einen Ordner speichern


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
    path_seg_gesamt = "/home/wolfda/Clinic_Data/Challenge/CT_PreTrain/Downstream/Part2_Val/lables"
    path_img_gesamt = "/home/wolfda/Clinic_Data/Challenge/CT_PreTrain/Downstream/Part2_Val/images"
    path_table = "/home/wolfda/Clinic_Data/Challenge/CT_PreTrain/Downstream/Part2_Val/ribfrac-val_part2-info.csv"

    # Todo: Pfade zum Speichern wählen: (voher erstellen)
    save_path = "/home/wolfda/Clinic_Data/Challenge/CT_PreTrain/Downstream/2D/DiffChan_Brueche_Gesamt/Mit_Bruch"

    # Todo: wählen für 3 Channel:
    # True: [DiffChen] Nimmt das mittlere Bild des Bruches + das davor und das dahinter
    # False: [SameChen] Nur das mittlere Bild des Bruches drei mal hintereinander
    three_differnt_channel = True


    # Tablle laden
    table = pd.read_csv(path_table)  # Komplette Tabelle einlesen
    # ToDo: Tabellen spalten anpassen!!!!!
    table = table.iloc[0:1770]  # Nur die relevanten Zeilen (Dann macht es später keine Probleme weil lehre Spalten als Nan angezeigt werden)

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

        double = np.array([]) # Schichten die mehr als 2 Klassen hat

        no_class = 0 # Anzahl der Bilder ohne Klasse
        is_class = 0 # Anzahl der Bilder mit Klasse
        double_class = 0 # Anzahl der Bilder mit doppelter Klasse

        print(np.unique(seg))

        for i in range(0, img.shape[0]): # durchläuft alle Schichten
            # 3D -> 2D
            img_cut = img[i, :, :]
            seg_cut = seg[i, :, :]

            # listet alle grauwerte auf die es findet
            classes = np.unique(seg_cut)
            #print(classes)

            # Schichten des img wo es seg gibt speichern (eine Liste pro klasse)
            if 1 in classes:
                class_1.append(img_cut)
                is_class+=1
                #print("1")
            elif 2 in classes:
                class_2.append(img_cut)
                is_class += 1
                #print("2")
            elif 3 in classes:
                class_3.append(img_cut)
                is_class += 1
                #print("3")
            elif 4 in classes:
                class_4.append(img_cut)
                is_class += 1
                #print("4")
            elif 5 in classes:
                class_5.append(img_cut)
                is_class += 1
                #print("5")
            elif 6 in classes:
                class_6.append(img_cut)
                is_class += 1
                #print("6")
            elif 7 in classes:
                class_7.append(img_cut)
                is_class += 1
            elif 8 in classes:
                class_8.append(img_cut)
                is_class += 1
            elif 9 in classes:
                class_9.append(img_cut)
                is_class += 1
            elif 10 in classes:
                class_10.append(img_cut)
                is_class += 1
            elif 11 in classes:
                class_11.append(img_cut)
                is_class += 1
            elif 12 in classes:
                class_12.append(img_cut)
                is_class += 1
            elif 13 in classes:
                class_13.append(img_cut)
                is_class += 1
            elif 14 in classes:
                class_14.append(img_cut)
                is_class += 1
            elif 15 in classes:
                class_15.append(img_cut)
                is_class += 1
            elif 16 in classes:
                class_16.append(img_cut)
                is_class += 1
            elif 17 in classes:
                class_17.append(img_cut)
                is_class += 1
            elif 18 in classes:
                class_18.append(img_cut)
                is_class += 1
            elif 19 in classes:
                class_19.append(img_cut)
                is_class += 1
            elif 20 in classes:
                class_20.append(img_cut)
                is_class += 1
            elif 21 in classes:
                class_21.append(img_cut)
                is_class += 1
            elif 22 in classes:
                class_22.append(img_cut)
                is_class += 1
            elif 23 in classes:
                class_23.append(img_cut)
                is_class += 1
            elif 24 in classes:
                class_24.append(img_cut)
                is_class += 1
            elif 25 in classes:
                class_25.append(img_cut)
                is_class += 1
            elif 26 in classes:
                class_26.append(img_cut)
                is_class += 1
            elif 27 in classes:
                class_27.append(img_cut)
                is_class += 1
            elif 28 in classes:
                class_28.append(img_cut)
                is_class += 1
            elif 29 in classes:
                class_29.append(img_cut)
                is_class += 1
            elif 30 in classes:
                class_30.append(img_cut)
                is_class += 1
            elif 31 in classes:
                class_31.append(img_cut)
                is_class += 1
            elif 32 in classes:
                class_32.append(img_cut)
                is_class += 1
            elif 33 in classes:
                class_33.append(img_cut)
                is_class += 1
            elif 34 in classes:
                class_34.append(img_cut)
                is_class += 1
            elif 35 in classes:
                class_35.append(img_cut)
                is_class += 1
            elif 36 in classes:
                class_36.append(img_cut)
                is_class += 1
            elif 37 in classes:
                class_37.append(img_cut)
                is_class += 1
            elif 38 in classes:
                class_38.append(img_cut)
                is_class += 1
            elif 39 in classes:
                class_39.append(img_cut)
                is_class += 1
            else:
                no_class+=1

        print("no class: ", no_class)
        print("is class: ", is_class)
        print("total: ", img.shape[0])


        for i in range(len(np.unique(seg))): # so häufig machen wie es Klassen gibt (so häufig wie es verschiedenen Values in seg gibt)

            if len(vars()['class_' + str(i)]) > 3: # checken list empty and len > 3 (damit wir mindestens 3 Channel haben)

                #print("Klasse", i)
                #print(len(vars()['class_' + str(i)]))
                # Mitte bestimmen
                midddle = len(vars()['class_' + str(i)]) / 2 # nimmt den richtigen class dict (class_1, class_2,...)

                if three_differnt_channel == True:
                    # Nur die 3 mittleren Schichten nehmen und in numpy array packen
                    image = np.empty([3, 512, 512])
                    image[0, ...] = vars()['class_' + str(i)][
                        int(midddle) - 1]  # nimmt den richtigen class dict (class_1, class_2,...)
                    image[1, ...] = vars()['class_' + str(i)][
                        int(midddle)]  # nimmt den richtigen class dict (class_1, class_2,...)
                    image[2, ...] = vars()['class_' + str(i)][
                        int(midddle) + 1]  # nimmt den richtigen class dict (class_1, class_2,...)
                else:
                    # 3 mal die Mittlere Schicht nehmen
                    image = np.empty([3, 512, 512])
                    image[0, ...] = vars()['class_' + str(i)][
                        int(midddle)]  # nimmt den richtigen class dict (class_1, class_2,...)
                    image[1, ...] = vars()['class_' + str(i)][
                        int(midddle)]  # nimmt den richtigen class dict (class_1, class_2,...)
                    image[2, ...] = vars()['class_' + str(i)][
                        int(midddle)]  # nimmt den richtigen class dict (class_1, class_2,...)

                # Normalization for jpeg/png
                image = interval_mapping(image, image.min(), image.max(), 0, 255)
                image = image.astype(np.uint8)
                image.astype(np.uint8)

                # Richtige Klasse aus csv
                table_class = table.loc[(table['public_id'] == name) & (table['label_id'] == i)]
                lable = table_class['label_code'].item()
                print(lable)

                # Save
                image = image.transpose(2, 1, 0)  # (Anzahl Schichten,x,y)

                path = save_path + "/" + str(name) + "_" + str(i) + "_" + "lable_" +str(lable) + ".jpeg"
                cv2.imwrite(path, image)






if __name__ == '__main__':
    main()

