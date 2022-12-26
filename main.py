
import random
import os
import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

random.seed(19)


# %% Zadatak 1

putanje_slike_vrste = {
    "anger": "Dataset/anger",
    "disgust": "Dataset/disgust",
    "fear": "Dataset/fear",
    "happiness": "Dataset/happiness",
    "neutral": "Dataset/neutral",
    "sadness": "Dataset/sadness",
    "surprise": "Dataset/surprise"}

vrste_broj = 7
uzorak_vrsta_obim = 15

podaci_ulazi = []
podaci_oznake = []

for vrsta, putanja in putanje_slike_vrste.items():
  datoteke_nazivi = os.listdir(putanja)
  for i in random.sample(range(len(datoteke_nazivi)), uzorak_vrsta_obim):
      putanja_slika = os.path.join(putanja, datoteke_nazivi[i])
      slika = cv2.imread(putanja_slika, 0)
      podaci_ulazi.append(np.array(slika))
      podaci_oznake.append(vrsta)

podaci_ulazi = np.array(podaci_ulazi) 

kodovi_vrste_brojevi = {
    "anger": 0, "disgust": 1, "fear": 2, "happiness": 3, "neutral" : 4, "sadness" : 5, "surprise" : 6}

podaci_oznake_kodirane = to_categorical(
    pd.Series(podaci_oznake).replace(kodovi_vrste_brojevi), vrste_broj)


knm = Sequential()
#PRVI SLOJ
knm.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), 
               padding="valid", activation="relu", use_bias=True, 
               input_shape=( 490, 640, 1)))

knm.add(MaxPooling2D(pool_size=(2, 2)))

# DRUGI SLOJ

knm.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), 
               padding="valid", activation="relu", use_bias=True))

knm.add(MaxPooling2D(pool_size=(2, 2)))


knm.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), 
               padding="valid", activation="relu", use_bias=True))

knm.add(MaxPooling2D(pool_size=(2, 2)))

knm.add(Flatten())

knm.add(Dense(7, activation="softmax"))

knm.summary()

udeo_obuka = 0.8
izbor_obuka = [i for i in random.sample(range(len(podaci_ulazi)), 
                                        int(udeo_obuka * 
                                            podaci_ulazi.shape[0]))]

indeksi_obuka = np.repeat(False, len(podaci_ulazi))
indeksi_obuka[izbor_obuka] = True
indeksi_test = np.logical_not(indeksi_obuka)

podaci_ulazi_obuka = podaci_ulazi[indeksi_obuka]
podaci_ulazi_test = podaci_ulazi[indeksi_test]

podaci_oznake_kodirane_obuka = podaci_oznake_kodirane[indeksi_obuka]
podaci_oznake_kodirane_test = podaci_oznake_kodirane[indeksi_test]


broj_epoha = 4
obim_grupe = 10

knm.compile(loss="categorical_crossentropy", optimizer="adam", 
            metrics=["accuracy"])

info = knm.fit(podaci_ulazi_obuka, podaci_oznake_kodirane_obuka,
               batch_size=obim_grupe, epochs=broj_epoha)

plt.figure()
plt.plot(info.epoch, info.history["accuracy"], "o:")
plt.ylim(0)
plt.xlabel("Epoha")
plt.ylabel("Tačnost")
plt.title("Performanse tokom obučavanja")

ev = knm.evaluate(podaci_ulazi_test, podaci_oznake_kodirane_test)
print("Performanse nad podacima za testiranje:")
for i in range(len(knm.metrics_names)):
    print("[{}] {:.3f}".format(knm.metrics_names[i], ev[i]))


broj_primer = 10

izbor_primer = random.sample(range(len(podaci_ulazi_test)), broj_primer)
primer_procena = knm.predict(podaci_ulazi_test[izbor_primer])
primer_oznake_procena = np.argmax(primer_procena, axis=1)
primer_oznake_stvarno = np.argmax(podaci_oznake_kodirane_test[izbor_primer], 
                                  axis=1)

kodovi_brojevi_vrste = {
    "anger": 0, "disgust": 1, "fear": 2, "happiness": 3, "neutral" : 4, "sadness" : 5, "surprise" : 6}

font_ispravno = {"color": "green"}
font_neispravno = {"color": "red"}

for i in range(len(izbor_primer)):
    indeks = izbor_primer[i]
    oznaka_stvarno = kodovi_brojevi_vrste[primer_oznake_stvarno[i]]
    oznaka_procena = kodovi_brojevi_vrste[primer_oznake_procena[i]]
    plt.figure()
    plt.imshow(podaci_ulazi_test[indeks])
    plt.xticks([])
    plt.yticks([])
    plt.title("Slika {} iz test skupa".format(indeks) + "\n" + 
              "Vrsta pirinča \"{}\"".format(oznaka_stvarno) + "\n" + 
              "Procena \"{}\"".format(oznaka_procena),
              fontdict=font_ispravno if oznaka_stvarno == oznaka_procena else 
              font_neispravno)
                  