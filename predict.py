import scipy
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding1D, Add
from keras.layers import PReLU, LeakyReLU, UpSampling1D, Conv1D
from keras.applications import VGG19
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import keras.backend as K
import joblib
import numpy.random as random
import scipy.signal as signal
import soundfile as sf
from load_data import LoadData  # Importa il modulo corretto

LRSHAPE = 129
HRSHAPE = 128

def build_generator():
    audio_lr = Input(shape=(32, LRSHAPE))
    c1 = Conv1D(filters=256, kernel_size=7, strides=2, padding='same')(audio_lr)
    b1 = BatchNormalization()(c1)
    a1 = LeakyReLU(alpha=0.2)(b1)

    c2 = Conv1D(filters=512, kernel_size=5, strides=2, padding='same')(a1)
    b2 = BatchNormalization()(c2)
    a2 = LeakyReLU(alpha=0.2)(b2)

    c3 = Conv1D(filters=512, kernel_size=3, strides=2, padding='same')(a2)
    b3 = BatchNormalization()(c3)
    a3 = LeakyReLU(alpha=0.2)(b3)

    c4 = Conv1D(filters=1024, kernel_size=3, strides=2, padding='same')(a3)
    b4 = BatchNormalization()(c4)
    a4 = LeakyReLU(alpha=0.2)(b4)

    c5 = Conv1D(filters=512, kernel_size=3, strides=1, padding='same')(a4)
    u5 = UpSampling1D(size=2)(c5)
    b5 = BatchNormalization()(u5)
    a5 = LeakyReLU(alpha=0.2)(b5)
    A5 = Add()([a5, a3])

    c6 = Conv1D(filters=512, kernel_size=5, strides=1, padding='same')(A5)
    u6 = UpSampling1D(size=2)(c6)
    b6 = BatchNormalization()(u6)
    a6 = LeakyReLU(alpha=0.2)(b6)
    A6 = Add()([a6, a2])

    c7 = Conv1D(filters=256, kernel_size=7, strides=1, padding='same')(A6)
    u7 = UpSampling1D(size=2)(c7)
    b7 = BatchNormalization()(u7)
    a7 = LeakyReLU(alpha=0.2)(b7)
    A7 = Add()([a7, a1])

    c8 = Conv1D(filters=HRSHAPE, kernel_size=7, strides=1, padding='same')(A7)
    u8 = UpSampling1D(size=2)(c8)
    b8 = BatchNormalization()(u8)
    a8 = LeakyReLU(alpha=0.2)(b8)

    c9 = Conv1D(filters=HRSHAPE, kernel_size=9, strides=1, padding='same')(a8)
    return Model(audio_lr, c9)

def reconstruct_low_high(X_low, X_high, X_low_phase):
    X_log_magnitude = np.hstack([X_low, X_high])
    flipped = X_log_magnitude[:, ::-1]  # Utilizza la simmetria
    X_log_magnitude = np.hstack([X_log_magnitude, flipped])

    flipped = -1 * X_low_phase[:, ::-1]
    X_phase = np.hstack([X_low_phase, flipped])
    flipped = -1 * X_phase[:, ::-1]
    X_phase = np.hstack([X_phase, flipped])
    _, n = X_log_magnitude.shape
    X_phase = X_phase[:, :n]
    return X_log_magnitude, X_phase

def reconstruct_low_high2(X_low, X_high, X_low_phase=None, X_high_phase=None):
    if X_high.shape[1] == 129:
        X_high = X_high[:, 1:]

    X_log_magnitude = np.hstack([X_low, X_high])
    flipped = X_log_magnitude[:, 1:-1][:, ::-1]  # Utilizza la simmetria
    X_log_magnitude = np.hstack([X_log_magnitude, flipped])

    if X_low_phase is not None and X_high_phase is not None:
        X_phase = np.hstack([X_low_phase, X_high_phase])
        flipped_phase = -1 * X_phase[:, 1:-1][:, ::-1]
        X_phase = np.hstack([X_phase, flipped_phase])
        return X_log_magnitude, X_phase
    else:
        return X_log_magnitude

def stft(x, **params):
    f, t, zxx = signal.stft(x, **params)
    return f, t, zxx

def stft_specgram(x, picname=None, **params):
    f, t, zxx = stft(x, **params)
    plt.pcolormesh(t, f, np.abs(zxx), cmap='gnuplot')
    plt.colorbar()
    plt.ylabel('Frequency (kHz)')
    plt.xlabel('Time (s)')
    plt.tight_layout()
    if picname:
        plt.title(picname)
    plt.show()
    return t, f, zxx

# Carica il file audio
data_test_path = os.path.abspath('./Audio/Input/guitar-01.wav')
waveform, sample_rate = sf.read(data_test_path)
print(f"Forma del waveform: {waveform.shape}, Frequenza di campionamento: {sample_rate}")

# Costruisci il modello del generatore
model = build_generator()
model.load_weights(os.path.abspath("Models/gen_epoch.h5"))

# Carica e istanzia la classe LoadData
data_loader = LoadData()

idx = 0  # Definisci idx prima di utilizzarlo

# Usa la waveform direttamente
if idx % 50 == 0:
    stft_specgram(waveform, "origin")
elif idx % 50 == 9:
    stft_specgram(waveform, "origin2")

# Esegui STFT utilizzando scipy.signal
_, _, X = stft(waveform, nperseg=512, noverlap=256)
X_log_magnitude, X_phase = data_loader.decompose_stft(X)  # Usa l'istanza data_loader per chiamare il metodo
X_low, X_high, X_low_phase, X_high_phase = data_loader.extract_low_high(X_log_magnitude, X_phase)

# Calcola la lunghezza corretta per X_low
correct_length = 129  # Questo è basato sulla forma prevista dal modello

# Troncatura o padding per ottenere la lunghezza corretta
if X_low.shape[1] > correct_length:
    X_low = X_low[:, :correct_length]  # Troncatura
else:
    padding = correct_length - X_low.shape[1]
    X_low = np.pad(X_low, ((0, 0), (0, padding)), mode='constant', constant_values=0)  # Padding

# Assicurati che X_low sia della forma (m // 32, 32, correct_length)
m = X_low.shape[0]
X_low = X_low[0:m // 32 * 32]
X_low = X_low.reshape(m // 32, 32, correct_length)

# Stampa la forma di X_low per verificare
print(X_low.shape)

# Prosegui con la previsione
Y_hat = model.predict(X_low)
X_low = X_low.reshape(m // 32 * 32, correct_length)
p, _, q = Y_hat.shape
Y_hat = Y_hat.reshape(p * _, q)
n_samples = (m // 32 * 32 + 1) * 256

Xhat_log_magnitude, Xhat_phase = reconstruct_low_high2(X_low, Y_hat, X_low_phase, X_high_phase)
Xhat_log_magnitude = Xhat_log_magnitude[:, :512]
Xhat_phase = Xhat_phase[:, :512]
Xhat = data_loader.compose_stft(Xhat_log_magnitude, Xhat_phase)  # Usa l'istanza data_loader per chiamare il metodo
xhat = data_loader.istft(Xhat, n_samples)  # Usa l'istanza data_loader per chiamare il metodo

if idx % 50 == 0:
    stft_specgram(xhat, "GAN")
elif idx % 50 == 9:
    stft_specgram(xhat, "GAN2")
sf.write(os.path.abspath('Audio/Output' + str(idx) + '.wav'), xhat, 48000)
idx += 1

a, b = Y_hat.shape
lsd, snr = [], []
for i in range(a):
    x1 = np.sqrt(np.average((X_high[i] - Y_hat[i]) ** 2))
    x2 = np.sum((X_high[i] - Y_hat[i]) ** 2)
    x3 = np.sum(X_high[i] ** 2)
    lsd.append(x1)
    snr.append(10 * np.log10(x3 / x2))
LSD.append(np.average(lsd))
SNR.append(np.average(snr))

print("SNR={0}, LSD={1}".format(np.average(SNR), np.average(LSD)))
