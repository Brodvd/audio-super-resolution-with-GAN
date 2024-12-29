import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Input, Flatten, Dense, Reshape, LeakyReLU, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
import numpy as np
import os
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.fft import fft
from python_speech_features import mfcc

# Funzione per estrarre le caratteristiche audio
def extract_features(audio, sample_rate):
    fft_spectrum = fft(audio)
    magnitude = np.abs(fft_spectrum)
    phase = np.angle(fft_spectrum)
    mfcc_features = mfcc(audio, samplerate=sample_rate, numcep=13)
    return magnitude, phase, mfcc_features

# Modifica della funzione di caricamento dei dati per includere le caratteristiche
def load_audio_data(damaged_folder, original_folder):
    damaged_files = [os.path.join(damaged_folder, f) for f in os.listdir(damaged_folder) if f.endswith('.wav')]
    original_files = [os.path.join(original_folder, f) for f in os.listdir(original_folder) if f.endswith('.wav')]
    
    damaged_data = []
    original_data = []
    
    for file in damaged_files:
        data, sample_rate = sf.read(file)
        magnitude, phase, mfcc_features = extract_features(data, sample_rate)
        damaged_data.append((magnitude, phase, mfcc_features))
        print(f"Caricato file low: {file}, Lunghezza: {len(data)}, Forma: {np.shape(data)}")
    
    for file in original_files:
        data, sample_rate = sf.read(file)
        magnitude, phase, mfcc_features = extract_features(data, sample_rate)
        original_data.append((magnitude, phase, mfcc_features))
        print(f"Caricato file originale: {file}, Lunghezza: {len(data)}, Forma: {np.shape(data)}")
    
    return np.array(damaged_data, dtype=object), np.array(original_data, dtype=object)

# Definizione della funzione slice_audio_data
def slice_audio_data(data, segment_length):
    sliced_data = []
    for magnitude, phase, mfcc_features in data:
        num_segments = len(magnitude) // segment_length
        for i in range(num_segments):
            segment = magnitude[i * segment_length:(i + 1) * segment_length]
            sliced_data.append(segment)
    return np.array(sliced_data)

# Funzione per plottare le perdite
def plot_losses(d_losses, g_losses):
    epochs = range(1, len(d_losses) + 1)
    plt.plot(epochs, d_losses, 'bo', label='Discriminator loss')
    plt.plot(epochs, g_losses, 'b', label='Generator loss')
    plt.title('Loss durante l\'addestramento')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Definizione delle funzioni build_generator e build_discriminator
def build_generator():
    model = tf.keras.Sequential()
    model.add(Dense(256, input_dim=100))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(2048))
    model.add(Reshape((2048, 1)))
    return model

def build_discriminator():
    model = tf.keras.Sequential()
    model.add(Conv1D(64, kernel_size=5, strides=2, input_shape=(2048, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv1D(128, kernel_size=5, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

# Modifica della funzione di addestramento per includere le caratteristiche
def train(generator, discriminator, epochs, batch_size, data):
    initial_lr = 0.0001
    optimizer = Adam(learning_rate=initial_lr, beta_1=0.5)
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    input = Input(shape=(100,))
    generated_audio = generator(input)
    discriminator.trainable = False
    validity = discriminator(generated_audio)
    
    combined = Model(input, validity)
    combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    d_losses, g_losses = [], []

    for epoch in range(epochs):
        if epoch >= 10:
            new_lr = initial_lr * (0.99 ** (epoch - 50))
            tf.keras.backend.set_value(optimizer.learning_rate, new_lr)
        
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_audio = data[idx]
        
        real_audio = real_audio[:, :2048].reshape((batch_size, 2048, 1))
        
        noise = np.random.normal(0, 1, (batch_size, 100))
        generated_audio = generator.predict(noise)
        
        generated_audio = generated_audio.reshape((batch_size, 2048, 1))
        
        d_loss_real = discriminator.train_on_batch(real_audio, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(generated_audio, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        g_loss = combined.train_on_batch(noise, np.ones((batch_size, 1)))

        d_losses.append(d_loss[0])
        g_losses.append(g_loss)
        
        print(f"{epoch}/{epochs} [D loss: {d_loss[0]} | D accuracy: {d_loss[1]}] [G loss: {g_loss}]")

    plot_losses(d_losses, g_losses)

    generator.save('./Models/latentGAN_generator_model_50_epoch_1.keras')
    discriminator.save('./Models/latentGAN_discriminator_model_50_epoch_1.keras')

# Caricamento dei dati e addestramento
damaged_folder = './Training-Data/GAN_latent_diffusion/Damaged'
original_folder = './Training-Data/GAN_latent_diffusion/High-Resolution'
damaged_data, original_data = load_audio_data(damaged_folder, original_folder)
segment_length = 2048
damaged_data_sliced = slice_audio_data(damaged_data, segment_length)
original_data_sliced = slice_audio_data(original_data, segment_length)

data = np.concatenate((damaged_data_sliced, original_data_sliced), axis=0)
print("Concatenazione riuscita.")

generator = build_generator()
discriminator = build_discriminator()
train(generator, discriminator, epochs=50, batch_size=32, data=data)
