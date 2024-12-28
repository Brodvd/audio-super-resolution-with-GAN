import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Conv2D, Conv2DTranspose, Input, Flatten, Dense, Reshape, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.layers import Dropout
import numpy as np
import os
import soundfile as sf
import matplotlib.pyplot as plt

# Definizione del generatore con diffusione latente

def build_generator():
    input = Input(shape=(100,))  # Input latente
    x = Dense(256)(input)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)
    x = Dense(2048)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape((2048, 1))(x)
    x = Conv1D(64, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)
    x = Conv1D(128, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)
    x = Conv1D(256, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)
    x = Conv1D(512, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)
    x = Conv1D(256, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)
    x = Conv1D(128, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)
    x = Conv1D(64, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)
    output = Conv1D(1, kernel_size=3, strides=1, padding='same', activation='tanh')(x)
    return Model(input, output)

# Definizione del discriminatore
def build_discriminator():
    input = Input(shape=(2048, 1))  # Assicurati che questa forma sia corretta per i tuoi dati
    x = Reshape((2048, 1, 1))(input)  # Aggiungi una dimensione extra
    x = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(input, x)

# Funzione di addestramento
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
        # Modifica il learning rate dopo 50 epoche
        if epoch >= 25:
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

def plot_losses(d_losses, g_losses):
    epochs = range(1, len(d_losses) + 1)
    plt.plot(epochs, d_losses, 'bo', label='Discriminator loss')
    plt.plot(epochs, g_losses, 'b', label='Generator loss')
    plt.title('Loss durante l\'addestramento')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Funzione per caricare i dati audio
def load_audio_data(damaged_folder, original_folder):
    damaged_files = [os.path.join(damaged_folder, f) for f in os.listdir(damaged_folder) if f.endswith('.wav')]
    original_files = [os.path.join(original_folder, f) for f in os.listdir(original_folder) if f.endswith('.wav')]
    
    damaged_data = []
    original_data = []
    
    for file in damaged_files:
        data, _ = sf.read(file)
        damaged_data.append(data)
        print(f"Caricato file low: {file}, Lunghezza: {len(data)}, Forma: {np.shape(data)}")
    
    for file in original_files:
        data, _ = sf.read(file)
        original_data.append(data)
        print(f"Caricato file originale: {file}, Lunghezza: {len(data)}, Forma: {np.shape(data)}")
    
    damaged_data = np.array(damaged_data)
    original_data = np.array(original_data)

    if damaged_data.ndim != original_data.ndim:
        print(f"Errore: Le dimensioni non corrispondono. damaged_data.ndim = {damaged_data.ndim}, original_data.ndim = {original_data.ndim}")
    else:
        print("File di Training ok")
    
    return damaged_data, original_data

def slice_audio_data(data, segment_length):
    sliced_data = []
    for audio in data:
        for i in range(0, len(audio) - segment_length + 1, segment_length):
            sliced_data.append(audio[i:i + segment_length])
    return np.array(sliced_data)

# Caricamento dei dati e addestramento
damaged_folder = './Training-Data/GAN_latent_diffusion/Damaged'
original_folder = './Training-Data/GAN_latent_diffusion/High-Resolution'
damaged_data, original_data = load_audio_data(damaged_folder, original_folder)
segment_length = 2048 # fissa
damaged_data_sliced = slice_audio_data(damaged_data, segment_length)
original_data_sliced = slice_audio_data(original_data, segment_length)

# Unisci i dati danneggiati e originali per l'addestramento
data = np.concatenate((damaged_data, original_data), axis=0)
print("Concatenazione riuscita.")

generator = build_generator()
discriminator = build_discriminator()
train(generator, discriminator, epochs=50, batch_size=32, data=data)
