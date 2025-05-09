# Import libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
import matplotlib.pyplot as plt

# Encoder class
class ConvEncoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(ConvEncoder, self).__init__()
        self.conv1 = layers.Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu')  # (28,28,1) -> (14,14,32)
        self.conv2 = layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu')  # (14,14,32) -> (7,7,64)
        self.flatten = layers.Flatten()  # (7,7,64) -> (3136,)
        self.dense = layers.Dense(128, activation='relu')  # Optional intermediate layer
        self.z_mean = layers.Dense(latent_dim)  # (128,) -> (latent_dim,)
        self.z_log_var = layers.Dense(latent_dim)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)

        # Reparameterization trick
        eps = tf.random.normal(shape=tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_log_var) * eps
        return z, z_mean, z_log_var

# Decoder class
class ConvDecoder(tf.keras.Model):
    def __init__(self):
        super(ConvDecoder, self).__init__()
        self.dense = layers.Dense(7 * 7 * 64, activation='relu')  # (latent_dim,) -> (3136,)
        self.reshape = layers.Reshape((7, 7, 64))  # (3136,) -> (7, 7, 64)

        self.deconv1 = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', activation='relu')  # (7,7,64) -> (14,14,64)
        self.deconv2 = layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', activation='relu')  # (14,14,64) -> (28,28,32)
        self.output_layer = layers.Conv2DTranspose(1, kernel_size=3, padding='same', activation='sigmoid')  # (28,28,32) -> (28,28,1)

    def call(self, z):
        x = self.dense(z)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        reconstructed = self.output_layer(x)
        return reconstructed


# VAE (Variational Autoencoder)
class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, beta=1.0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta  # Weight for KL divergence

    def call(self, x):
        z, z_mean, z_log_var = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed

    # Custom train step because we have custom loss
    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]  # ignore labels

        with tf.GradientTape() as tape:
            z, z_mean, z_log_var = self.encoder(data)
            reconstruction = self.decoder(z)

            # Reconstruction loss (BCE)
            reconstruction_loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(data, reconstruction), axis=[1, 2]) # BCE calculation reduces to (batch_size, 28, 28); use axis=1,2 for reduce_sum(...)

            # KL divergence loss
            kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)

            total_loss = tf.reduce_mean(reconstruction_loss + self.beta * kl_loss)

        # Backpropagation
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {
            "loss": total_loss,
            "reconstruction_loss": tf.reduce_mean(reconstruction_loss),
            "kl_loss": tf.reduce_mean(kl_loss),
        }

    # Custom test step (so we can log validation losses)
    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]  # ignore labels

        z, z_mean, z_log_var = self.encoder(data)
        reconstruction = self.decoder(z)

        reconstruction_loss = tf.reduce_sum(
            tf.keras.losses.binary_crossentropy(data, reconstruction), axis=[1, 2]
        )
        kl_loss = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
        )
        total_loss = tf.reduce_mean(reconstruction_loss + self.beta * kl_loss)

        return {
            "loss": total_loss,
            "reconstruction_loss": tf.reduce_mean(reconstruction_loss),
            "kl_loss": tf.reduce_mean(kl_loss),
        }


    
# Load data
def load_data():
    (x_train, _), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0 # Rescale
    x_train = x_train.reshape(-1, 28, 28, 1)

    # Return data
    return x_train

x_train = load_data()

# Train model
def train_model(latent_dim=10):

    # Initialize parts
    encoder = ConvEncoder(latent_dim)
    decoder = ConvDecoder()
    vae = VAE(encoder, decoder, beta=1.0) # Can play around with

    # Add an learning rate scheduler
    lr_schedule = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.00001
    )

    # Add early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)   

    # Compile with Adam optimizer
    vae.compile(optimizer=tf.keras.optimizers.Adam(), loss=None) # loss=None because we have custom loss in train_step()

    # Train
    vae.fit(x_train, epochs=50, batch_size=128, validation_split=0.2, callbacks=[lr_schedule, early_stopping])

    # Return model
    return vae

vae = train_model()

###----------------------------------------------------###

# Visualize results
def visualize_results(vae, x_train):
    num_samples = 10
    x_sample = x_train[:num_samples]

    # Reconstruct
    x_reconstructed = vae.predict(x_sample)

    # Plot
    plt.figure(figsize=(20,4))
    for i in range(num_samples):
        ax = plt.subplot(2, num_samples, i + 1)
        plt.imshow(x_sample[i].reshape(28, 28), cmap='gray')
        plt.title("Original")
        plt.axis('off')

        ax = plt.subplot(2, num_samples, i + 1 + num_samples)
        plt.imshow(x_reconstructed[i].reshape(28, 28), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("comparison.png")
    plt.show()

visualize_results(vae, x_train)