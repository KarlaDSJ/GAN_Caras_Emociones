from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, Reshape, Input, Concatenate, Embedding, BatchNormalization, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model
#from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import Adam #Para MAC
import numpy as np
import matplotlib.pyplot as plt


class GAN():
    def __init__(self, dataset):
        self.dataset = dataset
        self.noise_size = 48

        self.latent_dim = self.noise_size

        self.generator = self.define_generator(self.noise_size)
        self.discriminator = self.define_discriminator()
        self.gan_model = self.define_gan()


    def get_dataset_samples(self, n_samples):
        """Obtenemos n imagenes-reales de los datos"""
        images, labels = self.dataset
        ix = np.random.randint(0, images.shape[0], n_samples)
        X, labels = images[ix], labels[ix]
        y = np.ones((n_samples, 1))

        return [X, labels], y
 
    def generate_noise(self, n_samples, n_classes=7):
        """Creamos muestras de ruido"""
        #Creamos ruido
        x_input = np.random.randn(self.noise_size * self.noise_size * n_samples)
        #Ajustamos al tamaño de las imágenes reales
        z_input = x_input.reshape(n_samples, self.noise_size, self.noise_size)
        #Generamos etiquetas
        labels = np.random.randint(0, n_classes, n_samples)
        
        return [z_input, labels]


    def generate_fake_samples(self, n_samples): 
        #get the noise calling the function
        z_input, labels_input = self.generate_noise(self.latent_dim, n_samples)
        images = self.generator.predict([z_input, labels_input])
        #create class labes
        y = np.zeros((n_samples, 1))

        return [images, labels_input], y
    

    def plot_results(self, images, n_cols=None):
        '''visualizes fake images'''
        n_cols = n_cols or len(images)
        n_rows = (len(images) - 1) // n_cols + 1

        if images.shape[-1] == 1:
            images = np.squeeze(images, axis=-1)

        plt.figure(figsize=(n_cols, n_rows))
        
        for index, image in enumerate(images):
            plt.subplot(n_rows, n_cols, index + 1)
            plt.imshow(image, cmap = "binary")
            plt.axis("off")
        plt.show()

    # define the standalone generator model
    def define_generator(latent_dim, n_classes=10):
        # label input
        in_label = Input(shape=(1,))
        # embedding for categorical input
        li = Embedding(n_classes, 50)(in_label)
        # linear multiplication
        n_nodes = 7 * 7
        li = Dense(n_nodes)(li)
        # reshape to additional channel
        li = Reshape((7, 7, 1))(li)
        # image generator input
        in_lat = Input(shape=(latent_dim,))
        # foundation for 7x7 image
        n_nodes = 128 * 7 * 7
        gen = Dense(n_nodes)(in_lat)
        gen = LeakyReLU(alpha=0.2)(gen)
        gen = Reshape((7, 7, 128))(gen)
        # merge image gen and label input
        merge = Concatenate()([gen, li])
        # upsample to 14x14
        gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', 
                                            activation=LeakyReLU(alpha=0.2))(merge)
        gen = BatchNormalization()(gen)
        # upsample to 28x28
        gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', 
                                            activation=LeakyReLU(alpha=0.2))(gen)
        gen = BatchNormalization()(gen)
        # output
        out_layer = Conv2D(1, (7,7), activation='tanh', padding='same')(gen)
        # define model
        model = Model([in_lat, in_label], out_layer)
        return model
    
    def define_discriminator(in_shape=(48, 48), n_classes=7):
        in_label = Input(shape=(1,))
        li = Embedding(n_classes, 50)(in_label)
        n_nodes = in_shape[0] * in_shape[1]
        li= Dense(n_nodes)(li)
        li = Reshape((in_shape[0], in_shape[1], 1))(li)
        in_image = Input(shape=in_shape)
        merge = Concatenate()([in_image, li])
        #downsample
        fe= Conv2D(128, (3, 3), strides=(2, 2), padding='same', 
        activation = LeakyReLU(alpha=0.2))(merge)
        fe = Dropout(0.4)(fe)
        #downsample
        fe = Conv2D(128, (3, 3), strides=(2, 2), padding='same', 
        activation = LeakyReLU(alpha=0.2))(fe)
        fe = Dropout(0.4)(fe)
        fe = Flatten()(fe)
        out_layer = Dense(1, activation='sigmoid')(fe)

        model = Model([in_image, in_label], out_layer)
        opt = Adam(learning_rate=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        return model
    
            
    def define_gan(self):
        """Definimos una GAN condicional para generar imágenes"""
        #Hacemos que el discriminador no se entrene
        self.discriminator.trainable = False
        #Obtenemos ruido con su respectiva etiqueta del generador
        gen_noise, gen_label = self.generator.input
        #Obtenemos la predicción del generador
        gen_output = self.generator.output
        #Obtenmemos el resultado del discriminador con una imagen creada
        gan_output = self.discriminator([gen_output, gen_label])
        #Se lo pasamos al modelo de GAN
        model= Model([gen_noise, gen_label], gan_output)
        #Compilamos el modelo
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

        return model
    
    def train_gan(self, n_epochs=30, n_batch=512):
        steps = int(self.dataset[0].shape[0] / n_batch)
        half_batch = int(n_batch / 2)
        # manually enumerate epochs
        for e in range(n_epochs):
            # enumerate batches over the training set
            for s in range(steps):
                #TRAIN THE DISCRIMINATOR
                # get randomly selected 'real' samples
                [X_real, labels_real], y_real = self.get_dataset_samples(self.dataset, half_batch)
                # update discriminator model weights
                d_loss1, _ = self.discriminator.train_on_batch([X_real, labels_real], y_real)
                # generate 'fake' examples
                [X_fake, labels], y_fake = self.generate_fake_samples(self.generator, self.noise_size, half_batch)
                # update discriminator model weights
                d_loss2, _ = self.discriminator.train_on_batch([X_fake, labels], y_fake)

                #TRAIN THE GENERATOR
                # prepare points in latent space as input for the generator
                [z_input, labels_input] = self.generate_noise(self.noise_size, n_batch)
                # create inverted labels for the fake samples
                y_gan = np.ones((n_batch, 1))
                # update the generator via the discriminator's error
                g_loss = self.gan_model.train_on_batch([z_input, labels_input], y_gan)
                # summarize loss on this batch
                print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                (e+1, s+1, steps, d_loss1, d_loss2, g_loss))
                self.plot_results(X_fake, 8)  
            
        # save the generator model
        self.generator.save('cgan_generator.h5')