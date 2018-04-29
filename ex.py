#!/usr/bin/python3

from keras.layers import Input, Dense, Embedding, LSTM, concatenate, Lambda, RepeatVector
import keras.backend as kbe
from keras.models import Sequential, Model
from keras.optimizers import Adam
import tensorflow as tf

import sys

import numpy as np

class GAN():

    def __init__(self):
        self.sentence_length = 15
        self.vocabulary_size = 15000
        self.embedding_vector_length = 300
        self.latent_dimension = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', 
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes s1 as input and generated s2
        s1 = Input(shape=(self.sentence_length,))
        s2 = self.generator(s1)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator([s1,s2])
#
#        # The combined model  (stacked generator and discriminator) takes
#        # noise as input => generates images => determines validity 
#        self.combined = Model(z, valid)
#        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        s1 = Input(shape=(self.sentence_length,))

        embed = Embedding(output_dim=self.embedding_vector_length,
                input_dim=self.vocabulary_size,
                input_length=self.sentence_length)(s1)
        x = concatenate([embed,embed],axis=1)
        x = LSTM(self.latent_dimension,return_sequences=True)(x)
        x = LSTM(self.embedding_vector_length,return_sequences=True)(x)
        x = Lambda(lambda s: s[:,15:,:])(x)
        x = Dense(self.vocabulary_size,activation='softmax')(x)
        # need some way to convert to one hot vector
        # x = Lambda(lambda s:kbe.max(s,axis=-1))(x)

        model = Model(s1,x)
        model.summary()
        
        return model


    def build_discriminator(self):
        
        s1 = Input(shape=(self.sentence_length,), name='s1')
        s2 = Input(shape=(self.sentence_length,), name='s2')

        embed = Embedding(output_dim=self.embedding_vector_length,
                input_dim=self.vocabulary_size,
                input_length=self.sentence_length)
        x = concatenate([embed(s1),embed(s2)])
        x = LSTM(self.latent_dimension)(x)
        x = Dense(1,activation='sigmoid')(x)

        model = Model([s1,s2],x)
        model.summary()
        
        return model

    def load_data(self):

        X_train = np.array()
        y_train = np.array()
        X_test = np.array()
        y_test = np.array()

        return (X_train, y_train), (X_test, y_test)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        (X_train, y_train), (X_test, y_test) = self.load_data()

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

#            noise = np.random.normal(0, 1, (half_batch, 100))
#
#            # Generate a half batch of new images
#            gen_imgs = self.generator.predict(noise)
#
#            # Train the discriminator
#            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
#            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
#            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
#
#
#            # ---------------------
#            #  Train Generator
#            # ---------------------
#
#            noise = np.random.normal(0, 1, (batch_size, 100))
#
#            # The generator wants the discriminator to label the generated samples
#            # as valid (ones)
#            valid_y = np.array([1] * batch_size)
#
#            # Train the generator
#            g_loss = self.combined.train_on_batch(noise, valid_y)
#
#            # Plot the progress
#            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
#
#            # If at save interval => save generated image samples
#            if epoch % save_interval == 0:
#                self.save_imgs(epoch)

if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=30000, batch_size=32)
