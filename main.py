import artGAN

if __name__ == '__main__':

    # Parameters
    img_shape = (28,28,1)
    n_labels = 10
    batch_size = 64
    latent_dim = 100


    disc = artGAN.create_discriminator(img_shape, n_labels)
    generator = artGAN.create_generator(latent_dim, n_labels)
    artgan = artGAN.create_gan(disc, generator)
    disc.summary()
    artgan.summary()
    

    # a, b = artGAN.load_mnist()

    # for epoch in range(100):
    #     for batch in range(128):
    #         [x_train, y_train], real = generate_real_samples(dataset, 64)
    #         loss, _ = disc.train_on_batch([x_train,y_train], real)