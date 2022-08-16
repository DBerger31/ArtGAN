**ArtGAN: Art Generation Using Conditional Generative Adversarial Networks**

A user can generate art on our web app either based on five or ten different art styles at https://art-gan.herokuapp.com/
We used a keras framework to create two similar GAN models that effectively learn to classify different styles of art and produce art of a similar style.
In addition to upscaling the images with Super Resolution from OpenCV.
The Five style generator produces 192x192 images and the Ten style generator produces 112x112.

If you would like to test our model in the model folder we have various versions of the model that can be tested.
The most recent functioning model that should be tested is the file named artgan.py

The requirements you will need is:

tensorflow-cpu <br />
matplotlib <br />
opencv-contrib-python-headless <br />
numpy <br />

Along with a link to download our images.npy that can be found here _______________.

Team Members:<br />
Amy Tse<br />
Anthony Bi<br />
Daniel Berger<br />
