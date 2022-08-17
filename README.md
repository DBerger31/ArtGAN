**ArtGAN: Art Generation Using Conditional Generative Adversarial Networks**

A user can generate art on our web app either based on five or ten different art styles at https://art-gan.herokuapp.com/
We used a keras framework to create two similar GAN models that effectively learn to classify different styles of art and produce art of a similar style.
In addition to upscaling the images with Super Resolution from OpenCV.
The Five style generator produces 192x192 images and the Ten style generator produces 112x112.

The easiest method of running our model would be to visit our google colab and follow the steps displayed. <br />
<br />
https://colab.research.google.com/drive/1iX-hwK225jQ6f675gREH5UeihO8WlALJ?usp=sharing for Ten Styles <br />
<br />
https://colab.research.google.com/drive/1DV2Nx4ENeDwSNi-4ClcJrJa_7gVpi-RN?usp=sharing for Five Styles <br />
<br />
Otherwise, <br />

If you would like to test our model in the model folder we have various versions of the model that can be tested.
We have multiple iterations of our model saved in a folder called experimentalgans <br /> 
these gans arent functioning properly or produce less desirable images.

The two functioning models you would like to view are artgan.py which produces 28x28 images based on ten styles<br />
and 5gan.py that produces 64x64 based on five styles.

The requirements you will need is:
tensorflow-cpu <br />
matplotlib <br />
opencv-contrib-python-headless <br />
numpy <br />

Along with a link to download our images.npy that can be found here https://drive.google.com/drive/folders/1qhJqeMZ4mnuZ-sxyVM18zKXg5HJDp01O.

Team Members:<br />
Amy Tse<br />
Anthony Bi<br />
Daniel Berger<br />
