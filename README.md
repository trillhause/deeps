
# deeps

_Solving [Comma.ai's 2017 intern challenge](https://twitter.com/comma_ai/status/849131721572327424?lang=en)_

This approach uses optical flow analysis and Convolutional neural networks to estimate speed of a car from dashcam footage. We are provided with a training video (train.mp4) and the speed data at each frame  (train.txt).

- **train.mp4:** 20 fps video with 20400 frames, each frame is 640(w) x 840(h) x 3 (RGB)
- **train.txt:** Text file with speed data at each frame
- **test.mp4:** 20 fps video with 20400 frames, each frame is 640(w) x 840(h) x 3 (RGB)

_[Here](https://github.com/millingab/deeps/blob/master/Full%20Article.md) is a full article explaining the process in detail_

## Results

![result](https://media.giphy.com/media/2Kc6BtTNwRU6Q/200w_d.gif)

_This gif is speed up 3X. Checkout the [full video here](https://youtu.be/LUTn_I52SMQ)_

I got a Mean Square Error(MSE) of 3.48 on the training set, an MSE of 1.4 on the validation data, and an MSE of 3.5 on the test set. Even though the validation curve has some noise, the overall trend suggests that we are not overfitting the training set. 

![png](images/output_44_1.png)

This was a iterative process like all Deep Learning projects are. This is a list of things I tuned / experimented with:
- **Inputs:** rgb difference, hsv difference, sparse optical flow, dense optical flow
- **Hyperparameters:** epoch, batch_size, steps_per_epoch
- **Optimizers:** batch gradient descent, rms prop, momentum, adam optimizer

## Training parameters

**Input** dense optical flow between two image frames

**Optimizer:** Adam

**loss:** MSE

**epoch:** 25

**Samples per epoch:** 400

**Batches per sample** 32 images, 16 optical flow rgb_diffs

## What I learned from this project

1. Optical flow is a powerful tool to measure motion of objects in images.
2. Adam Optimizer converges faster because it uses momentum and RMS prop to modify the gradient during descent
3. Python generators can be used to yield batches of data. They are very efficient for image processing because most machines do not have enough memory to hold all the images when batching.
4. OpenCV is great for video and image processing.
5. Training on local machine is horrible for training algorithms. Always use cloud services. This also allows you to train multiple models in parallel. (Mo money you spend, mo problems you solve)
