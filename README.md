# GANs-study (Cornell University - MEng CS project)

**A study/report on the recent trend of Generative Adverserial Networks - their usages, advantages and disadvantages.**

We study the current state the generative adversarial networks (GANs) framework and try to understand the various processes behind them. We will have a look at the benefits of Gans, their drawbacks and possible future work. We mainly focus on the mnist, cifar10 and lung data set (from kaggle) that humans can relate to on a physical level. To this extent, we will look at some of the classic models (Vanilla GANs, Variational Auto Encoders) as well as some of the newer state-of-the-art models like WGans (Wasserstein GANs), DCGans (Deep convolutional GANs) and ACGans (Auxillary Classifier GANs). We have also provided some sample results in the form of images, mathematical estimations and computations regarding time and space complexities.

Generative models can often be difficult to train. Training a GAN requires finding a Nash equilibrium of a non-convex game with continuous, high dimensional parameters. GANs are typically trained using gradient descent techniques that are designed to find a low value of a cost function, rather than to find the Nash equilibrium of a game. When used to seek for a Nash equilibrium, these algorithms may fail to converge[1]. Thus we have compiled a study with some resources on GANS and their inner working (in a fairly abstract manner)

*Advisor - Professor Madeliene Udell*

*Project Team - Ishaan Jain (Me), Darpan Kalra*

### Credits
Some amazing codebases we used for this study

[Wiseodd - vanilla GANs, variations on Vanilla GANs and VAEs](https://github.com/wiseodd/generative-models)

[Carpedm - tensorflow implementation of DCGANs](https://github.com/carpedm20/DCGAN-tensorflow)

[Jacobil - keras implementation of DCGANs](https://github.com/jacobgil/keras-dcgan)

[Fchollet - keras implementation of ACGANs](https://github.com/fchollet/keras/tree/master/examples)
