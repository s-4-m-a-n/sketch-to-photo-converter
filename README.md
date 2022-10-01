# sketch-to-photo-converter
It deep learning model created by implementing conditional GANs i.e pix2pix. It can convert portrait sketch to a portrait photograph.

# How to use
- create a new folder named model

```$ mkdir model```

- download the trained model from -> https://drive.google.com/file/d/1eTgm0eiE3Ggabdqvl-amcrvzDzt52WuP/view?usp=sharing and put it into the model/


# Dataset
- CUHK sketch face dataset -> http://mmlab.ie.cuhk.edu.hk/archive/facesketch.html 
- pix2pix https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/ 

# Input Image
![sketch](https://github.com/s-4-m-a-n/sketch-to-photo-converter/blob/master/test/inputs/9.jpg)

# result
![output](https://github.com/s-4-m-a-n/sketch-to-photo-converter/blob/master/test/outputs/gen_9.png)

# Conclusion
The model is not robust yet and can not give the acceptable result, that is because of lack in dataset.
