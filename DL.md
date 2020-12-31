# ML strategy

## Practical:

- Scale the image down by factor 5 (1280x720 -> 256->144)
- Only analyze a half to a third of the image (256->144 -> 96 x 54 or 128x64.), centered around the last known ball position
- When last ball position is unknown, scan the whole image (6 to 9 tiles)
- Use MobileNet and transfer learning of last layer(s) to detect ball (and maybe blue and red players)
- Use Depth-Wise Convolution to save around 80% computations/parameters  (3x3 conv filter wise followed by 1x1 conv intra-filter)

## Wish: 

Use a Recurrent neural network, integrating ball movement as derivative. Output is a coordinate (x, y)

# Related Work/Links

https://arxiv.org/pdf/1912.05445  -> https://github.com/jac99/FootAndBall
https://github.com/DengPingFan/DAVSOD
https://paperswithcode.com/task/weakly-supervised-object-detection  <- the number one video-object-detection paper
https://www.sciencedirect.com/science/article/abs/pii/S123034021830146X
https://towardsdatascience.com/detecting-soccer-palyers-and-ball-retinantet-2ab5f997ab2

https://github.com/SteveMacenski/jetson_nano_detection_and_tracking <- mobilenet with 
