# ML strategy

## Practical:

- Scale the image down by factor 5 (1280x720 -> 256->144)
- Only analyze a half to a third of the image (256->144 -> 96 x 54), centered around the last known ball position
- Use MobileNet and transfer learning of last layer(s) to detect ball (and maybe blue and red players)

## Wish: 

Use a Recurrent neural network, integrating ball movement as integral

# Related Work/Links

https://arxiv.org/pdf/1912.05445  -> https://github.com/jac99/FootAndBall
https://github.com/DengPingFan/DAVSOD
https://paperswithcode.com/task/weakly-supervised-object-detection  <- the number one video-object-detection paper
https://www.sciencedirect.com/science/article/abs/pii/S123034021830146X
https://towardsdatascience.com/detecting-soccer-palyers-and-ball-retinantet-2ab5f997ab2

https://github.com/SteveMacenski/jetson_nano_detection_and_tracking <- mobilenet with 
