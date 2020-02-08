# tf-HOG
The histogram of oriented gradients (HOG) implemented in tensorflow. Based on the codes from [digamma-ai's tfcv](https://github.com/digamma-ai/tfcv), where I made the following changes:

1. Fixing the bugs when running with newer version tensorflow (>=1.4).
2. Completing the orientation function.
3. Modifying the bin partition function to suit the orientation function.
4. Automatically pad the input images.
5. Masking the unwanted gradients of edge pixels.
6. Visualizing HOG image (use the codes from skimage v0.14).
