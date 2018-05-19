# tf-HOG
The histogram of oriented gradients (HOG) implemented in tensorflow. Modifying the codes from [digamma-ai's tfcv](https://github.com/digamma-ai/tfcv), where I make the following changes:

1. Fix the bugs when using newer tensorflow version (>=1.4) and some other little bugs.
2. Modify the bin partition function.
3. Automatically padding images to valid size.
4. Complete the orientation part.
5. Masking the unwanted gradients of edge pixels.
6. Visualizing HOG image (borrow from skimage).
