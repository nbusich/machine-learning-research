AFFILIATIONS:
This research project is being conducted in collaboration with Professor Chieh Wu of Northeastern University, Ashwin Satish, Priyanka Adhikari,  Samantha Cheung, and Sophia Yang,
who are students at Northeastern University. 

INTRODUCTION:
We are working with a biology research lab at Northeastern that is studying Lyme Disease. In their research, they generate millions of images of bacteria. The images are then 
manually sorted into "Good" or "Bad" categories, referring to the quality of the image. This inefficient manual process obviously takes up huge amounts of time. We set out to 
automate this process using computer vision techniques.

METHODS:
I started by learning about kernel methods such as kernel Ridge Regression, kernel SVM, MMD as a measure of distribution similarity, and Hilbert Schmidt Independency Criterion (HSIC).
I also researched methods of distilling information from images into a representation vector. I implemented a Convolutional Autoencoder to condense the important features from
the input images into representation vectors. I am currently implementing the simCLR framework to generate representation vectors that use less labels to acheive similar results.

RESULTS:
With the representation vectors, I have implemented kernel SVM to classify the corresponding images as good or bad. The convolutional autoencoder combined with kernel SVM
trained on a preliminary dataset of about 60 images and corresponding labels resulted in in a test accuracy of 89%.

NEXT STEPS:
Lets look back to the original problem of reducing/eliminating manual image classification while still generating enough data for research. 
The key word is **enough** data for research. There are millions of images being generated, while 1000 images would probably suffice for the research being conducted. That means
we don't actually have to acheive 100% accuracy in order to solve the problem. We only need to classify bad images with 100% accuracy, to avoid bad data skewing results.
That means we can misclassify good images as bad, so long as we keep a reasonable amount (enough) images for the purposes of the research. Therefore, with a classification
algorithm other than kSVM (because it does not produce probabilities) we will keep only images where the model produces a high threshold of certainty for good or bad. The next
steps are therefore implementing a model other than kSVM using the representation vectors, and thresholding the probability necessary to be classified as good. I will implement
MMD/KDE to find the probability distribution of each class p_g(x), p_b(x). With a new image to be classified, a, the probability p(a| model = p_g) and p(a| model = p_b) 
will be obtained by plugging a into p_g(x) and p_b(x). Using that probability and Bayesian decision theory, we can obtain p(model = p_g | a) and p(model = p_b | a). Those are 
the probabilities that the image is in the good or bad class.
