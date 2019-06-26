# Simple-Fully-Convolutional-Network

This is a simple fully convolutional network following the FCN achitecture specifications found on https://github.com/RyanCodes44/Machine-Learning-Exercise/blob/master/ML%20Exercise%20Instructions.pdf.

Code is modified from: https://github.com/RyanCodes44/Machine-Learning-Exercise, to create a FCN (There a CNN was created instead of an FCN)

To run the code use the specify the parameters in the run_code()

train_image / train_label = the path to the image and the label (.tif, in this repository)
n_dims = dimensions of the height/width of the patches
save_dir = the path to where you want to save it
weights = path to weights file (.h5)
train / valid = set true if you want to train / set true if you want to do cross-validation (default is 5-fold cross-validation)
importance = the importance of a the 2-classes (default is [1,5], thus getting the roof class wrong will penalize 5x more than getting the non-roof class wrong)

in the function step_decay()

initial_lrate = you can specify the initial learning rate
drop = how much you decay the learning rate
epochs_drop = when to decay the learning rate (i.e. every X epochs)



