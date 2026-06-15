# IMPACT Lab DNN Model Truncation Emulator

University of South Alabama, IMPACT and VLSI lab design.

This code base is a direct source code reference for the manuscript "Flexible Bit-Truncation Memory for Approximate
Applications on the Edge"

This codebase is the direct product of the work from Dr. William Oswald, and Dr. md. Bipul Hossain.

The structure of this codebase is as follows:
## notebook_examples
contains all source code for the DNN truncation models emulated and evaluated in the paper. 

## src
contains the bare-bones example of an Alexnet model using the emulator. This can be used as a reference when attempting to apply the emulator logic to new models.

-- Short Description --
We evaluated the DNN perfor-mance using a custom in-house hardware emulator, emu-lating the effects of TrunMEM, utilizing TensorFlow as the base DNN processing library. We have reported the performance, number of Floating Point Operations (FLOPs), and number of learning parameters for the baseline and lightweight (i.e., when attention-based filter pruning was solely applied on the models) VGG-16 and ResNet-56 models applied on the CIFAR-10 and CIFAR-100 datasets