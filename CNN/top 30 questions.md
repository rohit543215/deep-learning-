Here are 30 CNN interview/coding questions, ordered from easy to hard:

**Easy — Conceptual Foundations**
1. What is a Convolutional Neural Network, and how is it different from a regular fully connected neural network?
2. What is a kernel/filter in a CNN, and what does it do?
3. What is the purpose of pooling layers? What's the difference between max pooling and average pooling?
4. Why do CNNs work well for image data specifically?
5. What is stride, and how does it affect the output size of a convolution?
6. What is padding, and what's the difference between "valid" and "same" padding?
7. Why is ReLU commonly used as the activation function in CNNs instead of sigmoid/tanh?
8. What is a feature map?
9. What does "flattening" mean in the context of a CNN architecture?
10. What is the role of the fully connected (dense) layer at the end of a CNN?

**Medium — Mechanics & Math**
11. Derive the formula for the output size of a convolution layer given input size, kernel size, stride, and padding.
12. How many parameters does a convolutional layer have, given kernel size, number of input channels, and number of output channels? Show a worked example.
13. Explain parameter sharing in CNNs. Why does it reduce overfitting compared to fully connected layers?
14. What is the receptive field of a neuron in a CNN, and how does it grow across layers?
15. What is a 1x1 convolution used for, and why would you use it?
16. Explain how backpropagation works through a convolutional layer.
17. Explain how backpropagation works through a max pooling layer (which neurons actually receive gradient?).
18. What is batch normalization, and why is it used in CNNs? Where is it typically placed relative to the activation function?
19. What is dropout, and how does it help prevent overfitting in CNNs?
20. What is data augmentation, and name at least 4 techniques used for image data.

**Hard — Architecture & Applied Reasoning**
21. Explain the vanishing gradient problem in deep CNNs, and how ResNet's skip connections address it.
22. Compare VGG, ResNet, and Inception architectures — what was the key innovation of each?
23. What is transfer learning? Explain the difference between feature extraction and fine-tuning with a pretrained CNN.
24. Why do deeper CNNs generally perform better, and what are the tradeoffs (compute, overfitting, degradation problem)?
25. Explain dilated (atrous) convolutions and a use case where they're preferred over standard convolutions.
26. What is the difference between a CNN used for classification vs one used for object detection (e.g. how does something like YOLO differ structurally)?
27. Explain how a CNN can be adapted for semantic segmentation (e.g. what changes are needed vs a classification CNN — think transposed convolutions/upsampling).
28. What is depthwise separable convolution, and why is it used in mobile-efficient architectures like MobileNet? (Compare parameter/computation cost to standard convolution.)
29. Given a CNN that's overfitting on a small dataset, walk through your full debugging/mitigation strategy (data, architecture, regularization, training approach).
30. Design a CNN architecture from scratch for a given task (e.g. classify 128x128 medical images into 5 classes) — walk through your layer choices, kernel sizes, depth, and reasoning at each step.

Want me to go through any of these Socratically — for example working through the output-size formula derivation (#11) or the parameter-sharing explanation (#13) the way we've been doing with your LeetCode problems?