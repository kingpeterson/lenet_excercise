## Tech challenge

### Goal

Build a C++ program that reads weights of the LeNet5 model and a input image. Then perform inference using the loaded wright and image. Classify the number within the image.

### Requirement

* Compile OpenCV 4 from source
* Must write the inference functionality from scratch. 
   * 3rd party matrix/tensor library disallowed
   * BLAS libraries disallowed
* Use STL where appropriate
* Use git submodules to handle external dependencies
* Must not cause memory access violations
* Clean instructions and CMakeLists.txt (or any build system you choose) for building the program

### Bonus
* Able to utilize multiple cores when inference. Without causing data race or corruption
   * Need not be faster then single threaded. As the overhead is way higher in such a small network
   * Multithreading helpers allowed. Ex: OpenMP, TBB, Parallel STL

### Specs

The weights of LeNet5 is stored in `lenet.json` under the current working directory. Your program should take an argument from the command line which points to a image.

Here is the model definition. The name (in comments) refers to the name in the json file.

```
# conv2d
model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=(32,32,1), padding='valid'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
# conv2d_1
model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
model.add(Flatten())
# dense
model.add(Dense(120, activation='tanh'))
# dense_1
model.add(Dense(84, activation='tanh'))
# dense_2
model.add(Dense(10, activation='sigmoid'))
```

Load the weight and image. Perform necessary calculations. Then print the resulting vector and clarification from the network. For example:

```
$ ./run_lenet images/1/img23.png # This is a 1

Loading LeNet model from lenet.json...
Loading image file images/1/23.png...
Inferencing...
Done

Resulting vector:
0       0.17786324
1       0.9957249
2       0.58365333
3       0.28783548
4       0.48930362
5       0.03686047
6       0.2652241
7       0.5161304
8       0.79401815
9       0.16537145

Classification: 1
```

