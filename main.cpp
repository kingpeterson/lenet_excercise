#include <iostream>
#include<string>
#include <leNet.h>

void buildModel() {
	std::vector<Layer> layers;

	Layer input_layer_1;
	input_layer_1.type = 'i'; input_layer_1.iChannel = 1; input_layer_1.iSizePic[0] = 32; input_layer_1.iSizePic[1] = 32;
	layers.push_back(input_layer_1);

	Layer convolutional_layer_2;
	convolutional_layer_2.type = 'c'; convolutional_layer_2.iChannel = 2; convolutional_layer_2.iSizeKer = 5;
	layers.push_back(convolutional_layer_2);

	Layer subsampling_layer_3;
	subsampling_layer_3.type = 's'; subsampling_layer_3.iSample = 2;
	layers.push_back(subsampling_layer_3);

	Layer convolutional_layer_4;
	convolutional_layer_4.type = 'c'; convolutional_layer_4.iChannel = 4; convolutional_layer_4.iSizeKer = 5;
	layers.push_back(convolutional_layer_4);

	Layer subsampling_layer_5;
	subsampling_layer_5.type = 's'; subsampling_layer_5.iSample = 2;
	layers.push_back(subsampling_layer_5);

	Layer fully_connected_layer_6;
	fully_connected_layer_6.type = 'f'; fully_connected_layer_6.iChannel = 120;
	layers.push_back(fully_connected_layer_6);

	Layer fully_connected_layer_7;
	fully_connected_layer_7.type = 'f'; fully_connected_layer_7.iChannel = 84;
	layers.push_back(fully_connected_layer_7);

	Layer fully_connected_layer_8;
	fully_connected_layer_8.type = 'f'; fully_connected_layer_8.iChannel = 10;
	layers.push_back(fully_connected_layer_8);
}

int main(int argc, char* argv[]) {
    // if (argc != 2) { 
    //     // We expect 2 arguments: the program name and the video path
    //     std::cerr << "Usage: " << argv[0] << "input video" << std::endl;
    //     return 1;
    // }
    std::cout << "Hello World!" << std::endl;
    return 1;
}