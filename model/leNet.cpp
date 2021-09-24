#include <leNet.h>
// #include <maths.h>
#include <time.h>
#include <iostream>


using namespace std;


// 初始化CNN类
CNN::CNN(const vector<Layer> &layers, float alpha, float eta, int batchsize, int epochs, activation_function_type activ_func_type, down_sample_type down_samp_type)
	:_layers(layers), 
	_alpha(alpha), 
	_eta(eta), 
	_batchsize(batchsize), 
	_epochs(epochs), 
	_activation_func_type(activ_func_type), 
	_down_sample_type(down_samp_type)
{
	// 依据网络结构设置CNN.layers, 初始化一个CNN网络
	init();

	_ERR.assign(_epochs, 0);// 将历次迭代的均方误差初始化为0
	_err = 0;// 将当前轮的当前批次的均方误差初始化为0

	cout << "CNN has initialised!" << endl;
}




// 依据网络结构设置CNN.layers, 初始化一个CNN网络
void CNN::init()
{
	// CNN网络层数
	int n = _layers.size();

	cout << "input layer " << 1 << " has initialised!" << endl;

	// 对CNN网络层数做循环
	for (int L = 1; L < n; L++)// 注意:这里实际上可以从第2层开始，所以L初始值是1不是0
	{
		// ======================================================================
		// 以下代码仅对第2, 4层(卷积层)有效

		if (_layers.at(L).type == 'c')
		{
			// 由前一层图像行列数,和本层卷积核尺度,计算本层图像行列数,二维向量
			_layers.at(L).iSizePic[0] = _layers.at(L - 1).iSizePic[0] - _layers.at(L).iSizeKer + 1;
			_layers.at(L).iSizePic[1] = _layers.at(L - 1).iSizePic[1] - _layers.at(L).iSizeKer + 1;

			// "前一层任意一个通道", 对应"本层所有通道"卷积核权值W(可训练参数)个数, 不包括加性偏置
			int fan_out = _layers.at(L).iChannel * pow(_layers.at(L).iSizeKer, 2);// 比如 4*5^2=4*25

			// "前一层所有通道", 对应"本层任意一个通道"卷积核权值W(可训练参数)个数, 不包括加性偏置
			int fan_in = _layers.at(L-1).iChannel * pow(_layers.at(L).iSizeKer, 2);

			// 对当前层的Ker和Ker_delta初始化维数，并赋初值
			_layers.at(L).Ker.resize(_layers.at(L - 1).iChannel);
			_layers.at(L).Ker_delta.resize(_layers.at(L - 1).iChannel);

			_layers.at(L).Ker_grad.resize(_layers.at(L - 1).iChannel);

			// 对本层输入通道数做循环
			for (int I = 0; I < _layers.at(L - 1).iChannel; I++)
			{
				_layers.at(L).Ker.at(I).resize(_layers.at(L).iChannel);
				_layers.at(L).Ker_delta.at(I).resize(_layers.at(L).iChannel);

				_layers.at(L).Ker_grad.at(I).resize(_layers.at(L).iChannel);// 这里仅仅初始化大小，不初始化值。值会在反向传播中给出

				// 对本层输出通道数做循环
				for (int J = 0; J < _layers.at(L).iChannel; J++)
				{
					double maximum = (double)sqrt(6.0f / (fan_in + fan_out));

					// "前一层所有通道",对"本层所有通道",层对层的全连接,卷积核权值W,进行均匀分布初始化,范围为:[-1,1]*sqrt(6/(fan_in+fan_out))
					_layers.at(L).Ker[I][J].set_rand(_layers.at(L).iSizeKer, _layers.at(L).iSizeKer, -maximum, maximum);
					_layers.at(L).Ker_delta[I][J].set_zero_same_size_as(_layers.at(L).Ker[I][J]);
				}
			}

			// 对本层输出通道加性偏置进行0值初始化
			_layers.at(L).B.assign(_layers.at(L).iChannel, 0);
			_layers.at(L).B_delta.assign(_layers.at(L).iChannel, 0);

			_layers.at(L).Delta.resize(_layers.at(L).iChannel);// 敏感度图的大小初始化，但不赋值

			cout << "convolutional layer " << L + 1 << " has initialised!" << endl;
		}

		// ======================================================================
		// 以下代码对第3,5层(下采样层)有效

		if (_layers.at(L).type == 's')
		{
			_layers.at(L).iSizePic[0] = floor((_layers.at(L - 1).iSizePic[0] + _layers.at(L).iSample - 1) / _layers.at(L).iSample);
			_layers.at(L).iSizePic[1] = floor((_layers.at(L - 1).iSizePic[1] + _layers.at(L).iSample - 1) / _layers.at(L).iSample);
			_layers.at(L).iChannel = _layers.at(L - 1).iChannel;

			// 以下代码用于下采样层的计算

			// 对本层输出通道乘性偏置进行1值初始化
			_layers.at(L).Beta.assign(_layers.at(L).iChannel, 1);
			_layers.at(L).Beta_delta.assign(_layers.at(L).iChannel, 0);

			// 对本层输出通道加性偏置进行0值初始化
			_layers.at(L).B.assign(_layers.at(L).iChannel, 0);
			_layers.at(L).B_delta.assign(_layers.at(L).iChannel, 0);

			_layers.at(L).Delta.resize(_layers.at(L).iChannel);// 敏感度图的大小初始化，但不赋值

			cout << "subsampling layer " << L + 1 << " has initialised!" << endl;
		}

		// ======================================================================
		// 本层是全连接层的前提下，三种情况：前一层是下采样层，前一层是卷积层，前一层是输入层

		if (_layers.at(L).type == 'f')
		{
			if (_layers.at(L - 1).type == 's' || _layers.at(L - 1).type == 'c' || _layers.at(L - 1).type == 'i')
			{
				// ------------------------------------------------------------------
				// 以下代码对第6层(过渡全连接层)有效

				// 当前层全连接输入个数 = 上一层每个通道的像素个数 * 上一层输入通道数
				int fvnum = _layers.at(L - 1).iSizePic[0] * _layers.at(L - 1).iSizePic[1] * _layers.at(L - 1).iChannel;
				// 当前输出层类别个数
				int onum = _layers.at(L).iChannel;

				double maximum = (double)sqrt(6.0f / (onum + fvnum));
				// 初始化当前层与上一层的连接权值
				_layers.at(L).W.set_rand(fvnum, onum, -maximum, maximum);// 注意是W[I列][J行],I为当前层全连接输入个数，J为当前层数目
				_layers.at(L).W_delta.set_zero_same_size_as(_layers.at(L).W);

				// 对本层输出通道加性偏置进行0值初始化
				_layers.at(L).B.assign(onum, 0);
				_layers.at(L).B_delta.assign(onum, 0);
			}
			else if (_layers.at(L - 1).type == 'f')
			{
				// ------------------------------------------------------------------
				// 以下代码对第7层(全连接层)有效。 对第8层也有效吧？

				// 当前层全连接输入个数 = 上一层输入通道数
				int fvnum = _layers.at(L - 1).iChannel;
				// 当前输出层类别个数
				int onum = _layers.at(L).iChannel;

				double maximum = (double)sqrt(6.0f / (onum + fvnum));
				// 初始化当前层与上一层的连接权值
				_layers.at(L).W.set_rand(fvnum, onum, -maximum, maximum);// 注意是W[列I][行J],I为上一层的数目，J为当前层数目
				_layers.at(L).W_delta.set_zero_same_size_as(_layers.at(L).W);

				// 对本层输出通道加性偏置进行0值初始化
				_layers.at(L).B.assign(onum, 0);
				_layers.at(L).B_delta.assign(onum, 0);
			}

			cout << "fully connected layer " << L + 1 << " has initialised!" << endl;
		}
	}
}
