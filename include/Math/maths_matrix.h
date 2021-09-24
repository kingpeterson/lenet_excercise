#pragma once
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp>  

using namespace cv;

void show_image_64FC1_as_matrix_double(const Mat &img);

// ע�⣺new�������������鲻��ʹ�øú�������Ϊnew������������ڴ��ַ��������
template <typename T>
void print_matrix(T *array, int row, int col)
{
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			cout << (double)*(array++) << ' ';
		}
		cout << endl;
	}
}


