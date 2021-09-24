#pragma once
#include <iostream>
#include <vector>
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp>  


using namespace std;
using namespace cv;

template <typename T>
class Array2D
{
public:

	// ����
	Array2D() { }

	Array2D(int col, int row, T value);

	Array2D(const vector<vector<T>> &vec_vec);

	// from image 64FC1
	Array2D(const Mat &img);

	// *************************** ���� ***************************************** //

	void from_image_64FC1(const Mat &img);

	// *************************** ��ֵ ***************************************** //

	void create(int col, int row, T value);

	void zeros(int col, int row);

	vector<T>& at(int col);// д

	const vector<T>& at(int col) const;// ��

	void push_back(vector<T> val);

	void get_specific_patch(const Array2D<T> &array2D, int size_col, int size_row, int pos_col, int pos_row);

	// *************************** ���� ***************************************** //

	void set_zero();

	void set_value(T val);

	void set_zero_same_size_as(const Array2D<T> &array2D);

	void clear();

	// ��һ��Ϊ0~1
	void normalize();

	// ����Ϊָ����Χ�ʹ�С�������
	void set_rand(int col, int row, double minimum, double maximum);

	Array2D<T> sampling(const int &sample_interval) const;

	void expand_to_full_size(int col_size, int row_size);

	vector<T> reshape_to_vector() const;

	void append_along_row(const Array2D<T> &array2D);

	// ���ｫһά��vector����������������matlab����repmatһ��
	static Array2D<T> repmat(const vector<T> &vec, int row, int col)// ��̬��Ա����
	{
		if (row <= 0 || col <= 0 || vec.size() <= 0)
		{
			cout << "size is zero!" << endl << "Array2D.repmat() failed!" << endl;
			Array2D<T> temp;
			return temp;
		}

		vector<T> vec_row;
		Array2D<T> array2D;

		int i;
		for (i = 0; i < row; i++)
		{
			vec_row.insert(vec_row.end(), vec.begin(), vec.end());
		}

		for (i = 0; i < col; i++)
		{
			array2D.push_back(vec_row);
		}

		return array2D;
	}

	// ����ת��
	Array2D<T> transpose() const;

	// ���ھ����˵ķ�ת
	Array2D<T> flip_xy() const;

	void class_0_to_9(int length);

	// ************************** ��ѧ���� **************************************** //

	Array2D<T> operator + (const Array2D<T> &array2D) const;

	Array2D<T> operator + (const T &val) const;

	Array2D<T> operator - (const Array2D<T> &array2D) const;

	// ���
	Array2D<T> operator * (const Array2D<T> &array2D) const;

	Array2D<T> operator * (const T &val) const;

	void add(const Array2D<T> &array2D);

	void dot_product(const Array2D<T> &array2D);

	// ����˷�
	Array2D<T> product(const Array2D<T> &array2D) const;

	T sum() const;

	vector<T> mean() const;

	Array2D<T> pow(const int power_num) const;

	// *************************** ��� ***************************************** //

	int size() const;

	void print() const;

	void show_image_64FC1() const;

	void show_image_64FC1(int time_msec) const;

	Mat to_Mat_64FC1() const;

	vector<vector<T>> get_array2D() const;

	// �õ�ÿһ�е����ֵ��λ��
	vector<int> max_index() const;

private:

	vector<vector<T>> _array2D;

};

typedef Array2D<double>                Array2Dd;
