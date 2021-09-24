// �������
#include <maths_convolution.h>


// ������������������������vector���ٶ�Ҫ��10%��
Array3Dd convolution(Array3Dd X, const Array2Dd &Ker, string shape)
{
	// 'full' - (default) returns the full N - D convolution
	// 'valid' - returns only the part of the result that can be
	//          computed without assuming zero - padded arrays.
	//          size(C, k) = max([nak - max(0, nbk - 1)], 0).

	if (shape != "valid" && shape != "full")
	{
		cout << "wrong convolution shape control!" << endl << "convolution() failed!" << endl;
		Array3Dd temp;
		return temp;
	}

	if (X.size() <= 0)
	{
		cout << "Array3Dd is wrong!" << endl << "convolution() failed!" << endl;
		Array3Dd temp;
		return temp;
	}

	int Ker_row = Ker.at(0).size();
	int Ker_col = Ker.size();

	if (shape == "full")
	{
		X.expand_to_full_size(Ker_col, Ker_row);
	}

	int X_page = X.size();
	int X_row = X.at(0).at(0).size();
	int X_col = X.at(0).size();

	int i, j, k;

	if (shape == "valid" && (X_row < Ker_row || X_col < Ker_col))
	{
		cout << "X size is smaller than Ker size!" << endl << "convolution() failed!" << endl;
		Array3Dd temp;
		return temp;
	}

	// �����������������conv����ʼ��Ϊ0
	int conv_row = X.at(0).at(0).size() - Ker.at(0).size() + 1;
	int conv_col = X.at(0).size() - Ker.size() + 1;
	Array3Dd convn(X_page, conv_col, conv_row, 0);

	double *arr_X = new double[X_page * X_row * X_col]();
	double *arr_Ker = new double[Ker_row * Ker_col]();

	for (i = 0; i < X_page; i++)
	{
		for (j = 0; j < X_row; j++)
		{
			for (k = 0; k < X_col; k++)
			{
				// ��arr_X��ֵ
				arr_X[i * (X_row * X_col) + j * X_col + k] = X.at(i).at(k).at(j);

				// ��arr_Ker��ֵ
				if ((i == 0) && (j < Ker_row) && (k < Ker_col))
				{
					// x,y��ͬʱ��ת
					arr_Ker[j * Ker_col + k] = Ker.at(Ker_col - 1 - k).at(Ker_row - 1 - j);
				}
			}
		}
	}

	int row, col;
	for (i = 0; i < X_page; i++)
	{
		for (j = 0; j < conv_row; j++)
		{
			for (k = 0; k < conv_col; k++)
			{
				// �����������(j,k)���ֵ
				double sum_ijk = 0;
				for (row = j; row < j + Ker_row; row++)
				{
					for (col = k; col < k + Ker_col; col++)
					{
						sum_ijk += arr_X[i * (X_row * X_col) + row * X_col + col] * arr_Ker[(row - j) * Ker_col + (col - k)];
					}
				}
				convn.at(i).at(k).at(j) = sum_ijk;
			}
		}
	}

	delete[] arr_X;
	delete[] arr_Ker;

	return convn;
}


// ��size(a, 3) = size(b, 3), ����ʽ�������άΪ1, ��ʾ����ѵ�������ĵ��Ӻ�(�������㷨), ���Ҫ����������ƽ��
// ���Ƕ�ÿһ��ͼ���������Ȼ�������
Array2Dd convolution(const Array3Dd &X, const Array3Dd &Ker, string shape)
{
	// 'full' - (default) returns the full N - D convolution
	// 'valid' - returns only the part of the result that can be
	//          computed without assuming zero - padded arrays.
	//          size(C, k) = max([nak - max(0, nbk - 1)], 0).

	if (shape != "valid" && shape != "full")
	{
		cout << "wrong convolution shape control!" << endl << "convolution() failed!" << endl;
		Array2Dd temp;
		return temp;
	}

	int page_X = X.size();
	int page_Ker = Ker.size();

	if (page_X != page_Ker)
	{
		cout << "page size not equal!" << endl << "convolution() failed!" << endl;
		Array2Dd temp;
		return temp;
	}

	Array2Dd sum;

	for (int i = 0; i < page_X; ++i)
	{
		sum.add(convolution(X.at(i), Ker.at(i), shape));
	}

	return sum;
}


// ������������������������vector���ٶ�Ҫ��30����
Array2Dd convolution(Array2Dd X, Array2Dd Ker, string shape)
{
	// 'full' - (default) returns the full N - D convolution
	// 'valid' - returns only the part of the result that can be
	//          computed without assuming zero - padded arrays.
	//          size(C, k) = max([nak - max(0, nbk - 1)], 0).

	if (shape != "valid" && shape != "full")
	{
		cout << "wrong convolution shape control!" << endl << "convolution() failed!" << endl;
		Array2Dd temp;
		return temp;
	}

	int Ker_row = Ker.at(0).size();
	int Ker_col = Ker.size();

	if (shape == "full")
	{
		X.expand_to_full_size(Ker_col, Ker_row);
	}

	int X_row = X.at(0).size();
	int X_col = X.size();

	if (shape == "valid" && (X_row < Ker_row || X_col < Ker_col))
	{
		cout << "X size is smaller than Ker size!" << endl << "convolution() failed!" << endl;
		Array2Dd temp;
		return temp;
	}

	// �����������������conv����ʼ��Ϊ0
	int conv_row = X.at(0).size() - Ker.at(0).size() + 1;
	int conv_col = X.size() - Ker.size() + 1;
	Array2Dd conv(conv_col, conv_row, 0);

	double *arr_X = new double[X_row * X_col]();
	double *arr_Ker = new double[Ker_row * Ker_col]();

	int i, j;

	for (i = 0; i < X_row; i++)
	{
		for (j = 0; j < X_col; j++)
		{
			// ��arr_X��ֵ
			arr_X[i * X_col + j] = X.at(j).at(i);

			// ��arr_Ker��ֵ
			if ((i < Ker_row) && (j < Ker_col))
			{
				// x,y��ͬʱ��ת
				arr_Ker[i * Ker_col + j] = Ker.at(Ker_col - 1 - j).at(Ker_row - 1 - i);
			}
		}
	}

	int row, col;
	for (i = 0; i < conv_row; i++)
	{
		for (j = 0; j < conv_col; j++)
		{
			// �����������(i,j)���ֵ
			double sum_ij = 0;
			for (row = i; row < i + Ker_row; row++)
			{
				for (col = j; col < j + Ker_col; col++)
				{
					sum_ij += arr_X[row * X_col + col] * arr_Ker[(row - i) * Ker_col + (col - j)];
				}
			}
			conv.at(j).at(i) = sum_ij;
		}
	}

	delete[] arr_X;
	delete[] arr_Ker;

	return conv;
}

