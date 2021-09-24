#include <maths_vector.h>
#include <iostream>
#include <time.h>
#include<algorithm>


// �õ�n~m��vector
vector<double> get_vector_n2m(int n, int m)
{
	vector<double> vector_n2m;

	if (n > m)
	{
		cout << "n > m in function get_vector_double_n2m, so do nothing and return!" << endl;
	}

	for (int i = n; i <= m; i++)
	{
		vector_n2m.push_back(i);
	}

	return vector_n2m;
}


// ����matlab�е�randperm���������õ�0~num-1֮��Ĵ���˳��������
vector<int> randperm_vector(int num)
{
	vector<int> serial_num;

	for (int i = 0; i < num; i++)
	{
		serial_num.push_back(i);
	}

	int j, temp;

	srand((unsigned)time(NULL));//srand()��������һ���Ե�ǰʱ�俪ʼ���������
	for (int i = num; i > 1; i--)
	{
		j = rand() % i;
		temp = serial_num.at(i - 1);
		serial_num.at(i - 1) = serial_num.at(j);
		serial_num.at(j) = temp;
	}

	return serial_num;
}
