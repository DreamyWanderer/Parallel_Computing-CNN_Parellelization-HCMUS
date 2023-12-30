#include "./src/utils.h"
#include "./src/layer.h"
#include "./src/layer/conv.h"
#include <iostream>

using namespace std;

/* int main() {
    Conv* C1 = new Conv(1, 28, 28, 6, 5, 5, 1);

    Eigen::Matrix<int, 27, 1> mat;
    mat << 1, 2, 0, 1, 1, 3, 0, 2, 2,
            0, 2, 1, 0, 3, 2, 1, 1, 0,
            1, 2, 1, 0, 1, 3, 3, 3, 2;
    Eigen::Matrix<float, 12, 2> w;
    w << 1, 1, 2, 2, 1, 1, 1, 1, 0, 1, 1, 0,
            1, 0, 0, 1, 2, 1, 2, 1, 1, 2, 2, 0;
    C1->weight = w;
    C1->weight.resize(12, 2);

    // Print the weight matrix
    cout << "weight: " << endl;
    cout << C1->weight << endl;


    return 0;
} */
