#include "test_gaussian_filler.h"
#include <MetaNN/meta_nn.h>
#include <cassert>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
void test_gaussian_filler1()
{
    cout << "test gaussian filler case 1 ...";
    
    GaussianFiller filler(1.5, 3.3);
    Matrix<float, DeviceTags::CPU> mat (1000, 3000);
    filler.Fill(mat, 1000, 3000);
    
    float mean = 0;
    for (size_t i = 0; i < mat.RowNum(); ++i)
    {
        for (size_t j = 0; j < mat.ColNum(); ++j)
        {
            mean += mat(i, j);
        }
    }
    mean /= mat.RowNum() * mat.ColNum();
    
    float var = 0;
    for (size_t i = 0; i < mat.RowNum(); ++i)
    {
        for (size_t j = 0; j < mat.ColNum(); ++j)
        {
            var += (mat(i, j) - mean) * (mat(i, j) - mean);
        }
    }
    var /= mat.RowNum() * mat.ColNum();
    
    // mean = 1.5, std = 3.3
    cout << "mean-delta = " << fabs(mean-1.5) << " std-delta = " << fabs(sqrt(var)-3.3) << ' ';
    cout << "done" << endl;
}
}

void test_gaussian_filler()
{
    test_gaussian_filler1();
}
