#include "test_var_scale_filter.h"
#include <MetaNN/meta_nn.h>
#include <cassert>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
void test_xavier_filler1()
{
    cout << "test xavier filler case 1 ...";
    
    XavierFiller<PolicyContainer<PUniformVarScale>> filler;
    Matrix<float, DeviceTags::CPU> mat (100, 200);
    filler.Fill(mat, 100, 200);
    
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
    
    // std = 0.0816 (sqrt(1/150))
    cout << "mean-delta = " << fabs(mean) << " std-delta = " << fabs(sqrt(var)-0.0816) << ' ';
    cout << "done" << endl;
}

void test_xavier_filler2()
{
    cout << "test xavier filler case 2 ...";
    
    XavierFiller<PolicyContainer<PNormVarScale>> filler;
    Matrix<float, DeviceTags::CPU> mat (100, 200);
    filler.Fill(mat, 100, 200);
    
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
    
    // std = 0.0816 (sqrt(1/150))
    cout << "mean-delta = " << fabs(mean) << " std = " << fabs(sqrt(var)-0.0816) << ' ';
    cout << "done" << endl;
}

void test_msra_filler1()
{
    cout << "test msra filler case 1 ...";
    
    MSRAFiller<> filler;
    Matrix<float, DeviceTags::CPU> mat (100, 200);
    filler.Fill(mat, 100, 200);
    
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
    
    // std = 0.1414 (sqrt(2/50))
    cout << "mean-delta = " << fabs(mean) << " std-delta = " << fabs(sqrt(var)-0.1414) << ' ';
    cout << "done" << endl;
}
}

void test_var_scale_filter()
{
    test_xavier_filler1();
    test_xavier_filler2();
    test_msra_filler1();
}