#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_var_scale_filler1()
    {
        cout << "test val scale filler case 1 (XavierFiller) ...";
    
        XavierFiller filler;
        Matrix<float, DeviceTags::CPU> mat(1000, 3000);
        filler.Fill(mat);
    
        float mean = 0;
        for (size_t i = 0; i < mat.Shape()[0]; ++i)
        {
            for (size_t j = 0; j < mat.Shape()[1]; ++j)
            {
                mean += mat(i, j);
            }
        }
        mean /= mat.Shape().Count();
    
        float var = 0;
        for (size_t i = 0; i < mat.Shape()[0]; ++i)
        {
            for (size_t j = 0; j < mat.Shape()[1]; ++j)
            {
                var += (mat(i, j) - mean) * (mat(i, j) - mean);
            }
        }
        var /= mat.Shape().Count();

        // mean = 0, std = sqrt(2 / (1000 + 3000))
        cout << "mean-delta = " << fabs(mean) << " std-delta = " << fabs(sqrt(var) - sqrt(1.0 / 2000)) << ' ';
        cout << "done" << endl;
    }
    
    void test_var_scale_filler2()
    {
        cout << "test val scale filler case 2 (MSRAFiller) ...";
    
        MSRAFiller filler;
        Matrix<float, DeviceTags::CPU> mat(1000, 3000);
        filler.Fill(mat);
    
        float mean = 0;
        for (size_t i = 0; i < mat.Shape()[0]; ++i)
        {
            for (size_t j = 0; j < mat.Shape()[1]; ++j)
            {
                mean += mat(i, j);
            }
        }
        mean /= mat.Shape().Count();
    
        float var = 0;
        for (size_t i = 0; i < mat.Shape()[0]; ++i)
        {
            for (size_t j = 0; j < mat.Shape()[1]; ++j)
            {
                var += (mat(i, j) - mean) * (mat(i, j) - mean);
            }
        }
        var /= mat.Shape().Count();

        // mean = 0, std = sqrt(2 / 1000)
        cout << "mean-delta = " << fabs(mean) << " std-delta = " << fabs(sqrt(var) - sqrt(2.0 / 1000)) << ' ';
        cout << "done" << endl;
    }
}

namespace Test::Model::ParamInitializer
{
    void test_var_scale_filler()
    {
        test_var_scale_filler1();
        test_var_scale_filler2();
    }
}