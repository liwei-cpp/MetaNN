#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_gaussian_filler1()
    {
        cout << "test gaussian filler case 1 ...";
    
        GaussianFiller filler(1.5, 3.3);
        Matrix<float, DeviceTags::CPU> mat(1000, 3000);
        filler.Fill(mat);
    
        float mean = 0;
        for (size_t i = 0; i < mat.Shape().RowNum(); ++i)
        {
            for (size_t j = 0; j < mat.Shape().ColNum(); ++j)
            {
                mean += mat(i, j);
            }
        }
        mean /= mat.Shape().Count();
    
        float var = 0;
        for (size_t i = 0; i < mat.Shape().RowNum(); ++i)
        {
            for (size_t j = 0; j < mat.Shape().ColNum(); ++j)
            {
                var += (mat(i, j) - mean) * (mat(i, j) - mean);
            }
        }
        var /= mat.Shape().Count();

        // mean = 1.5, std = 3.3
        cout << "mean-delta = " << fabs(mean-1.5) << " std-delta = " << fabs(sqrt(var)-3.3) << ' ';
        cout << "done" << endl;
}
}

namespace Test::Model::ParamInitializer
{
    void test_gaussian_filler()
    {
        test_gaussian_filler1();
    }
}