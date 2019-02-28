#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_uniform_filler1()
    {
        cout << "test uniform filler case 1 ...";
    
        UniformFiller filler(-1, 1);
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

        // mean = 0, std = 2^2/12 = 1/3
        cout << "mean-delta = " << fabs(mean) << " std-delta = " << fabs(sqrt(var) - sqrt(1.0f / 3)) << ' ';
        cout << "done" << endl;
}
}

namespace Test::Model::ParamInitializer
{
    void test_uniform_filler()
    {
        test_uniform_filler1();
    }
}