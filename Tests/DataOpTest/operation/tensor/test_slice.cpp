#include <data_gen.h>
#include <MetaNN/meta_nn.h>
#include <calculate_tags.h>
#include <cmath>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_slice_case1()
    {
        cout << "Test slice: vector -> scalar\t";
        
        auto ori = GenTensor<CheckElement>(-1, 0.1, 10);
        auto trans = Tanh(ori);
        for (size_t i = 0; i < 10; ++i)
        {
            auto res = Evaluate(trans[i]);
            assert(fabs(res.Value() - tanh(ori[i].Value())) < 0.001f);
        }

        cout << "done" << endl;
    }

    void test_slice_case2()
    {
        cout << "Test slice: batch matrix -> matrix\t";
        
        auto ori = GenTensor<CheckElement>(-1, 0.1, 10, 3, 7);
        auto trans = Tanh(ori);
        for (size_t i = 0; i < 10; ++i)
        {
            auto res = Evaluate(trans[i]);
            for (size_t j = 0; j < 3; ++j)
            {
                for (size_t k = 0; k < 7; ++k)
                {
                    assert(fabs(res(j, k) - tanh(ori[i](j, k))) < 0.001f);
                }
            }
        }

        cout << "done" << endl;
    }
}

namespace Test::Operation::Tensor
{
    void test_slice()
    {
        test_slice_case1();
        test_slice_case2();
    }
}