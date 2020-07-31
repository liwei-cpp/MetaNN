#include <data_gen.h>
#include <MetaNN/meta_nn.h>
#include <calculate_tags.h>
#include <cmath>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_permute_case1()
    {
        cout << "Test permute case 1 ...\t";
        auto ori = GenTensor<CheckElement>(0, 1, 3, 2, 5);
        auto op = Permute<PolicyContainer<PDimArrayIs<2, 0, 1>>>(ori);
        assert(op.Shape()[0] == 5);
        assert(op.Shape()[1] == 3);
        assert(op.Shape()[2] == 2);
        
        auto res = Evaluate(op);
        assert(res.Shape()[0] == 5);
        assert(res.Shape()[1] == 3);
        assert(res.Shape()[2] == 2);
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j< 2; ++j)
            {
                for (size_t k = 0; k < 5; ++k)
                {
                    assert(fabs(ori(i, j, k) - res(k, i, j)) <= 0.001f);
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_permute_case2()
    {
        cout << "Test permute case 2 ...\t";
        Tensor<int, DeviceTags::CPU, 2> ori(2, 3);
        ori.SetValue(0, 0, 1);
        ori.SetValue(0, 1, 2);
        ori.SetValue(0, 2, 3);
        ori.SetValue(1, 0, 4);
        ori.SetValue(1, 1, 5);
        ori.SetValue(1, 2, 6);
        
        auto op = Permute<PolicyContainer<PDimArrayIs<1, 0>>>(ori);
        assert(op.Shape()[0] == 3);
        assert(op.Shape()[1] == 2);
        
        auto res = Evaluate(op);
        assert(res(0, 0) == 1);
        assert(res(1, 0) == 2);
        assert(res(2, 0) == 3);
        assert(res(0, 1) == 4);
        assert(res(1, 1) == 5);
        assert(res(2, 1) == 6);
        cout << "done" << endl;
    }
    
    void test_permute_case3()
    {
        cout << "Test permute case 3 (inverse operation of permute)...\t";
        auto ori = GenTensor<CheckElement>(0, 1, 3, 2, 5);
        auto op = Permute<PolicyContainer<PDimArrayIs<2, 0, 1>>>(ori);
        auto op2 = PermuteInv<PolicyContainer<PDimArrayIs<2, 0, 1>>>(op);
        assert(op2.Shape()[0] == 3);
        assert(op2.Shape()[1] == 2);
        assert(op2.Shape()[2] == 5);
        
        auto res = Evaluate(op2);
        assert(res.Shape()[0] == 3);
        assert(res.Shape()[1] == 2);
        assert(res.Shape()[2] == 5);
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j< 2; ++j)
            {
                for (size_t k = 0; k < 5; ++k)
                {
                    assert(fabs(ori(i, j, k) - res(i, j, k)) <= 0.001f);
                }
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Operation::Tensor
{
    void test_permute()
    {
        test_permute_case1();
        test_permute_case2();
        test_permute_case3();
    }
}