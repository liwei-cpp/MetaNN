#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <cmath>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_zero_tensor1()
    {
        cout << "Test zero tensor case 1 (scalar) ...\t";
        static_assert(IsScalar<ZeroTensor<CheckElement, CheckDevice, 0>>);
        static_assert(IsScalar<ZeroTensor<CheckElement, CheckDevice, 0> &>);
        static_assert(IsScalar<ZeroTensor<CheckElement, CheckDevice, 0> &&>);
        static_assert(IsScalar<const ZeroTensor<CheckElement, CheckDevice, 0> &>);
        static_assert(IsScalar<const ZeroTensor<CheckElement, CheckDevice, 0> &&>);
    
        ZeroTensor<CheckElement, CheckDevice, 0> val1;
        assert(val1 == val1);
    
        auto x = Evaluate(val1);
        assert(x.Value() == 0);
        cout << "done" << endl;
    }
    
    void test_zero_tensor2()
    {
        cout << "Test zero tensor case 2 (matrix) ...\t";
        static_assert(IsMatrix<ZeroTensor<CheckElement, CheckDevice, 2>>);
        static_assert(IsMatrix<ZeroTensor<CheckElement, CheckDevice, 2> &>);
        static_assert(IsMatrix<ZeroTensor<CheckElement, CheckDevice, 2> &&>);
        static_assert(IsMatrix<const ZeroTensor<CheckElement, CheckDevice, 2> &>);
        static_assert(IsMatrix<const ZeroTensor<CheckElement, CheckDevice, 2> &&>);

        ZeroTensor<CheckElement, CheckDevice, 2> rm(10, 20);
        assert(rm.Shape()[0] == 10);
        assert(rm.Shape()[1] == 20);

        const auto& evalHandle = rm.EvalRegister();
        EvalPlan<DeviceTags::CPU>::Inst().Eval();

        auto rm1 = evalHandle.Data();
        for (size_t i=0; i<10; ++i)
        {
            for (size_t j=0; j<20; ++j)
            {
                assert(rm1(i, j) == 0);
            }
        }

        cout << "done" << endl;
    }
    
    void test_zero_tensor3()
    {
        cout << "Test zero tensor case 3 (3d array) ...\t";
        static_assert(IsThreeDArray<ZeroTensor<CheckElement, CheckDevice, 3>>);
        static_assert(IsThreeDArray<ZeroTensor<CheckElement, CheckDevice, 3>&>);
        static_assert(IsThreeDArray<ZeroTensor<CheckElement, CheckDevice, 3>&&>);
        static_assert(IsThreeDArray<const ZeroTensor<CheckElement, CheckDevice, 3>&>);
        static_assert(IsThreeDArray<const ZeroTensor<CheckElement, CheckDevice, 3>&&>);

        ZeroTensor<CheckElement, CheckDevice, 3> rm;
        assert(rm.Shape()[0] == 0);
        assert(rm.Shape()[1] == 0);
        assert(rm.Shape()[2] == 0);

        rm = ZeroTensor<CheckElement, CheckDevice, 3>(5, 10, 20);
        assert(rm.Shape()[0] == 5);
        assert(rm.Shape()[1] == 10);
        assert(rm.Shape()[2] == 20);

        auto res = Evaluate(rm);
        for (size_t p = 0; p < 5; ++p)
        {
            for (size_t i=0; i<10; ++i)
            {
                for (size_t j=0; j<20; ++j)
                {
                    assert(res(p, i, j) == 0);
                }
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Data
{
    void test_zero_tensor()
    {
        test_zero_tensor1();
        test_zero_tensor2();
        test_zero_tensor3();
    }
}