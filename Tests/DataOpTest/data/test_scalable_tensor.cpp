#include <MetaNN/meta_nn.h>
#include <calculate_tags.h>
#include <cmath>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_scalable_tensor_case1()
    {
        cout << "Test scalable tensor case 1 (scalar array) ...\t";
        using TCardinal = Scalar<CheckElement, CheckDevice>;
        
        static_assert(IsVector<ScalableTensor<TCardinal>>);
        static_assert(IsVector<ScalableTensor<TCardinal> &>);
        static_assert(IsVector<ScalableTensor<TCardinal> &&>);
        static_assert(IsVector<const ScalableTensor<TCardinal> &>);
        static_assert(IsVector<const ScalableTensor<TCardinal> &&>);

        auto rm1 = ScalableTensor<TCardinal>();
        assert(rm1.Shape()[0] == 0);
        assert(rm1.IsEmpty());

        rm1.PushBack(TCardinal{3});
        rm1.PushBack(TCardinal{8});
        rm1.PushBack(TCardinal{2});
        assert(rm1.Shape()[0] == 3);
        assert(!rm1.IsEmpty());

        auto evalHandle = rm1.EvalRegister();
        EvalPlan::Inst().Eval();
        auto rm2 = evalHandle.Data();
    
        assert(rm2(0) == 3);
        assert(rm2(1) == 8);
        assert(rm2(2) == 2);
        cout << "done" << endl;
    }

    void test_scalable_tensor_case2()
    {
        cout << "Test scalable tensor case 2 (matrix array)...\t";
        using TCardinal = Matrix<CheckElement, CheckDevice>;
        
        static_assert(IsTensorWithDim<ScalableTensor<TCardinal>, 3>);
        static_assert(IsTensorWithDim<ScalableTensor<TCardinal> &, 3>);
        static_assert(IsTensorWithDim<ScalableTensor<TCardinal> &&, 3>);
        static_assert(IsTensorWithDim<const ScalableTensor<TCardinal> &, 3>);
        static_assert(IsTensorWithDim<const ScalableTensor<TCardinal> &&, 3>);

        ScalableTensor<TCardinal> rm1(10, 20);
        assert(rm1.Shape()[0] == 0);
        assert(rm1.IsEmpty());

        int c = 0;
        Matrix<CheckElement, CheckDevice> me1(10, 20);
        Matrix<CheckElement, CheckDevice> me2(10, 20);
        Matrix<CheckElement, CheckDevice> me3(10, 20);
        for (size_t i = 0; i < 10; ++i)
        {
            for (size_t j = 0; j < 20; ++j)
            {
                me1.SetValue(i, j, (CheckElement)(c++));
                me2.SetValue(i, j, (CheckElement)(c++));
                me3.SetValue(i, j, (CheckElement)(c++));
            }
        }
        rm1.PushBack(me1);
        rm1.PushBack(me2);
        rm1.PushBack(me3);
        assert(rm1.Shape()[0] == 3);
        assert(!rm1.IsEmpty());

        auto evalHandle = rm1.EvalRegister();
        EvalPlan::Inst().Eval();
        auto rm2 = evalHandle.Data();

        for (size_t i = 0; i < 10; ++i)
        {
            for (size_t j = 0; j < 20; ++j)
            {
                assert(rm1[0](i, j) == me1(i, j));
                assert(rm1[1](i, j) == me2(i, j));
                assert(rm1[2](i, j) == me3(i, j));
            }
        }
        cout << "done" << endl;
    }

    void test_scalable_tensor_case3()
    {
        cout << "Test scalable tensor case 3  (array of 3d array) ...\t";
        using TCardinal = ThreeDArray<CheckElement, CheckDevice>;
        
        static_assert(IsTensorWithDim<ScalableTensor<TCardinal>, 4>);
        static_assert(IsTensorWithDim<ScalableTensor<TCardinal> &, 4>);
        static_assert(IsTensorWithDim<ScalableTensor<TCardinal> &&, 4>);
        static_assert(IsTensorWithDim<const ScalableTensor<TCardinal> &, 4>);
        static_assert(IsTensorWithDim<const ScalableTensor<TCardinal> &&, 4>);

        ScalableTensor<TCardinal> rm1(7, 10, 20);
        assert(rm1.Shape()[0] == 0);
        assert(rm1.IsEmpty());

        int c = 0;
        TCardinal me1(7, 10, 20);
        TCardinal me2(7, 10, 20);
        TCardinal me3(7, 10, 20);
        
        for (size_t p = 0; p < 7; ++p)
        {
            for (size_t i = 0; i < 10; ++i)
            {
                for (size_t j = 0; j < 20; ++j)
                {
                    me1.SetValue(p, i, j, (CheckElement)(c++));
                    me2.SetValue(p, i, j, (CheckElement)(c++));
                    me3.SetValue(p, i, j, (CheckElement)(c++));
                }
            }
        }
        rm1.PushBack(me1);
        rm1.PushBack(me2);
        rm1.PushBack(me3);
        assert(rm1.Shape()[0] == 3);
        assert(!rm1.IsEmpty());

        auto evalHandle = rm1.EvalRegister();
        EvalPlan::Inst().Eval();
        auto rm2 = evalHandle.Data();

        for (size_t p = 0; p < 7; ++p)
        {
            for (size_t i = 0; i < 10; ++i)
            {
                for (size_t j = 0; j < 20; ++j)
                {
                    assert(rm1[0](p, i, j) == me1(p, i, j));
                    assert(rm1[1](p, i, j) == me2(p, i, j));
                    assert(rm1[2](p, i, j) == me3(p, i, j));
                }
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Data
{
    void test_scalable_tensor()
    {
        test_scalable_tensor_case1();
        test_scalable_tensor_case2();
        test_scalable_tensor_case3();
    }
}