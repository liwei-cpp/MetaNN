#include <data_gen.h>
#include <MetaNN/meta_nn.h>
#include <calculate_tags.h>
#include <cmath>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_add_case1()
    {
        cout << "Test add case 1 (scalar)\t";
        Scalar<CheckElement, CheckDevice> ori1(3);
        Scalar<CheckElement, CheckDevice> ori2(9);
        auto op = ori1 + ori2;
        static_assert(IsScalar<decltype(op)>);
        
        auto res = Evaluate(op);
        static_assert(IsScalar<decltype(res)>);
        assert(fabs(res.Value() - 12) < 0.001f);
        cout << "done" << endl;
    }
    
    void test_add_case2()
    {
        cout << "Test add case 2 (matrix)\t";
        auto ori1 = GenTensor<CheckElement>(-100, 3, 10, 7);
        auto ori2 = GenTensor<CheckElement>(1, 1.5, 10, 7);
        auto op = ori1 + ori2;
        static_assert(IsMatrix<decltype(op)>);
        assert(op.Shape()[0] == 10);
        assert(op.Shape()[1] == 7);
        
        auto res = Evaluate(op);
        static_assert(IsMatrix<decltype(res)>);
        assert(res.Shape()[0] == 10);
        assert(res.Shape()[1] == 7);
        
        for (size_t i = 0; i < 10; ++i)
        {
            for (size_t k = 0; k < 7; ++k)
            {
                auto check = ori1(i, k) + ori2(i, k);
                assert(fabs(check - res(i, k)) < 0.001f);
            }
        }
        cout << "done" << endl;
    }
    
    void test_add_case3()
    {
        cout << "Test add case 3 (add with number)\t";
        {
            auto ori1 = GenTensor<CheckElement>(-100, 3, 10, 7);
            auto op = ori1 + 3;
            static_assert(IsMatrix<decltype(op)>);
            assert(op.Shape()[0] == 10);
            assert(op.Shape()[1] == 7);
        
            auto res = Evaluate(op);
            static_assert(IsMatrix<decltype(res)>);
            assert(res.Shape()[0] == 10);
            assert(res.Shape()[1] == 7);
        
            for (size_t i = 0; i < 10; ++i)
            {
                for (size_t k = 0; k < 7; ++k)
                {
                    auto check = ori1(i, k) + 3;
                    assert(fabs(check - res(i, k)) < 0.001f);
                }
            }
        }
        {
            auto ori1 = GenTensor<CheckElement>(-100, 3, 10, 7);
            auto op = 3 + ori1;
            static_assert(IsMatrix<decltype(op)>);
            assert(op.Shape()[0] == 10);
            assert(op.Shape()[1] == 7);
        
            auto res = Evaluate(op);
            static_assert(IsMatrix<decltype(res)>);
            assert(res.Shape()[0] == 10);
            assert(res.Shape()[1] == 7);
        
            for (size_t i = 0; i < 10; ++i)
            {
                for (size_t k = 0; k < 7; ++k)
                {
                    auto check = ori1(i, k) + 3;
                    assert(fabs(check - res(i, k)) < 0.001f);
                }
            }
        }
        cout << "done" << endl;
    }

    void test_add_case4()
    {
        cout << "Test add case 4 (add with broadcast)\t";
        auto ori1 = GenTensor<CheckElement>(-100, 3, 10, 7);
        auto ori2 = GenTensor<CheckElement>(1, 1.5, 7);
        auto op = ori1 + ori2;
        static_assert(IsMatrix<decltype(op)>);
        assert(op.Shape()[0] == 10);
        assert(op.Shape()[1] == 7);
        
        auto res = Evaluate(op);
        static_assert(IsMatrix<decltype(res)>);
        assert(res.Shape()[0] == 10);
        assert(res.Shape()[1] == 7);
        
        for (size_t i = 0; i < 10; ++i)
        {
            for (size_t k = 0; k < 7; ++k)
            {
                auto check = ori1(i, k) + ori2(k);
                assert(fabs(check - res(i, k)) < 0.001f);
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Operation::Elwentwise
{
    void test_add()
    {
        test_add_case1();
        test_add_case2();
        test_add_case3();
        test_add_case4();
    }
}