#include <data_gen.h>
#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <cmath>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_acos_case1()
    {
        cout << "Test acos case 1 (scalar)\t";
        {
            Scalar<CheckElement, CheckDevice> ori(0.3348);
            auto op = Acos(ori);
            auto res = Evaluate(op);
            assert(fabs(res.Value() - 1.2294) < 0.001f);
        }
        cout << "done" << endl;
    }
    
    void test_acos_case2()
    {
        cout << "Test acos case 2 (matrix)\t";
        auto ori = GenMatrix<CheckElement>(10, 7, -1, 0.01);
        auto op = Acos(ori);
        static_assert(IsMatrix<decltype(op)>);
        assert(op.Shape().RowNum() == 10);
        assert(op.Shape().ColNum() == 7);
        
        auto res = Evaluate(op);
        static_assert(IsMatrix<decltype(res)>);
        assert(res.Shape().RowNum() == 10);
        assert(res.Shape().ColNum() == 7);
        
        for (size_t i = 0; i < 10; ++i)
        {
            for (size_t k = 0; k < 7; ++k)
            {
                auto value = std::acos(ori(i, k));
                assert(fabs(res(i, k) - value) < 0.001f);
            }
        }
        cout << "done" << endl;
    }
    
    void test_acos_case3()
    {
        cout << "Test acos case 3 (3d-array)\t";
        auto ori = GenThreeDArray<CheckElement>(2, 10, 7, -1, 0.01);
        auto op = Acos(ori);
        static_assert(IsThreeDArray<decltype(op)>);
        assert(op.Shape().PageNum() == 2);
        assert(op.Shape().RowNum() == 10);
        assert(op.Shape().ColNum() == 7);
        
        auto res = Evaluate(op);
        static_assert(IsThreeDArray<decltype(res)>);
        assert(res.Shape().PageNum() == 2);
        assert(res.Shape().RowNum() == 10);
        assert(res.Shape().ColNum() == 7);
        
        for (size_t p = 0; p < 2; ++p)
        {
            for (size_t i = 0; i < 10; ++i)
            {
                for (size_t k = 0; k < 7; ++k)
                {
                    auto value = std::acos(ori(p, i, k));
                    assert(fabs(res(p, i, k) - value) < 0.001f);
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_acos_case4()
    {
        cout << "Test acos case 4 (batch scalar)\t";
        auto ori = GenBatchScalar<CheckElement>(10, -1, 0.1);
        auto op = Acos(ori);
        static_assert(IsBatchScalar<decltype(op)>);
        assert(op.Shape().BatchNum() == 10);
        
        auto res = Evaluate(op);
        static_assert(IsBatchScalar<decltype(res)>);
        assert(res.Shape().BatchNum() == 10);
        
        for (size_t i = 0; i < 10; ++i)
        {
            auto value = std::acos(ori[i].Value());
            assert(fabs(res[i].Value() - value) < 0.001f);
        }
        cout << "done" << endl;
    }
}

namespace Test::Operators
{
    void test_acos()
    {
        test_acos_case1();
        test_acos_case2();
        test_acos_case3();
        test_acos_case4();
    }
}