#include <data_gen.h>
#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <cmath>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_tanh_case1()
    {
        cout << "Test tanh case 1 (scalar)\t";
        {
            Scalar<CheckElement, CheckDevice> ori(1.3122);
            auto op = Tanh(ori);
            auto res = Evaluate(op);
            assert(fabs(res.Value() - 0.8648) < 0.001f);
        }
        cout << "done" << endl;
    }
    
    void test_tanh_case2()
    {
        cout << "Test tanh case 2 (matrix)\t";
        auto ori = GenMatrix<CheckElement>(10, 7, -1, 0.01);
        auto op = Tanh(ori);
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
                auto value = std::tanh(ori(i, k));
                assert(fabs(res(i, k) - value) < 0.001f);
            }
        }
        cout << "done" << endl;
    }
    
    void test_tanh_case3()
    {
        cout << "Test tanh case 3 (3d-array)\t";
        auto ori = GenThreeDArray<CheckElement>(2, 10, 7, -1, 0.01);
        auto op = Tanh(ori);
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
                    auto value = std::tanh(ori(p, i, k));
                    assert(fabs(res(p, i, k) - value) < 0.001f);
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_tanh_case4()
    {
        cout << "Test tanh case 4 (batch scalar)\t";
        auto ori = GenBatchScalar<CheckElement>(10, -1, 0.1);
        auto op = Tanh(ori);
        static_assert(IsBatchScalar<decltype(op)>);
        assert(op.Shape().BatchNum() == 10);
        
        auto res = Evaluate(op);
        static_assert(IsBatchScalar<decltype(res)>);
        assert(res.Shape().BatchNum() == 10);
        
        for (size_t i = 0; i < 10; ++i)
        {
            auto value = std::tanh(ori[i].Value());
            assert(fabs(res[i].Value() - value) < 0.001f);
        }
        cout << "done" << endl;
    }
    
    void test_tanh_case5()
    {
        cout << "Test tanh case 5 (batch matrix)\t";
        auto ori = GenBatchMatrix<CheckElement>(2, 10, 7, -0.9, 0.01);
        auto op = Tanh(ori);
        static_assert(IsBatchMatrix<decltype(op)>);
        assert(op.Shape().BatchNum() == 2);
        assert(op.Shape().RowNum() == 10);
        assert(op.Shape().ColNum() == 7);
        
        auto res = Evaluate(op);
        static_assert(IsBatchMatrix<decltype(res)>);
        assert(res.Shape().BatchNum() == 2);
        assert(res.Shape().RowNum() == 10);
        assert(res.Shape().ColNum() == 7);
        
        for (size_t p = 0; p < 2; ++p)
        {
            for (size_t i = 0; i < 10; ++i)
            {
                for (size_t k = 0; k < 7; ++k)
                {
                    auto value = std::tanh(ori[p](i, k));
                    assert(fabs(res[p](i, k) - value) < 0.001f);
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_tanh_case6()
    {
        cout << "Test tanh case 6 (batch 3d-array)\t";
        auto ori = GenBatchThreeDArray<CheckElement>(2, 3, 10, 7);
        auto op = Tanh(ori);
        static_assert(IsBatchThreeDArray<decltype(op)>);
        assert(op.Shape().BatchNum() == 2);
        assert(op.Shape().PageNum() == 3);
        assert(op.Shape().RowNum() == 10);
        assert(op.Shape().ColNum() == 7);
        
        auto res = Evaluate(op);
        static_assert(IsBatchThreeDArray<decltype(res)>);
        assert(res.Shape().BatchNum() == 2);
        assert(res.Shape().PageNum() == 3);
        assert(res.Shape().RowNum() == 10);
        assert(res.Shape().ColNum() == 7);
        
        for (size_t b = 0; b < 2; ++b)
        {
            for (size_t p = 0; p < 3; ++p)
            {
                for (size_t i = 0; i < 10; ++i)
                {
                    for (size_t k = 0; k < 7; ++k)
                    {
                        auto value = tanh(ori[b](p, i, k));
                        assert(fabs(res[b](p, i, k) - value) < 0.001f);
                    }
                }
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Operators
{
    void test_tanh()
    {
        test_tanh_case1();
        test_tanh_case2();
        test_tanh_case3();
        test_tanh_case4();
        test_tanh_case5();
        test_tanh_case6();
    }
}