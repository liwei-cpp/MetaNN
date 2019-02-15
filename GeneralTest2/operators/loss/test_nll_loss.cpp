#include <data_gen.h>
#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <cmath>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_nll_loss_case1()
    {
        cout << "Test NLL loss case 1 (scalar)\t";
        {
            Scalar<CheckElement, CheckDevice> weight(2);
            Scalar<CheckElement, CheckDevice> input(3);
            auto op = NLLLoss(weight, input);
            auto res = Evaluate(op);
            
            auto check = -2 * log(3);
            assert(fabs(res.Value() - check) < 0.001f);
        }        
        cout << "done" << endl;
    }
    
    void test_nll_loss_case2()
    {
        cout << "Test NLL loss case 2 (matrix)\t";
        auto weight = GenMatrix<CheckElement>(10, 7, -100, 3);
        auto input = GenMatrix<CheckElement>(10, 7, 1, 0.1);
        auto op = NLLLoss(weight, input);
        static_assert(IsScalar<decltype(op)>);
        
        auto res = Evaluate(op);
        static_assert(IsScalar<decltype(res)>);
        
        CheckElement check{};
        for (size_t i = 0; i < 10; ++i)
        {
            for (size_t j = 0; j < 7; ++j)
            {
                check -= weight(i, j) * log(input(i, j));
            }
        }
        assert(fabs(check - res.Value()) < 0.001f);
        cout << "done" << endl;
    }
    
    void test_nll_loss_case3()
    {
        cout << "Test NLL loss case 2 (batch matrix)\t";
        auto weight = GenBatchMatrix<CheckElement>(3, 10, 7, -100, 3);
        auto input = GenBatchMatrix<CheckElement>(3, 10, 7, 1, 0.1);
        auto op = NLLLoss(weight, input);
        static_assert(IsScalar<decltype(op)>);
        
        auto res = Evaluate(op);
        static_assert(IsScalar<decltype(res)>);
        
        CheckElement check{};
        for (size_t b = 0; b < 3; ++b)
        {
            for (size_t i = 0; i < 10; ++i)
            {
                for (size_t j = 0; j < 7; ++j)
                {
                    check -= weight[b](i, j) * log(input[b](i, j));
                }
            }
        }
        // batch average
        check /= 3;
        assert(fabs(check - res.Value()) < 0.001f);
        cout << "done" << endl;
    }
    
    void test_nll_loss_case4()
    {
        cout << "Test NLL loss case 2 (batch matrix sequence)\t";
        auto weight = GenBatchMatrixSequence<CheckElement>(std::vector{3, 5, 7}, 10, 7, 0, 0.001);
        auto input = GenBatchMatrixSequence<CheckElement>(std::vector{3, 5, 7}, 10, 7, 1, 0.1);
        auto op = NLLLoss(weight, input);
        static_assert(IsScalar<decltype(op)>);
        
        auto res = Evaluate(op);
        static_assert(IsScalar<decltype(res)>);
        
        CheckElement check{};
        for (size_t i = 0; i < 10; ++i)
        {
            for (size_t j = 0; j < 7; ++j)
            {
                for (size_t b = 0; b < 3; ++b)
                {
                    check -= weight[0][b](i, j) * log(input[0][b](i, j));
                }
                for (size_t b = 0; b < 5; ++b)
                {
                    check -= weight[1][b](i, j) * log(input[1][b](i, j));
                }
                for (size_t b = 0; b < 7; ++b)
                {
                    check -= weight[2][b](i, j) * log(input[2][b](i, j));
                }
            }
        }
        // average
        check /= (3 + 5 + 7);
        assert(fabs(check - res.Value()) < 0.001f);
        cout << "done" << endl;
    }
}

namespace Test::Operators
{
    void test_nll_loss()
    {
        test_nll_loss_case1();
        test_nll_loss_case2();
        test_nll_loss_case3();
        test_nll_loss_case4();
    }
}