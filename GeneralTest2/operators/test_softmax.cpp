#include <data_gen.h>
#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <cmath>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_softmax_case1()
    {
        cout << "Test softmax case 1 ...\t";
        auto input = GenMatrix<float>(1, 20, 0, 0.001f);
        auto op = Softmax(input);
        auto res = Evaluate(op);

        float sum = 0;
        for (size_t i = 0; i < 20; ++i)
        {
            sum += exp(input(0, i));
        }

        for (size_t i = 0; i < 20; ++i)
        {
            assert(fabs(res(0, i) - exp(input(0, i)) / sum) < 0.0001);
        }

        cout << "done" << endl;
    }
    
    void test_softmax_case2()
    {
        cout << "Test softmax case 2 ...\t";
        {
            auto rm1 = GenMatrix<float>(1, 20, 0, 0.001f);
            auto res = Softmax(rm1);
            auto res2 = Softmax(rm1);

            assert(res == res2);

            auto cm1 = Evaluate(res);
            auto cm2 = Evaluate(res);
            assert(cm1 == cm2);
        }
        {
            auto rm1 = GenMatrix<float>(1, 20, 0, 0.001f);
            auto res = Softmax(rm1);
            auto res2 = res;

            assert(res == res2);

            const auto& evalHandle1 = res.EvalRegister();
            const auto& evalHandle2 = res2.EvalRegister();
            EvalPlan<DeviceTags::CPU>::Eval();

            auto cm1 = evalHandle1.Data();
            auto cm2 = evalHandle2.Data();
            assert(cm1 == cm2);
        }
        cout << "done" << endl;
    }
    
    void test_softmax_case3()
    {
        cout << "Test softmax case 3 ...\t";
        auto rm1 = GenBatchMatrix<float>(7, 1, 20, 0, 0.001f);
        auto t = Softmax(rm1);
        auto t_r = Evaluate(t);

        for (size_t b = 0; b < 7; ++b)
        {
            float sum = 0;
            for (size_t i = 0; i < 20; ++i)
            {
                sum += exp(rm1[b](0, i));
            }

            for (size_t i = 0; i < 20; ++i)
            {
                assert(fabs(t_r[b](0, i) - exp(rm1[b](0, i)) / sum) < 0.0001);
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Operators
{
    void test_softmax()
    {
        test_softmax_case1();
        test_softmax_case2();
        test_softmax_case3();
    }
}