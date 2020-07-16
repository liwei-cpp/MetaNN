#include <MetaNN/meta_nn.h>
#include <calculate_tags.h>
#include <cmath>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_bias_vector_case1()
    {
        cout << "Test bias vector case 1...\t";
        static_assert(IsVector<BiasVector<Scalar<CheckElement, CheckDevice>>>);
        static_assert(IsVector<BiasVector<Scalar<CheckElement, CheckDevice>> &>);
        static_assert(IsVector<BiasVector<Scalar<CheckElement, CheckDevice>> &&>);
        static_assert(IsVector<const BiasVector<Scalar<CheckElement, CheckDevice>> &>);
        static_assert(IsVector<const BiasVector<Scalar<CheckElement, CheckDevice>> &&>);

        BiasVector rm(100, 37, Scalar<CheckElement, CheckDevice>{0.3});
        assert(rm.Shape()[0] == 100);
        assert(rm.HotPos() == 37);

        auto rm1 = Evaluate(rm);
        for (size_t j=0; j<100; ++j)
        {
            if (j != 37)
            {
                assert(fabs(rm1(j)) < 0.001);
            }
            else
            {
                assert(fabs(rm1(j) - 0.3) < 0.001);
            }
        }
        cout << "done" << endl;
    }

    void test_bias_vector_case2()
    {
        cout << "Test bias vector vector case 2...\t";
        BiasVector rm1(100, 37, Scalar<CheckElement, CheckDevice>{0.3});
        BiasVector rm2(50, 16, Scalar<CheckElement, CheckDevice>{0.1});

        auto evalRes1 = rm1.EvalRegister();
        auto evalRes2 = rm2.EvalRegister();

        EvalPlan::Inst().Eval();
        for (size_t j = 0; j < 100; ++j)
        {
            if (j == 37)
            {
                assert(fabs(evalRes1.Data()(j) - 0.3) < 0.001f);
            }
            else
            {
                assert(fabs(evalRes1.Data()(j)) < 0.001f);
            }
        }

        for (size_t j = 0; j < 50; ++j)
        {
            if (j == 16)
            {
                assert(fabs(evalRes2.Data()(j) - 0.1) < 0.001f);
            }
            else
            {
                assert(fabs(evalRes2.Data()(j)) < 0.001f);
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Data
{
    void test_bias_vector()
    {
        test_bias_vector_case1();
        test_bias_vector_case2();
    }
}