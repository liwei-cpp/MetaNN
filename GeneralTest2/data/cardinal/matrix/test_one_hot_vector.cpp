#include <data/cardinal/matrix/test_one_hot_vector.h>
#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_one_hot_vector_case1()
    {
        cout << "Test one hot vector case 1...\t";
        static_assert(IsMatrix<OneHotVector<CheckElement, CheckDevice>>, "Test Error");
        static_assert(IsMatrix<OneHotVector<CheckElement, CheckDevice> &>, "Test Error");
        static_assert(IsMatrix<OneHotVector<CheckElement, CheckDevice> &&>, "Test Error");
        static_assert(IsMatrix<const OneHotVector<CheckElement, CheckDevice> &>, "Test Error");
        static_assert(IsMatrix<const OneHotVector<CheckElement, CheckDevice> &&>, "Test Error");

        auto rm = OneHotVector<CheckElement, CheckDevice>(100, 37);
        assert(rm.Shape().RowNum() == 1);
        assert(rm.Shape().ColNum() == 100);
        assert(rm.HotPos() == 37);

        auto rm1 = Evaluate(rm);
        for (size_t i=0; i<1; ++i)
        {
            for (size_t j=0; j<100; ++j)
            {
                if (j != 37)
                {
                    assert(rm1(i, j) == 0);
                }
                else
                {
                    assert(rm1(i, j) == 1);
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_one_hot_vector_case2()
    {
        cout << "Test one hot vector case 2...\t";
        auto rm1 = OneHotVector<int, CheckDevice>(100, 37);
        auto rm2 = OneHotVector<int, CheckDevice>(50, 16);

        auto evalRes1 = rm1.EvalRegister();
        auto evalRes2 = rm2.EvalRegister();

        EvalPlan<CheckDevice>::Eval();
        for (size_t j = 0; j < 100; ++j)
        {
            if (j == 37)
            {
                assert(evalRes1.Data()(0, j) == 1);
            }
            else
            {
                assert(evalRes1.Data()(0, j) == 0);
            }
        }

        for (size_t j = 0; j < 50; ++j)
        {
            if (j == 16)
            {
                assert(evalRes2.Data()(0, j) == 1);
            }
            else
            {
                assert(evalRes2.Data()(0, j) == 0);
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Data::Cardinal::Matrix
{
    void test_one_hot_vector()
    {
        test_one_hot_vector_case1();
        test_one_hot_vector_case2();
    }
}