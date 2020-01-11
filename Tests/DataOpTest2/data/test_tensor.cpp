#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_tensor_case1()
    {
        cout << "Test tensor case 1 (matrix)...\t";
        static_assert(IsMatrix<Matrix<CheckElement, CheckDevice>>);
        static_assert(IsMatrix<Matrix<CheckElement, CheckDevice>&>);
        static_assert(IsMatrix<Matrix<CheckElement, CheckDevice>&&>);
        static_assert(IsMatrix<const Matrix<CheckElement, CheckDevice>&>);
        static_assert(IsMatrix<const Matrix<CheckElement, CheckDevice>&&>);

        Matrix<CheckElement, CheckDevice> rm;
        assert(rm.Shape()[0] == 0);
        assert(rm.Shape()[1] == 0);

        rm = Matrix<CheckElement, CheckDevice>(10, 20);
        assert(rm.Shape()[0] == 10);
        assert(rm.Shape()[1] == 20);

        int c = 0;
        for (size_t i=0; i<10; ++i)
        {
            for (size_t j=0; j<20; ++j)
            {
                rm.SetValue(i, j, (float)(c++));
            }
        }

        const Matrix<CheckElement, CheckDevice> rm2 = rm;
        c = 0;
        for (size_t i=0; i<10; ++i)
        {
            for (size_t j=0; j<20; ++j)
                assert(rm2(i, j) == c++);
        }
        cout << "done" << endl;
    }

    void test_tensor_case2()
    {
        cout << "Test tensor case 2 (matrix)...\t";
        Matrix<CheckElement, CheckDevice> rm1(10, 20);
        int c = 0;
        for (size_t i = 0; i < 10; ++i)
        {
            for (size_t j = 0; j < 20; ++j)
            {
                rm1.SetValue(i, j, (float)(c++));
            }
        }

        Matrix<CheckElement, CheckDevice> rm2(3, 7);
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 7; ++j)
            {
                rm2.SetValue(i, j, (float)(c++));
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Data
{
    void test_tensor()
    {
        test_tensor_case1();
        test_tensor_case2();
    }
}