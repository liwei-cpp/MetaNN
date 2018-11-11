#include <data/cardinal/matrix/test_matrix.h>
#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_matrix_case1()
    {
        cout << "Test matrix case 1...\t";
        static_assert(IsMatrix<Matrix<CheckElement, CheckDevice>>, "Test Error");
        static_assert(IsMatrix<Matrix<CheckElement, CheckDevice>&>, "Test Error");
        static_assert(IsMatrix<Matrix<CheckElement, CheckDevice>&&>, "Test Error");
        static_assert(IsMatrix<const Matrix<CheckElement, CheckDevice>&>, "Test Error");
        static_assert(IsMatrix<const Matrix<CheckElement, CheckDevice>&&>, "Test Error");

        Matrix<CheckElement, CheckDevice> rm;
        assert(rm.Shape().RowNum() == 0);
        assert(rm.Shape().ColNum() == 0);

        rm = Matrix<CheckElement, CheckDevice>(10, 20);
        assert(rm.Shape().RowNum() == 10);
        assert(rm.Shape().ColNum() == 20);

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
    
    void test_matrix_case2()
    {
        cout << "Test matrix case 2...\t";
        auto rm1 = Matrix<CheckElement, CheckDevice>(10, 20);
        int c = 0;
        for (size_t i = 0; i < 10; ++i)
        {
            for (size_t j = 0; j < 20; ++j)
            {
                rm1.SetValue(i, j, (float)(c++));
            }
        }

        auto rm2 = Matrix<CheckElement, CheckDevice>(3, 7);
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

namespace Test::Data::Cardinal::Matrix
{
    void test_matrix()
    {
        test_matrix_case1();
        test_matrix_case2();
    }
}