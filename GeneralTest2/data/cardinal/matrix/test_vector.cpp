#include <data/cardinal/matrix/test_vector.h>
#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_vector_case1()
    {
        cout << "Test vector case 1...\t";
        static_assert(IsMatrix<Vector<CheckElement, CheckDevice>>);
        static_assert(IsMatrix<Vector<CheckElement, CheckDevice>&>);
        static_assert(IsMatrix<Vector<CheckElement, CheckDevice>&&>);
        static_assert(IsMatrix<const Vector<CheckElement, CheckDevice>&>);
        static_assert(IsMatrix<const Vector<CheckElement, CheckDevice>&&>);

        Vector<CheckElement, CheckDevice> rm;
        assert(rm.Shape().RowNum() == 0);
        assert(rm.Shape().ColNum() == 0);

        rm = Vector<CheckElement, CheckDevice>::CreateWithShape(20);
        assert(rm.Shape().RowNum() == 1);
        assert(rm.Shape().ColNum() == 20);

        int c = 0;
        for (size_t j=0; j<20; ++j)
        {
            rm.SetValue((float)(c++), j);
        }

        const Vector<CheckElement, CheckDevice> rm2 = rm;
        c = 0;
        for (size_t j=0; j<20; ++j)
            assert(rm2(j) == c++);
        cout << "done" << endl;
    }
    
    void test_vector_case2()
    {
        cout << "Test vector case 2...\t";
        auto rm1 = Vector<CheckElement, CheckDevice>::CreateWithShape(20);
        int c = 0;
        for (size_t j = 0; j < 20; ++j)
        {
            rm1.SetValue(j, (float)(c++));
        }
    
        Matrix<CheckElement, CheckDevice> res = Evaluate(rm1);
        assert(res.Shape().RowNum() == 1);
        assert(res.Shape().ColNum() == 20);
        c = 0;
        for (size_t j = 0; j < 20; ++j)
        {
            assert(res(0, j) == c++);
        }
        cout << "done" << endl;
    }
}

namespace Test::Data::Cardinal::Matrix
{
    void test_vector()
    {
        test_vector_case1();
        test_vector_case2();
    }
}