#include <MetaNN/meta_nn.h>
#include <calculate_tags.h>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_vector_case1()
    {
        cout << "Test vector case 1...\t";
        static_assert(IsVector<Vector<CheckElement, CheckDevice>>);
        static_assert(IsVector<Vector<CheckElement, CheckDevice>&>);
        static_assert(IsVector<Vector<CheckElement, CheckDevice>&&>);
        static_assert(IsVector<const Vector<CheckElement, CheckDevice>&>);
        static_assert(IsVector<const Vector<CheckElement, CheckDevice>&&>);

        Vector<CheckElement, CheckDevice> rm;
        assert(rm.Shape()[0] == 0);

        rm = Vector<CheckElement, CheckDevice>(20);
        assert(rm.Shape()[0] == 20);

        int c = 0;
        for (size_t j=0; j<20; ++j)
        {
            rm.SetValue(j, (float)(c++));
        }

        const Vector<CheckElement, CheckDevice> rm2 = rm;
        c = 0;
        for (size_t j=0; j<20; ++j)
            assert(rm2(j) == c++);
        cout << "done" << endl;
    }
    
    void test_matrix_case1()
    {
        cout << "Test matrix case 1...\t";
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

    void test_matrix_case2()
    {
        cout << "Test matrix case 2...\t";
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
    
    void test_3d_array_case1()
    {
        cout << "Test 3d array case 1...\t";
        static_assert(IsThreeDArray<ThreeDArray<CheckElement, CheckDevice>>, "Test Error");
        static_assert(IsThreeDArray<ThreeDArray<CheckElement, CheckDevice>&>, "Test Error");
        static_assert(IsThreeDArray<ThreeDArray<CheckElement, CheckDevice>&&>, "Test Error");
        static_assert(IsThreeDArray<const ThreeDArray<CheckElement, CheckDevice>&>, "Test Error");
        static_assert(IsThreeDArray<const ThreeDArray<CheckElement, CheckDevice>&&>, "Test Error");

        ThreeDArray<CheckElement, CheckDevice> rm;
        assert(rm.Shape()[0] == 0);
        assert(rm.Shape()[1] == 0);
        assert(rm.Shape()[2] == 0);

        rm = ThreeDArray<CheckElement, CheckDevice>(5, 10, 20);
        assert(rm.Shape()[0] == 5);
        assert(rm.Shape()[1] == 10);
        assert(rm.Shape()[2] == 20);

        int c = 0;
        for (size_t p = 0; p < 5; ++p)
        {
            for (size_t i=0; i<10; ++i)
            {
                for (size_t j=0; j<20; ++j)
                {
                    rm.SetValue(p, i, j, (float)(c++));
                }
            }
        }

        const ThreeDArray<CheckElement, CheckDevice> rm2 = rm;
        c = 0;
        for (size_t p = 0; p < 5; ++p)
        {
            for (size_t i=0; i<10; ++i)
            {
                for (size_t j=0; j<20; ++j)
                    assert(rm2(p, i, j) == c++);
            }
        }

        auto evalHandle = rm.EvalRegister();
        auto cm = evalHandle.Data();

        for (size_t p = 0; p < cm.Shape()[0]; ++p)
        {
            for (size_t i=0; i < cm.Shape()[1]; ++i)
            {
                for (size_t j = 0; j < cm.Shape()[2]; ++j)
                {
                    assert(cm(p, i, j) == rm(p, i, j));
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_batch_scalar_case1()
    {
        cout << "Test static batch scalar case 1...\t";
        static_assert(IsTensorWithDim<Tensor<CheckElement, CheckDevice, 1>, 1>);
        static_assert(IsTensorWithDim<Tensor<CheckElement, CheckDevice, 1>&, 1>);
        static_assert(IsTensorWithDim<Tensor<CheckElement, CheckDevice, 1>&&, 1>);
        static_assert(IsTensorWithDim<const Tensor<CheckElement, CheckDevice, 1>&, 1>);
        static_assert(IsTensorWithDim<const Tensor<CheckElement, CheckDevice, 1>&&, 1>);

        Tensor<CheckElement, CheckDevice, 1> check;
        assert(check.Shape()[0] == 0);

        check = Tensor<CheckElement, CheckDevice, 1>(13);
        assert(check.Shape()[0] == 13);

        int c = 0;
        for (size_t i=0; i<13; ++i)
        {
            check.SetValue(i, (float)(c++));
        }

        const Tensor<CheckElement, CheckDevice, 1> c2 = check;
        c = 0;
        for (size_t i=0; i<13; ++i)
        {
            assert(c2(i) == (float)(c));
            assert(c2[i].Value() == (float)(c++));
        }

        auto evalHandle = check.EvalRegister();
        auto cm = evalHandle.Data();

        for (size_t i = 0; i < cm.Shape()[0]; ++i)
        {
            assert(cm[i] == check[i]);
        }
        cout << "done" << endl;
    }
    
    void test_batch_matrix_case1()
    {
        cout << "Test static batch matrix case 1...\t";
        static_assert(IsTensorWithDim<Tensor<int, CheckDevice, 3>, 3>);
        static_assert(IsTensorWithDim<Tensor<int, CheckDevice, 3>&, 3>);
        static_assert(IsTensorWithDim<Tensor<int, CheckDevice, 3>&&, 3>);
        static_assert(IsTensorWithDim<const Tensor<int, CheckDevice, 3>&, 3>);
        static_assert(IsTensorWithDim<const Tensor<int, CheckDevice, 3>&&, 3>);
    
        Tensor<int, CheckDevice, 3> data(10, 13, 35);
        assert(data.AvailableForWrite());
        assert(data.Shape()[0] == 10);
        assert(data.Shape()[1] == 13);
        assert(data.Shape()[2] == 35);
        for (size_t i = 0; i < 10; ++i)
        {
            for (size_t j = 0; j < 13; ++j)
            {
                for (size_t k = 0; k < 35; ++k)
                {
                    data.SetValue(i, j, k, (int)(i * 1000 + j * 100 + k));
                }
            }
        }
    
        for (size_t i = 0; i < 10; ++i)
        {
            for (size_t j = 0; j < 13; ++j)
            {
                for (size_t k = 0; k < 35; ++k)
                {
                    assert(data[i](j, k) == (int)(i * 1000 + j * 100 + k));
                }
            }
        }
        cout << "done" << endl;
    }

    void test_batch_matrix_case2()
    {
        cout << "Test static batch matrix case 2...\t";
        
        Tensor<CheckElement, CheckDevice, 3> rm1(3, 10, 20);
        assert(rm1.Shape()[0] == 3);

        int c = 0;
        Matrix<CheckElement, CheckDevice> me1(10, 20);
        Matrix<CheckElement, CheckDevice> me2(10, 20);
        Matrix<CheckElement, CheckDevice> me3(10, 20);
        for (size_t i = 0; i < 10; ++i)
        {
            for (size_t j = 0; j < 20; ++j)
            {
                me1.SetValue(i, j, (float)(c++));
                me2.SetValue(i, j, (float)(c++));
                me3.SetValue(i, j, (float)(c++));
                rm1.SetValue(0, i, j, me1(i, j));
                rm1.SetValue(1, i, j, me2(i, j));
                rm1.SetValue(2, i, j, me3(i, j));
            }
        }
    
        for (size_t i = 0; i < 10; ++i)
        {
            for (size_t j = 0; j < 20; ++j)
            {
                assert(rm1[0](i, j) == me1(i, j));
                assert(rm1[1](i, j) == me2(i, j));
                assert(rm1[2](i, j) == me3(i, j));
            }
        }
        cout << "done" << endl;
    }
    
    void test_batch_3d_array_case1()
    {
        cout << "Test static batch 3d array case 1...\t";
        static_assert(IsTensorWithDim<Tensor<int, CheckDevice, 4>, 4>);
        static_assert(IsTensorWithDim<Tensor<int, CheckDevice, 4>&, 4>);
        static_assert(IsTensorWithDim<Tensor<int, CheckDevice, 4>&&, 4>);
        static_assert(IsTensorWithDim<const Tensor<int, CheckDevice, 4>&, 4>);
        static_assert(IsTensorWithDim<const Tensor<int, CheckDevice, 4>&&, 4>);
    
        Tensor<int, CheckDevice, 4> data(10, 7, 13, 35);
        assert(data.AvailableForWrite());
        assert(data.Shape()[0] == 10);
        assert(data.Shape()[1] == 7);
        assert(data.Shape()[2] == 13);
        assert(data.Shape()[3] == 35);
    
        for (size_t i = 0; i < 10; ++i)
        {
            for (size_t p = 0; p < 7; ++p)
            {
                for (size_t j = 0; j < 13; ++j)
                {
                    for (size_t k = 0; k < 35; ++k)
                    {
                        data.SetValue(i, p, j, k, (int)(p * 33 + i * 1000 + j * 100 + k));
                    }
                }
            }
        }
    
        for (size_t i = 0; i < 10; ++i)
        {
            for (size_t p = 0; p < 7; ++p)
            {
                for (size_t j = 0; j < 13; ++j)
                {
                    for (size_t k = 0; k < 35; ++k)
                    {
                        assert(data[i](p, j, k) == (int)(p * 33 + i * 1000 + j * 100 + k));
                    }
                }
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Data
{
    void test_tensor()
    {
        test_vector_case1();
        
        test_matrix_case1();
        test_matrix_case2();
        
        test_3d_array_case1();
        
        test_batch_scalar_case1();
        test_batch_matrix_case1();
        test_batch_matrix_case2();
        test_batch_3d_array_case1();
    }
}