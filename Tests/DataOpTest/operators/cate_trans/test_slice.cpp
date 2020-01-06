#include <data_gen.h>
#include <MetaNN/meta_nn.h>
#include <calculate_tags.h>
#include <cmath>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_slice_case1()
    {
        cout << "Test slice: batch scalar -> scalar\t";
        
        auto ori = GenBatchScalar<CheckElement>(10, -1, 0.1);
        auto trans = Tanh(ori);
        for (size_t i = 0; i < 10; ++i)
        {
            auto res = Evaluate(trans[i]);
            assert(fabs(res.Value() - tanh(ori[i].Value())) < 0.001f);
        }

        cout << "done" << endl;
    }
    
    void test_slice_case2()
    {
        cout << "Test slice: scalar sequence -> scalar\t";
        
        auto ori = GenScalarSequence<CheckElement>(10, -1, 0.1);
        auto trans = Tanh(ori);
        for (size_t i = 0; i < 10; ++i)
        {
            auto res = Evaluate(trans[i]);
            assert(fabs(res.Value() - tanh(ori[i].Value())) < 0.001f);
        }

        cout << "done" << endl;
    }
    
    void test_slice_case3()
    {
        cout << "Test slice: batch scalar sequence -> scalar\t";
        
        auto ori = GenBatchScalarSequence<CheckElement>(std::vector{3, 2, 5}, -1, 0.1);
        auto trans = Tanh(ori);
        for (size_t i = 0; i < 3; ++i)
        {
            auto res = Evaluate(trans[0][i]);
            assert(fabs(res.Value() - tanh(ori[0][i].Value())) < 0.001f);
        }
        
        for (size_t i = 0; i < 2; ++i)
        {
            auto res = Evaluate(trans[1][i]);
            assert(fabs(res.Value() - tanh(ori[1][i].Value())) < 0.001f);
        }
        
        for (size_t i = 0; i < 5; ++i)
        {
            auto res = Evaluate(trans[2][i]);
            assert(fabs(res.Value() - tanh(ori[2][i].Value())) < 0.001f);
        }

        cout << "done" << endl;
    }
    
    void test_slice_case4()
    {
        cout << "Test slice: batch matrix -> matrix\t";
        
        auto ori = GenBatchMatrix<CheckElement>(10, 3, 7, -1, 0.1);
        auto trans = Tanh(ori);
        for (size_t i = 0; i < 10; ++i)
        {
            auto res = Evaluate(trans[i]);
            for (size_t j = 0; j < 3; ++j)
            {
                for (size_t k = 0; k < 7; ++k)
                {
                    assert(fabs(res(j, k) - tanh(ori[i](j, k))) < 0.001f);
                }
            }
        }

        cout << "done" << endl;
    }
    
    void test_slice_case5()
    {
        cout << "Test slice: matrix sequence -> matrix\t";
        
        auto ori = GenMatrixSequence<CheckElement>(10, 3, 7, -1, 0.1);
        auto trans = Tanh(ori);
        for (size_t i = 0; i < 10; ++i)
        {
            auto res = Evaluate(trans[i]);
            for (size_t j = 0; j < 3; ++j)
            {
                for (size_t k = 0; k < 7; ++k)
                {
                    assert(fabs(res(j, k) - tanh(ori[i](j, k))) < 0.001f);
                }
            }
        }

        cout << "done" << endl;
    }

    void test_slice_case6()
    {
        cout << "Test slice: batch matrix sequence -> matrix\t";
        
        auto ori = GenBatchMatrixSequence<CheckElement>(std::vector{3, 2, 5}, 3, 7, -1, 0.1);
        auto trans = Tanh(ori);
        for (size_t i = 0; i < 3; ++i)
        {
            auto res = Evaluate(trans[0][i]);
            for (size_t j = 0; j < 3; ++j)
            {
                for (size_t k = 0; k < 7; ++k)
                {
                    assert(fabs(res(j, k) - tanh(ori[0][i](j, k))) < 0.001f);
                }
            }
        }
        
        for (size_t i = 0; i < 2; ++i)
        {
            auto res = Evaluate(trans[1][i]);
            for (size_t j = 0; j < 3; ++j)
            {
                for (size_t k = 0; k < 7; ++k)
                {
                    assert(fabs(res(j, k) - tanh(ori[1][i](j, k))) < 0.001f);
                }
            }
        }
        
        for (size_t i = 0; i < 5; ++i)
        {
            auto res = Evaluate(trans[2][i]);
            for (size_t j = 0; j < 3; ++j)
            {
                for (size_t k = 0; k < 7; ++k)
                {
                    assert(fabs(res(j, k) - tanh(ori[2][i](j, k))) < 0.001f);
                }
            }
        }

        cout << "done" << endl;
    }
    
    void test_slice_case7()
    {
        cout << "Test slice: batch 3d-array -> 3d-array\t";
        
        auto ori = GenBatchThreeDArray<CheckElement>(10, 3, 5, 7, -1, 0.1);
        auto trans = Tanh(ori);
        for (size_t i = 0; i < 10; ++i)
        {
            auto res = Evaluate(trans[i]);
            for (size_t j = 0; j < 3; ++j)
            {
                for (size_t k = 0; k < 5; ++k)
                {
                    for (size_t l = 0; l < 7; ++l)
                    {
                        assert(fabs(res(j, k, l) - tanh(ori[i](j, k, l))) < 0.001f);
                    }
                }
            }
        }

        cout << "done" << endl;
    }
    
    void test_slice_case8()
    {
        cout << "Test slice: 3d-array sequence -> 3d-array\t";
        
        auto ori = GenThreeDArraySequence<CheckElement>(10, 3, 5, 7, -1, 0.1);
        auto trans = Tanh(ori);
        for (size_t i = 0; i < 10; ++i)
        {
            auto res = Evaluate(trans[i]);
            for (size_t j = 0; j < 3; ++j)
            {
                for (size_t k = 0; k < 5; ++k)
                {
                    for (size_t l = 0; l < 7; ++l)
                    {
                        assert(fabs(res(j, k, l) - tanh(ori[i](j, k, l))) < 0.001f);
                    }
                }
            }
        }

        cout << "done" << endl;
    }
    
    void test_slice_case9()
    {
        cout << "Test slice: 3d-array sequence -> 3d-array\t";
        
        auto ori = GenBatchThreeDArraySequence<CheckElement>(std::vector{3, 2, 5}, 3, 5, 7, -1, 0.1);
        auto trans = Tanh(ori);
        for (size_t i = 0; i < 3; ++i)
        {
            auto res = Evaluate(trans[0][i]);
            for (size_t j = 0; j < 3; ++j)
            {
                for (size_t k = 0; k < 5; ++k)
                {
                    for (size_t l = 0; l < 7; ++l)
                    {
                        assert(fabs(res(j, k, l) - tanh(ori[0][i](j, k, l))) < 0.001f);
                    }
                }
            }
        }
        
        for (size_t i = 0; i < 2; ++i)
        {
            auto res = Evaluate(trans[1][i]);
            for (size_t j = 0; j < 3; ++j)
            {
                for (size_t k = 0; k < 5; ++k)
                {
                    for (size_t l = 0; l < 7; ++l)
                    {
                        assert(fabs(res(j, k, l) - tanh(ori[1][i](j, k, l))) < 0.001f);
                    }
                }
            }
        }
        
        for (size_t i = 0; i < 5; ++i)
        {
            auto res = Evaluate(trans[2][i]);
            for (size_t j = 0; j < 3; ++j)
            {
                for (size_t k = 0; k < 5; ++k)
                {
                    for (size_t l = 0; l < 7; ++l)
                    {
                        assert(fabs(res(j, k, l) - tanh(ori[2][i](j, k, l))) < 0.001f);
                    }
                }
            }
        }

        cout << "done" << endl;
    }
}

namespace Test::Operators::CateTrans
{
    void test_slice()
    {
        test_slice_case1();
        test_slice_case2();
        test_slice_case3();

        test_slice_case4();
        test_slice_case5();
        test_slice_case6();
        
        test_slice_case7();
        test_slice_case8();
        test_slice_case9();
    }
}