#include <operators/test_collapse.h>
#include <data_gen.h>
#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <cmath>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_scalar_collapse_case1()
    {
        cout << "Test collapse: scalar -> scalar (plain collapse)\t";
        
        Scalar<CheckElement, CheckDevice> ori;
        auto dup = Collapse(ori, ori.Shape());
        assert(dup == ori);
        cout << "done" << endl;
    }
    
    void test_scalar_collapse_case2()
    {
        cout << "Test collapse: matrix -> scalar\t";
        
        auto ori = GenMatrix<CheckElement>(10, 3);
        auto col = Collapse(ori, Shape<CategoryTags::Scalar>{});
        
        static_assert(std::is_same_v<DataCategory<decltype(col)>, CategoryTags::Scalar>);
        
        auto eval = Evaluate(col);
        CheckElement check{};
        for (size_t i = 0; i < 10; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                check += ori(i, j);
            }
        }
        assert(fabs(check - eval.Value()) < 0.001);
        cout << "done" << endl;
    }
    
    void test_scalar_collapse_case3()
    {
        cout << "Test collapse: 3d-array -> scalar\t";
        
        auto ori = GenThreeDArray<CheckElement>(7, 10, 3);
        auto col = Collapse(ori, Shape<CategoryTags::Scalar>{});
        
        static_assert(std::is_same_v<DataCategory<decltype(col)>, CategoryTags::Scalar>);
        
        auto eval = Evaluate(col);
        CheckElement check{};
        for (size_t p = 0; p < 7; ++p)
        {
            for (size_t i = 0; i < 10; ++i)
            {
                for (size_t j = 0; j < 3; ++j)
                {
                    check += ori(p, i, j);
                }
            }
        }
        assert(fabs(check - eval.Value()) < 0.001);
        cout << "done" << endl;
    }
    
    void test_scalar_collapse_case4()
    {
        cout << "Test collapse: batch scalar -> scalar\t";
        
        auto ori = GenBatchScalar<CheckElement>(37);
        auto col = Collapse(ori, Shape<CategoryTags::Scalar>{});
        
        static_assert(std::is_same_v<DataCategory<decltype(col)>, CategoryTags::Scalar>);
        
        auto eval = Evaluate(col);
        CheckElement check{};
        for (size_t p = 0; p < 37; ++p)
        {
            check += ori[p].Value();
        }
        assert(fabs(check - eval.Value()) < 0.001);
        cout << "done" << endl;
    }
    
    void test_scalar_collapse_case5()
    {
        cout << "Test collapse: batch matrix -> scalar\t";
        auto ori = GenBatchMatrix<CheckElement>(7, 10, 3);
        auto col = Collapse(ori, Shape<CategoryTags::Scalar>{});
        
        static_assert(std::is_same_v<DataCategory<decltype(col)>, CategoryTags::Scalar>);
        
        auto eval = Evaluate(col);
        CheckElement check{};
        
        for (size_t p = 0; p < 7; ++p)
        {
            for (size_t i = 0; i < 10; ++i)
            {
                for (size_t j = 0; j < 3; ++j)
                {
                    check += ori[p](i, j);
                }
            }
        }
        assert(fabs(check - eval.Value()) < 0.001);
        cout << "done" << endl;
    }
    
    void test_scalar_collapse_case6()
    {
        cout << "Test collapse: batch 3d-array -> scalar\t";
        auto ori = GenBatchThreeDArray<CheckElement>(7, 5, 10, 3);
        auto col = Collapse(ori, Shape<CategoryTags::Scalar>{});
        
        static_assert(std::is_same_v<DataCategory<decltype(col)>, CategoryTags::Scalar>);
        
        auto eval = Evaluate(col);
        CheckElement check{};

        for (size_t b = 0; b < 7; ++b)
        {
            for (size_t p = 0; p < 5; ++p)
            {
                for (size_t i = 0; i < 10; ++i)
                {
                    for (size_t j = 0; j < 3; ++j)
                    {
                        check += ori[b](p, i, j);
                    }
                }
            }
        }
        assert(fabs(check - eval.Value()) < 0.001);
        cout << "done" << endl;
    }
    
    void test_scalar_collapse_case7()
    {
        cout << "Test collapse: scalar sequence -> scalar\t";
        auto ori = GenScalarSequence<CheckElement>(37);
        auto col = Collapse(ori, Shape<CategoryTags::Scalar>{});
        
        static_assert(std::is_same_v<DataCategory<decltype(col)>, CategoryTags::Scalar>);
        
        auto eval = Evaluate(col);
        CheckElement check{};
        for (size_t p = 0; p < 37; ++p)
        {
            check += ori[p].Value();
        }
        assert(fabs(check - eval.Value()) < 0.001);
        cout << "done" << endl;
    }
    
    void test_scalar_collapse_case8()
    {
        cout << "Test collapse: matrix sequence -> scalar\t";
        auto ori = GenMatrixSequence<CheckElement>(7, 10, 3);
        auto col = Collapse(ori, Shape<CategoryTags::Scalar>{});
        
        static_assert(std::is_same_v<DataCategory<decltype(col)>, CategoryTags::Scalar>);
        
        auto eval = Evaluate(col);
        CheckElement check{};
        
        for (size_t p = 0; p < 7; ++p)
        {
            for (size_t i = 0; i < 10; ++i)
            {
                for (size_t j = 0; j < 3; ++j)
                {
                    check += ori[p](i, j);
                }
            }
        }
        assert(fabs(check - eval.Value()) < 0.001);
        cout << "done" << endl;
    }
    
    void test_scalar_collapse_case9()
    {
        cout << "Test collapse: 3d-array sequence -> scalar\t";
        auto ori = GenThreeDArraySequence<CheckElement>(7, 5, 10, 3);
        auto col = Collapse(ori, Shape<CategoryTags::Scalar>{});
        
        static_assert(std::is_same_v<DataCategory<decltype(col)>, CategoryTags::Scalar>);
        
        auto eval = Evaluate(col);
        CheckElement check{};

        for (size_t b = 0; b < 7; ++b)
        {
            for (size_t p = 0; p < 5; ++p)
            {
                for (size_t i = 0; i < 10; ++i)
                {
                    for (size_t j = 0; j < 3; ++j)
                    {
                        check += ori[b](p, i, j);
                    }
                }
            }
        }
        assert(fabs(check - eval.Value()) < 0.001);
        cout << "done" << endl;
    }
    
    void test_scalar_collapse_case10()
    {
        cout << "Test collapse: batch scalar sequence -> scalar\t";
        auto ori = GenBatchScalarSequence<CheckElement>(std::vector{3, 5, 7});
        auto col = Collapse(ori, Shape<CategoryTags::Scalar>{});
        
        static_assert(std::is_same_v<DataCategory<decltype(col)>, CategoryTags::Scalar>);
        
        auto eval = Evaluate(col);
        CheckElement check{};
        for (size_t p = 0; p < 3; ++p)
        {
            check += ori[0][p].Value();
        }
        for (size_t p = 0; p < 5; ++p)
        {
            check += ori[1][p].Value();
        }
        for (size_t p = 0; p < 7; ++p)
        {
            check += ori[2][p].Value();
        }
        assert(fabs(check - eval.Value()) < 0.001);
        cout << "done" << endl;
    }
    
    void test_scalar_collapse_case11()
    {
        cout << "Test collapse: batch matrix sequence -> scalar\t";
        auto ori = GenBatchMatrixSequence<CheckElement>(std::vector{3, 5, 7}, 10, 4);
        auto col = Collapse(ori, Shape<CategoryTags::Scalar>{});
        
        static_assert(std::is_same_v<DataCategory<decltype(col)>, CategoryTags::Scalar>);
        
        auto eval = Evaluate(col);
        CheckElement check{};
        for (size_t i = 0; i < 10; ++i)
        {
            for (size_t j = 0; j < 4; ++j)
            {
                for (size_t p = 0; p < 3; ++p)
                {
                    check += ori[0][p](i, j);
                }
                for (size_t p = 0; p < 5; ++p)
                {
                    check += ori[1][p](i, j);
                }
                for (size_t p = 0; p < 7; ++p)
                {
                    check += ori[2][p](i, j);
                }
            }
        }
        assert(fabs(check - eval.Value()) < 0.001);
        cout << "done" << endl;
    }
    
    void test_scalar_collapse_case12()
    {
        cout << "Test collapse: batch 3d-array sequence -> scalar\t";
        auto ori = GenBatchThreeDArraySequence<CheckElement>(std::vector{3, 5, 7}, 7, 10, 4);
        auto col = Collapse(ori, Shape<CategoryTags::Scalar>{});
        
        static_assert(std::is_same_v<DataCategory<decltype(col)>, CategoryTags::Scalar>);
        
        auto eval = Evaluate(col);
        CheckElement check{};
        for (size_t p = 0; p < 7; ++p)
        {
            for (size_t i = 0; i < 10; ++i)
            {
                for (size_t j = 0; j < 4; ++j)
                {
                    for (size_t s = 0; s < 3; ++s)
                    {
                        check += ori[0][s](p, i, j);
                    }
                    for (size_t s = 0; s < 5; ++s)
                    {
                        check += ori[1][s](p, i, j);
                    }
                    for (size_t s = 0; s < 7; ++s)
                    {
                        check += ori[2][s](p, i, j);
                    }
                }
            }
        }
        assert(fabs(check - eval.Value()) < 0.001);
        cout << "done" << endl;
    }
    
    void test_matrix_collapse_case1()
    {
        cout << "Test collapse: matrix -> matrix (plain collapse)\t";
        
        Matrix<CheckElement, CheckDevice> ori(3, 7);
        auto col = Collapse(ori, ori.Shape());
        assert(col == ori);
        cout << "done" << endl;
    }
    
    void test_matrix_collapse_case2()
    {
        cout << "Test collapse: 3d-array -> matrix\t";
        
        auto ori = GenThreeDArray<CheckElement>(10, 7, 3);
        auto col = Collapse(ori, Shape<CategoryTags::Matrix>(7, 3));
        static_assert(std::is_same_v<DataCategory<decltype(col)>, CategoryTags::Matrix>);
        assert(col.Shape().RowNum() == 7);
        assert(col.Shape().ColNum() == 3);
        
        auto eval = Evaluate(col);
        assert(col.Shape().RowNum() == 7);
        assert(col.Shape().ColNum() == 3);
        for (size_t i = 0; i < 7; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                CheckElement check{};
                for (size_t p = 0; p < 10; ++p)
                {
                    check += ori(p, i, j);
                }
                assert(fabs(check - eval(i, j)) < 0.001);
            }
        }
        cout << "done" << endl;
    }
    
    void test_matrix_collapse_case3()
    {
        cout << "Test collapse: batch matrix -> matrix\t";
        
        auto ori = GenBatchMatrix<CheckElement>(10, 7, 3);
        auto col = Collapse(ori, Shape<CategoryTags::Matrix>(7, 3));
        static_assert(std::is_same_v<DataCategory<decltype(col)>, CategoryTags::Matrix>);
        assert(col.Shape().RowNum() == 7);
        assert(col.Shape().ColNum() == 3);
        
        auto eval = Evaluate(col);
        assert(eval.Shape().RowNum() == 7);
        assert(eval.Shape().ColNum() == 3);
        for (size_t i = 0; i < 7; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                CheckElement check{};
                for (size_t p = 0; p < 10; ++p)
                {
                    check += ori[p](i, j);
                }
                assert(fabs(check - eval(i, j)) < 0.001);
            }
        }
        cout << "done" << endl;
    }
    
    void test_matrix_collapse_case4()
    {
        cout << "Test collapse: batch 3d-array -> matrix\t";
        
        auto ori = GenBatchThreeDArray<CheckElement>(10, 5, 7, 3);
        auto col = Collapse(ori, Shape<CategoryTags::Matrix>(7, 3));
        static_assert(std::is_same_v<DataCategory<decltype(col)>, CategoryTags::Matrix>);
        assert(col.Shape().RowNum() == 7);
        assert(col.Shape().ColNum() == 3);
        
        auto eval = Evaluate(col);
        assert(eval.Shape().RowNum() == 7);
        assert(eval.Shape().ColNum() == 3);
        for (size_t i = 0; i < 7; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                CheckElement check{};
                for (size_t b = 0; b < 10; ++b)
                {
                    for (size_t p = 0; p < 5; ++p)
                    {
                        check += ori[b](p, i, j);
                    }
                }
                assert(fabs(check - eval(i, j)) < 0.001);
            }
        }
        cout << "done" << endl;
    }
    
    void test_matrix_collapse_case5()
    {
        cout << "Test collapse: matrix sequence -> matrix\t";
        
        auto ori = GenMatrixSequence<CheckElement>(10, 7, 3);
        auto col = Collapse(ori, Shape<CategoryTags::Matrix>(7, 3));
        static_assert(std::is_same_v<DataCategory<decltype(col)>, CategoryTags::Matrix>);
        assert(col.Shape().RowNum() == 7);
        assert(col.Shape().ColNum() == 3);
        
        auto eval = Evaluate(col);
        assert(eval.Shape().RowNum() == 7);
        assert(eval.Shape().ColNum() == 3);
        for (size_t i = 0; i < 7; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                CheckElement check{};
                for (size_t p = 0; p < 10; ++p)
                {
                    check += ori[p](i, j);
                }
                assert(fabs(check - eval(i, j)) < 0.001);
            }
        }
        cout << "done" << endl;
    }
    
    void test_matrix_collapse_case6()
    {
        cout << "Test collapse: 3d-array sequence -> matrix\t";
        
        auto ori = GenThreeDArraySequence<CheckElement>(10, 5, 7, 3);
        auto col = Collapse(ori, Shape<CategoryTags::Matrix>(7, 3));
        static_assert(std::is_same_v<DataCategory<decltype(col)>, CategoryTags::Matrix>);
        assert(col.Shape().RowNum() == 7);
        assert(col.Shape().ColNum() == 3);
        
        auto eval = Evaluate(col);
        assert(eval.Shape().RowNum() == 7);
        assert(eval.Shape().ColNum() == 3);
        
        for (size_t i = 0; i < 7; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                CheckElement check{};
                for (size_t b = 0; b < 10; ++b)
                {
                    for (size_t p = 0; p < 5; ++p)
                    {
                        check += ori[b](p, i, j);
                    }
                }
                assert(fabs(check - eval(i, j)) < 0.001);
            }
        }
        cout << "done" << endl;
    }
    
    void test_matrix_collapse_case7()
    {
        cout << "Test collapse: batch matrix sequence -> matrix\t";
        
        auto ori = GenBatchMatrixSequence<CheckElement>(std::vector{3, 7, 11}, 7, 3);
        auto col = Collapse(ori, Shape<CategoryTags::Matrix>(7, 3));
        static_assert(std::is_same_v<DataCategory<decltype(col)>, CategoryTags::Matrix>);
        assert(col.Shape().RowNum() == 7);
        assert(col.Shape().ColNum() == 3);
        
        auto eval = Evaluate(col);
        assert(eval.Shape().RowNum() == 7);
        assert(eval.Shape().ColNum() == 3);
        
        for (size_t i = 0; i < 7; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                CheckElement check{};
                for (size_t p = 0; p < 3; ++p)
                {
                    check += ori[0][p](i, j);
                }
                for (size_t p = 0; p < 7; ++p)
                {
                    check += ori[1][p](i, j);
                }
                for (size_t p = 0; p < 11; ++p)
                {
                    check += ori[2][p](i, j);
                }
                assert(fabs(check - eval(i, j)) < 0.001);
            }
        }
        cout << "done" << endl;
    }
    
    void test_matrix_collapse_case8()
    {
        cout << "Test collapse: batch 3d-array sequence -> matrix\t";
        
        auto ori = GenBatchThreeDArraySequence<CheckElement>(std::vector{3, 7, 11}, 5, 7, 3);
        auto col = Collapse(ori, Shape<CategoryTags::Matrix>(7, 3));
        static_assert(std::is_same_v<DataCategory<decltype(col)>, CategoryTags::Matrix>);
        assert(col.Shape().RowNum() == 7);
        assert(col.Shape().ColNum() == 3);
        
        auto eval = Evaluate(col);
        assert(eval.Shape().RowNum() == 7);
        assert(eval.Shape().ColNum() == 3);
        
        for (size_t i = 0; i < 7; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                CheckElement check{};
                for (size_t p = 0; p < 5; ++p)
                {
                    for (size_t s = 0; s < 3; ++s)
                    {
                        check += ori[0][s](p, i, j);
                    }
                    for (size_t s = 0; s < 7; ++s)
                    {
                        check += ori[1][s](p, i, j);
                    }
                    for (size_t s = 0; s < 11; ++s)
                    {
                        check += ori[2][s](p, i, j);
                    }
                }
                assert(fabs(check - eval(i, j)) < 0.001);
            }
        }
        cout << "done" << endl;
    }
    
    void test_3d_array_collapse_case1()
    {
        cout << "Test collapse: 3d-array -> 3d-array (plain collapse)\t";
        
        ThreeDArray<CheckElement, CheckDevice> ori(5, 7, 3);
        auto col = Collapse(ori, ori.Shape());
        assert(col == ori);
        cout << "done" << endl;
    }
    
    void test_3d_array_collapse_case2()
    {
        cout << "Test collapse: batch 3d-array -> 3d-array\t";
        
        auto ori = GenBatchThreeDArray<CheckElement>(10, 5, 7, 3);
        auto col = Collapse(ori, Shape<CategoryTags::ThreeDArray>(5, 7, 3));
        static_assert(std::is_same_v<DataCategory<decltype(col)>, CategoryTags::ThreeDArray>);
        assert(col.Shape().PageNum() == 5);
        assert(col.Shape().RowNum() == 7);
        assert(col.Shape().ColNum() == 3);
        
        auto eval = Evaluate(col);
        assert(eval.Shape().PageNum() == 5);
        assert(eval.Shape().RowNum() == 7);
        assert(eval.Shape().ColNum() == 3);
        
        for (size_t p = 0; p < 5; ++p)
        {
            for (size_t i = 0; i < 7; ++i)
            {
                for (size_t j = 0; j < 3; ++j)
                {
                    CheckElement check{};
                    for (size_t b = 0; b < 10; ++b)
                    {
                        check += ori[b](p, i, j);
                    }
                    assert(fabs(check - eval(p, i, j)) < 0.001);
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_3d_array_collapse_case3()
    {
        cout << "Test collapse: 3d-array sequence -> 3d-array\t";
        
        auto ori = GenThreeDArraySequence<CheckElement>(10, 5, 7, 3);
        auto col = Collapse(ori, Shape<CategoryTags::ThreeDArray>(5, 7, 3));
        static_assert(std::is_same_v<DataCategory<decltype(col)>, CategoryTags::ThreeDArray>);
        assert(col.Shape().PageNum() == 5);
        assert(col.Shape().RowNum() == 7);
        assert(col.Shape().ColNum() == 3);
        
        auto eval = Evaluate(col);
        assert(eval.Shape().PageNum() == 5);
        assert(eval.Shape().RowNum() == 7);
        assert(eval.Shape().ColNum() == 3);
        
        for (size_t p = 0; p < 5; ++p)
        {
            for (size_t i = 0; i < 7; ++i)
            {
                for (size_t j = 0; j < 3; ++j)
                {
                    CheckElement check{};
                    for (size_t b = 0; b < 10; ++b)
                    {
                        check += ori[b](p, i, j);
                    }
                    assert(fabs(check - eval(p, i, j)) < 0.001);
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_3d_array_collapse_case4()
    {
        cout << "Test collapse: batch 3d-array sequence -> 3d-array\t";
        
        auto ori = GenBatchThreeDArraySequence<CheckElement>(std::vector{3, 7, 11}, 5, 7, 3);
        auto col = Collapse(ori, Shape<CategoryTags::ThreeDArray>(5, 7, 3));
        static_assert(std::is_same_v<DataCategory<decltype(col)>, CategoryTags::ThreeDArray>);
        assert(col.Shape().PageNum() == 5);
        assert(col.Shape().RowNum() == 7);
        assert(col.Shape().ColNum() == 3);
        
        auto eval = Evaluate(col);
        assert(eval.Shape().PageNum() == 5);
        assert(eval.Shape().RowNum() == 7);
        assert(eval.Shape().ColNum() == 3);
        
        for (size_t p = 0; p < 5; ++p)
        {
            for (size_t i = 0; i < 7; ++i)
            {
                for (size_t j = 0; j < 3; ++j)
                {
                    CheckElement check{};
                    for (size_t s = 0; s < 3; ++s)
                    {
                        check += ori[0][s](p, i, j);
                    }
                    for (size_t s = 0; s < 7; ++s)
                    {
                        check += ori[1][s](p, i, j);
                    }
                    for (size_t s = 0; s < 11; ++s)
                    {
                        check += ori[2][s](p, i, j);
                    }
                    assert(fabs(check - eval(p, i, j)) < 0.001);
                }
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Operators
{
    void test_collapse()
    {
        test_scalar_collapse_case1();
        test_scalar_collapse_case2();
        test_scalar_collapse_case3();
        test_scalar_collapse_case4();
        test_scalar_collapse_case5();
        test_scalar_collapse_case6();
        test_scalar_collapse_case7();
        test_scalar_collapse_case8();
        test_scalar_collapse_case9();
        test_scalar_collapse_case10();
        test_scalar_collapse_case11();
        test_scalar_collapse_case12();
        
        test_matrix_collapse_case1();
        test_matrix_collapse_case2();
        test_matrix_collapse_case3();
        test_matrix_collapse_case4();
        test_matrix_collapse_case5();
        test_matrix_collapse_case6();
        test_matrix_collapse_case7();
        test_matrix_collapse_case8();
        
        test_3d_array_collapse_case1();
        test_3d_array_collapse_case2();
        test_3d_array_collapse_case3();
        test_3d_array_collapse_case4();
    }
}