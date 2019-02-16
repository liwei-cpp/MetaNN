#include <data_gen.h>
#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <cmath>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_scalar_duplicate_case1()
    {
        cout << "Test duplicate: scalar -> scalar (plain duplicate)\t";
        
        Scalar<CheckElement, CheckDevice> ori;
        auto dup = Duplicate(ori, ori.Shape());
        assert(dup == ori);
        cout << "done" << endl;
    }
    
    void test_scalar_duplicate_case2()
    {
        cout << "Test duplicate: scalar -> matrix\t";
        Scalar<CheckElement, CheckDevice> ori(3);
        
        auto dup = Duplicate(ori, Shape<CategoryTags::Matrix>(10, 3));
        static_assert(std::is_same_v<DataCategory<decltype(dup)>, CategoryTags::Matrix>);
        assert(dup.Shape().RowNum() == 10);
        assert(dup.Shape().ColNum() == 3);
        
        auto eval = Evaluate(dup);
        assert(eval.Shape().RowNum() == 10);
        assert(eval.Shape().ColNum() == 3);
        for (size_t i = 0; i < 10; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                assert(fabs(eval(i, j) - 3) < 0.001);
            }
        }

        cout << "done" << endl;
    }
    
    void test_scalar_duplicate_case3()
    {
        cout << "Test duplicate: scalar -> 3d-array\t";
        Scalar<CheckElement, CheckDevice> ori(3);
        
        auto dup = Duplicate(ori, Shape<CategoryTags::ThreeDArray>(7, 10, 3));
        static_assert(std::is_same_v<DataCategory<decltype(dup)>, CategoryTags::ThreeDArray>);
        assert(dup.Shape().PageNum() == 7);
        assert(dup.Shape().RowNum() == 10);
        assert(dup.Shape().ColNum() == 3);
        
        auto eval = Evaluate(dup);
        assert(eval.Shape().PageNum() == 7);
        assert(eval.Shape().RowNum() == 10);
        assert(eval.Shape().ColNum() == 3);
        for (size_t p = 0; p < 7; ++p)
        {
            for (size_t i = 0; i < 10; ++i)
            {
                for (size_t j = 0; j < 3; ++j)
                {
                    assert(fabs(eval(p, i, j) - 3) < 0.001);
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_scalar_duplicate_case4()
    {
        cout << "Test duplicate: scalar -> batch scalar\t";
        Scalar<CheckElement, CheckDevice> ori(3);
        
        auto dup = Duplicate(ori, Shape<CategoryTags::BatchScalar>(7));
        static_assert(std::is_same_v<DataCategory<decltype(dup)>, CategoryTags::BatchScalar>);
        assert(dup.Shape().BatchNum() == 7);
        
        auto eval = Evaluate(dup);
        assert(eval.Shape().BatchNum() == 7);
        for (size_t p = 0; p < 7; ++p)
        {
            assert(fabs(eval[p].Value() - 3) < 0.001);
        }
        cout << "done" << endl;
    }
    
    void test_scalar_duplicate_case5()
    {
        cout << "Test duplicate: scalar -> batch matrix\t";
        Scalar<CheckElement, CheckDevice> ori(3);
        
        auto dup = Duplicate(ori, Shape<CategoryTags::BatchMatrix>(7, 10, 3));
        static_assert(std::is_same_v<DataCategory<decltype(dup)>, CategoryTags::BatchMatrix>);
        assert(dup.Shape().BatchNum() == 7);
        assert(dup.Shape().RowNum() == 10);
        assert(dup.Shape().ColNum() == 3);
        
        auto eval = Evaluate(dup);
        assert(eval.Shape().BatchNum() == 7);
        assert(eval.Shape().RowNum() == 10);
        assert(eval.Shape().ColNum() == 3);
        for (size_t p = 0; p < 7; ++p)
        {
            for (size_t i = 0; i < 10; ++i)
            {
                for (size_t j = 0; j < 3; ++j)
                {
                    assert(fabs(eval[p](i, j) - 3) < 0.001);
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_scalar_duplicate_case6()
    {
        cout << "Test duplicate: scalar -> batch 3d-array\t";
        Scalar<CheckElement, CheckDevice> ori(3);
        
        auto dup = Duplicate(ori, Shape<CategoryTags::BatchThreeDArray>(7, 5, 10, 3));
        static_assert(std::is_same_v<DataCategory<decltype(dup)>, CategoryTags::BatchThreeDArray>);
        assert(dup.Shape().BatchNum() == 7);
        assert(dup.Shape().PageNum() == 5);
        assert(dup.Shape().RowNum() == 10);
        assert(dup.Shape().ColNum() == 3);
        
        auto eval = Evaluate(dup);
        assert(eval.Shape().BatchNum() == 7);
        assert(eval.Shape().PageNum() == 5);
        assert(eval.Shape().RowNum() == 10);
        assert(eval.Shape().ColNum() == 3);
        for (size_t b = 0; b < 7; ++b)
        {
            for (size_t p = 0; p < 5; ++p)
            {
                for (size_t i = 0; i < 10; ++i)
                {
                    for (size_t j = 0; j < 3; ++j)
                    {
                        assert(fabs(eval[b](p, i, j) - 3) < 0.001);
                    }
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_scalar_duplicate_case7()
    {
        cout << "Test duplicate: scalar -> scalar sequence\t";
        Scalar<CheckElement, CheckDevice> ori(3);
        
        auto dup = Duplicate(ori, Shape<CategoryTags::ScalarSequence>(7));
        static_assert(std::is_same_v<DataCategory<decltype(dup)>, CategoryTags::ScalarSequence>);
        assert(dup.Shape().Length() == 7);
        
        auto eval = Evaluate(dup);
        assert(eval.Shape().Length() == 7);
        for (size_t p = 0; p < 7; ++p)
        {
            assert(fabs(eval[p].Value() - 3) < 0.001);
        }
        cout << "done" << endl;
    }
    
    void test_scalar_duplicate_case8()
    {
        cout << "Test duplicate: scalar -> matrix sequence\t";
        Scalar<CheckElement, CheckDevice> ori(3);
        
        auto dup = Duplicate(ori, Shape<CategoryTags::MatrixSequence>(7, 10, 3));
        static_assert(std::is_same_v<DataCategory<decltype(dup)>, CategoryTags::MatrixSequence>);
        assert(dup.Shape().Length() == 7);
        assert(dup.Shape().RowNum() == 10);
        assert(dup.Shape().ColNum() == 3);
        
        auto eval = Evaluate(dup);
        assert(eval.Shape().Length() == 7);
        assert(eval.Shape().RowNum() == 10);
        assert(eval.Shape().ColNum() == 3);
        for (size_t p = 0; p < 7; ++p)
        {
            for (size_t i = 0; i < 10; ++i)
            {
                for (size_t j = 0; j < 3; ++j)
                {
                    assert(fabs(eval[p](i, j) - 3) < 0.001);
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_scalar_duplicate_case9()
    {
        cout << "Test duplicate: scalar -> 3d-array sequence\t";
        Scalar<CheckElement, CheckDevice> ori(3);
        
        auto dup = Duplicate(ori, Shape<CategoryTags::ThreeDArraySequence>(7, 5, 10, 3));
        static_assert(std::is_same_v<DataCategory<decltype(dup)>, CategoryTags::ThreeDArraySequence>);
        assert(dup.Shape().Length() == 7);
        assert(dup.Shape().PageNum() == 5);
        assert(dup.Shape().RowNum() == 10);
        assert(dup.Shape().ColNum() == 3);
        
        auto eval = Evaluate(dup);
        assert(eval.Shape().Length() == 7);
        assert(eval.Shape().PageNum() == 5);
        assert(eval.Shape().RowNum() == 10);
        assert(eval.Shape().ColNum() == 3);
        for (size_t b = 0; b < 7; ++b)
        {
            for (size_t p = 0; p < 5; ++p)
            {
                for (size_t i = 0; i < 10; ++i)
                {
                    for (size_t j = 0; j < 3; ++j)
                    {
                        assert(fabs(eval[b](p, i, j) - 3) < 0.001);
                    }
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_scalar_duplicate_case10()
    {
        cout << "Test duplicate: scalar -> batch scalar sequence\t";
        Scalar<CheckElement, CheckDevice> ori(3);
        
        auto dup = Duplicate(ori, Shape<CategoryTags::BatchScalarSequence>({3, 5, 7}));
        static_assert(std::is_same_v<DataCategory<decltype(dup)>, CategoryTags::BatchScalarSequence>);
        assert(dup.Shape().SeqLenContainer().size() == 3);
        assert(dup.Shape().SeqLenContainer()[0] == 3);
        assert(dup.Shape().SeqLenContainer()[1] == 5);
        assert(dup.Shape().SeqLenContainer()[2] == 7);
        
        auto eval = Evaluate(dup);
        assert(eval.Shape().SeqLenContainer().size() == 3);
        assert(eval.Shape().SeqLenContainer()[0] == 3);
        assert(eval.Shape().SeqLenContainer()[1] == 5);
        assert(eval.Shape().SeqLenContainer()[2] == 7);
        for (size_t p = 0; p < 3; ++p)
        {
            assert(fabs(eval[0][p].Value() - 3) < 0.001);
        }
        for (size_t p = 0; p < 5; ++p)
        {
            assert(fabs(eval[1][p].Value() - 3) < 0.001);
        }
        for (size_t p = 0; p < 7; ++p)
        {
            assert(fabs(eval[2][p].Value() - 3) < 0.001);
        }
        cout << "done" << endl;
    }
    
    void test_scalar_duplicate_case11()
    {
        cout << "Test duplicate: scalar -> batch matrix sequence\t";
        Scalar<CheckElement, CheckDevice> ori(3);
        
        auto dup = Duplicate(ori, Shape<CategoryTags::BatchMatrixSequence>({3, 5, 7}, 10, 4));
        static_assert(std::is_same_v<DataCategory<decltype(dup)>, CategoryTags::BatchMatrixSequence>);
        assert(dup.Shape().SeqLenContainer().size() == 3);
        assert(dup.Shape().SeqLenContainer()[0] == 3);
        assert(dup.Shape().SeqLenContainer()[1] == 5);
        assert(dup.Shape().SeqLenContainer()[2] == 7);
        assert(dup.Shape().RowNum() == 10);
        assert(dup.Shape().ColNum() == 4);
        
        auto eval = Evaluate(dup);
        assert(eval.Shape().SeqLenContainer().size() == 3);
        assert(eval.Shape().SeqLenContainer()[0] == 3);
        assert(eval.Shape().SeqLenContainer()[1] == 5);
        assert(eval.Shape().SeqLenContainer()[2] == 7);
        assert(eval.Shape().RowNum() == 10);
        assert(eval.Shape().ColNum() == 4);
        
        for (size_t i = 0; i < 10; ++i)
        {
            for (size_t j = 0; j < 4; ++j)
            {
                for (size_t p = 0; p < 3; ++p)
                {
                    assert(fabs(eval[0][p](i, j) - 3) < 0.001);
                }
                for (size_t p = 0; p < 5; ++p)
                {
                    assert(fabs(eval[1][p](i, j) - 3) < 0.001);
                }
                for (size_t p = 0; p < 7; ++p)
                {
                    assert(fabs(eval[2][p](i, j) - 3) < 0.001);
                }
            }
        }
        
        cout << "done" << endl;
    }
    
    void test_scalar_duplicate_case12()
    {
        cout << "Test duplicate: scalar -> batch 3d-array sequence\t";
        Scalar<CheckElement, CheckDevice> ori(3);
        
        auto dup = Duplicate(ori, Shape<CategoryTags::BatchThreeDArraySequence>({3, 5, 7}, 7, 10, 4));
        static_assert(std::is_same_v<DataCategory<decltype(dup)>, CategoryTags::BatchThreeDArraySequence>);
        assert(dup.Shape().SeqLenContainer().size() == 3);
        assert(dup.Shape().SeqLenContainer()[0] == 3);
        assert(dup.Shape().SeqLenContainer()[1] == 5);
        assert(dup.Shape().SeqLenContainer()[2] == 7);
        assert(dup.Shape().PageNum() == 7);
        assert(dup.Shape().RowNum() == 10);
        assert(dup.Shape().ColNum() == 4);
        
        auto eval = Evaluate(dup);
        assert(eval.Shape().SeqLenContainer().size() == 3);
        assert(eval.Shape().SeqLenContainer()[0] == 3);
        assert(eval.Shape().SeqLenContainer()[1] == 5);
        assert(eval.Shape().SeqLenContainer()[2] == 7);
        assert(dup.Shape().PageNum() == 7);
        assert(eval.Shape().RowNum() == 10);
        assert(eval.Shape().ColNum() == 4);
        
        for (size_t p = 0; p < 7; ++p)
        {
            for (size_t i = 0; i < 10; ++i)
            {
                for (size_t j = 0; j < 4; ++j)
                {
                    for (size_t s = 0; s < 3; ++s)
                    {
                        assert(fabs(eval[0][s](p, i, j) - 3) < 0.001);
                    }
                    for (size_t s = 0; s < 5; ++s)
                    {
                        assert(fabs(eval[1][s](p, i, j) - 3) < 0.001);
                    }
                    for (size_t s = 0; s < 7; ++s)
                    {
                        assert(fabs(eval[2][s](p, i, j) - 3) < 0.001);
                    }
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_matrix_duplicate_case1()
    {
        cout << "Test duplicate: matrix -> matrix (plain duplicate)\t";
        
        Matrix<CheckElement, CheckDevice> ori(3, 7);
        auto dup = Duplicate(ori, ori.Shape());
        assert(dup == ori);
        cout << "done" << endl;
    }
    
    void test_matrix_duplicate_case2()
    {
        cout << "Test duplicate: matrix -> 3d-array\t";
        
        auto ori = GenMatrix<CheckElement>(7, 3);
        auto dup = Duplicate(ori, Shape<CategoryTags::ThreeDArray>(10, 7, 3));
        static_assert(std::is_same_v<DataCategory<decltype(dup)>, CategoryTags::ThreeDArray>);
        assert(dup.Shape().PageNum() == 10);
        assert(dup.Shape().RowNum() == 7);
        assert(dup.Shape().ColNum() == 3);
        
        auto eval = Evaluate(dup);
        assert(eval.Shape().PageNum() == 10);
        assert(eval.Shape().RowNum() == 7);
        assert(eval.Shape().ColNum() == 3);
        for (size_t p = 0; p < 10; ++p)
        {
            for (size_t i = 0; i < 7; ++i)
            {
                for (size_t j = 0; j < 3; ++j)
                {
                    assert(fabs(eval(p, i, j) - ori(i, j)) < 0.001);
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_matrix_duplicate_case3()
    {
        cout << "Test duplicate: matrix -> batch matrix\t";
        
        auto ori = GenMatrix<CheckElement>(7, 3);
        auto dup = Duplicate(ori, Shape<CategoryTags::BatchMatrix>(10, 7, 3));
        static_assert(std::is_same_v<DataCategory<decltype(dup)>, CategoryTags::BatchMatrix>);
        assert(dup.Shape().BatchNum() == 10);
        assert(dup.Shape().RowNum() == 7);
        assert(dup.Shape().ColNum() == 3);
        
        auto eval = Evaluate(dup);
        assert(eval.Shape().BatchNum() == 10);
        assert(eval.Shape().RowNum() == 7);
        assert(eval.Shape().ColNum() == 3);
        for (size_t p = 0; p < 10; ++p)
        {
            for (size_t i = 0; i < 7; ++i)
            {
                for (size_t j = 0; j < 3; ++j)
                {
                    assert(fabs(eval[p](i, j) - ori(i, j)) < 0.001);
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_matrix_duplicate_case4()
    {
        cout << "Test duplicate: matrix -> batch 3d-array\t";
        
        auto ori = GenMatrix<CheckElement>(7, 3);
        auto dup = Duplicate(ori, Shape<CategoryTags::BatchThreeDArray>(10, 5, 7, 3));
        static_assert(std::is_same_v<DataCategory<decltype(dup)>, CategoryTags::BatchThreeDArray>);
        assert(dup.Shape().BatchNum() == 10);
        assert(dup.Shape().PageNum() == 5);
        assert(dup.Shape().RowNum() == 7);
        assert(dup.Shape().ColNum() == 3);
        
        auto eval = Evaluate(dup);
        assert(eval.Shape().BatchNum() == 10);
        assert(eval.Shape().PageNum() == 5);
        assert(eval.Shape().RowNum() == 7);
        assert(eval.Shape().ColNum() == 3);
        for (size_t b = 0; b < 10; ++b)
        {
            for (size_t p = 0; p < 5; ++p)
            {
                for (size_t i = 0; i < 7; ++i)
                {
                    for (size_t j = 0; j < 3; ++j)
                    {
                        assert(fabs(eval[b](p, i, j) - ori(i, j)) < 0.001);
                    }
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_matrix_duplicate_case5()
    {
        cout << "Test duplicate: matrix -> matrix sequence\t";
        
        auto ori = GenMatrix<CheckElement>(7, 3);
        auto dup = Duplicate(ori, Shape<CategoryTags::MatrixSequence>(10, 7, 3));
        static_assert(std::is_same_v<DataCategory<decltype(dup)>, CategoryTags::MatrixSequence>);
        assert(dup.Shape().Length() == 10);
        assert(dup.Shape().RowNum() == 7);
        assert(dup.Shape().ColNum() == 3);
        
        auto eval = Evaluate(dup);
        assert(eval.Shape().Length() == 10);
        assert(eval.Shape().RowNum() == 7);
        assert(eval.Shape().ColNum() == 3);
        for (size_t p = 0; p < 10; ++p)
        {
            for (size_t i = 0; i < 7; ++i)
            {
                for (size_t j = 0; j < 3; ++j)
                {
                    assert(fabs(eval[p](i, j) - ori(i, j)) < 0.001);
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_matrix_duplicate_case6()
    {
        cout << "Test duplicate: matrix -> 3d-array sequence\t";
        
        auto ori = GenMatrix<CheckElement>(7, 3);
        auto dup = Duplicate(ori, Shape<CategoryTags::ThreeDArraySequence>(10, 5, 7, 3));
        static_assert(std::is_same_v<DataCategory<decltype(dup)>, CategoryTags::ThreeDArraySequence>);
        assert(dup.Shape().Length() == 10);
        assert(dup.Shape().PageNum() == 5);
        assert(dup.Shape().RowNum() == 7);
        assert(dup.Shape().ColNum() == 3);
        
        auto eval = Evaluate(dup);
        assert(eval.Shape().Length() == 10);
        assert(eval.Shape().PageNum() == 5);
        assert(eval.Shape().RowNum() == 7);
        assert(eval.Shape().ColNum() == 3);
        for (size_t b = 0; b < 10; ++b)
        {
            for (size_t p = 0; p < 5; ++p)
            {
                for (size_t i = 0; i < 7; ++i)
                {
                    for (size_t j = 0; j < 3; ++j)
                    {
                        assert(fabs(eval[b](p, i, j) - ori(i, j)) < 0.001);
                    }
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_matrix_duplicate_case7()
    {
        cout << "Test duplicate: matrix -> batch matrix sequence\t";
        
        auto ori = GenMatrix<CheckElement>(7, 3);
        auto dup = Duplicate(ori, Shape<CategoryTags::BatchMatrixSequence>({3, 7, 11}, 7, 3));
        static_assert(std::is_same_v<DataCategory<decltype(dup)>, CategoryTags::BatchMatrixSequence>);
        assert(dup.Shape().SeqLenContainer().size() == 3);
        assert(dup.Shape().SeqLenContainer()[0] == 3);
        assert(dup.Shape().SeqLenContainer()[1] == 7);
        assert(dup.Shape().SeqLenContainer()[2] == 11);
        assert(dup.Shape().RowNum() == 7);
        assert(dup.Shape().ColNum() == 3);
        
        auto eval = Evaluate(dup);
        assert(eval.Shape().SeqLenContainer().size() == 3);
        assert(eval.Shape().SeqLenContainer()[0] == 3);
        assert(eval.Shape().SeqLenContainer()[1] == 7);
        assert(eval.Shape().SeqLenContainer()[2] == 11);
        assert(eval.Shape().RowNum() == 7);
        assert(eval.Shape().ColNum() == 3);
        
        for (size_t i = 0; i < 7; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                for (size_t p = 0; p < 3; ++p)
                {
                    assert(fabs(eval[0][p](i, j) - ori(i, j)) < 0.001);
                }
                for (size_t p = 0; p < 7; ++p)
                {
                    assert(fabs(eval[1][p](i, j) - ori(i, j)) < 0.001);
                }
                for (size_t p = 0; p < 11; ++p)
                {
                    assert(fabs(eval[2][p](i, j) - ori(i, j)) < 0.001);
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_matrix_duplicate_case8()
    {
        cout << "Test duplicate: matrix -> batch 3d-array sequence\t";
        
        auto ori = GenMatrix<CheckElement>(7, 3);
        auto dup = Duplicate(ori, Shape<CategoryTags::BatchThreeDArraySequence>({3, 7, 11}, 5, 7, 3));
        static_assert(std::is_same_v<DataCategory<decltype(dup)>, CategoryTags::BatchThreeDArraySequence>);
        assert(dup.Shape().SeqLenContainer().size() == 3);
        assert(dup.Shape().SeqLenContainer()[0] == 3);
        assert(dup.Shape().SeqLenContainer()[1] == 7);
        assert(dup.Shape().SeqLenContainer()[2] == 11);
        assert(dup.Shape().PageNum() == 5);
        assert(dup.Shape().RowNum() == 7);
        assert(dup.Shape().ColNum() == 3);
        
        auto eval = Evaluate(dup);
        assert(eval.Shape().SeqLenContainer().size() == 3);
        assert(eval.Shape().SeqLenContainer()[0] == 3);
        assert(eval.Shape().SeqLenContainer()[1] == 7);
        assert(eval.Shape().SeqLenContainer()[2] == 11);
        assert(eval.Shape().PageNum() == 5);
        assert(eval.Shape().RowNum() == 7);
        assert(eval.Shape().ColNum() == 3);
        for (size_t p = 0; p < 5; ++p)
        {
            for (size_t i = 0; i < 7; ++i)
            {
                for (size_t j = 0; j < 3; ++j)
                {
                    for (size_t s = 0; s < 3; ++s)
                    {
                        assert(fabs(eval[0][s](p, i, j) - ori(i, j)) < 0.001);
                    }
                    for (size_t s = 0; s < 7; ++s)
                    {
                        assert(fabs(eval[1][s](p, i, j) - ori(i, j)) < 0.001);
                    }
                    for (size_t s = 0; s < 11; ++s)
                    {
                        assert(fabs(eval[2][s](p, i, j) - ori(i, j)) < 0.001);
                    }
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_3d_array_duplicate_case1()
    {
        cout << "Test duplicate: 3d-array -> 3d-array (plain duplicate)\t";
        
        ThreeDArray<CheckElement, CheckDevice> ori(5, 7, 3);
        auto dup = Duplicate(ori, ori.Shape());
        assert(dup == ori);
        cout << "done" << endl;
    }
    
    void test_3d_array_duplicate_case2()
    {
        cout << "Test duplicate: 3d-array -> batch 3d-array\t";
        
        auto ori = GenThreeDArray<CheckElement>(5, 7, 3);
        auto dup = Duplicate(ori, Shape<CategoryTags::BatchThreeDArray>(10, 5, 7, 3));
        static_assert(std::is_same_v<DataCategory<decltype(dup)>, CategoryTags::BatchThreeDArray>);
        assert(dup.Shape().BatchNum() == 10);
        assert(dup.Shape().PageNum() == 5);
        assert(dup.Shape().RowNum() == 7);
        assert(dup.Shape().ColNum() == 3);
        
        auto eval = Evaluate(dup);
        assert(eval.Shape().BatchNum() == 10);
        assert(eval.Shape().PageNum() == 5);
        assert(eval.Shape().RowNum() == 7);
        assert(eval.Shape().ColNum() == 3);
        for (size_t b = 0; b < 10; ++b)
        {
            for (size_t p = 0; p < 5; ++p)
            {
                for (size_t i = 0; i < 7; ++i)
                {
                    for (size_t j = 0; j < 3; ++j)
                    {
                        assert(fabs(eval[b](p, i, j) - ori(p, i, j)) < 0.001);
                    }
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_3d_array_duplicate_case3()
    {
        cout << "Test duplicate: 3d-array -> 3d-array sequence\t";
        
        auto ori = GenThreeDArray<CheckElement>(5, 7, 3);
        auto dup = Duplicate(ori, Shape<CategoryTags::ThreeDArraySequence>(10, 5, 7, 3));
        static_assert(std::is_same_v<DataCategory<decltype(dup)>, CategoryTags::ThreeDArraySequence>);
        assert(dup.Shape().Length() == 10);
        assert(dup.Shape().PageNum() == 5);
        assert(dup.Shape().RowNum() == 7);
        assert(dup.Shape().ColNum() == 3);
        
        auto eval = Evaluate(dup);
        assert(eval.Shape().Length() == 10);
        assert(eval.Shape().PageNum() == 5);
        assert(eval.Shape().RowNum() == 7);
        assert(eval.Shape().ColNum() == 3);
        for (size_t b = 0; b < 10; ++b)
        {
            for (size_t p = 0; p < 5; ++p)
            {
                for (size_t i = 0; i < 7; ++i)
                {
                    for (size_t j = 0; j < 3; ++j)
                    {
                        assert(fabs(eval[b](p, i, j) - ori(p, i, j)) < 0.001);
                    }
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_3d_array_duplicate_case4()
    {
        cout << "Test duplicate: 3d-array -> batch 3d-array sequence\t";
        
        auto ori = GenThreeDArray<CheckElement>(5, 7, 3);
        auto dup = Duplicate(ori, Shape<CategoryTags::BatchThreeDArraySequence>({3, 7, 11}, 5, 7, 3));
        static_assert(std::is_same_v<DataCategory<decltype(dup)>, CategoryTags::BatchThreeDArraySequence>);
        assert(dup.Shape().SeqLenContainer().size() == 3);
        assert(dup.Shape().SeqLenContainer()[0] == 3);
        assert(dup.Shape().SeqLenContainer()[1] == 7);
        assert(dup.Shape().SeqLenContainer()[2] == 11);
        assert(dup.Shape().PageNum() == 5);
        assert(dup.Shape().RowNum() == 7);
        assert(dup.Shape().ColNum() == 3);
        
        auto eval = Evaluate(dup);
        assert(eval.Shape().SeqLenContainer().size() == 3);
        assert(eval.Shape().SeqLenContainer()[0] == 3);
        assert(eval.Shape().SeqLenContainer()[1] == 7);
        assert(eval.Shape().SeqLenContainer()[2] == 11);
        assert(eval.Shape().PageNum() == 5);
        assert(eval.Shape().RowNum() == 7);
        assert(eval.Shape().ColNum() == 3);
        
        for (size_t p = 0; p < 5; ++p)
        {
            for (size_t i = 0; i < 7; ++i)
            {
                for (size_t j = 0; j < 3; ++j)
                {
                    for (size_t s = 0; s < 3; ++s)
                    {
                        assert(fabs(eval[0][s](p, i, j) - ori(p, i, j)) < 0.001);
                    }
                    for (size_t s = 0; s < 7; ++s)
                    {
                        assert(fabs(eval[1][s](p, i, j) - ori(p, i, j)) < 0.001);
                    }
                    for (size_t s = 0; s < 11; ++s)
                    {
                        assert(fabs(eval[2][s](p, i, j) - ori(p, i, j)) < 0.001);
                    }
                }
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Operators
{
    void test_duplicate()
    {
        test_scalar_duplicate_case1();
        test_scalar_duplicate_case2();
        test_scalar_duplicate_case3();
        test_scalar_duplicate_case4();
        test_scalar_duplicate_case5();
        test_scalar_duplicate_case6();
        test_scalar_duplicate_case7();
        test_scalar_duplicate_case8();
        test_scalar_duplicate_case9();
        test_scalar_duplicate_case10();
        test_scalar_duplicate_case11();
        test_scalar_duplicate_case12();
        
        test_matrix_duplicate_case1();
        test_matrix_duplicate_case2();
        test_matrix_duplicate_case3();
        test_matrix_duplicate_case4();
        test_matrix_duplicate_case5();
        test_matrix_duplicate_case6();
        test_matrix_duplicate_case7();
        test_matrix_duplicate_case8();
        
        test_3d_array_duplicate_case1();
        test_3d_array_duplicate_case2();
        test_3d_array_duplicate_case3();
        test_3d_array_duplicate_case4();
    }
}