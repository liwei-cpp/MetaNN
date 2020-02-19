#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <cassert>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_shape1()
    {
        cout << "Test shape case 1 (equality)...\t";
        Shape<3> sh;
        assert(sh[0] == 0);
        assert(sh[1] == 0);
        assert(sh[2] == 0);
        Shape<3> sh2;
        assert(sh == sh2);

        assert(sh != Shape(1,2,3));
        
        assert(sh != Shape<4>());
        
        assert(Shape(1,2,3) == Shape<3>(1,2,3));
        cout << "done" << endl;
    }
    
    void test_shape2()
    {
        cout << "Test shape case 2 (count)...\t";
        Shape<3> sh;
        static_assert(sh.DimNum == 3);
        assert(sh.Count() == 0);

        Shape<3> sh2(7, 2, 3);
        static_assert(sh2.DimNum == 3);
        assert(sh2.Count() == 42);
        
        Shape<0> sh3;
        static_assert(sh3.DimNum == 0);
        assert(sh3.Count() == 1);
        cout << "done" << endl;
    }
    
    void test_shape3()
    {
        cout << "Test shape case 3 (index to offset)...\t";
        Shape<3> sh(7, 2, 3);
        
        size_t id = 0;
        for (size_t i = 0; i < 7; ++i)
        {
            for (size_t j = 0; j < 2; ++j)
            {
                for (size_t k = 0; k < 3; ++k)
                {
                    assert(sh.IndexToOffset(i, j, k) == id);
                    ++id;
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_shape4()
    {
        cout << "Test shape case 4 (index access)...\t";
        Shape<3> sh(7, 2, 3);
        assert(sh[0] == 7);
        assert(sh[1] == 2);
        assert(sh[2] == 3);
        cout << "done" << endl;
    }
    
    void test_shape5()
    {
        cout << "Test shape case 5 (OffsetToIndex)...\t";
        Shape<3> sh(7, 2, 3);
        
        size_t input = 0;
        for (size_t i = 0; i < 7; ++i)
        {
            for (size_t j = 0; j < 2; ++j)
            {
                for (size_t k = 0; k < 3; ++k)
                {
                    auto res = sh.OffsetToIndex(input);
                    assert(res[0] == i);
                    assert(res[1] == j);
                    assert(res[2] == k);
                    ++input;
                }
            }
        }
        cout << "done" << endl;
    }
    
    void test_shape6()
    {
        cout << "Test shape case 6 (Shift Index with non-negative offset)...\t";
        Shape<3> sh(7, 2, 3);
        size_t totalCount = sh.Count();
        
        for (size_t i = 0; i < 7; ++i)
        {
            for (size_t j = 0; j < 2; ++j)
            {
                for (size_t k = 0; k < 3; ++k)
                {
                    size_t curPos = sh.IndexToOffset(i, j, k);
                    size_t checkBuf = totalCount - curPos;
                    for (size_t c = 0; c < checkBuf; ++c)
                    {
                        std::array<size_t, 3> idx{i, j, k};
                        sh.ShiftIndex(idx, c);
                        assert(sh.IndexToOffset(idx) == curPos + c);
                    }
                }
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Data::Facilities
{
    void test_shape()
    {
        test_shape1();
        test_shape2();
        test_shape3();
        test_shape4();
        test_shape5();
        test_shape6();
    }
}