#include <data_gen.h>
#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <cmath>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_tile_case1()
    {
        cout << "Test tile case 1 (h/v expand)...\t";
        Shape<2> aimShape(2, 2);

        Vector<int, DeviceTags::CPU> ori(2);
        ori.SetValue(0, 0);
        ori.SetValue(1, 1);
        {
            auto op = Tile<PolicyContainer<DimArrayIs<0>>>(ori, aimShape);
            
            auto res = Evaluate(op);
            static_assert(IsMatrix<decltype(res)>);
            assert(res.Shape()[0] == 2);
            assert(res.Shape()[1] == 2);
            
            assert(res(0, 0) == 0);
            assert(res(0, 1) == 1);
            assert(res(1, 0) == 0);
            assert(res(1, 1) == 1);
        }
        
        {
            auto op = Tile<PolicyContainer<DimArrayIs<1>>>(ori, aimShape);
            
            auto res = Evaluate(op);
            static_assert(IsMatrix<decltype(res)>);
            assert(res.Shape()[0] == 2);
            assert(res.Shape()[1] == 2);
            
            assert(res(0, 0) == 0);
            assert(res(0, 1) == 0);
            assert(res(1, 0) == 1);
            assert(res(1, 1) == 1);
        }
        
        cout << "done" << endl;
    }

    void test_tile_case2()
    {
        cout << "Test tile case 2 (no expand)...\t";
        Matrix<int, DeviceTags::CPU> ori(2, 3);
        ori.SetValue(0, 0, 1); ori.SetValue(0, 1, 2); ori.SetValue(0, 2, 3);
        ori.SetValue(1, 0, 4); ori.SetValue(1, 1, 5); ori.SetValue(1, 2, 6);
        {
            auto op = Tile(ori, Shape<2>(2, 6));
            
            auto res = Evaluate(op);
            static_assert(IsMatrix<decltype(res)>);
            assert(res.Shape()[0] == 2);
            assert(res.Shape()[1] == 6);
            
            assert(res(0, 0) == 1); assert(res(0, 1) == 2); assert(res(0, 2) == 3);
            assert(res(0, 3) == 1); assert(res(0, 4) == 2); assert(res(0, 5) == 3);

            assert(res(1, 0) == 4); assert(res(1, 1) == 5); assert(res(1, 2) == 6);
            assert(res(1, 3) == 4); assert(res(1, 4) == 5); assert(res(1, 5) == 6);
        }
        
        {
            auto op = Tile(ori, Shape<2>(4, 3));
            
            auto res = Evaluate(op);
            static_assert(IsMatrix<decltype(res)>);
            assert(res.Shape()[0] == 4);
            assert(res.Shape()[1] == 3);
            
            assert(res(0, 0) == 1); assert(res(0, 1) == 2); assert(res(0, 2) == 3);
            assert(res(1, 0) == 4); assert(res(1, 1) == 5); assert(res(1, 2) == 6);
            assert(res(2, 0) == 1); assert(res(2, 1) == 2); assert(res(2, 2) == 3);
            assert(res(3, 0) == 4); assert(res(3, 1) == 5); assert(res(3, 2) == 6);            
        }
        cout << "done" << endl;
    }
    
    void test_tile_case3()
    {
        cout << "Test tile case 3 (general)...\t";
        auto ori = GenTensor<int>(0, 1, 2, 5, 3);
        auto op = Tile<PolicyContainer<DimArrayIs<0, 3>>>(ori, Shape<5>(7, 4, 15, 4, 3));
        static_assert(DataCategory<decltype(op)>::DimNum == 5);;
        assert(op.Shape() == Shape<5>(7, 4, 15, 4, 3));

        auto res = Evaluate(op);
        static_assert(DataCategory<decltype(res)>::DimNum == 5);;
        assert(res.Shape() == Shape<5>(7, 4, 15, 4, 3));

        for (size_t a1 = 0; a1 < 7; ++a1)
            for (size_t a2 = 0; a2 < 4; ++a2)
                for (size_t a3 = 0; a3 < 15; ++a3)
                    for (size_t a4 = 0; a4 < 4; ++a4)
                        for (size_t a5 = 0; a5 < 3; ++a5)
                        {
                            assert(res(a1, a2, a3, a4, a5) == ori(a2 % 2, a3 % 5, a5 % 3));
                        }
        cout << "done" << endl;
    }
}

namespace Test::Operators::Tensor
{
    void test_tile()
    {
        test_tile_case1();
        test_tile_case2();
        test_tile_case3();
    }
}