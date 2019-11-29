#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_constant_filler1()
    {
        cout << "test constant filler case 1 ...";
        ConstantFiller filler;
        Matrix<CheckElement, CheckDevice> data(3, 7);
        filler.Fill(data);
        
        assert(data.Shape().RowNum() == 3);
        assert(data.Shape().ColNum() == 7);
        
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 7; ++j)
            {
                assert(fabs(data(i, j)) < 0.001f);
            }
        }
        cout << "done" << endl;
    }
    
    void test_constant_filler2()
    {
        cout << "test constant filler case 2 ...";
        ConstantFiller filler(3);
        Matrix<CheckElement, CheckDevice> data(3, 7);
        filler.Fill(data);
        
        assert(data.Shape().RowNum() == 3);
        assert(data.Shape().ColNum() == 7);
        
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 7; ++j)
            {
                assert(fabs(data(i, j) - 3) < 0.001f);
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Model::ParamInitializer
{
    void test_constant_filler()
    {
        test_constant_filler1();
        test_constant_filler2();
    }
}