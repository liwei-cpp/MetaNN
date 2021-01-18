#include <data_gen.h>
#include <MetaNN/meta_nn.h>
#include <calculate_tags.h>
#include <cmath>
#include <iostream>
using namespace std;
using namespace MetaNN;

namespace
{
    void test_acosh_case1()
    {
        cout << "Test acosh case 1 (scalar)\t";
        {
            Scalar<CheckElement, CheckDevice> ori(1.2f);
            auto op = Acosh(ori);
            auto res = Evaluate(op);
            assert(fabs(res.Value() - 0.6224) < 0.001f);
        }
        cout << "done" << endl;
    }

    void test_acosh_case2()
    {
        cout << "Test acosh case 2 (vector)\t";
        {
            Vector<CheckElement, CheckDevice> ori(5);
            ori.SetValue(0, 1.2f);
            ori.SetValue(1, 3.0f);
            ori.SetValue(2, 4.0f);
            ori.SetValue(3, 5.0f);
            ori.SetValue(4, 6.0f);
            auto op = Acosh(ori);
            auto res = Evaluate(op);
            assert(fabs(res(0) - 0.6224) < 0.001f);
            assert(fabs(res(1) - 1.7627) < 0.001f);
            assert(fabs(res(2) - 2.0634) < 0.001f);
            assert(fabs(res(3) - 2.2924) < 0.001f);
            assert(fabs(res(4) - 2.4779) < 0.001f);
        }
        cout << "done" << endl;
    }
    
    void test_acosh_case3()
    {
        cout << "Test acosh case 3 (matrix)\t";
        auto ori = GenTensor<CheckElement>(static_cast<CheckElement>(1.1), static_cast<CheckElement>(0.01), 10, 7);
        auto op = Acosh(ori);
        static_assert(IsMatrix<decltype(op)>);
        assert(op.Shape()[0] == 10);
        assert(op.Shape()[1] == 7);
        
        auto res = Evaluate(op);
        static_assert(IsMatrix<decltype(res)>);
        assert(res.Shape()[0] == 10);
        assert(res.Shape()[1] == 7);
        
        for (size_t i = 0; i < 10; ++i)
        {
            for (size_t k = 0; k < 7; ++k)
            {
                auto value = std::acosh(ori(i, k));
                assert(fabs(res(i, k) - value) < 0.001f);
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Operation::Math
{
    void test_acosh()
    {
        test_acosh_case1();
        test_acosh_case2();
        test_acosh_case3();
    }
}

namespace
{
    void test_acosh_grad_case1()
    {
        cout << "Test acosh-grad case 1 (scalar)\t";
        {
            Scalar<CheckElement, CheckDevice> grad(1);
            Scalar<CheckElement, CheckDevice> ori(static_cast<CheckElement>(1.2));
            auto op = AcoshGrad(grad, ori);
            auto res = Evaluate(op);
            
            assert(fabs(res.Value() - 1.5076) < 0.001f);
        }
        cout << "done" << endl;
    }

    void test_acosh_grad_case2()
    {
        cout << "Test acosh-grad case 2 (vector)\t";
        {
            Scalar<CheckElement, CheckDevice> grad(1);
            Vector<CheckElement, CheckDevice> ori(5);
            ori.SetValue(0, 1.2f);
            ori.SetValue(1, 3.0f);
            ori.SetValue(2, 4.0f);
            ori.SetValue(3, 5.0f);
            ori.SetValue(4, 6.0f);
            auto op = AcoshGrad(grad, ori);
            auto res = Evaluate(op);
            
            assert(fabs(res(0) - 1.5076) < 0.001f);
            assert(fabs(res(1) - 0.3536) < 0.001f);
            assert(fabs(res(2) - 0.2582) < 0.001f);
            assert(fabs(res(3) - 0.2041) < 0.001f);
            assert(fabs(res(4) - 0.1690) < 0.001f);
        }
        cout << "done" << endl;
    }
    
    void test_acosh_grad_case3()
    {
        cout << "Test acosh-grad case 3 (matrix)\t";
        auto grad = GenTensor<CheckElement>(0, 1, 10, 7);
        auto ori = GenTensor<CheckElement>(static_cast<CheckElement>(1.5), static_cast<CheckElement>(0.01), 10, 7);
        auto op = AcoshGrad(grad, ori);
        static_assert(IsMatrix<decltype(op)>);
        assert(op.Shape()[0] == 10);
        assert(op.Shape()[1] == 7);
        
        auto res = Evaluate(op);
        static_assert(IsMatrix<decltype(res)>);
        assert(res.Shape()[0] == 10);
        assert(res.Shape()[1] == 7);
        
        for (size_t i = 0; i < 10; ++i)
        {
            for (size_t k = 0; k < 7; ++k)
            {
                auto check = grad(i, k) / std::sqrt(ori(i, k) * ori(i, k) - 1);
                assert(fabs(res(i, k) - check) < 0.001f);
            }
        }
        cout << "done" << endl;
    }
}

namespace Test::Operation::Math
{
    void test_acosh_grad()
    {
        test_acosh_grad_case1();
        test_acosh_grad_case2();
        test_acosh_grad_case3();
    }
}