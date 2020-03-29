#pragma once

namespace Test::Operation::Loss
{
    void test_nll_loss();   void test_nll_loss_grad();

    void test()
    {
        test_nll_loss();
        test_nll_loss_grad();
    }
}