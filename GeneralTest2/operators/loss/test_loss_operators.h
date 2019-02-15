#pragma once

namespace Test::Operators
{
    void test_nll_loss();   void test_nll_loss_grad();
    void test_loss_operators()
    {
        test_nll_loss();
        test_nll_loss_grad();
    }
}