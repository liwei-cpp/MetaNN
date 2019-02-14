#pragma once

#include <operators/loss/test_nll_loss.h>
#include <operators/loss/test_nll_loss_grad.h>

namespace Test::Operators
{
    void test_loss_operators()
    {
        test_nll_loss();
        test_nll_loss_grad();
    }
}