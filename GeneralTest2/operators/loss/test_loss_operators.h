#pragma once

#include <operators/loss/test_nll_loss.h>

namespace Test::Operators
{
    void test_loss_operators()
    {
        test_nll_loss();
    }
}