#pragma once

namespace Test::Data::Cardinal::Matrix
{
    void test_matrix();
    void test_trival_matrix();
    void test_vector();
    void test_one_hot_vector();

    inline void test()
    {
        test_matrix();
        test_trival_matrix();
        test_vector();
        test_one_hot_vector();
    }
}
