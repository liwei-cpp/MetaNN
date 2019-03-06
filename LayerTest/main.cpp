#include <src_rec/_.h>

namespace Test::Layer
{
    void test_elementary();
    void test_facilities();
    void test_loss();
}

int main()
{
    Test::Layer::test_elementary();
    Test::Layer::test_facilities();
    Test::Layer::test_loss();
    Test::Layer::SrcRec::test();
}
