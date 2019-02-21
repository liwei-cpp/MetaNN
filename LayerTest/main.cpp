#include <facilities/test_facilities.h>
#include <elementary/test_elementary.h>
#include <loss/test_loss.h>
#include <src_rec/test_src_rec.h>

int main()
{
    Test::Layer::test_facilities();
    Test::Layer::test_elementary();
    Test::Layer::test_loss();
    Test::Layer::test_src_rec();
}
