#include <facilities/test_facilities.h>
#include <elementary/test_elementary.h>
#include <loss/test_loss.h>

int main()
{
    Test::Layer::test_facilities();
    Test::Layer::test_elementary();
    Test::Layer::test_loss();
}
