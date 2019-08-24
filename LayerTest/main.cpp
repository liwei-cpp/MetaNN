#include <compose/_.h>
#include <elementary/_.h>
#include <facilities/_.h>
#include <loss/_.h>
#include <src_rec/_.h>

int main()
{
    Test::Layer::Compose::Test();
    Test::Layer::Elementary::Test();
    Test::Layer::Facilities::Test();
    Test::Layer::Loss::Test();
    Test::Layer::SrcRec::test();
}
