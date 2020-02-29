#include <compose/_.h>
#include <elementary/_.h>
#include <loss/_.h>
#include <math/_.h>
#include <recurrent/_.h>
#include <source/_.h>

int main()
{
    Test::Layer::Compose::Test();
    Test::Layer::Elementary::Test();
    Test::Layer::Loss::Test();
    Test::Layer::Math::Test();
    Test::Layer::Recurrent::Test();
    Test::Layer::Source::test();
}
