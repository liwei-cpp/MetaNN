#include <MetaNN/meta_nn2.h>
using namespace MetaNN;

namespace
{
namespace TestFind
{
namespace Case0
{
    using Map1 = MultiMap::Insert<std::tuple<>, int, double>;
    using Map2 = MultiMap::Insert<Map1, int, short>;
    using Check = MultiMap::Find<Map2, int>;
    static_assert(std::is_same_v<Check, Helper::ValueSequence<double, short>>);
}
}
}