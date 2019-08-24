#include <MetaNN/meta_nn2.h>
using namespace MetaNN;

namespace
{
namespace TestFind
{
namespace Case0
{
    using Map1 = ContMetaFun::MultiMap::Insert<std::tuple<>, int, double>;
    using Map2 = ContMetaFun::MultiMap::Insert<Map1, int, short>;
    using Check = ContMetaFun::MultiMap::Find<Map2, int>;
    static_assert(std::is_same_v<Check, ContMetaFun::Helper::ValueSequence<double, short>>);
}
}
}