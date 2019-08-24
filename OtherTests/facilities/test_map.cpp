#include <MetaNN/meta_nn2.h>
using namespace MetaNN;

namespace
{
template <typename... Params>
struct MyVector;

namespace TestCreateFromItems
{
namespace Case0
{
    using Check1 = MyVector<int, short, double>;
    using CheckRes = ContMetaFun::Map::CreateFromItems<Check1, std::add_lvalue_reference>;
    static_assert(std::is_same_v<CheckRes,
                                 std::tuple<ContMetaFun::Helper::KVBinder<int&, int>,
                                            ContMetaFun::Helper::KVBinder<short&, short>,
                                            ContMetaFun::Helper::KVBinder<double&, double>>>);
}
}

namespace TestFind
{
namespace Case0
{
    using Check1 = std::tuple<ContMetaFun::Helper::KVBinder<int, int*>,
                              ContMetaFun::Helper::KVBinder<short, short*>,
                              ContMetaFun::Helper::KVBinder<double, double*>>;
    static_assert(std::is_same_v<ContMetaFun::Map::Find<Check1, int>, int*>);
    static_assert(std::is_same_v<ContMetaFun::Map::Find<Check1, double>, double*>);
    static_assert(std::is_same_v<ContMetaFun::Map::Find<Check1, bool>, void>);
}
}
}