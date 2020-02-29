#include <MetaNN/meta_nn.h>
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
    using CheckRes = Map::CreateFromItems<Check1, std::add_lvalue_reference>;
    static_assert(std::is_same_v<CheckRes,
                                 std::tuple<Helper::KVBinder<int&, int>,
                                            Helper::KVBinder<short&, short>,
                                            Helper::KVBinder<double&, double>>>);
}
}

namespace TestFind
{
namespace Case0
{
    using Check1 = std::tuple<Helper::KVBinder<int, int*>,
                              Helper::KVBinder<short, short*>,
                              Helper::KVBinder<double, double*>>;
    static_assert(std::is_same_v<Map::Find<Check1, int>, int*>);
    static_assert(std::is_same_v<Map::Find<Check1, double>, double*>);
    static_assert(std::is_same_v<Map::Find<Check1, bool>, void>);
}
}
}