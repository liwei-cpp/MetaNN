#include <MetaNN/meta_nn.h>
using namespace MetaNN;

namespace
{
template <typename... Params>
struct MyVector;

namespace TestAt
{
namespace Case0
{
    using Check1 = MyVector<int, short, double>;
    static_assert(std::is_same_v<Sequential::At<Check1, 0>, int>);
    static_assert(std::is_same_v<Sequential::At<Check1, 1>, short>);
    static_assert(std::is_same_v<Sequential::At<Check1, 2>, double>);
}
}

namespace TestOrder
{
namespace Case0
{
    using Check1 = MyVector<int, short, double>;
    static_assert(Sequential::Order<Check1, int> == 0);
    static_assert(Sequential::Order<Check1, short> == 1);
    static_assert(Sequential::Order<Check1, double> == 2);
}
}

namespace TestSet
{
namespace Case0
{
    using Check1 = MyVector<int, short, double>;

    using Res1 = Sequential::Set<Check1, 0, bool>;
    static_assert(std::is_same_v<Res1, MyVector<bool, short, double>>);

    using Res2 = Sequential::Set<Check1, 1, bool>;
    static_assert(std::is_same_v<Res2, MyVector<int, bool, double>>);

    using Res3 = Sequential::Set<Check1, 2, bool>;
    static_assert(std::is_same_v<Res3, MyVector<int, short, bool>>);
}
}

namespace TestTransform
{
namespace Case0
{
    using Check1 = MyVector<int, short, double>;
    using Res1 = Sequential::Transform<Check1, std::add_lvalue_reference, std::tuple>;
    static_assert(std::is_same_v<Res1, std::tuple<int&, short&, double&>>);
}
}
}

