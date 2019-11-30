#include <MetaNN/meta_nn.h>
using namespace MetaNN;

namespace
{
    template <typename... Params>
    struct MyVector;

    namespace TestEqual
    {
        using Check1 = std::tuple<int, double, short>;
        using Check2 = MyVector<double, short, int>;
        static_assert(Set::IsEqual<Check1, Check2>);
        
        using Check3 = std::tuple<int, double>;
        static_assert(!Set::IsEqual<Check1, Check3>);
        
        using Check4 = std::tuple<int, double, char>;
        static_assert(!Set::IsEqual<Check1, Check4>);
    }
}