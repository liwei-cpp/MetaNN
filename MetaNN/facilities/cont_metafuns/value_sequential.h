#pragma once
#include <MetaNN/facilities/cont_metafuns/helpers.h>

namespace MetaNN::ValueSequential
{
    template <typename TValueSeq, auto val>
    struct Contains_;
    
    template <template<auto...> class TValueCont, auto val, auto... vals>
    struct Contains_<TValueCont<vals...>, val>
    {
        constexpr static bool value = ((vals == val) || ...);
    };
    
    template <typename TValueSeq, auto val>
    constexpr static bool Contains = Contains_<TValueSeq, val>::value;
}