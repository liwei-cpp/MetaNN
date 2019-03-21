#pragma once
#include <MetaNN/facilities/cont_metafuns/helpers.h>
#include <MetaNN/facilities/cont_metafuns/sequential.h>
#include <tuple>

namespace MetaNN::ContMetaFun::Map
{
// Create from items ======================================================================
namespace NSCreateFromItems
{
    template <template<typename> typename KeyPicker>
    struct KVCreator
    {
        template <typename TItem>
        struct apply
        {
            using type = ContMetaFun::Helper::KVBinder<typename KeyPicker<TItem>::type, TItem>;
        };
    };
}
    template <typename TItemCont, template<typename> typename KeyPicker,
              template<typename...> typename TOutCont>
    struct CreateFromItems_
    {
        template <typename TItem>
        using CurKVCreator = typename NSCreateFromItems::KVCreator<KeyPicker>::template apply<TItem>;

        using type
            = ContMetaFun::Sequential::Transform<TItemCont,
                                                 CurKVCreator,
                                                 TOutCont>;
    };
    
    template <typename TItemCont,
              template<typename> typename KeyPicker,
              template<typename...> typename TOutCont = std::tuple>
    using CreateFromItems = typename CreateFromItems_<TItemCont, KeyPicker, TOutCont>::type;
}