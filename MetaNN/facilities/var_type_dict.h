#pragma once

#include <MetaNN/facilities/null_param.h>
#include <MetaNN/facilities/traits.h>
#include <MetaNN/facilities/cont_metafuns/sequential.h>
#include <memory>
#include <type_traits>
#include <vector>

namespace MetaNN
{
template <typename...TParameters>
struct VarTypeDict
{
    template <typename...TTypes>
    struct Values
    {
    public:
        Values() = default;

        Values(std::shared_ptr<void> (&&input)[sizeof...(TTypes)])
        {
            for (size_t i = 0; i < sizeof...(TTypes); ++i)
            {
                m_tuple[i] = std::move(input[i]);
            }
        }

    public:
        template <typename TTag, typename TVal>
        auto Set(TVal&& val) &&
        {
            constexpr static auto TagPos = ContMetaFun::Sequential::Order<VarTypeDict, TTag>;
            
            using rawVal = std::decay_t<TVal>;
            rawVal* tmp = new rawVal(std::forward<TVal>(val));
            m_tuple[TagPos] = std::shared_ptr<void>(tmp,
                                    [](void* ptr){
                                        rawVal* nptr = static_cast<rawVal*>(ptr);
                                        delete nptr;
                                    });

            using new_type = ContMetaFun::Sequential::Set<Values, TagPos, rawVal>;
            return new_type(std::move(m_tuple));
        }
        
        template <typename TTag>
        auto& Get() const
        {
            constexpr static auto TagPos = ContMetaFun::Sequential::Order<VarTypeDict, TTag>;
            using AimType = ContMetaFun::Sequential::At<Values, TagPos>;

            void* tmp = m_tuple[TagPos].get();
            AimType* res = static_cast<AimType*>(tmp);
            return *res;
        }
        
        template <typename TTag>
        auto Get() &&
        {
            constexpr static auto TagPos = ContMetaFun::Sequential::Order<VarTypeDict, TTag>;
            using AimType = ContMetaFun::Sequential::At<Values, TagPos>;

            void* tmp = m_tuple[TagPos].get();
            AimType* res = static_cast<AimType*>(tmp);
            return std::move(*res);
        }
        
        template <typename TTag>
        using ValueType = ContMetaFun::Sequential::At<Values, ContMetaFun::Sequential::Order<VarTypeDict, TTag>>;

    private:
        std::shared_ptr<void> m_tuple[sizeof...(TTypes)];
    };

public:
    static auto Create()
    {
        using type = ContMetaFun::Sequential::Create<Values, NullParameter, sizeof...(TParameters)>;
        return type{};
    }
};
}
