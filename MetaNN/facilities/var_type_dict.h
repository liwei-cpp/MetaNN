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
        using Keys = VarTypeDict;
        
        template <typename TKey>
        using ValueType = Sequential::At<Values, Sequential::Order<VarTypeDict, TKey>>;
        
        template <typename TKey>
        constexpr static bool IsValueEmpty = std::is_same_v<ValueType<TKey>, NullParameter>;

    public:
        Values() = default;
        
        Values(Values&& val)
        {
            for (size_t i = 0; i < sizeof...(TTypes); ++i)
            {
                m_tuple[i] = std::move(val.m_tuple[i]);
            }
        }
        
        Values(const Values&) = default;
        Values& operator= (const Values&) = default;
        Values& operator= (Values&&) = default;

        Values(std::shared_ptr<void> (&&input)[sizeof...(TTypes)])
        {
            for (size_t i = 0; i < sizeof...(TTypes); ++i)
            {
                m_tuple[i] = std::move(input[i]);
            }
        }

    public:
        template <typename TTag, typename... TParams>
        void Update(TParams&&... p_params)
        {
            constexpr static auto TagPos = Sequential::Order<VarTypeDict, TTag>;
            using rawVal = Sequential::At<Values, TagPos>;
            rawVal* tmp = new rawVal(std::forward<TParams>(p_params)...);
            m_tuple[TagPos] = std::shared_ptr<void>(tmp,
                                    [](void* ptr){
                                        rawVal* nptr = static_cast<rawVal*>(ptr);
                                        delete nptr;
                                    });
            return;
        }

        template <typename TTag, typename TVal>
        auto Set(TVal&& val) &&
        {
            constexpr static auto TagPos = Sequential::Order<VarTypeDict, TTag>;

            using rawVal = RemConstRef<TVal>;
            rawVal* tmp = new rawVal(std::forward<TVal>(val));
            m_tuple[TagPos] = std::shared_ptr<void>(tmp,
                                    [](void* ptr){
                                        rawVal* nptr = static_cast<rawVal*>(ptr);
                                        delete nptr;
                                    });
            
            if constexpr (std::is_same_v<rawVal, Sequential::At<Values, TagPos>>)
            {
                return *this;
            }
            else
            {
                using new_type = Sequential::Set<Values, TagPos, rawVal>;
                return new_type(std::move(m_tuple));
            }
        }
        
        template <typename TTag>
        const auto& Get() const
        {
            constexpr static auto TagPos = Sequential::Order<VarTypeDict, TTag>;
            using AimType = Sequential::At<Values, TagPos>;

            void* tmp = m_tuple[TagPos].get();
            if (!tmp)
                throw std::runtime_error("Empty Value.");
            AimType* res = static_cast<AimType*>(tmp);
            return *res;
        }
        
        template <typename TTag>
        auto& Get()
        {
            constexpr static auto TagPos = Sequential::Order<VarTypeDict, TTag>;
            using AimType = Sequential::At<Values, TagPos>;

            void* tmp = m_tuple[TagPos].get();
            if (!tmp)
                throw std::runtime_error("Empty Value.");

            AimType* res = static_cast<AimType*>(tmp);
            return *res;
        }

    private:
        std::shared_ptr<void> m_tuple[(sizeof...(TTypes) == 0) ? 1 : sizeof...(TTypes)];
    };

public:
    static auto Create()
    {
        using type = Sequential::Create<Values, NullParameter, sizeof...(TParameters)>;
        return type{};
    }
};
}
