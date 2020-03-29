#pragma once
#include <MetaNN/data/facilities/category_tags.h>
#include <MetaNN/data/facilities/device_tags.h>
#include <MetaNN/facilities/traits.h>
#include <iterator>
#include <type_traits>

namespace MetaNN
{
    template <typename T>
    struct IsIterator_
    {   
        template <typename R>
        static std::true_type Test(typename std::iterator_traits<R>::iterator_category*);

        template <typename R>
        static std::false_type Test(...);

        static constexpr bool value = decltype(Test<T>(nullptr))::value;
    };

    template <typename T>
    constexpr bool IsIterator = IsIterator_<T>::value;

    template <typename T1, typename T2,
              std::enable_if_t<IsValidCategoryTag<typename T1::CategoryTag>>* = nullptr,
              std::enable_if_t<IsValidCategoryTag<typename T2::CategoryTag>>* = nullptr,
              std::enable_if_t<!std::is_same_v<T1, T2>>* = nullptr
            >
    bool operator== (const T1&, const T2&)
    {
        return false;
    }

    template <typename T1, typename T2,
              std::enable_if_t<IsValidCategoryTag<typename T1::CategoryTag>>* = nullptr,
              std::enable_if_t<IsValidCategoryTag<typename T1::CategoryTag>>* = nullptr
             >
    bool operator!= (const T1& val1, const T2& val2)
    {
        return !(val1 == val2);
    }
    
    template <typename TData>
    using ShapeType = RemConstRef<decltype(std::declval<TData>().Shape())>;
}
