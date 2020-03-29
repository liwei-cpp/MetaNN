#pragma once

#include <MetaNN/facilities/traits.h>
#include <type_traits>

namespace MetaNN
{
    /// data types
    namespace CategoryTags
    {
        struct OutOfCategory;
    
        template <size_t uDim>
        struct Tensor
        {
            constexpr static size_t DimNum = uDim;
        };
        
        using Scalar = Tensor<0>;
        using Vector = Tensor<1>;
        using Matrix = Tensor<2>;
    }

    template <typename T>
    constexpr bool IsValidCategoryTag = false;

    template <size_t uDim>
    constexpr bool IsValidCategoryTag<CategoryTags::Tensor<uDim>> = true;

    template <typename TElem, typename TDevice, size_t uDim> class Tensor;

    template <typename TElem, typename TDevice, size_t uDim>
    struct PrincipalDataType_
    {
        using type = Tensor<TElem, TDevice, uDim>;
    };

    template <typename TCategory, typename TElem, typename TDevice>
    using PrincipalDataType = typename PrincipalDataType_<TElem, TDevice, TCategory::DimNum>::type;

    template <typename T>
    struct DataCategory_
    {
        template <typename R>
        static typename R::CategoryTag Test(typename R::CategoryTag*);

        template <typename R>
        static CategoryTags::OutOfCategory Test(...);

        using type = decltype(Test<RemConstRef<T>>(nullptr));
    };

    template <typename T>
    using DataCategory = typename DataCategory_<T>::type;
    
    template <typename T, size_t uDim>
    constexpr bool IsTensorWithDim = std::is_same_v<DataCategory<T>, CategoryTags::Tensor<uDim>>;

    template <typename T>
    constexpr bool IsScalar = IsTensorWithDim<T, 0>;

    template <typename T>
    constexpr bool IsVector = IsTensorWithDim<T, 1>;

    template <typename T>
    constexpr bool IsMatrix = IsTensorWithDim<T, 2>;

    template <typename T>
    constexpr bool IsThreeDArray = IsTensorWithDim<T, 3>;
}
