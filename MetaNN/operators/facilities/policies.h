#pragma once

#include <MetaNN/policies/policy_macro_begin.h>
namespace MetaNN
{
    struct DimPolicy
    {
        using MajorClass = DimPolicy;

        struct IsKeepDimValueCate;
        static constexpr bool IsKeepDim = false;
        
        struct DimArrayValueCate;
        struct DimBitArrayValueCate;
        
        struct ResDimNumValueCate;
        static constexpr size_t ResDimNum = 0;
    };
    ValuePolicyObj(PKeepDim,             DimPolicy, IsKeepDim, true);
    ValuePolicyObj(PNoKeepDIm,           DimPolicy, IsKeepDim, false);
    ValuePolicyTemplate(PKeepDimValueIs, DimPolicy, IsKeepDim);
    ValuePolicyTemplate(PResDimNumIs,    DimPolicy, ResDimNum);

    template <size_t... uDims>
    struct DimArrayIs : virtual public DimPolicy
    {
        using MinorClass = DimPolicy::DimArrayValueCate;
        static constexpr std::array<size_t, sizeof...(uDims)> DimArray{uDims...};
    };

    template <bool... uDims>
    struct DimBitArrayIs : virtual public DimPolicy
    {
        using MinorClass = DimPolicy::DimBitArrayValueCate;
        static constexpr std::array<bool, sizeof...(uDims)> DimBitArray{uDims...};
    };
}
#include <MetaNN/policies/policy_macro_end.h>