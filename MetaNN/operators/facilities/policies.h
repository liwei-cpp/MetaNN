#pragma once

namespace MetaNN
{
    struct DimPolicy;

    template <size_t... uDims>
    struct DimPolicyIs
    {
        using MajorClass = DimPolicy;
        using MinorClass = DimPolicy;

    public:
        static constexpr std::array<size_t, sizeof...(uDims)> DimArray{uDims...};
    };
}