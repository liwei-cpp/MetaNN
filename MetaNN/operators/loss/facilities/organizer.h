#pragma once

#include <MetaNN/data/facilities/tags.h>

namespace MetaNN
{
struct GenLossOperCategory_
{
    using type = CategoryTags::Scalar;
};

class GenLossOperShapeInfo
{
public:
    template <typename TOpTag, typename THead, typename...TRemain>
    GenLossOperShapeInfo(const OperAuxParams<TOpTag, CategoryTags::Scalar>&,
                         const THead& head, const TRemain&... rem)
    {
        static_assert((std::is_same_v<decltype(head.Shape()), decltype(rem.Shape())> && ...));
        assert(((head.Shape() == rem.Shape()) && ...));
    }
    
    const auto& Shape() const
    {
        static MetaNN::Shape<CategoryTags::Scalar> inst;
        return inst;
    }
};
}