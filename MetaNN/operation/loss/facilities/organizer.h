#pragma once

#include <MetaNN/data/facilities/category_tags.h>
#include <MetaNN/data/facilities/traits.h>

namespace MetaNN
{
struct GenLossOperCategory_
{
    using type = CategoryTags::Tensor<0>;
};

class GenLossOperShapeInfo
{
public:
    template <typename TOpTag, typename THead, typename...TRemain>
    GenLossOperShapeInfo(const OperAuxParams<TOpTag,
                                             typename THead::ElementType,
                                             CategoryTags::Tensor<0>>&,
                         const THead& head, const TRemain&... rem)
    { }
    
    const auto& Shape() const
    {
        static MetaNN::Shape<0> inst;
        return inst;
    }
};
}