#pragma once

#include <MetaNN/data/facilities/tags.h>
#include <MetaNN/data/facilities/traits.h>

namespace MetaNN
{
// FP
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

// BP
template <typename TOpTag, typename TOperGrad, typename TOperHead, typename... TOperands>
constexpr bool IsValidLossBP
    = (std::is_same_v<DataCategory<TOperHead>, DataCategory<TOperands>> && ...) && 
      (((IsScalar<TOperGrad>)      && (IsCardinal<TOperHead>)) ||
       ((IsBatchScalar<TOperGrad>) && (IsBatchCardinal<TOperHead>)) ||
       ((IsScalarSequence<TOperGrad>) && (IsCardinalSequence<TOperHead>)) ||
       ((IsBatchScalarSequence<TOperGrad>) && (IsBatchCardinalSequence<TOperHead>)));

template <typename TCate>
class GenLossBPOperShapeInfo
{
public:
    template <typename TAux, typename TGrad, typename THead, typename...TRemain>
    GenLossBPOperShapeInfo(const TAux&, const TGrad&, const THead& head, const TRemain&... rem)
        : m_shape(head.Shape())
    {
        static_assert((std::is_same_v<decltype(head.Shape()), decltype(rem.Shape())> && ...));
        assert(((m_shape == rem.Shape()) && ...));
    }
    
    const auto& Shape() const
    {
        return m_shape;
    }
    
private:
    MetaNN::Shape<TCate> m_shape;
};
}