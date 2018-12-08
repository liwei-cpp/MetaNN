#pragma once

#include <MetaNN/data/facilities/shape.h>
#include <MetaNN/data/facilities/traits.h>
#include <cassert>

namespace MetaNN
{
// data category calculation
template <typename TOpTag, typename... TOperands>
struct OperCategory_
{
    static_assert(DependencyFalse<TOpTag>, "Operand container is empty");
};

template <typename TOpTag, typename THeadCate, typename...TRemainCate>
struct OperCategory_<TOpTag, THeadCate, TRemainCate...>
{
    static_assert((std::is_same_v<THeadCate, TRemainCate> && ...), "Data category mismatch.");
    using type = THeadCate;
};

template <typename TOpTag, typename... TOperands>
using OperCateCal = typename OperCategory_<TOpTag, DataCategory<TOperands>...>::type;

// ElementType
template <typename TOpTag, typename...TOperands>
struct OperElementType_
{
    static_assert(DependencyFalse<TOpTag>, "Operand container is empty");
};

template <typename TOpTag, typename TOp1, typename...TOperands>
struct OperElementType_<TOpTag, TOp1, TOperands...>
{
    using type = typename TOp1::ElementType;
};

// DeviceType
template <typename TOpTag, typename...TOperands>
struct OperDeviceType_
{
    static_assert(DependencyFalse<TOpTag>, "Operand container is empty");
};

template <typename TOpTag, typename TOp1, typename...TOperands>
struct OperDeviceType_<TOpTag, TOp1, TOperands...>
{
    using type = typename TOp1::DeviceType;
};

// operator auxiliary parameters
template <typename TOpTag, typename TCate>
class OperAuxParams
{
public:
    bool operator == (const OperAuxParams&) const
    {
        return true;
    }
};

// Shape
template <typename TOpTag, typename TCate>
class OperShapeInfo
{
public:
    template <typename THead, typename...TRemain>
    OperShapeInfo(const OperAuxParams<TOpTag, TCate>&, const THead& head, const TRemain&... rem)
        : m_shape(head.Shape())
    {
        static_assert((std::is_same_v<decltype(m_shape), decltype(rem.Shape())> && ...));
        assert(((m_shape == rem.Shape()) && ...));
    }
    
    const auto& Shape() const
    {
        return m_shape;
    }
    
private:
    MetaNN::Shape<TCate> m_shape;
};


// operator calculate sequence container
template <typename...TCases>
struct OperSeqContainer;

template <typename TOpTag>
struct OperSeq_;
}