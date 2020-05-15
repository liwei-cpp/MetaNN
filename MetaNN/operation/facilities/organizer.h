#pragma once

#include <MetaNN/data/facilities/shape.h>
#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/operation/facilities/instance_id.h>
#include <cassert>

namespace MetaNN
{
    template <typename... TOperands> struct OperandContainer;

    // operator validation check
    template <typename TOpTag, typename... TOperands>
    constexpr bool IsValidOper = ((IsValidCategoryTag<DataCategory<TOperands>>) && ...);

    // data category calculation
    template <typename TFirstCate, typename... TCategories>
    struct PickCommonCategory_
    {
        using type = TFirstCate;
    };

    template <typename TFirstCate, typename TSecondCate, typename... TRemainCates>
    struct PickCommonCategory_<TFirstCate, TSecondCate, TRemainCates...>
    {
        using TCompRes = std::conditional_t<(TFirstCate::DimNum > TSecondCate::DimNum),
                                            TFirstCate, TSecondCate>;
        using type = typename PickCommonCategory_<TCompRes, TRemainCates...>::type;
    };

    template <typename TOpTag, typename TPolicy, typename... TOperands>
    struct OperCategory_ : PickCommonCategory_<TOperands...>
    {};

    template <typename TOpTag, typename TPolicy, typename... TOperands>
    using OperCateCal = typename OperCategory_<TOpTag, TPolicy, DataCategory<TOperands>...>::type;

    // ElementType
    template <typename TOpTag, typename TOp1, typename...TOperands>
    struct OperElementType_
    {
        using type = typename TOp1::ElementType;
    };

    // DeviceType
    template <typename TOpTag, typename TOp1, typename...TOperands>
    struct OperDeviceType_
    {
        using type = typename TOp1::DeviceType;
    };

    // operator auxiliary parameters
    template <typename TOpTag, typename TElem, typename TCate>
    class OperAuxParams
    {
    public:
        bool operator == (const OperAuxParams&) const
        {
            return true;
        }
    };

    template <typename TValue>
    struct OperAuxValue
    {
    public:
        OperAuxValue(TValue val)
            : m_value(val)
            , m_instID(InstanceID::Get())
        {}

        const auto& Value() const
        {
            return m_value;
        }

        bool operator== (const OperAuxValue& val) const
        {
            return m_instID == val.m_instID;
        }

    private:
        TValue m_value;
        size_t m_instID;
    };

    template <typename TShape1, typename TShape2>
    bool IsBroadcastMatch(const TShape1& shape1, const TShape2& shape2)
    {
        if constexpr ((TShape1::DimNum == 0) || (TShape2::DimNum == 0))
        {
            return true;
        }
        else if constexpr(TShape1::DimNum > TShape2::DimNum)
        {
            return IsBroadcastMatch(shape2, shape1);
        }
        else
        {
            auto it1 = shape1.rbegin();
            auto it2 = shape2.rbegin();
            while (it1 != shape1.rend())
            {
                if (*it1 != *it2) return false;
                ++it1;
                ++it2;
            }
            return true;
        }
    }

    namespace NSOperShapeInfo
    {
        template <typename TShape>
        auto CommonShape(const TShape& shape)
        {
            return shape;
        }
        
        template <typename TShape1, typename TShape2, typename... TShapes>
        auto CommonShape(const TShape1& shape1, const TShape2& shape2, const TShapes&... shapes)
        {
            assert(IsBroadcastMatch(shape1, shape2));
            if constexpr (TShape1::DimNum > TShape2::DimNum)
            {
                return CommonShape(shape1, shapes...);
            }
            else
            {
                return CommonShape(shape2, shapes...);
            }
        }
    }

    // Shape
    template <typename TOpTag, typename TCate, typename TPolicies>
    class OperShapeInfo
    {
    public:
        template <typename TOperAuxParams, typename... TOperands>
        OperShapeInfo(const TOperAuxParams&, const TOperands&... operands)
            : m_shape(NSOperShapeInfo::CommonShape((operands.Shape())...))
        {}

        const auto& Shape() const
        {
            return m_shape;
        }

    private:
        MetaNN::Shape<TCate::DimNum> m_shape;
    };

    // operator calculate sequence container
    template <typename...TCases>
    struct OperCalAlgoChain;

    template <typename TOpTag>
    struct OperSeq_;
}