#pragma once

#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/evaluate/eval_plan.h>
#include <MetaNN/facilities/_.h>
#include <MetaNN/operation/facilities/_.h>
#include <MetaNN/policies/_.h>
#include <cassert>
#include <type_traits>

namespace MetaNN::OpTags
{
    struct Dot;
}

namespace MetaNN
{
namespace OperDot::NSCaseGen
{
    template <typename TInputHandle1, typename TInputHandle2, typename TOutputHandle, typename TPolicy>
    class EvalItem : public BaseEvalItem
    {
    public:
        using CategoryTag = CategoryTagFromHandle<TOutputHandle>;

        EvalItem(TInputHandle1 operand1, TInputHandle2 operand2, 
                 TOutputHandle outputHandle, Shape<CategoryTag::DimNum> shape)
            : BaseEvalItem(TypeID<EvalItem>(),
                           {operand1.DataPtr(), operand2.DataPtr()},
                           outputHandle.DataPtr())
            , m_operand1(std::move(operand1))
            , m_operand2(std::move(operand2))
            , m_outputHandle(std::move(outputHandle))
            , m_outputShape(std::move(shape))
        {}
        
        const TInputHandle1 m_operand1;
        const TInputHandle2 m_operand2;
        TOutputHandle m_outputHandle;
        Shape<CategoryTag::DimNum> m_outputShape;
    };

    template <typename TInputHandle1, typename TInputHandle2, typename TOutputHandle, typename TPolicy>
    class EvalGroup : public TrivialEvalGroup<EvalItem<TInputHandle1, TInputHandle2, TOutputHandle, TPolicy>>
    {
        using EvalItemType = EvalItem<TInputHandle1, TInputHandle2, TOutputHandle, TPolicy>;
    protected:
        virtual void EvalInternalLogic(EvalItemType& evalItem) final override
        {
            const auto& in1 = evalItem.m_operand1.Data();
            const auto& in2 = evalItem.m_operand2.Data();

            constexpr size_t modDimNum = PolicySelect<DimPolicy, TPolicy>::ModifyDimNum;
            size_t contractCount = 1;
            for (size_t id = 0; id < modDimNum; ++id)
            {
                contractCount *= in2.Shape()[id];
            }
            assert(in1.Shape().Count() % contractCount == 0);
            assert(in2.Shape().Count() % contractCount == 0);
            const size_t remCount1 = in1.Shape().Count() / contractCount;
            const size_t remCount2 = in2.Shape().Count() / contractCount;

            using ResType = typename TOutputHandle::DataType;
            using ElementType = typename ResType::ElementType;
            ResType out(evalItem.m_outputShape);

            auto low_in1 = LowerAccess(in1);
            const ElementType* mem_in1 = low_in1.RawMemory();

            auto low_in2 = LowerAccess(in2);
            const ElementType* mem_in2 = low_in2.RawMemory();

            auto low_out = LowerAccess(out);
            ElementType* mem_out = low_out.MutableRawMemory();

            static_assert(std::is_same_v<DeviceTypeFromHandle<TOutputHandle>, DeviceTags::CPU>, "Currently only CPU is supported");

            for (size_t i = 0; i < remCount1; ++i)
            {
                for (size_t j = 0; j < remCount2; ++j)
                {
                    mem_out[i * remCount2 + j] = 0;
                    for (size_t l = 0; l < contractCount; ++l)
                    {
                        mem_out[i * remCount2 + j] += mem_in1[i * contractCount + l] * mem_in2[l * remCount2 + j];
                    }
                }
            }
            evalItem.m_outputHandle.SetData(std::move(out));
        }
    };
}

    template <typename TOperand1, typename TOperand2>
    constexpr bool IsValidOper<OpTags::Dot, TOperand1, TOperand2> =
        (DataCategory<TOperand1>::DimNum >= 1) &&
        (DataCategory<TOperand2>::DimNum >= 1);

    template <typename TPolicy, typename TOperand1, typename TOperand2>
    struct OperCategory_<OpTags::Dot, TPolicy, TOperand1, TOperand2>
    {
        constexpr static size_t modDimNum = PolicySelect<DimPolicy, TPolicy>::ModifyDimNum;
        constexpr static size_t OriDim = TOperand1::DimNum + TOperand2::DimNum;
        using type = CategoryTags::Tensor<OriDim - modDimNum * 2>;
    };

    template <typename TCate, typename TPolicies>
    class OperShapeInfo<OpTags::Dot, TCate, TPolicies>
    {
    public:
        template <typename TOperAuxParams, typename TOperand1, typename TOperand2>
        OperShapeInfo(const TOperAuxParams&, const TOperand1& operand1, const TOperand2& operand2)
        {
            if constexpr(TCate::DimNum > 0)
            {
                constexpr static size_t modDimNum = PolicySelect<DimPolicy, TPolicies>::ModifyDimNum;
                constexpr static size_t op1Dims = DataCategory<TOperand1>::DimNum;
                constexpr static size_t op2Dims = DataCategory<TOperand2>::DimNum;

                size_t p = 0;
                for (size_t i = 0; i < op1Dims - modDimNum; ++i)
                {
                    m_shape[p++] = operand1.Shape()[i];
                }
            
                for (size_t i = modDimNum; i < op2Dims; ++i)
                {
                    m_shape[p++] = operand2.Shape()[i];
                }
            }
        }
    
        const auto& Shape() const
        {
            return m_shape;
        }
    
    private:
        MetaNN::Shape<TCate::DimNum> m_shape;
    };

    template <>
    struct OperSeq_<OpTags::Dot>
    {
        using type = OperCalAlgoChain<TailCalculator<OperDot::NSCaseGen::EvalItem,
                                                     OperDot::NSCaseGen::EvalGroup,
                                                     PolicyContainer<PPassPolicy, PPassShape>>>;
    };

    template <typename TPolicy = PolicyContainer<>,
              typename TP1, typename TP2,
              std::enable_if_t<IsValidOper<OpTags::Dot, TP1, TP2>>* = nullptr>
    auto Dot(TP1&& p_m1, TP2&& p_m2)
    {
        constexpr size_t modDimNum = PolicySelect<DimPolicy, TPolicy>::ModifyDimNum;
        static_assert(DataCategory<TP1>::DimNum >= modDimNum);
        static_assert(DataCategory<TP2>::DimNum >= modDimNum);
        
        for (size_t id1 = DataCategory<TP1>::DimNum - modDimNum, id2 = 0;
             id2 < modDimNum; ++id1, ++id2)
        {
            if (p_m1.Shape()[id1] != p_m2.Shape()[id2])
                throw std::runtime_error("Dot shape mismatch");
        }

        using ResType = Operation<OpTags::Dot, OperandContainer<RemConstRef<TP1>, RemConstRef<TP2>>,
                                 PolicyContainer<PModifyDimNumIs<modDimNum>>>;
        return ResType(std::forward<TP1>(p_m1), std::forward<TP2>(p_m2));
    }
}
