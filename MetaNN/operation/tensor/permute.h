#pragma once

#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/evaluate/eval_plan.h>
#include <MetaNN/operation/facilities/_.h>
#include <MetaNN/policies/_.h>
#include <MetaNN/facilities/_.h>
#include <cassert>
#include <type_traits>

namespace MetaNN::OpTags
{
    struct Permute;
}

namespace MetaNN
{
    namespace OperPermute::NSCaseGen
    {
        template <typename TInputHandle, typename TOutputHandle, typename TPolicies>
        class EvalItem : public BaseEvalItem
        {
        public:
            using CategoryTag = CategoryTagFromHandle<TOutputHandle>;

            EvalItem(TInputHandle oriHandle, TOutputHandle outputHandle, Shape<CategoryTag::DimNum> shape)
                : BaseEvalItem(TypeID<EvalItem>(),
                               {oriHandle.DataPtr()}, outputHandle.DataPtr())
                , m_inputHandle(std::move(oriHandle))
                , m_outputHandle(std::move(outputHandle))
                , m_outShape(std::move(shape))
            {}
        
            const TInputHandle m_inputHandle;
            TOutputHandle m_outputHandle;
            Shape<CategoryTag::DimNum> m_outShape;
        };
        
        template <typename TInputHandle, typename TOutputHandle, typename TPolicies>
        class EvalGroup : public TrivialEvalGroup<EvalItem<TInputHandle, TOutputHandle, TPolicies>>
        {
            using EvalItemType = EvalItem<TInputHandle, TOutputHandle, TPolicies>;
            constexpr static size_t uDim = EvalItemType::CategoryTag::DimNum;

        protected:
            virtual void EvalInternalLogic(EvalItemType& evalItem) final override
            {
                const auto& in = evalItem.m_inputHandle.Data();
                const auto& oriShape = in.Shape();
                
                constexpr auto dims = PolicySelect<DimPolicy, TPolicies>::DimArray;
                
                using ResType = typename TOutputHandle::DataType;
                using ElementType = typename ResType::ElementType;
                ResType out(evalItem.m_outShape);

                const size_t count = oriShape.Count();
                assert(count == evalItem.m_outShape.Count());
                
                if (count == 0)
                {
                    evalItem.m_outputHandle.SetData(std::move(out));
                    return;
                }

                std::array<size_t, uDim> aimGaps;
                auto rit1 = evalItem.m_outShape.rbegin();
                auto rit2 = aimGaps.rbegin();
                *rit2 = 1;
                while (true)
                {
                    auto rit2_n = rit2;
                    ++rit2_n;
                    if (rit2_n == aimGaps.rend()) break;
                    *rit2_n = (*rit1) * (*rit2);
                    ++rit1;
                    ++rit2;
                }


                auto low_in = LowerAccess(in);
                const ElementType* mem_in = low_in.RawMemory();

                auto low_out = LowerAccess(out);
                ElementType* mem_out = low_out.MutableRawMemory();

                static_assert(std::is_same_v<DeviceTypeFromHandle<TOutputHandle>, DeviceTags::CPU>, "Currently only CPU is supported");
                
                mem_out[0] = mem_in[0];
                
                std::array<size_t, uDim> oriPos{};
                std::array<size_t, uDim> aimPos;
                for (size_t i = 1; i < count; ++i)
                {
                    oriShape.ShiftIndex(oriPos);
                    CalAimPos(oriPos, dims, aimPos);
                    size_t index = evalItem.m_outShape.IndexToOffset(aimPos);
                    mem_out[index] = mem_in[i];
                }
                evalItem.m_outputHandle.SetData(std::move(out));
            }
        private:
            void CalAimPos(const std::array<size_t, uDim>& oriPos, const std::array<size_t, uDim>& dims,
                           std::array<size_t, uDim>& aimPos) const
            {
                for (size_t i = 0; i < uDim; ++i)
                {
                    aimPos[i] = oriPos[dims[i]];
                }
            }
        };
    }
    
    template <>
    struct OperSeq_<OpTags::Permute>
    {
        using type = OperCalAlgoChain<TailCalculator<OperPermute::NSCaseGen::EvalItem,
                                                     OperPermute::NSCaseGen::EvalGroup,
                                                     PolicyContainer<PPassPolicy, PPassShape>>>;
    };

    template <typename TP>
    constexpr bool IsValidOper<OpTags::Permute, TP> = (IsValidCategoryTag<DataCategory<TP>>) &&
                                                      (DataCategory<TP>::DimNum > 1);

    template <typename TCate, typename TPolicies>
    class OperShapeInfo<OpTags::Permute, TCate, TPolicies>
    {
        constexpr static size_t uDim = TCate::DimNum;
    public:
        template <typename TOperAuxParams, typename TOperand>
        OperShapeInfo(const TOperAuxParams&, const TOperand& operand)
        {
            constexpr auto dims = PolicySelect<DimPolicy, TPolicies>::DimArray;
            auto prevShape = operand.Shape();
            
            for (size_t i = 0; i < uDim; ++i)
            {
                m_shape[i] = prevShape[dims[i]];
            }
        }
    
        const auto& Shape() const
        {
            return m_shape;
        }
    
    private:
        MetaNN::Shape<uDim> m_shape;
    };

    namespace OperPermute
    {
        template <size_t uDim>
        constexpr bool ValidDims(std::array<size_t, uDim> dims)
        {
            std::array<bool, uDim> checkBuf{};
            for (size_t i = 0; i < uDim; ++i)
            {
                if (dims[i] >= uDim) return false;
                checkBuf[i] = true;
            }
            
            for (size_t i = 0; i < uDim; ++i)
            {
                if (!checkBuf[i]) return false;
            }
            return true;
        }
        
        template <size_t uDim>
        constexpr bool TrivialDims(std::array<size_t, uDim> dims)
        {
            for (size_t i = 0; i < uDim; ++i)
            {
                if (dims[i] != i) return false;
            }
            return true;
        }
    }

    template <typename TDimPolicy, typename TP,
              std::enable_if_t<IsValidOper<OpTags::Permute, TP>>* = nullptr>
    auto Permute(TP&& oper)
    {
        constexpr auto dims = PolicySelect<DimPolicy, TDimPolicy>::DimArray;
        static_assert(OperPermute::ValidDims(dims));
        if constexpr (OperPermute::TrivialDims(dims))
        {
            return std::forward<TP>(oper);
        }
        else
        {
            using rawOp = RemConstRef<TP>;
            using PDim = PickPolicyOjbect<TDimPolicy, DimPolicy, DimPolicy::DimArrayValueCate>;
            using ResType = Operation<OpTags::Permute, OperandContainer<rawOp>, PolicyContainer<PDim>>;
            return ResType(std::forward<TP>(oper));
        }
    }
    
    template <typename TP,
              std::enable_if_t<IsValidOper<OpTags::Permute, TP>>* = nullptr>
    auto Transpose(TP&& oper)
    {
        static_assert(DataCategory<TP>::DimNum == 2);
        return Permute<PolicyContainer<PDimArrayIs<1, 0>>>(std::forward<TP>(oper));
    }
    
    namespace OperPermute
    {
        template <typename TIndexSeq, typename TDimArray>
        struct DimInv_;
        
        template <size_t... Index, typename TDimArray>
        struct DimInv_<std::index_sequence<Index...>, TDimArray>
        {
            using type = PDimArrayIs<ValueSequential::Order<TDimArray, Index>...>;
        };
    }

    template <typename TDimPolicy, typename TP,
              std::enable_if_t<IsValidOper<OpTags::Permute, TP>>* = nullptr>
    auto PermuteInv(TP&& oper)
    {
        constexpr auto dims = PolicySelect<DimPolicy, TDimPolicy>::DimArray;
        static_assert(OperPermute::ValidDims(dims));
        if constexpr (OperPermute::TrivialDims(dims))
        {
            return std::forward<TP>(oper);
        }
        else
        {
            using PDim = PickPolicyOjbect<TDimPolicy, DimPolicy, DimPolicy::DimArrayValueCate>;
            using PModDim = typename OperPermute::DimInv_<std::make_index_sequence<dims.size()>,
                                                          PDim>::type;
            return Permute<PolicyContainer<PModDim>>(std::forward<TP>(oper));
        }
    }
}