#pragma once
#include <MetaNN/operators/facilities/policies.h>
#include <MetaNN/policies/policy_selector.h>
#include <MetaNN/facilities/cont_metafuns/value_sequential.h>
#include <numeric>

namespace MetaNN::OpTags
{
    struct ReduceSum;
}

namespace MetaNN
{
    namespace OperReduceSum
    {
        template <typename TPDim, typename TIndexes>
        struct DimArrToBitHelper_;
        
        template <typename TPDim, size_t... I>
        struct DimArrToBitHelper_<TPDim, std::index_sequence<I...>>
        {
            using type = DimBitArrayIs<ValueSequential::Contains<TPDim, I>...>;
        };

        template <typename TPolicy, size_t uDimNum>
        struct DimArrToBit_
        {
            using PDim = PickPolicyOjbect<TPolicy, DimPolicy, DimPolicy::DimArrayValueCate>;
            using type = typename DimArrToBitHelper_<PDim, std::make_index_sequence<uDimNum>>::type;
        };

        template <size_t uTrueBound, typename TIndexes>
        struct ResDimToBitHelper_;
        
        template <size_t uTrueBound, size_t... I>
        struct ResDimToBitHelper_<uTrueBound, std::index_sequence<I...>>
        {
            using type = DimBitArrayIs<(I < uTrueBound)...>;
        };
        
        template <typename TPolicy, size_t uDimNum>
        struct ResDimToBit_
        {
            constexpr static size_t ResDimNum = PolicySelect<DimPolicy, TPolicy>::ResDimNum;
            static_assert(ResDimNum <= uDimNum);
            using type = typename ResDimToBitHelper_<uDimNum - ResDimNum, std::make_index_sequence<uDimNum>>::type;
        };

        template <size_t uDimNum>
        constexpr bool IsTrivalDimBits(std::array<bool, uDimNum> dimBits)
        {
            for (size_t i = 0; i < uDimNum; ++i)
            {
                if (dimBits[i]) return false;
            }
            return true;
        }

        template <size_t uDimNum>
        constexpr size_t AccuDimBits(std::array<bool, uDimNum> dimBits)
        {
            size_t res = 0;
            for (size_t i = 0; i < uDimNum; ++i)
            {
                if (dimBits[i]) ++res;
            }
            return res;
        }
        
        template <template<auto...> class Cont, auto DimBits, typename>
        struct DimBits2VaridicTemp_;
        
        template <template<auto...> class Cont, auto DimBits, int... Is>
        struct DimBits2VaridicTemp_<Cont, DimBits, Helper::IndexSequence<Is...>>
        {
            using type = Cont<DimBits[Is]...>;
        };
    }
    
    namespace OperReduceSum::NSCaseGen
    {
        template <typename TInputHandle, typename TOutputHandle, typename TPolicies>
        class EvalItem : public BaseEvalItem<DeviceTypeFromHandle<TOutputHandle>>
        {
            using BaseType = BaseEvalItem<DeviceTypeFromHandle<TOutputHandle>>;
        public:
            using CategoryTag = CategoryTagFromHandle<TOutputHandle>;
            
            template <typename TAuxParams>
            EvalItem(TInputHandle oriHandle, TOutputHandle outputHandle, Shape<CategoryTag::DimNum> shape, const TAuxParams&)
                : BaseType(std::type_index(typeid(EvalItem)),
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
        class EvalGroup : public TrivalEvalGroup<EvalItem<TInputHandle, TOutputHandle, TPolicies>>
        {
            using EvalItemType = EvalItem<TInputHandle, TOutputHandle, TPolicies>;

        protected:
            virtual void EvalInternalLogic(EvalItemType& evalItem) final override
            {
                const auto& in = evalItem.m_inputHandle.Data();
                const auto& oriShape = in.Shape();
                
                constexpr auto dimBits = PolicySelect<DimPolicy, TPolicies>::DimBitArray;
                constexpr size_t OriDim = dimBits.size();
                constexpr size_t AimDim = CategoryTagFromHandle<TOutputHandle>::DimNum;
                
                using ResType = typename TOutputHandle::DataType;
                using ElementType = typename ResType::ElementType;
                ResType out(evalItem.m_outShape);

                const size_t count = oriShape.Count();
                if (count == 0)
                {
                    evalItem.m_outputHandle.SetData(std::move(out));
                    return;
                }
                
                auto low_in = LowerAccess(in);
                const ElementType* mem_in = low_in.RawMemory();
                auto low_out = LowerAccess(out);
                ElementType* mem_out = low_out.MutableRawMemory();
                
                static_assert(std::is_same_v<DeviceTypeFromHandle<TOutputHandle>, DeviceTags::CPU>, "Currently only CPU is supported");
                if constexpr (AimDim)
                {
                    std::fill(mem_out, mem_out + evalItem.m_outShape.Count(), ElementType{});
                    mem_out[0] = mem_in[0];

                    std::array<size_t, OriDim> oriPos{};
                    std::array<size_t, AimDim> aimPos{};
                    for (size_t i = 1; i < count; ++i)
                    {
                        oriShape.ShiftIndex(oriPos);
                        size_t p = 0;
                        for (size_t j = 0; j < OriDim; ++j)
                        {
                            if (!dimBits[j])
                            {
                                aimPos[p++] = oriPos[j];
                            }
                        }
                        mem_out[evalItem.m_outShape.IndexToOffset(aimPos)] += mem_in[i];
                    }
                }
                else
                {
                    mem_out[0] = std::accumulate(mem_in, mem_in + count, ElementType{});
                }
                evalItem.m_outputHandle.SetData(std::move(out));
            }
        };
    }

    template <>
    struct OperSeq_<OpTags::ReduceSum>
    {
        using type = OperCalAlgoChain<TailCalculator<OperReduceSum::NSCaseGen::EvalItem, OperReduceSum::NSCaseGen::EvalGroup>>;
    };

    template <typename TCate, typename TPolicies>
    class OperShapeInfo<OpTags::ReduceSum, TCate, TPolicies>
    {
        constexpr static size_t uDim = TCate::DimNum;
    public:
        template <typename TOperand>
        OperShapeInfo(const OperAuxParams<OpTags::ReduceSum, TCate>&,
                      const TOperand& operand)
        {
            if constexpr(uDim != 0)
            {
                constexpr auto dimBits = PolicySelect<DimPolicy, TPolicies>::DimBitArray;
                auto prevShape = operand.Shape();

                size_t p = 0;
                for (size_t i = 0; i < dimBits.size(); ++i)
                {
                    if (!dimBits[i])
                    {
                        m_shape[p++] = prevShape[i];
                    }
                }
                assert(p == uDim);
            }
        }
    
        const auto& Shape() const
        {
            return m_shape;
        }
    
    private:
        MetaNN::Shape<uDim> m_shape;
    };
    
    template <typename TPolicy, typename TOperand>
    struct OperCategory_<OpTags::ReduceSum, TPolicy, TOperand>
    {
        constexpr static size_t OriDim = TOperand::DimNum;
        constexpr static auto DimBits = PolicySelect<DimPolicy, TPolicy>::DimBitArray;
        constexpr static size_t AccuBits = OperReduceSum::AccuDimBits(DimBits);
        static_assert(OriDim >= AccuBits);
        using type = CategoryTags::Tensor<OriDim - AccuBits>;
    };
    
    template <typename TP>
    constexpr bool IsValidOper<OpTags::ReduceSum, TP> = (IsValidCategoryTag<DataCategory<TP>>) &&
                                                        (DataCategory<TP>::DimNum > 0);
    
    template <typename TPolicy, typename TP,
              typename = std::enable_if_t<IsValidOper<OpTags::ReduceSum, TP>>>
    auto ReduceSum(TP&& oper)
    {
        constexpr bool HasDimArray = HasNonTrivalPolicy<TPolicy, DimPolicy, DimPolicy::DimArrayValueCate>;
        constexpr bool HasResDimNum = HasNonTrivalPolicy<TPolicy, DimPolicy, DimPolicy::ResDimNumValueCate>;

        static_assert((int)HasDimArray + (int)HasResDimNum == 1, "DimArray or ResDimValue should be set");
        
        using TDimBits = typename CompileTimeSwitch<std::integer_sequence<bool, HasDimArray, HasResDimNum>,
                                                    std::tuple<OperReduceSum::DimArrToBit_<TPolicy, DataCategory<TP>::DimNum>,
                                                               OperReduceSum::ResDimToBit_<TPolicy, DataCategory<TP>::DimNum>>>::type;
        constexpr auto DimBits = TDimBits::DimBitArray;
        if constexpr (OperReduceSum::IsTrivalDimBits(DimBits))
        {
            return std::forward<TP>(oper);
        }
        else
        {
            using rawOp = RemConstRef<TP>;
            using ResType = Operator<OpTags::ReduceSum, OperandContainer<rawOp>,
                                     PolicyContainer<TDimBits>>;
            return ResType(std::forward<TP>(oper));
        }
    }
}