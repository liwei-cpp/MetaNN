#pragma once

#include <MetaNN/operators/facilities/operator_frame.h>
#include <MetaNN/facilities/cont_metafuns/helpers.h>

namespace MetaNN
{
    template <template<typename...> class EvalItem,
              template<typename...> class EvalGroup,
              template<typename> class EvalDispatcher = TrivalEvalItemDispatcher>
    struct TailCalculator
    {
        template <typename TCaseTail, typename TEvalRes, typename TOp>
        static void EvalRegister(TEvalRes& evalRes, const TOp& oper)
        {
            static_assert(std::is_same_v<TCaseTail, OperCalAlgoChain<>>,
                          "General case is not the last one");

            const auto& operands = oper.OperandTuple();
            constexpr size_t tupleSize = Sequential::Size<RemConstRef<decltype(operands)>>;
            using IndexSeq = Helper::MakeIndexSequence<(int)tupleSize>;
            constexpr IndexSeq* dummyParam = nullptr;
        
            auto operandHandles = GetOperandHandles(operands, dummyParam);
            DoEvalRegister(std::move(operandHandles), evalRes.Handle(), oper.Shape(), oper.AuxParams(), dummyParam);
        }
    private:
        template <typename TOpTuple, template<int...> class IndCont, int... Index>
        static auto GetOperandHandles(const TOpTuple& opers, const IndCont<Index...>*)
        {
            using ResType = std::tuple<RemConstRef<decltype(std::get<Index>(opers).EvalRegister())>...>;
            return ResType{std::get<Index>(opers).EvalRegister()...};
        }
    
        template <typename TOperHandleTuple, typename TResHandle, typename TShape, typename TAuxParams,
                  template<int...> class IndCont, int... Index>
        static auto DoEvalRegister(TOperHandleTuple operHandles, TResHandle resHandle, 
                                   const TShape& shape, const TAuxParams& auxParams, const IndCont<Index...>*)
        {
            using DeviceType = DeviceTypeFromHandle<TResHandle>;
        
            using ItemType = EvalItem<RemConstRef<decltype(std::get<Index>(operHandles))>..., 
                                      RemConstRef<TResHandle>>;
            using GroupType = EvalGroup<RemConstRef<decltype(std::get<Index>(operHandles))>..., 
                                      RemConstRef<TResHandle>>;
            using DispatcherType = EvalDispatcher<GroupType>;

            auto item = std::make_unique<ItemType>(std::move(std::get<Index>(operHandles))... ,
                                                   std::move(resHandle), shape, auxParams);
            EvalPlan<DeviceType>::Inst().template Register<DispatcherType>(std::move(item));
        }
    };
}