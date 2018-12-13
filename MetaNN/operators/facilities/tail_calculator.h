#pragma once

#include <MetaNN/operators/facilities/operator_frame.h>
#include <MetaNN/facilities/cont_metafuns/helpers.h>

namespace MetaNN
{
    template <template<typename...> class EvalUnitCont, template<typename> class EvalGroupCont = TrivalEvalGroup>
    struct TailCalculator
    {
        template <typename TCaseTail, typename TEvalRes, typename TOp>
        static void EvalRegister(TEvalRes& evalRes, const TOp& oper)
        {
            static_assert(std::is_same_v<TCaseTail, OperSeqContainer<>>,
                          "General case is not the last one");

            const auto& operands = oper.GetOperandTuple();
            constexpr size_t tupleSize = ArraySize<RemConstRef<decltype(operands)>>;
            using IndexSeq = ContMetaFun::Helper::MakeIndexSequence<(int)tupleSize>;
            constexpr IndexSeq* dummyParam = nullptr;
        
            auto operandHandles = GetOperandHandles(operands, dummyParam);
            DoEvalRegister(std::move(operandHandles), evalRes.Handle(), dummyParam);
        }
    private:
        template <typename TOpTuple, template<int...> class IndCont, int... Index>
        static auto GetOperandHandles(const TOpTuple& opers, const IndCont<Index...>*)
        {
            using ResType = std::tuple<RemConstRef<decltype(std::get<Index>(opers).EvalRegister())>...>;
            return ResType{std::get<Index>(opers).EvalRegister()...};
        }
    
        template <typename TOperHandleTuple, typename TResHandle, template<int...> class IndCont, int... Index>
        static auto DoEvalRegister(TOperHandleTuple operHandles, TResHandle resHandle, const IndCont<Index...>*)
        {
            using DeviceType = DeviceTypeFromHandle<TResHandle>;
        
            using UnitType = EvalUnitCont<RemConstRef<decltype(std::get<Index>(operHandles))>..., 
                                          RemConstRef<TResHandle>>;
            using GroupType = EvalGroupCont<UnitType>;
        
            std::vector<const void*> depVec{(std::get<Index>(operHandles).DataPtr())...};
            const void* dataPtr = resHandle.DataPtr();
            UnitType unit(std::move(std::get<Index>(operHandles))... , std::move(resHandle));
            EvalPlan<DeviceType>::template Register<GroupType>(std::move(unit), dataPtr, {depVec});
        }
    };
}