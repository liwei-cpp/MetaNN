#pragma once

#include <MetaNN/operation/facilities/operation_frame.h>
#include <MetaNN/facilities/cont_metafuns/helpers.h>
#include <MetaNN/policies/_.h>

namespace MetaNN
{
    struct TailCalculatorPolicy
    {
        using MajorClass = TailCalculatorPolicy;

        struct IsPassPolicyValueCate;
        static constexpr bool IsPassPolicy = false;
        
        struct IsPassShapeValueCate;
        static constexpr bool IsPassShape = false;
        
        struct IsPassAuxParamValueCate;
        static constexpr bool IsPassAuxParam = false;
        
        struct DispatcherTempCate;
        template <typename TGroup>
        using Dispatcher = TrivialEvalItemDispatcher<TGroup>;
    };
#include <MetaNN/policies/policy_macro_begin.h>
    ValuePolicyObj(PPassPolicy,     TailCalculatorPolicy, IsPassPolicy,    true);
    ValuePolicyObj(PNoPassPolicy,   TailCalculatorPolicy, IsPassPolicy,    false);
    ValuePolicyObj(PPassShape,      TailCalculatorPolicy, IsPassShape,     true);
    ValuePolicyObj(PNoPassShape,    TailCalculatorPolicy, IsPassShape,     false);
    ValuePolicyObj(PPassAuxParam,   TailCalculatorPolicy, IsPassAuxParam,  true);
    ValuePolicyObj(PNoPassAuxParam, TailCalculatorPolicy, IsPassAuxParam,  false);
#include <MetaNN/policies/policy_macro_end.h>
    template <template<typename> class T>
    struct PDispatcherIs : virtual public TailCalculatorPolicy
    {
        using MinorClass = TailCalculatorPolicy::DispatcherTempCate;
        template <typename TGroup>
        using Dispatcher = T<TGroup>;
    };

    namespace NSTailCalculator
    {
        template <bool IsPassingPolicy,
                  template<typename...> class EvalItem, template<typename...> class EvalGroup,
                  typename TPolicy, typename... TOtherParams>
        struct PickEvalTypes_
        {
            using ItemType = EvalItem<TOtherParams..., TPolicy>;
            using GroupType = EvalGroup<TOtherParams..., TPolicy>;
        };

        template <template<typename...> class EvalItem, template<typename...> class EvalGroup,
                  typename TPolicy, typename... TOtherParams>
        struct PickEvalTypes_<false, EvalItem, EvalGroup, TPolicy, TOtherParams...>
        {
            using ItemType = EvalItem<TOtherParams...>;
            using GroupType = EvalGroup<TOtherParams...>;
        };
        
        template <bool passShape, bool passAuxParam, typename TItemType, typename TShape, typename TAuxParam,
                  typename... TOtherParams>
        auto CreateEvalItem(TShape&& p_shape, TAuxParam&& p_auxParam, TOtherParams&&... others)
        {
            if constexpr (passShape && passAuxParam)
            {
                return std::make_unique<TItemType>(std::forward<TOtherParams>(others)...,
                                                   std::forward<TShape>(p_shape), std::forward<TAuxParam>(p_auxParam));
            }
            else if constexpr (passShape)
            {
                return std::make_unique<TItemType>(std::forward<TOtherParams>(others)...,
                                                   std::forward<TShape>(p_shape));
            }
            else if constexpr (passAuxParam)
            {
                return std::make_unique<TItemType>(std::forward<TOtherParams>(others)...,
                                                   std::forward<TAuxParam>(p_auxParam));
            }
            else
            {
                return std::make_unique<TItemType>(std::forward<TOtherParams>(others)...);
            }
        }
    }

    template <template<typename...> class EvalItem,
              template<typename...> class EvalGroup,
              typename TPolicy = PolicyContainer<>>
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
            DoEvalRegister<typename TOp::Policies>(std::move(operandHandles), evalRes.Handle(), oper.Shape(), oper.AuxParams(), dummyParam);
        }
    private:
        template <typename TOpTuple, template<int...> class IndCont, int... Index>
        static auto GetOperandHandles(const TOpTuple& opers, const IndCont<Index...>*)
        {
            using ResType = std::tuple<RemConstRef<decltype(std::get<Index>(opers).EvalRegister())>...>;
            return ResType{std::get<Index>(opers).EvalRegister()...};
        }
    
        template <typename TPolicies, typename TOperHandleTuple, typename TResHandle,
                  typename TShape, typename TAuxParams,
                  template<int...> class IndCont, int... Index>
        static auto DoEvalRegister(TOperHandleTuple operHandles, TResHandle resHandle, 
                                   const TShape& shape, const TAuxParams& auxParams, const IndCont<Index...>*)
        {
            using namespace NSTailCalculator;
            constexpr bool IsPassPolicy = PolicySelect<TailCalculatorPolicy, TPolicy>::IsPassPolicy;
            using ItemType = typename PickEvalTypes_<IsPassPolicy, EvalItem, EvalGroup, TPolicies,
                                                     RemConstRef<decltype(std::get<Index>(operHandles))>...,
                                                     RemConstRef<TResHandle>>::ItemType;
            using GroupType = typename PickEvalTypes_<IsPassPolicy, EvalItem, EvalGroup, TPolicies,
                                                      RemConstRef<decltype(std::get<Index>(operHandles))>...,
                                                      RemConstRef<TResHandle>>::GroupType;
            auto item = CreateEvalItem<PolicySelect<TailCalculatorPolicy, TPolicy>::IsPassShape,
                                       PolicySelect<TailCalculatorPolicy, TPolicy>::IsPassAuxParam,
                                       ItemType>(shape, auxParams,
                                                 std::move(std::get<Index>(operHandles))...,
                                                 std::move(resHandle));

            using TDispatcher = typename PolicySelect<TailCalculatorPolicy, TPolicy>::template Dispatcher<GroupType>;
            EvalPlan::Inst().Register<TDispatcher>(std::move(item));
        }
    };
}