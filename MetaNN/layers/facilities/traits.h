#pragma once

#include <stack>
#include <string>
#include <type_traits>
#include <MetaNN/facilities/var_type_dict.h>
#include <MetaNN/layers/facilities/layer_in_map.h>
#include <MetaNN/facilities/cont_metafuns/sequential.h>

namespace MetaNN::LayerTraits
{
template <typename TWeight, typename TGradStack, typename TGradCollector>
void ParamGradCollect(const std::string& name, const TWeight& weight, TGradStack& gradStack,
                      TGradCollector& col)
{
    size_t stackSize = gradStack.size();
    if (stackSize == 0) return;
    
    if (stackSize == 1)
    {
        auto g = gradStack.top();
        gradStack.pop();
        col.Collect(name, weight, std::move(g));
        return;
    }
    else
    {
        ScalableTensor<RemConstRef<typename TGradStack::value_type>> dBatch(weight.Shape());
        while (!gradStack.empty())
        {
            auto g = gradStack.top();
            gradStack.pop();
            dBatch.PushBack(std::move(g));
        }
        auto tmp = ReduceSum<PolicyContainer<PModifyDimNumIs<1>>>(dBatch);
        col.Collect(name, weight, std::move(tmp));
        return;
    }
}

template <typename TTypeMap, typename TKey, typename TCont>
auto PickItemFromCont(TCont&& cont)
{
    auto itemOri = std::forward<TCont>(cont).template Get<TKey>();
    static_assert(!std::is_same_v<decltype(itemOri), NullParameter>);
    
    using TAim = typename TTypeMap::template Find<TKey>;
    if constexpr (std::is_same_v<TAim, NullParameter>)
    {
        return itemOri;
    }
    else
    {
        if constexpr (!IsValidCategoryTag<TAim>)
        {
            return TAim(itemOri);
        }
        else if constexpr (IsDynamic<TAim>)
        {
            auto res = MakeDynamic(std::move(itemOri));
            static_assert(std::is_same_v<decltype(res), TAim>);
            return res;
        }
        else
        {
            static_assert(std::is_same_v<RemConstRef<decltype(itemOri)>, TAim>);
            return itemOri;
        }
    }
}

namespace NSLayerInMapTrasfer
{
template <typename TVarTypeDictOutter, typename TVarTypeDictInner, typename... TKVs>
struct CreateVarTypeDict_;

template <typename... TOutters, typename... TInners>
struct CreateVarTypeDict_<std::tuple<TOutters...>, std::tuple<TInners...>>
{
    using type = typename VarTypeDict<TOutters...>::template Values<TInners...>;
};


template <typename TVarTypeDictOutter, typename TVarTypeDictInner, typename TCur, typename... TKVs>
struct CreateVarTypeDict_<TVarTypeDictOutter, TVarTypeDictInner, TCur, TKVs...>
{
    using NewOutter = Sequential::PushBack<TVarTypeDictOutter, typename TCur::KeyType>;
    using NewInner = Sequential::PushBack<TVarTypeDictInner, typename TCur::ValueType>;
    using type = typename CreateVarTypeDict_<NewOutter, NewInner, TKVs...>::type;
};

template <typename TVarTypeDict, typename TKeySet>
struct VarTypeDict2IOMap_;

template <typename TVarTypeDict, typename... TKeys>
struct VarTypeDict2IOMap_<TVarTypeDict, VarTypeDict<TKeys...>>
{
    using type = LayerInMap<LayerKV<TKeys, typename TVarTypeDict::template ValueType<TKeys>>
                            ...>;
};

template <typename TLayer, typename TLayerInMap>
struct LayerInMapForwardTransfer_;

template <typename TLayer, typename... TKVs>
struct LayerInMapForwardTransfer_<TLayer, LayerInMap<TKVs...>>
{
    using TVarTypeDictFill = typename CreateVarTypeDict_<std::tuple<>, std::tuple<>, TKVs...>::type;    
    using TForwardRes = decltype(std::declval<TLayer>().FeedForward(std::declval<TVarTypeDictFill>()));
    using KeySet = typename TForwardRes::Keys;
    using type = typename VarTypeDict2IOMap_<TForwardRes, KeySet>::type;
};
}

template <typename TLayer>
using LayerOutputItemTypes = typename NSLayerInMapTrasfer::LayerInMapForwardTransfer_<TLayer, typename TLayer::InputMap>::type;

template <typename TStoreType, bool store>
using LayerInternalBuf = std::conditional_t<store, std::stack<TStoreType>, NullParameter>;

template <typename TData, typename TInput>
auto Collapse(TInput&& p_input)
{
    using AimCategory = DataCategory<TData>;
    if constexpr (IsValidCategoryTag<AimCategory>)
    {
        constexpr size_t AimDim = AimCategory::DimNum;
        constexpr size_t OriDim = DataCategory<TInput>::DimNum;
        static_assert(OriDim >= AimDim);
        if constexpr (OriDim == AimDim)
        {
            return std::forward<TInput>(p_input);
        }
        else
        {
            return ReduceSum<PolicyContainer<PModifyDimNumIs<OriDim - AimDim>>>(std::forward<TInput>(p_input));
        }
    }
    else
    {
        return NullParameter{};
    }
}

namespace NSShapeChecker
{
    template <typename TShape, bool bTrigger>
    class ShapeChecker_
    {
    public:
        template <typename TData>
        void PushDataShape(const TData&) {}
        
        template <typename TData>
        void CheckDataShape(const TData&) {}
        
        void AssertEmpty() const {}
        
        void Pop() {}
    };
    
    template <typename TShape>
    class ShapeChecker_<TShape, true>
    {
    public:
        template <typename TData>
        void PushDataShape(const TData& data)
        {
            m_buffer.push(data.Shape());
        }
        
        template <typename TData>
        void CheckDataShape(const TData& data)
        {
            if (m_buffer.empty())
            {
                throw std::runtime_error("ShapeStack is empty, cannot check shape.");
            }
            if (!(data.Shape() == m_buffer.top()))
            {
                throw std::runtime_error("Shape check fail.");
            }
        }
        
        void AssertEmpty() const
        {
            if (!m_buffer.empty())
            {
                throw std::runtime_error("Shape checker is not empty.");
            }
        }
        
        void Pop()
        {
            if (m_buffer.empty())
            {
                throw std::runtime_error("ShapeStack is empty, cannot check shape.");
            }
            m_buffer.pop();
        }
        
    private:
        std::stack<TShape> m_buffer;
    };

    template <typename TData, bool bTrigger>
    struct DataToShape_
    {
        using type = ShapeChecker_<void, false>;
    };

#ifdef METANN_CHECKSHAPE
    template <typename TData>
    struct DataToShape_<TData, true>
    {
        using type = ShapeChecker_<ShapeType<TData>, true>;
    };
#endif
}

template <typename TData, bool bTrigger>
using ShapeChecker = typename NSShapeChecker::DataToShape_<TData, bTrigger && (IsValidCategoryTag<TData>)>::type;

template <typename T>
void PopoutFromStackHelper(std::stack<T>& stack)
{
    stack.pop();
}

template <typename TShape, bool bTrigger>
void PopoutFromStackHelper(NSShapeChecker::ShapeChecker_<TShape, bTrigger>& stack)
{
    stack.Pop();
}

template <typename... TDataStacks>
void PopoutFromStack(TDataStacks&&... stacks)
{
    (PopoutFromStackHelper(std::forward<TDataStacks>(stacks)), ...);
}

template <typename T>
void CheckStackEmptyHelper(const std::stack<T>& stack)
{
    if (!stack.empty())
    {
        throw std::runtime_error("Stack is not empty.");
    }
}

template <typename TShape, bool bTrigger>
void CheckStackEmptyHelper(const NSShapeChecker::ShapeChecker_<TShape, bTrigger>& stack)
{
    stack.AssertEmpty();
}

template <typename... TDataStacks>
void CheckStackEmpty(const TDataStacks&... stacks)
{
    (CheckStackEmptyHelper(stacks), ...);
}
}