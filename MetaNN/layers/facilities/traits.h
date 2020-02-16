#pragma once

#include <stack>
#include <type_traits>
#include <MetaNN/facilities/var_type_dict.h>
#include <MetaNN/layers/facilities/layer_io_map.h>
#include <MetaNN/facilities/cont_metafuns/sequential.h>

namespace MetaNN::LayerTraits
{
template <typename TWeight, typename TGradStack, typename TGradCollector>
void ParamGradCollect(std::string_view name, const TWeight& weight, TGradStack& gradStack,
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
        auto tmp = Collapse(dBatch, weight.Shape());
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

namespace NSLayerIOMapTrasfer
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
    using type = LayerIOMap<LayerKV<TKeys, typename TVarTypeDict::template ValueType<TKeys>>
                            ...>;
};

template <typename TLayer, typename TLayerIOMap>
struct LayerIOMapForwardTransfer_;

template <typename TLayer, typename... TKVs>
struct LayerIOMapForwardTransfer_<TLayer, LayerIOMap<TKVs...>>
{
    using TVarTypeDictFill = typename CreateVarTypeDict_<std::tuple<>, std::tuple<>, TKVs...>::type;    
    using TForwardRes = decltype(std::declval<TLayer>().FeedForward(std::declval<TVarTypeDictFill>()));
    using KeySet = typename TForwardRes::Keys;
    using type = typename VarTypeDict2IOMap_<TForwardRes, KeySet>::type;
};
}

template <typename TLayer>
using LayerOutputItemTypes = typename NSLayerIOMapTrasfer::LayerIOMapForwardTransfer_<TLayer, typename TLayer::InputMap>::type;

template <typename TStoreType, bool store>
using LayerInternalBuf = std::conditional_t<store, std::stack<TStoreType>, NullParameter>;
/*
namespace NSShapePromote
{
    template <typename T>
    constexpr size_t ShapeIndex = (size_t)-1;
    
    template <>
    constexpr size_t ShapeIndex<Shape<CategoryTags::Scalar>> = 0;
    
    template <>
    constexpr size_t ShapeIndex<Shape<CategoryTags::Matrix>> = 1;
    
    template <>
    constexpr size_t ShapeIndex<Shape<CategoryTags::ThreeDArray>> = 2;
    
    template <typename TSubCate>
    constexpr size_t ShapeIndex<Shape<CategoryTags::Batch<TSubCate>>> = 10 + ShapeIndex<Shape<TSubCate>>;
    
    template <typename TSubCate>
    constexpr size_t ShapeIndex<Shape<CategoryTags::Sequence<TSubCate>>> = 10 + ShapeIndex<Shape<TSubCate>>;
    
    template <typename TSubCate>
    constexpr size_t ShapeIndex<Shape<CategoryTags::BatchSequence<TSubCate>>> = 100 + ShapeIndex<Shape<TSubCate>>;
    
    template <typename TShape1, typename TShape2>
    auto ShapePromoteHelper(const TShape1& shape1, const TShape2& shape2)
    {
        if constexpr (ShapeIndex<TShape1> > ShapeIndex<TShape2>)
        {
            return ShapePromoteHelper(shape2, shape1);
        }
        else if constexpr (ShapeIndex<TShape1> == ShapeIndex<TShape2>)
        {
            if (shape1 != shape2)
            {
                throw std::runtime_error("Shape promote error: shape mismatch.");
            }
            return shape1;
        }
        else if constexpr ((std::is_same_v<TShape1, Shape<CategoryTags::Scalar>>) &&
                           (ShapeIndex<TShape2> >= 0))
        {
            return shape2;
        }
        else if constexpr ((std::is_same_v<TShape1, Shape<CategoryTags::Matrix>>) &&
                           (ShapeIndex<TShape2> >= 1))
        {
            if ((shape1.RowNum() != shape2.RowNum()) || (shape1.ColNum() != shape2.ColNum()))
            {
                throw std::runtime_error("Shape promote error: shape mismatch.");
            }
            return shape2;
        }
        else if constexpr ((std::is_same_v<TShape1, Shape<CategoryTags::ThreeDArray>>) &&
                           (ShapeIndex<TShape2> >= 2))
        {
            if ((shape1.RowNum() != shape2.RowNum()) ||
                (shape1.ColNum() != shape2.ColNum()) ||
                (shape1.PageNum() != shape2.PageNum()))
            {
                throw std::runtime_error("Shape promote error: shape mismatch.");
            }
            return shape2;
        }
        else
        {
            static_assert(DependencyFalse<TShape1>);
        }
    }
    
    template <typename TShape>
    auto ShapePromote_(const TShape& s)
    {
        return s;
    }

    template <typename TShape, typename TData1, typename... TDatas>
    auto ShapePromote_(const TShape& shape, const TData1& data, const TDatas&... rem)
    {
        if constexpr (IsOutOfDataCategory<TData1>)
        {
            return ShapePromote_(shape, rem...);
        }
        else
        {
            auto res = NSShapePromote::ShapePromoteHelper(shape, data.Shape());
            return ShapePromote_(res, rem...);
        }
    }
}

template <typename TData>
auto ShapePromote(const TData& data)
{
    static_assert(IsInDataCategory<TData>, "All data types are invalid.");
    return data.Shape();
}

template <typename TDataHead, typename... TData>
auto ShapePromote(const TDataHead& head, const TData&... data)
{
    if constexpr (IsOutOfDataCategory<TDataHead>)
    {
        return ShapePromote(data...);
    }
    else
    {
        return NSShapePromote::ShapePromote_(head.Shape(), data...);
    }
}
*/
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