#include <MetaNN/meta_nn2.h>
#include <calculate_tags.h>
#include <data_gen.h>
#include <cassert>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
    using CommonInputMap = LayerIOMap<LayerKV<LeftOperand, Matrix<CheckElement, CheckDevice>>,
                                      LayerKV<RightOperand, Matrix<CheckElement, CheckDevice>>
                                     >;
    
    void test_dot_layer1()
    {
        cout << "Test dot layer case 1 ...\t";
        using RootLayer = MakeInferLayer<DotLayer>;
        static_assert(!RootLayer::IsFeedbackOutput, "Test Error");
        static_assert(!RootLayer::IsUpdate, "Test Error");

        RootLayer layer("root");

        auto i1 = GenMatrix<CheckElement>(2, 3, -3.3f, 0.1f);
        auto i2 = GenMatrix<CheckElement>(3, 4, -0.7f, 1.3f);
        auto input = LayerInputCont<RootLayer>().Set<LeftOperand>(i1)
                                                .Set<RightOperand>(i2);

        LayerNeutralInvariant(layer);

        auto out = layer.FeedForward(input);
        auto res = Evaluate(out.Get<LayerOutput>());
        assert(res.Shape().RowNum() == 2);
        assert(res.Shape().ColNum() == 4);
        
        auto check = Evaluate(Dot(i1, i2));
        for (size_t i = 0; i < 2; ++i)
        {
            for (size_t j = 0; j < 4; ++j)
            {
                assert(fabs(res(i, j) - check(i, j)) < 0.001);
            }
        }

        auto out_grad = layer.FeedBackward(NullParameter{});
        auto fb1 = out_grad.Get<LeftOperand>();
        auto fb2 = out_grad.Get<RightOperand>();
        static_assert(std::is_same<decltype(fb1), NullParameter>::value, "Test error");
        static_assert(std::is_same<decltype(fb2), NullParameter>::value, "Test error");

        LayerNeutralInvariant(layer);
        cout << "done" << endl;
    }
}

namespace Test::Layer::Elementary
{
    void test_dot_layer()
    {
        test_dot_layer1();
    }
}
