#include "test_softmax_derivative.h"
#include "../facilities/data_gen.h"
#include "../facilities/calculate_tags.h"

#include <MetaNN/meta_nn.h>
#include <cassert>
#include <iostream>
using namespace MetaNN;
using namespace std;

namespace
{
void test_softmax_derivative1()
{
    cout << "Test softmax derivative case 1 ...\t";
    auto mSout = GenMatrix<float>(1, 4, 1.0f, 0.0001f);
    auto mGrad = GenMatrix<float>(1, 4, .3f, 0.005f);
    auto t = VecSoftmaxDerivative(mGrad, mSout);
    auto t_r = Evaluate(t);

    Matrix<float, CheckDevice> helper(4, 4);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 4; ++j)
        {
            if (i == j)
            {
                helper.SetValue(i, j, mSout(0, i) * (1 - mSout(0, i)));
            }
            else
            {
                helper.SetValue(i, j, -mSout(0, i) * mSout(0, j));
            }
        }
    }
    helper = Evaluate(Dot(mGrad, helper));
    for (size_t i = 0; i < 4; ++i)
    {
        assert(fabs(t_r(0, i) - helper(0, i)) < 0.0001);
    }

    mSout = GenMatrix<float>(111, 113, 1.1f, 0.0001f);
    mGrad = Matrix<float, CheckDevice>(111, 113);
    mGrad = mGrad.SubMatrix(27, 28, 41, 45);
    mSout = mSout.SubMatrix(17, 18, 31, 35);
    t = VecSoftmaxDerivative(mGrad, mSout);
    t_r = Evaluate(t);

    helper = Matrix<float, CheckDevice>(4, 4);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 4; ++j)
        {
            if (i == j)
            {
                helper.SetValue(i, j, mSout(0, i) * (1 - mSout(0, i)));
            }
            else
            {
                helper.SetValue(i, j, -mSout(0, i) * mSout(0, j));
            }
        }
    }
    helper = Evaluate(Dot(mGrad, helper));
    for (size_t i = 0; i < 4; ++i)
    {
        assert(fabs(t_r(0, i) - helper(0, i)) < 0.0001);
    }
    cout << "done" << endl;
}

void test_softmax_derivative2()
{
    cout << "Test softmax derivative case 2 ...\t";
    {
        auto bSout = GenMatrix<float>(1, 4, 1.0f, 0.0001f);
        auto bGrad = GenMatrix<float>(1, 4, 0.3f, 0.007f);
        auto res = VecSoftmaxDerivative(bGrad, bSout);
        auto res2 = VecSoftmaxDerivative(bGrad, bSout);

        assert(res == res2);

        auto cm1 = Evaluate(res);
        auto cm2 = Evaluate(res);
        assert(cm1 == cm2);
    }
    {
        auto bSout = GenMatrix<float>(1, 4, 1.0f, 0.0001f);
        auto bGrad = GenMatrix<float>(1, 4, 0.3f, 0.007f);
        auto res = VecSoftmaxDerivative(bGrad, bSout);
        auto res2 = res;

        assert(res == res2);

        auto handle1 = res.EvalRegister();
        auto handle2 = res2.EvalRegister();
        EvalPlan<CheckDevice>::Eval();

        auto cm1 = handle1.Data();
        auto cm2 = handle2.Data();
        assert(cm1 == cm2);
    }
    cout << "done" << endl;
}

void test_softmax_derivative3()
{
    cout << "Test softmax derivative case 3 ...\t";
    {
        InjectPolicy<SoftmaxLayer, PFeedbackOutput> layer1;
        InjectPolicy<NegativeLogLikelihoodLayer, PFeedbackOutput> layer2;
        auto layer1Input = LayerIO::Create().Set<LayerIO>(GenMatrix<float>(1, 5, 1, 1));
        auto layer1Output = layer1.FeedForward(layer1Input).Get<LayerIO>();

        auto target = GenMatrix<float>(1, 5, 0.1f, -0.3f);
        auto layer2Input = CostLayerIn::Create()
                                       .Set<CostLayerIn>(layer1Output)
                                       .Set<CostLayerLabel>(target);

        auto layer2Output = layer2.FeedForward(layer2Input).Get<LayerIO>();

        auto layer2GradOutput = layer2.FeedBackward(LayerIO::Create().Set<LayerIO>(Scalar<float>(0.7))).Get<CostLayerIn>();
        auto layer1GradOutput = layer1.FeedBackward(LayerIO::Create().Set<LayerIO>(layer2GradOutput)).Get<LayerIO>();

        auto check = Evaluate(layer1GradOutput);

        auto softRes = Evaluate(layer1Output);

        float sum = 0;
        for (size_t i = 0; i < 5; ++i)
        {
            sum += target(0, i);
        }

        for (size_t i = 0; i < 5; ++i)
        {
            float compare = softRes(0, i) * sum - target(0, i);
            assert(fabs(check(0, i) - compare * 0.7f) <= 0.0001);
        }
    }
    {
        InjectPolicy<SoftmaxLayer, PFeedbackOutput, PBatchMode> layer1;
        InjectPolicy<NegativeLogLikelihoodLayer, PFeedbackOutput, PBatchMode> layer2;
        auto layer1Input = LayerIO::Create().Set<LayerIO>(GenBatchMatrix<float>(1, 5, 7, 1, 1));
        auto layer1Output = layer1.FeedForward(layer1Input).Get<LayerIO>();

        auto target = GenBatchMatrix<float>(1, 5, 7, 0.1f, -0.3f);
        auto layer2Input = CostLayerIn::Create()
                                       .Set<CostLayerIn>(layer1Output)
                                       .Set<CostLayerLabel>(target);

        auto layer2Output = layer2.FeedForward(layer2Input).Get<LayerIO>();

        auto scale = MakeDuplicate(7, Scalar<float>(0.7f));
        auto layer2GradOutput = layer2.FeedBackward(LayerIO::Create().Set<LayerIO>(scale)).Get<CostLayerIn>();
        auto layer1GradOutput = layer1.FeedBackward(LayerIO::Create().Set<LayerIO>(layer2GradOutput)).Get<LayerIO>();

        auto check = Evaluate(layer1GradOutput);

        auto softRes = Evaluate(layer1Output);

        for (size_t b = 0; b < 7; ++b)
        {
            float sum = 0;
            for (size_t i = 0; i < 5; ++i)
            {
                sum += target[b](0, i);
            }

            for (size_t i = 0; i < 5; ++i)
            {
                float compare = softRes[b](0, i) * sum - target[b](0, i);
                assert(fabs(check[b](0, i) - compare * 0.7f) <= 0.0001);
            }
        }
    }
    cout << "done" << endl;
}

void test_softmax_derivative4()
{
    cout << "Test softmax derivative case 4 ...\t";
    {
        InjectPolicy<SoftmaxLayer, PFeedbackOutput> layer1;
        InjectPolicy<NegativeLogLikelihoodLayer, PFeedbackOutput> layer2;

        auto layer1Input = LayerIO::Create().Set<LayerIO>(GenMatrix<float>(1, 5, 1, 1));
        auto layer1Output = layer1.FeedForward(layer1Input).Get<LayerIO>();

        auto target = OneHotVector<float, CheckDevice>(5, 3);
        auto layer2Input = CostLayerIn::Create()
                                       .Set<CostLayerIn>(layer1Output)
                                       .Set<CostLayerLabel>(target);

        auto layer2Output = layer2.FeedForward(layer2Input).Get<LayerIO>();

        auto layer2GradOutput = layer2.FeedBackward(LayerIO::Create().Set<LayerIO>(Scalar<float>(0.7))).Get<CostLayerIn>();
        auto layer1GradOutput = layer1.FeedBackward(LayerIO::Create().Set<LayerIO>(layer2GradOutput)).Get<LayerIO>();

        auto check = Evaluate(layer1GradOutput);

        auto softRes = Evaluate(layer1Output);

        for (size_t i = 0; i < 5; ++i)
        {
            float compare = softRes(0, i);
            if (i == 3)
            {
                compare -= 1;
            }
            assert(fabs(check(0, i) - compare * 0.7f) <= 0.0001);
        }
    }
    cout << "done" << endl;
}

void test_softmax_derivative5()
{
    cout << "Test softmax derivative case 5 ...\t";
    auto mSout = GenBatchMatrix<float>(1, 4, 7, 1.0f, 0.0001f);
    auto mGrad = GenBatchMatrix<float>(1, 4, 7, .3f, 0.005f);
    auto t = VecSoftmaxDerivative(mGrad, mSout);
    auto t_r = Evaluate(t);

    for (size_t b = 0; b < 7; ++b)
    {
        Matrix<float, CheckDevice> helper(4, 4);
        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j < 4; ++j)
            {
                if (i == j)
                {
                    helper.SetValue(i, j, mSout[b](0, i) * (1 - mSout[b](0, i)));
                }
                else
                {
                    helper.SetValue(i, j, -mSout[b](0, i) * mSout[b](0, j));
                }
            }
        }
        helper = Evaluate(Dot(mGrad[b], helper));
        for (size_t i = 0; i < 4; ++i)
        {
            assert(fabs(t_r[b](0, i) - helper(0, i)) < 0.0001);
        }
    }
    cout << "done" << endl;
}
}

void test_softmax_derivative()
{
    test_softmax_derivative1();
    test_softmax_derivative2();
    test_softmax_derivative3();
    test_softmax_derivative4();
    test_softmax_derivative5();
}
