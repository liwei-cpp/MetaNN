#include <MetaNN/meta_nn.h>
#include <cassert>
#include <iostream>
#include <map>
using namespace MetaNN;
using namespace std;

namespace
{
double weight_input[] =
    {-0.09661144, -0.33417121, +0.29103363, +0.50999618, -0.23182100, +0.24369442,
     +0.66899598, +0.36953580, -0.51747060, -0.26284057, +0.28918779, +0.55565894,
     +0.66740763, +0.21685427, -0.09278971, -0.41988072, +0.70715761, -0.09363341,
     +0.34229791, -0.22610682, +0.58880997, +0.60940993, +0.13732076, +0.65681124};

double weight_update[] =
    {-0.66653228, -0.47682026, -0.67518288, -0.40545496, -0.36147988, +0.01941973,
     +0.69609451, +0.25287008, -0.74357712, +0.72910321, +0.06159645, +0.25983655,
     +0.04027188, +0.60454583, -0.06552815, +0.62309813, -0.19332850, +0.04896450,
     -0.64127958, -0.23409408, -0.67684901, +0.54364264, +0.40317559, -0.71374387};

double weight_reset[] =
    {-0.53187048, +0.77293050, +0.17450088, -0.00994116, +0.04143310, -0.33568737,
     -0.15345633, -0.43647453, -0.31598645, +0.73241997, +0.41814876, -0.22857052,
     -0.57352620, +0.02003479, +0.21309507, -0.32143161, -0.15456516, +0.47672486,
     -0.60603082, +0.52535045, +0.03762549, +0.42038560, +0.60655487, +0.64915311};

double trans_input[] =
    {+0.70399410, +0.13278353, -0.19926924, +0.04356903, +0.35045522, +0.52788538,
     -0.63065779, -0.45202187, +0.07422507, +0.36386842, +0.18230617, +0.46815771,
     +0.52402252, -0.47640398, +0.11233097, -0.27676845, -0.65701407, +0.67802221,
     -0.60481840, -0.15317315, -0.06706792, +0.69611615, +0.35044616, +0.34480113,
     -0.70122135, +0.58410794, +0.26500583, +0.10885358, +0.47127050, +0.57044607,
     +0.59831005, +0.45211726, -0.56619442, +0.53402656, +0.60157329, +0.68390757};

double trans_reset[] =
    {-0.14363223, +0.63675302, +0.16913497, -0.44189492, -0.56120068, +0.26084465,
     +0.44514757, +0.59415108, -0.30962875, -0.31645194, -0.52880931, -0.16519666,
     +0.26052433, -0.49828398, +0.40446800, +0.07982349, -0.00644308, +0.35322934,
     +0.58120221, +0.53890330, -0.27229568, -0.11808503, +0.36836761, -0.18573779,
     -0.02476197, +0.19951814, -0.07490581, -0.46724612, +0.68554229, -0.29110119,
     -0.40188420, -0.09623235, -0.38734370, +0.57530755, +0.61518854, -0.37863937};

double trans_update[] =
    {+0.11948478, -0.47503161, -0.45790726, +0.19722390, -0.60824418, -0.63365418,
     -0.36145496, +0.34658331, -0.36760733, +0.36732060, -0.41768077, -0.48393381,
     -0.49159506, -0.60170573, +0.42115003, -0.57490540, -0.05455971, +0.70709771,
     +0.32830757, +0.63654321, +0.32902294, -0.51632595, +0.45209199, -0.41814247,
     -0.52965844, -0.63281918, +0.22141445, +0.02858138, +0.10368818, +0.55148059,
     +0.41502982, +0.03049517, +0.66101068, -0.59646982, +0.36144584, -0.52966756};

double x_np_0[] = {+0.06803755, -0.02112342, +0.05661985, +0.05968801,
                   +0.08232947, -0.06048973, -0.03295545, +0.05364592};

double x_np_1[] = {-0.04444506, +0.01079400, -0.00452059, +0.02577419,
                   -0.02704310, +0.00268018, +0.09044594, +0.08323903};

double x_np_2[] = {+0.02714235, +0.04345939, -0.07167949, +0.02139378,
                   -0.09673989, -0.05142265, -0.07255369, +0.06083535};

double fres_0[] = {+0.57356697, +0.67493051, +0.29173079, +0.91266167, +0.84974384, +0.98380929,
                   +0.55533367, +0.67668849, +0.30710635, +0.92483586, +0.82781047, +0.98345935};

double fres_1[] = {+0.08489671, +0.46299744, +0.03604220, +0.81269711, +0.74108052, +0.94202799,
                   +0.11790840, +0.45316586, +0.06124194, +0.82058430, +0.75081849, +0.94535780};

double fres_2[] = {-0.25375497, +0.34222481, -0.00472301, +0.74003857, +0.61487955, +0.86550480,
                   -0.23477679, +0.34515449, +0.01637596, +0.74365497, +0.62452525, +0.86549950};

double check_up_grad[] = {+0.48058826, +0.54069245, +0.13667561, +0.38400191, +0.35677141, +0.12427084,
                          +0.45646650, +0.57106030, +0.13630769, +0.43481576, +0.38776311, +0.16771854,
                          +0.47375917, +0.49797082, +0.13775741, +0.32429212, +0.32719505, +0.09297007,
                          +0.43777096, +0.62169272, +0.13541570, +0.51348776, +0.43199876, +0.22456039,
                          +0.43995771, +0.60675037, +0.13567832, +0.49119174, +0.42036501, +0.21021718,
                          +0.43016893, +0.63637161, +0.13516138, +0.53686970, +0.44570836, +0.24260066};

double check_reset_grad[] = {-0.22573137, +0.00850728, +0.17660162, -0.16386601, -0.25808012, -0.50331688,
                             -0.23692220, -0.00422012, +0.18385042, -0.22045365, -0.31345046, -0.59030676,
                             -0.20450531, +0.01547305, +0.16353773, -0.11952476, -0.21175016, -0.41683775,
                             -0.25851166, -0.01967195, +0.19749409, -0.29627436, -0.38922000, -0.71662235,
                             -0.25175691, -0.01601686, +0.19325995, -0.27675560, -0.36948425, -0.68223709,
                             -0.26449898, -0.02474073, +0.20129845, -0.32011148, -0.41287124, -0.75509369};

double check_input_grad[] = {-0.36018932, -0.65483344, -0.26400137, -0.59727877, -0.86820155, -0.21833292,
                             -0.41349605, -0.95949709, -0.32700250, -0.87920940, -1.28881669, -0.36934814,
                             -0.21169925, -0.32412082, -0.16103597, -0.29028437, -0.43350112, -0.09657308,
                             -0.13432479, -0.58535790, -0.13005032, -0.54306334, -0.79876238, -0.27465478,
                             -0.28591725, -0.97712284, -0.25488314, -0.90252972, -1.32828522, -0.43310648,
                             -0.18858892, -0.58639282, -0.16131029, -0.54136020, -0.79330409, -0.25235096};

double wi_grad_check[] = {-0.04264588, -0.01554555, -0.03181295, -0.01012502, -0.02110841, +0.00796517,
                          +0.02034394, +0.02356272, +0.01869094, +0.01973352, +0.03446945, +0.00592688,
                          -0.02382816, -0.01072955, -0.00267675, -0.01318462, +0.00054733, +0.00702775,
                          -0.02760490, -0.08487224, -0.02311299, -0.07875377, -0.11316016, -0.03615509};

double wu_grad_check[] = {+0.03632787, +0.02715230, +0.01062525, +0.00892575, +0.01664936, -0.00250984,
                          -0.01864217, -0.01733737, -0.00582790, -0.00920409, -0.01183222, -0.00222107,
                          +0.01316092, +0.00946078, +0.00133530, +0.00743668, +0.00213212, -0.00179100,
                          +0.02557347, +0.03541989, +0.00772383, +0.02940711, +0.02389065, +0.0124517};

double wr_grad_check[] = {-0.01041598, +0.00345089, +0.00921119, +0.00381854, -0.00276361, -0.00800987,
                          +0.00684211, -0.00091481, -0.00587776, +0.00235688, +0.00611044, +0.01163567,
                          -0.00513169, +0.00180021, +0.00335244, +0.00136408, -0.00014323, -0.00600464,
                          -0.01480510, -0.00092086, +0.01130431, -0.01644023, -0.02170095, -0.04062329};


void comp(const Batch<float, DeviceTags::CPU, CategoryTags::Matrix>& v1,
          const Batch<float, DeviceTags::CPU, CategoryTags::Matrix>& v2)
{
    assert(v1.RowNum() == v2.RowNum());
    assert(v1.ColNum() == v2.ColNum());
    assert(v1.BatchNum() == v2.BatchNum());

    size_t maxR = 0;
    size_t maxC = 0;
    size_t maxD = 0;
    float vv1 = v1[0](0, 0);
    float vv2 = v2[0](0, 0);
    float diff = fabs(vv1 - vv2);
    for (size_t i = 0; i < v1.RowNum(); ++i)
    {
        for (size_t j = 0; j < v1.ColNum(); ++j)
        {
            for (size_t k = 0; k < v1.BatchNum(); ++k)
            {
                float val = fabs(v1[k](i, j) - v2[k](i, j));
                if (val > diff)
                {
                    diff = val;
                    maxR = i;
                    maxC = j;
                    maxD = k;
                    vv1 = v1[k](i, j);
                    vv2 = v2[k](i, j);
                }
            }
        }
    }

    cout << maxR << ' ' << maxC << ' ' << maxD
         << ": (" << vv1 << ' ' << vv2 << ") " << diff << endl;
}

void comp(const Matrix<float, DeviceTags::CPU>& v1, const Matrix<float, DeviceTags::CPU>& v2)
{
    assert(v1.RowNum() == v2.RowNum());
    assert(v1.ColNum() == v2.ColNum());

    size_t maxR = 0;
    size_t maxC = 0;
    float vv1 = v1(0, 0);
    float vv2 = v2(0, 0);
    float diff = fabs(vv1 - vv2);
    for (size_t i = 0; i < v1.RowNum(); ++i)
    {
        for (size_t j = 0; j < v1.ColNum(); ++j)
        {
            float val = fabs(v1(i, j) - v2(i, j));
            if (val > diff)
            {
                diff = val;
                maxR = i;
                maxC = j;
                vv1 = v1(i, j);
                vv2 = v2(i, j);
            }
        }
    }

    cout << maxR << ' ' << maxC
         << ": (" << vv1 << ' ' << vv2 << ") " << diff << endl;
}

void test_gru_with_bptt()
{
    std::cerr << "test gru case 1 ..." << std::endl;
    using GRUKernel = InjectPolicy<RecurrentLayer, PUpdate, PFeedbackOutput>;

    GRUKernel gru("trans", 4, 6);
    map<std::string, Matrix<float, DeviceTags::CPU>> params;
    params["trans-W"] = Matrix<float, DeviceTags::CPU>(4, 6);
    params["trans-Wz"] = Matrix<float, DeviceTags::CPU>(4, 6);
    params["trans-Wr"] = Matrix<float, DeviceTags::CPU>(4, 6);
    
    params["trans-U"] = Matrix<float, DeviceTags::CPU>(6, 6);
    params["trans-Uz"] = Matrix<float, DeviceTags::CPU>(6, 6);
    params["trans-Ur"] = Matrix<float, DeviceTags::CPU>(6, 6);

    size_t id = 0;
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 6; ++j)
        {
            params["trans-W"].SetValue(i, j, (float)(weight_input[id]));
            params["trans-Wz"].SetValue(i, j, (float)(weight_update[id]));
            params["trans-Wr"].SetValue(i, j, (float)(weight_reset[id]));
            ++id;
        }
    }

    id = 0;
    for (size_t i = 0; i < 6; ++i)
    {
        for (size_t j = 0; j <6; ++j)
        {
            params["trans-U"].SetValue(i, j, trans_input[id]);
            params["trans-Ur"].SetValue(i, j, trans_reset[id]);
            params["trans-Uz"].SetValue(i, j, trans_update[id]);
            ++id;
        }
    }
    
    auto initializer = MakeInitializer<float>();
    for (const auto& [k, v] : params)
    {
        initializer.SetMatrix(k, v);
    }
    params.clear();
    
    gru.Init(initializer, params);

    Batch<float, DeviceTags::CPU, CategoryTags::Matrix> x0(2, 1, 4);
    Batch<float, DeviceTags::CPU, CategoryTags::Matrix> x1(2, 1, 4);
    Batch<float, DeviceTags::CPU, CategoryTags::Matrix> x2(2, 1, 4);
    id = 0;
    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j < 4; ++j)
        {
            x0.SetValue(i, 0, j, x_np_0[id]);
            x1.SetValue(i, 0, j, x_np_1[id]);
            x2.SetValue(i, 0, j, x_np_2[id]);
            ++id;
        }
    }

    Batch<float, DeviceTags::CPU, CategoryTags::Matrix> fres0(2, 1, 6);
    Batch<float, DeviceTags::CPU, CategoryTags::Matrix> fres1(2, 1, 6);
    Batch<float, DeviceTags::CPU, CategoryTags::Matrix> fres2(2, 1, 6);
    id = 0;
    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j < 6; ++j)
        {
            fres0.SetValue(i, 0, j, fres_0[id]);
            fres1.SetValue(i, 0, j, fres_1[id]);
            fres2.SetValue(i, 0, j, fres_2[id]);
            ++id;
        }
    }

    using TResData = DynamicData<float, DeviceTags::CPU, CategoryTags::Matrix>;
    using TResHandle = typename TResData::ResHandleType;
    
    Array<Matrix<float, DeviceTags::CPU>> forward0(1, 6);
    Array<Matrix<float, DeviceTags::CPU>> forward1(1, 6);
    Array<Matrix<float, DeviceTags::CPU>> forward2(1, 6);
    
    for (size_t batchSize = 0; batchSize < 2; ++batchSize)
    {
        vector<TResData> fData;
        
        auto gru0 = GRUKernel::InputType::Create()
                        .Set<LayerIO>(x0[batchSize])
                        .Set<RnnLayerHiddenBefore>(MakeTrivalMatrix<float, DeviceTags::CPU>(1, 6, 1));
        fData.push_back(MakeDynamic(gru.FeedForward(gru0).Get<LayerIO>()));
        
        auto gru1 = GRUKernel::InputType::Create()
                        .Set<LayerIO>(x1[batchSize]);
        fData.push_back(MakeDynamic(gru.FeedForward(gru1).Get<LayerIO>()));
        
        auto gru2 = GRUKernel::InputType::Create()
                        .Set<LayerIO>(x2[batchSize]);
        fData.push_back(MakeDynamic(gru.FeedForward(gru2).Get<LayerIO>()));
        
        auto grad_gru2 = gru.FeedBackward(LayerIO::Create().Set<LayerIO>(fData[2] * Scalar<int>(-1)));        
        auto grad_gru1 = gru.FeedBackward(LayerIO::Create().Set<LayerIO>(fData[1] * Scalar<int>(-1)));        
        auto grad_gru0 = gru.FeedBackward(LayerIO::Create().Set<LayerIO>(fData[0] * Scalar<int>(-1)));
        
        vector<TResHandle> fHandle;
        for (auto& p : fData)
        {
            fHandle.push_back(p.EvalRegister());
        }
        EvalPlan<DeviceTags::CPU>::Eval();        
        forward0.push_back(fHandle[0].Data());
        forward1.push_back(fHandle[1].Data());
        forward2.push_back(fHandle[2].Data());
    }
    
    cout << "\tCompare for forward 0: ", comp(Evaluate(forward0), fres0);
    cout << "\tCompare for forward 1: ", comp(Evaluate(forward1), fres1);
    cout << "\tCompare for forward 2: ", comp(Evaluate(forward2), fres2);

    GradCollector<float, DeviceTags::CPU> grad_collector;
    gru.GradCollect(grad_collector);
    assert(grad_collector.size() == 6);

    id = 0;
    Matrix<float, DeviceTags::CPU> trans_update_grad(6, 6);
    Matrix<float, DeviceTags::CPU> trans_reset_grad(6, 6);
    Matrix<float, DeviceTags::CPU> trans_input_grad(6, 6);
    for (size_t i = 0; i < 6; ++i)
    {
        for (size_t j = 0; j < 6; ++j)
        {
            trans_update_grad.SetValue(i, j, (float)(check_up_grad[id]));
            trans_reset_grad.SetValue(i, j, (float)(check_reset_grad[id]));
            trans_input_grad.SetValue(i, j, (float)(check_input_grad[id]));
            ++id;
        }
    }

    id = 0;
    Matrix<float, DeviceTags::CPU> wi_grad(4, 6);
    Matrix<float, DeviceTags::CPU> wu_grad(4, 6);
    Matrix<float, DeviceTags::CPU> wr_grad(4, 6);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 6; ++j)
        {
            wi_grad.SetValue(i, j, (float)(wi_grad_check[id]));
            wu_grad.SetValue(i, j, (float)(wu_grad_check[id]));
            wr_grad.SetValue(i, j, (float)(wr_grad_check[id]));
            ++id;
        }
    }

    for (auto& p : grad_collector)
    {
        auto info_g = Evaluate(Collapse(p.grad));
        if (params["trans-Uz"] == p.weight)
        {
            cout << "\tCompare for trans-update grad: ", comp(trans_update_grad, info_g);
        }
        else if (params["trans-Ur"] == p.weight)
        {
            cout << "\tCompare for trans-reset grad: ", comp(trans_reset_grad, info_g);
        }
        else if (params["trans-U"] == p.weight)
        {
            cout << "\tCompare for trans-input grad: ", comp(trans_input_grad, info_g);
        }
        else if (params["trans-W"] == p.weight)
        {
            cout << "\tCompare for input grad: ", comp(wi_grad, info_g);
        }
        else if (params["trans-Wz"] == p.weight)
        {
            cout << "\tCompare for update grad: ", comp(wu_grad, info_g);
        }
        else if (params["trans-Wr"] == p.weight)
        {
            cout << "\tCompare for reset grad: ", comp(wr_grad, info_g);
        }
        else
        {
            assert(false);
        }
    }
    
    gru.NeutralInvariant();
    std::cerr << "test gru case 1 done" << std::endl;
}
}

void test_gru()
{
    test_gru_with_bptt();
}
