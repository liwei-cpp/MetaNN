#include "policies/test_change_policy.h"
#include "policies/test_policy_operations.h"
#include "policies/test_policy_selector.h"
#include "facilities/test_var_type_dict.h"
#include "evaluate/test_eval_plan.h"
#include "data/test_array.h"
#include "data/test_duplicate.h"
#include "data/test_scalar.h"
#include "data/test_general_matrix.h"
#include "data/test_one_hot_vector.h"
#include "data/test_trival_matrix.h"
#include "data/test_zero_matrix.h"
#include "data/test_batch_scalar.h"
#include "data/test_batch_matrix.h"
#include "operators/test_abs.h"
#include "operators/test_add.h"
#include "operators/test_collapse.h"
#include "operators/test_divide.h"
#include "operators/test_dot.h"
#include "operators/test_element_mul.h"
#include "operators/test_interpolate.h"
#include "operators/test_negative_log_likelihood.h"
#include "operators/test_negative_log_likelihood_derivative.h"
#include "operators/test_sigmoid.h"
#include "operators/test_sigmoid_derivative.h"
#include "operators/test_sign.h"
#include "operators/test_softmax.h"
#include "operators/test_softmax_derivative.h"
#include "operators/test_substract.h"
#include "operators/test_tanh.h"
#include "operators/test_tanh_derivative.h"
#include "operators/test_transpose.h"
#include "layers/elementary/test_abs_layer.h"
#include "layers/elementary/test_add_layer.h"
#include "layers/elementary/test_bias_layer.h"
#include "layers/elementary/test_element_mul_layer.h"
#include "layers/elementary/test_interpolate_layer.h"
#include "layers/elementary/test_sigmoid_layer.h"
#include "layers/elementary/test_softmax_layer.h"
#include "layers/elementary/test_tanh_layer.h"
#include "layers/elementary/test_weight_layer.h"
#include "layers/cost/test_negative_log_likelihood_layer.h"
#include "layers/compose/test_compose_kernel.h"
#include "layers/compose/test_linear_layer.h"
#include "layers/compose/test_single_layer.h"
#include "layers/recurrent/test_gru.h"
#include "layers/recurrent/test_gru_2.h"
#include "model_rel/param_initializer/test_constant_filler.h"
#include "model_rel/param_initializer/test_gaussian_filler.h"
#include "model_rel/param_initializer/test_var_scale_filter.h"

int main(int argc, char **argv)
{
    test_change_policy();
    test_policy_operations();
    test_policy_selector();

    test_var_type_dict();
    test_eval_plan();

	test_scalar();
    test_general_matrix();
    test_one_hot_vector();
    test_trival_matrix();
    test_zero_matrix();
    test_array();
    test_duplicate();
    test_batch_scalar();
    test_batch_matrix();

    test_abs();
    test_add();
    test_collapse();
    test_divide();
    test_dot();
    test_element_mul();
    test_interpolate();
    test_negative_log_likelihood();
    test_negative_log_likelihood_derivative();
    test_sigmoid();
    test_sigmoid_derivative();
    test_sign();
    test_softmax();
    test_softmax_derivative();
    test_substract();
    test_tanh();
    test_tanh_derivative();
    test_transpose();

    test_abs_layer();
    test_add_layer();
    test_bias_layer();
    test_element_mul_layer();
    test_interpolate_layer();
    test_sigmoid_layer();
    test_softmax_layer();
    test_tanh_layer();
    test_weight_layer();

    test_negative_log_likelihood_layer();

    test_compose_kernel();
    test_linear_layer();
    test_single_layer();

    test_gru();
    test_gru_2();

    test_constant_filler();
    test_gaussian_filler();
    test_var_scale_filter();
	return 0;
}

