// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "op.hpp"

using namespace TemplateExtension;

constexpr ngraph::NodeTypeInfo SparseConvOp::type_info;

//! [op:ctor]
SparseConvOp::SparseConvOp(
    const ngraph::Output<ngraph::Node>& features,
    const ngraph::Output<ngraph::Node>& inp_pos,
    const ngraph::Output<ngraph::Node>& out_pos,
    const ngraph::Output<ngraph::Node>& kernel,
    const ngraph::Output<ngraph::Node>& offset
)
    : Op({features, inp_pos, out_pos, kernel, offset}) {
    constructor_validate_and_infer_types();
}
//! [op:ctor]

//! [op:validate]
void SparseConvOp::validate_and_infer_types() {
    auto outShape = get_input_partial_shape(2);
    auto kernelShape = get_input_partial_shape(3);
    outShape[1] = kernelShape[4];
    set_output_type(0, get_input_element_type(0), outShape);
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ngraph::Node> SparseConvOp::clone_with_new_inputs(const ngraph::OutputVector &new_args) const {
    if (new_args.size() != 5) {
        throw ngraph::ngraph_error("Incorrect number of new arguments");
    }
    return std::make_shared<SparseConvOp>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), new_args.at(4));
}
//! [op:copy]

//! [op:visit_attributes]
bool SparseConvOp::visit_attributes(ngraph::AttributeVisitor &visitor) {
    return true;
}
//! [op:visit_attributes]
