// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "op.hpp"

using namespace TemplateExtension;

constexpr ngraph::NodeTypeInfo LSTSQOp::type_info;

//! [op:ctor]
LSTSQOp::LSTSQOp(
    const ngraph::Output<ngraph::Node>& B,
    const ngraph::Output<ngraph::Node>& A
)
    : Op({B, A}) {
    constructor_validate_and_infer_types();
}
//! [op:ctor]

//! [op:validate]
void LSTSQOp::validate_and_infer_types() {
    auto outShape = get_input_partial_shape(0);
    outShape[0] = 2;
    set_output_type(0, get_input_element_type(0), outShape);
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ngraph::Node> LSTSQOp::clone_with_new_inputs(const ngraph::OutputVector &new_args) const {
    if (new_args.size() != 2) {
        throw ngraph::ngraph_error("Incorrect number of new arguments");
    }
    return std::make_shared<LSTSQOp>(new_args.at(0), new_args.at(1));
}
//! [op:copy]

//! [op:visit_attributes]
bool LSTSQOp::visit_attributes(ngraph::AttributeVisitor &visitor) {
    return true;
}
//! [op:visit_attributes]
