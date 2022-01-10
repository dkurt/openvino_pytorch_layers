// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "op.hpp"

using namespace TemplateExtension;

constexpr ngraph::NodeTypeInfo CalculateGridOp::type_info;

//! [op:ctor]
CalculateGridOp::CalculateGridOp(const ngraph::Output<ngraph::Node>& inp_pos)
    : Op({inp_pos}) {
    constructor_validate_and_infer_types();
}
//! [op:ctor]

//! [op:validate]
void CalculateGridOp::validate_and_infer_types() {
    auto outShape = get_input_partial_shape(0);
    set_output_type(0, get_input_element_type(0), outShape);
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ngraph::Node> CalculateGridOp::clone_with_new_inputs(const ngraph::OutputVector &new_args) const {
    if (new_args.size() != 1) {
        throw ngraph::ngraph_error("Incorrect number of new arguments");
    }
    return std::make_shared<CalculateGridOp>(new_args.at(0));
}
//! [op:copy]

//! [op:visit_attributes]
bool CalculateGridOp::visit_attributes(ngraph::AttributeVisitor &visitor) {
    return true;
}
//! [op:visit_attributes]
