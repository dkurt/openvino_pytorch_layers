// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "op.hpp"

using namespace TemplateExtension;

constexpr ngraph::NodeTypeInfo ComplexMulOp::type_info;

//! [op:ctor]
ComplexMulOp::ComplexMulOp(const ngraph::Output<ngraph::Node>& inp0,
                         const ngraph::Output<ngraph::Node>& inp1,
                         bool _is_conj) : Op({inp0, inp1}) {
    constructor_validate_and_infer_types();
    is_conj = _is_conj;
}
//! [op:ctor]

//! [op:validate]
void ComplexMulOp::validate_and_infer_types() {
    auto outShape = get_input_partial_shape(1);
    set_output_type(0, get_input_element_type(1), outShape);
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ngraph::Node> ComplexMulOp::clone_with_new_inputs(const ngraph::OutputVector &new_args) const {
    if (new_args.size() != 2) {
        throw ngraph::ngraph_error("Incorrect number of new arguments");
    }
    return std::make_shared<ComplexMulOp>(new_args.at(0), new_args.at(1), is_conj);
}
//! [op:copy]

//! [op:visit_attributes]
bool ComplexMulOp::visit_attributes(ngraph::AttributeVisitor &visitor) {
    visitor.on_attribute("is_conj", is_conj);
    return true;
}
//! [op:visit_attributes]
