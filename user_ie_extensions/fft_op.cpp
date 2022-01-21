// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "op.hpp"

using namespace TemplateExtension;

constexpr ngraph::NodeTypeInfo FFTOp::type_info;
constexpr ngraph::NodeTypeInfo IFFTOp::type_info;

//! [op:ctor]
FFTOp::FFTOp(const ngraph::Output<ngraph::Node>& inp, const ngraph::Output<ngraph::Node>& dims, bool _inverse, bool _centered) : Op({inp, dims}) {
    constructor_validate_and_infer_types();
    inverse = _inverse;
    centered = _centered;
}
//! [op:ctor]

IFFTOp::IFFTOp(const ngraph::Output<ngraph::Node>& inp, const ngraph::Output<ngraph::Node>& dims, bool _inverse, bool _centered) : FFTOp({inp, dims, true, _centered}) {}

//! [op:validate]
void FFTOp::validate_and_infer_types() {
    auto outShape = get_input_partial_shape(0);
    set_output_type(0, get_input_element_type(0), outShape);
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ngraph::Node> FFTOp::clone_with_new_inputs(const ngraph::OutputVector &new_args) const {
    if (new_args.size() != 2) {
        throw ngraph::ngraph_error("Incorrect number of new arguments");
    }
    return std::make_shared<FFTOp>(new_args.at(0), new_args.at(1), inverse, centered);
}
//! [op:copy]

//! [op:visit_attributes]
bool FFTOp::visit_attributes(ngraph::AttributeVisitor &visitor) {
    visitor.on_attribute("inverse", inverse);
    visitor.on_attribute("centered", centered);
    return true;
}
//! [op:visit_attributes]
