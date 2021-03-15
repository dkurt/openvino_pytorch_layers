// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "op.hpp"

using namespace TemplateExtension;

constexpr ngraph::NodeTypeInfo GridSampleOp::type_info;

//! [op:ctor]
GridSampleOp::GridSampleOp(const ngraph::Output<ngraph::Node>& inp,
                           const ngraph::Output<ngraph::Node>& grid) : Op({inp, grid}) {
    constructor_validate_and_infer_types();
}
//! [op:ctor]

//! [op:validate]
void GridSampleOp::validate_and_infer_types() {
    auto outShape = get_input_partial_shape(0);  // NC
    // Grid input has a shape NxHxWx2
    auto gridShape = get_input_partial_shape(1).to_shape();
    outShape[2] = gridShape[1];  // H
    outShape[3] = gridShape[2];  // W
    set_output_type(0, get_input_element_type(0), outShape);
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ngraph::Node> GridSampleOp::clone_with_new_inputs(const ngraph::OutputVector &new_args) const {
    if (new_args.size() != 2) {
        throw ngraph::ngraph_error("Incorrect number of new arguments");
    }
    return std::make_shared<GridSampleOp>(new_args.at(0), new_args.at(1));
}
//! [op:copy]

//! [op:visit_attributes]
bool GridSampleOp::visit_attributes(ngraph::AttributeVisitor &visitor) {
    return true;
}
//! [op:visit_attributes]
