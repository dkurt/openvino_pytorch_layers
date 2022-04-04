// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

//! [op:common_include]
#include <openvino/op/op.hpp>
//! [op:common_include]
//! [op:frontend_include]
#ifdef OPENVINO_ONNX_FRONTEND_ENABLED
#    include <openvino/frontend/onnx/extension/op.hpp>
#endif
//! [op:frontend_include]

//! [op:header]
namespace TemplateExtension {

class SparseConv : public ov::op::Op {
public:
    OPENVINO_OP("SparseConv");

#ifdef OPENVINO_ONNX_FRONTEND_ENABLED
    OPENVINO_FRAMEWORK_MAP(onnx)
#endif

    SparseConv() = default;
    SparseConv(const ov::Output<ov::Node>& features,
               const ov::Output<ov::Node>& inp_pos,
               const ov::Output<ov::Node>& out_pos,
               const ov::Output<ov::Node>& kernel,
               const ov::Output<ov::Node>& offset);
    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
};
//! [op:header]

}  // namespace TemplateExtension
