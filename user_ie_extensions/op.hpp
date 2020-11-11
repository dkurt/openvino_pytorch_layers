// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/ngraph.hpp>

//! [op:header]
namespace TemplateExtension {

class UnpoolOp : public ngraph::op::Op {
public:
    static constexpr ngraph::NodeTypeInfo type_info{"MaxPoolGrad", 0};
    const ngraph::NodeTypeInfo& get_type_info() const override { return type_info;  }

    UnpoolOp() = default;
    UnpoolOp(const ngraph::Output<ngraph::Node>& poolInp,
             const ngraph::Output<ngraph::Node>& poolOut,
             const ngraph::Output<ngraph::Node>& inp,
             const ngraph::Output<ngraph::Node>& shape);
    void validate_and_infer_types() override;
    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
    bool visit_attributes(ngraph::AttributeVisitor& visitor) override;
};

class FFTOp : public ngraph::op::Op {
public:
    static constexpr ngraph::NodeTypeInfo type_info{"FFT", 0};
    const ngraph::NodeTypeInfo& get_type_info() const override { return type_info;  }

    FFTOp() = default;
    FFTOp(const ngraph::Output<ngraph::Node>& inp, bool inverse);
    void validate_and_infer_types() override;
    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
    bool visit_attributes(ngraph::AttributeVisitor& visitor) override;

    bool inverse;
};
//! [op:header]

}  // namespace TemplateExtension
