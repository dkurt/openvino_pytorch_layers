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
    FFTOp(const ngraph::Output<ngraph::Node>& inp, bool inverse, bool centered, std::vector<int64_t> dim);
    void validate_and_infer_types() override;
    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
    bool visit_attributes(ngraph::AttributeVisitor& visitor) override;

    bool inverse, centered;
    std::vector<int64_t> dim;
};

class IFFTOp : public FFTOp {
public:
    static constexpr ngraph::NodeTypeInfo type_info{"IFFT", 0};

    IFFTOp() = default;
    IFFTOp(const ngraph::Output<ngraph::Node>& inp, bool inverse, bool centered, std::vector<int64_t> dim);
};

class GridSampleOp : public ngraph::op::Op {
public:
    static constexpr ngraph::NodeTypeInfo type_info{"GridSample", 0};
    const ngraph::NodeTypeInfo& get_type_info() const override { return type_info;  }

    GridSampleOp() = default;
    GridSampleOp(const ngraph::Output<ngraph::Node>& inp,
                 const ngraph::Output<ngraph::Node>& grid);
    void validate_and_infer_types() override;
    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
    bool visit_attributes(ngraph::AttributeVisitor& visitor) override;
};

class ComplexMulOp : public ngraph::op::Op {
public:
    static constexpr ngraph::NodeTypeInfo type_info{"ComplexMultiplication", 0};
    const ngraph::NodeTypeInfo& get_type_info() const override { return type_info;  }

    ComplexMulOp() = default;
    ComplexMulOp(const ngraph::Output<ngraph::Node>& inp0,
                const ngraph::Output<ngraph::Node>& inp1);
    void validate_and_infer_types() override;
    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
    bool visit_attributes(ngraph::AttributeVisitor& visitor) override;
};

class SparseConvOp : public ngraph::op::Op {
public:
    static constexpr ngraph::NodeTypeInfo type_info{"SparseConv", 0};
    const ngraph::NodeTypeInfo& get_type_info() const override { return type_info;  }

    SparseConvOp() = default;
    SparseConvOp(const ngraph::Output<ngraph::Node>& features,
                 const ngraph::Output<ngraph::Node>& inp_pos,
                 const ngraph::Output<ngraph::Node>& out_pos,
                 const ngraph::Output<ngraph::Node>& kernel,
                 const ngraph::Output<ngraph::Node>& offset);
    void validate_and_infer_types() override;
    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
    bool visit_attributes(ngraph::AttributeVisitor& visitor) override;
};

class SparseConvTransposeOp : public ngraph::op::Op {
public:
    static constexpr ngraph::NodeTypeInfo type_info{"SparseConvTranspose", 0};
    const ngraph::NodeTypeInfo& get_type_info() const override { return type_info;  }

    SparseConvTransposeOp() = default;
    SparseConvTransposeOp(const ngraph::Output<ngraph::Node>& features,
                          const ngraph::Output<ngraph::Node>& inp_pos,
                          const ngraph::Output<ngraph::Node>& out_pos,
                          const ngraph::Output<ngraph::Node>& kernel,
                          const ngraph::Output<ngraph::Node>& offset);
    void validate_and_infer_types() override;
    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
    bool visit_attributes(ngraph::AttributeVisitor& visitor) override;
};

class CalculateGridOp : public ngraph::op::Op {
public:
    static constexpr ngraph::NodeTypeInfo type_info{"calculate_grid", 0};
    const ngraph::NodeTypeInfo& get_type_info() const override { return type_info;  }

    CalculateGridOp() = default;
    CalculateGridOp(const ngraph::Output<ngraph::Node>& inp_pos);
    void validate_and_infer_types() override;
    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
    bool visit_attributes(ngraph::AttributeVisitor& visitor) override;
};

//! [op:header]

}  // namespace TemplateExtension
