// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/core/extension.hpp>
#include <openvino/core/op_extension.hpp>
#include <openvino/frontend/extension.hpp>

#include "unpool.hpp"
#include "sparse_conv.hpp"
#include "sparse_conv_transpose.hpp"
#include "complex_mul.hpp"

// clang-format off
//! [ov_extension:entry_point]
OPENVINO_CREATE_EXTENSIONS(
    std::vector<ov::Extension::Ptr>({
        // Register operation itself, required to be read from IR
        std::make_shared<ov::OpExtension<TemplateExtension::SparseConv>>(),
        std::make_shared<ov::OpExtension<TemplateExtension::SparseConvTranspose>>(),
        std::make_shared<ov::OpExtension<TemplateExtension::ComplexMultiplication>>(),

        // Register operaton mapping, required when converted from framework model format
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::SparseConv>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::SparseConvTranspose>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::ComplexMultiplication>>()
    }));
//! [ov_extension:entry_point]
// clang-format on
