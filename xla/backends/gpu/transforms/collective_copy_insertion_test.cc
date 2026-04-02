/* Copyright 2024 The OpenXLA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/backends/gpu/transforms/collective_copy_insertion.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/side_effect_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"


namespace xla::gpu {
namespace {

using CollectiveCopyInsertionTest = HloHardwareIndependentTestBase;

TEST_F(CollectiveCopyInsertionTest, RdmaCollectiveIsAddedCopy) {
  const absl::string_view hlo_string = R"(
  HloModule composite

  %sum {
    %a = f32[] parameter(0)
    %b = f32[] parameter(1)
    ROOT %add = f32[] add(%a, %b)
  }

  ENTRY %main () -> f32[] {
    %p1 = f32[] constant(42)
    ROOT ar = f32[] all-reduce(%p1), replica_groups={{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}}, to_apply=%sum
  })";

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  int64_t num_visible_devices_per_process = 4;
  se::GpuComputeCapability gpu_version = se::CudaComputeCapability(8, 0);
  Compiler::CompileOptions compile_options;

  CollectiveCopyInsertion cci_pass(gpu_version, num_visible_devices_per_process, &compile_options);

  TF_ASSERT_OK_AND_ASSIGN(bool mutated, cci_pass.Run(module.get()));
  absl::StatusOr<bool> filecheck_result = RunFileCheck(module->ToString({}), R"(
  // CHECK: %p1 = f32[] constant(42)
  // CHECK: %copy = f32[] copy(%p1), frontend_attributes={_xla_stream_annotation="collective"}
  // CHECK: ROOT %ar = f32[] all-reduce(%copy)
  )");
  ASSERT_OK(filecheck_result.status());
  EXPECT_TRUE(*filecheck_result);

  ASSERT_TRUE(mutated);
}

TEST_F(CollectiveCopyInsertionTest, NoneRdmaCollectiveIsNotAddedCopy) {
  const absl::string_view hlo_string = R"(
  HloModule composite

  %sum {
    %a = f32[] parameter(0)
    %b = f32[] parameter(1)
    ROOT %add = f32[] add(%a, %b)
  }

  ENTRY %main () -> f32[] {
    %p1 = f32[] constant(42)
    ROOT ar = f32[] all-reduce(%p1), replica_groups={{0,1,2,3}}, to_apply=%sum
  })";

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  int64_t num_visible_devices_per_process = 4;
  se::GpuComputeCapability gpu_version = se::CudaComputeCapability(8, 0);
  Compiler::CompileOptions compile_options;

  CollectiveCopyInsertion cci_pass(gpu_version, num_visible_devices_per_process, &compile_options);

  TF_ASSERT_OK_AND_ASSIGN(bool mutated, cci_pass.Run(module.get()));
  absl::StatusOr<bool> filecheck_result = RunFileCheck(module->ToString({}), R"(
  // CHECK: %p1 = f32[] constant(42)
  // CHECK NOT: %copy = f32[] copy(%p1), frontend_attributes={_xla_stream_annotation="collective"}
  // CHECK: ROOT %ar = f32[] all-reduce(%copy)
  )");
  ASSERT_OK(filecheck_result.status());
  EXPECT_TRUE(*filecheck_result);

  ASSERT_TRUE(mutated);
}

}  // namespace
}  // namespace xla::gpu
