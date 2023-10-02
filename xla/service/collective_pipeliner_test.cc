/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/collective_pipeliner.h"
#include "xla/service/algebraic_simplifier.h"
#include "xla/service/hlo_dce.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/service/hlo_parser.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/statusor.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/util.h"

namespace xla {
namespace {

using ::testing::_;
namespace op = xla::testing::opcode_matchers;

class CollectivePipelinerTest : public HloTestBase {
 public:
  CollectivePipelinerTest() {
    const int64_t kNumReplicas = 4;
    const int64_t kNumComputations = 2;
    config_ = GetModuleConfigForTest(/*replica_count=*/kNumReplicas,
                                     /*num_partitions=*/kNumComputations);
  }

 protected:
  const HloPredicate IsAllGather = HloPredicateIsOp<HloOpcode::kAllGather>;
  HloModuleConfig config_;
};

StatusOr<bool> RunOptimizer(
    HloModule* module, bool last_run, int64_t level_to_operate_on = 0,
    bool pipeline_use_tree = false, bool process_different_sized_ops = true,
    CollectivePipeliner::PipeliningDirection direction =
        CollectivePipeliner::PipeliningDirection::kForward,
    HloPredicate should_process = HloPredicateIsOp<HloOpcode::kAllReduce>) {
  CollectivePipeliner::Config config = {
      /*level_to_operate_on=*/level_to_operate_on,
      /*max_pipelining_per_loop=*/INT64_MAX,
      /*last_run=*/last_run,
      /*pipeline_use_tree=*/pipeline_use_tree,
      /*process_different_sized_ops=*/process_different_sized_ops,
      /*direction=*/
      direction,
      /*should_process=*/should_process,
  };
  HloPassPipeline pass("optimizer");
  pass.AddPass<HloVerifier>(/*layout_sensitive=*/false,
                            /*allow_mixed_precision=*/false);
  pass.AddPass<CollectivePipeliner>(config);
  pass.AddPass<HloVerifier>(/*layout_sensitive=*/false,
                            /*allow_mixed_precision=*/false);
  return pass.Run(module);
}

TEST_F(CollectivePipelinerTest, TransformIncrementIndexByOne) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ar.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(module.get(), /*last_run=*/true).value());
  XLA_VLOG_LINES(1, module->ToString());
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::DynamicUpdateSlice(_, op::AllReduce(), _, _, _));
  const HloInstruction* sliced = root->operand(1)->operand(0);
  EXPECT_EQ(sliced->opcode(), HloOpcode::kDynamicSlice);
  const HloInstruction* index = sliced->operand(1);
  EXPECT_EQ(index->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(index->tuple_index(), 3);
  const HloInstruction* while_inst = index->operand(0);
  EXPECT_EQ(while_inst->opcode(), HloOpcode::kWhile);
  const HloInstruction* while_root =
      while_inst->while_body()->root_instruction();
  EXPECT_EQ(while_root->opcode(), HloOpcode::kTuple);
  const HloInstruction* dyn_upd = while_root->operand(1);
  EXPECT_EQ(dyn_upd->opcode(), HloOpcode::kDynamicUpdateSlice);
  const HloInstruction* dyn_upd2 = dyn_upd->operand(0);
  EXPECT_EQ(dyn_upd2->opcode(), HloOpcode::kDynamicUpdateSlice);
  const HloInstruction* prev_ar = dyn_upd2->operand(1);
  EXPECT_EQ(prev_ar->opcode(), HloOpcode::kAllReduce);
  const HloInstruction* dyn_slice_top = prev_ar->operand(0);
  EXPECT_EQ(dyn_slice_top->opcode(), HloOpcode::kDynamicSlice);
  const HloInstruction* get_tuple_value = dyn_slice_top->operand(0);
  const HloInstruction* get_tuple_index = dyn_slice_top->operand(1);
  EXPECT_EQ(get_tuple_value->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(get_tuple_index->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(get_tuple_value->tuple_index(), 1);
  EXPECT_EQ(get_tuple_index->tuple_index(), 3);
}

TEST_F(CollectivePipelinerTest, TransformIncrementIndexByOneNotFirstIdx) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[8,3,128], bf16[8,3,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[8,3,128], bf16[8,3,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[8,3,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[8,3,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[8,1,128] dynamic-slice(get-tuple-element.5, constant.2561, select.1348, constant.2561), dynamic_slice_sizes={8,1,128}
  mul = bf16[8,1,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[8,1,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  dynamic-update-slice.35 = bf16[8,3,128] dynamic-update-slice(get-tuple-element.395, ar.1, constant.2561, select.1348, constant.2561)
  ROOT tuple = (s32[], bf16[8,3,128], bf16[8,3,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[8,3,128] parameter(0)
  tuple = (s32[], bf16[8,3,128], bf16[8,3,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[8,3,128], bf16[8,3,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[8,3,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(module.get(), /*last_run=*/true).value());
  XLA_VLOG_LINES(1, module->ToString());
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::DynamicUpdateSlice(_, op::AllReduce(), _, _, _));
  const HloInstruction* sliced = root->operand(1)->operand(0);
  EXPECT_EQ(sliced->opcode(), HloOpcode::kDynamicSlice);
  const HloInstruction* index = sliced->operand(2);
  EXPECT_EQ(index->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(index->tuple_index(), 3);
  const HloInstruction* while_inst = index->operand(0);
  EXPECT_EQ(while_inst->opcode(), HloOpcode::kWhile);
  const HloInstruction* while_root =
      while_inst->while_body()->root_instruction();
  EXPECT_EQ(while_root->opcode(), HloOpcode::kTuple);
  const HloInstruction* dyn_upd = while_root->operand(1);
  EXPECT_EQ(dyn_upd->opcode(), HloOpcode::kDynamicUpdateSlice);
  const HloInstruction* dyn_upd2 = dyn_upd->operand(0);
  EXPECT_EQ(dyn_upd2->opcode(), HloOpcode::kDynamicUpdateSlice);
  const HloInstruction* prev_ar = dyn_upd2->operand(1);
  EXPECT_EQ(prev_ar->opcode(), HloOpcode::kAllReduce);
  const HloInstruction* dyn_slice_top = prev_ar->operand(0);
  EXPECT_EQ(dyn_slice_top->opcode(), HloOpcode::kDynamicSlice);
  const HloInstruction* get_tuple_value = dyn_slice_top->operand(0);
  const HloInstruction* get_tuple_index = dyn_slice_top->operand(2);
  EXPECT_EQ(get_tuple_value->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(get_tuple_index->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(get_tuple_value->tuple_index(), 1);
  EXPECT_EQ(get_tuple_index->tuple_index(), 3);
}

TEST_F(CollectivePipelinerTest, TransformIncrementByTwo) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(2)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)

  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ar.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(module.get(), /*last_run=*/true).value());
  XLA_VLOG_LINES(1, module->ToString());
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::DynamicUpdateSlice(_, op::AllReduce(), _, _, _));
  const HloInstruction* sliced = root->operand(1)->operand(0);
  EXPECT_EQ(sliced->opcode(), HloOpcode::kDynamicSlice);
  const HloInstruction* index = sliced->operand(1);
  EXPECT_EQ(index->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(index->tuple_index(), 3);
  const HloInstruction* while_inst = index->operand(0);
  EXPECT_EQ(while_inst->opcode(), HloOpcode::kWhile);
  const HloInstruction* while_root =
      while_inst->while_body()->root_instruction();
  EXPECT_EQ(while_root->opcode(), HloOpcode::kTuple);
  const HloInstruction* dyn_upd = while_root->operand(1);
  EXPECT_EQ(dyn_upd->opcode(), HloOpcode::kDynamicUpdateSlice);
  const HloInstruction* dyn_upd2 = dyn_upd->operand(0);
  EXPECT_EQ(dyn_upd2->opcode(), HloOpcode::kDynamicUpdateSlice);
  const HloInstruction* prev_ar = dyn_upd2->operand(1);
  EXPECT_EQ(prev_ar->opcode(), HloOpcode::kAllReduce);
  const HloInstruction* dyn_slice_top = prev_ar->operand(0);
  EXPECT_EQ(dyn_slice_top->opcode(), HloOpcode::kDynamicSlice);
  const HloInstruction* get_tuple_value = dyn_slice_top->operand(0);
  const HloInstruction* get_tuple_index = dyn_slice_top->operand(1);
  EXPECT_EQ(get_tuple_value->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(get_tuple_index->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(get_tuple_value->tuple_index(), 1);
  EXPECT_EQ(get_tuple_index->tuple_index(), 3);
}

TEST_F(CollectivePipelinerTest, NoTransformCantProveIndexDoesntWrap) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(4)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ar.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(-1)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_FALSE(RunOptimizer(module.get(), /*last_run=*/true).value());
  XLA_VLOG_LINES(1, module->ToString());
}

TEST_F(CollectivePipelinerTest, TransformNegativeIndexIterationToZero) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(0)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ar.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(-3)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(module.get(), /*last_run=*/false).value());
  XLA_VLOG_LINES(1, module->ToString());
  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::DynamicUpdateSlice(
                        _,
                        op::CustomCall(op::AllReduce(op::DynamicSlice(
                                           op::GetTupleElement(op::While()),
                                           op::GetTupleElement(),
                                           op::Constant(), op::Constant())),
                                       op::Constant()),
                        op::GetTupleElement(), op::Constant(), op::Constant()));
}

TEST_F(CollectivePipelinerTest, EscapedInputNoTransform) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[1,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(0)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[1,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=3
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.911 = bf16[1,8,128] dynamic-slice(get-tuple-element.395, constant.2561, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.395, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ar.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[1,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, dynamic-slice.911, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(-3)
  p0 = bf16[3,8,128] parameter(0)
  cc = bf16[] constant(0)
  c1 = bf16[1,8,128] broadcast(cc), dimensions={}
  c2 = bf16[3,8,128] broadcast(cc), dimensions={}
  tuple = (s32[], bf16[3,8,128], bf16[1,8,128], bf16[3,8,128]) tuple(c0, p0, c1, c2)
  while = (s32[], bf16[3,8,128], bf16[1,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_FALSE(RunOptimizer(module.get(), /*last_run=*/true).value());
}

TEST_F(CollectivePipelinerTest, TransformWithAg) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(0)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  rs.1 = bf16[1,1,128] reduce-scatter(mul), replica_groups={}, to_apply=add, channel_id=1, dimensions={1}
  ag.1 = bf16[1,8,128] all-gather(rs.1), replica_groups={}, channel_id=2, dimensions={1}
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ag.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(-3)
  p0 = bf16[3,8,128] parameter(0)
  cc = bf16[] constant(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(module.get(), /*last_run=*/true, 0,
                           /*pipeline_use_tree=*/false,
                           /*process_different_sized_ops=*/true,
                           CollectivePipeliner::PipeliningDirection::kForward,
                           IsAllGather)
                  .value());
  XLA_VLOG_LINES(1, module->ToString());
  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::DynamicUpdateSlice(
                        _, op::AllGather(op::GetTupleElement(op::While())),
                        op::GetTupleElement(), op::Constant(), op::Constant()));
}

TEST_F(CollectivePipelinerTest, TransformWithAgWithFormatting) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,9,128], bf16[3,9,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(0)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,9,128], bf16[3,9,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,9,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,9,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,9,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,9,128}
  mul = bf16[1,9,128] multiply(dynamic-slice.99, dynamic-slice.99)
  cpd = bf16[] constant(0)
  %pd = bf16[1,16,128] pad(mul, cpd), padding=0_0x0_7x0_0
  rs.1 = bf16[1,2,128] reduce-scatter(pd), replica_groups={}, to_apply=add, channel_id=1, dimensions={1}
  ag.1 = bf16[1,16,128] all-gather(rs.1), replica_groups={}, channel_id=2, dimensions={1}
  slc = bf16[1,9,128] slice(ag.1), slice={[0:1], [0:9], [0:128]}
  dynamic-update-slice.35 = bf16[3,9,128] dynamic-update-slice(get-tuple-element.395, slc, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,9,128], bf16[3,9,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(-3)
  p0 = bf16[3,9,128] parameter(0)
  cc = bf16[] constant(0)
  tuple = (s32[], bf16[3,9,128], bf16[3,9,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,9,128], bf16[3,9,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,9,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(module.get(), /*last_run=*/true, 0,
                           /*pipeline_use_tree=*/false,
                           /*process_different_sized_ops=*/true,
                           CollectivePipeliner::PipeliningDirection::kForward,
                           IsAllGather)
                  .value());
  XLA_VLOG_LINES(1, module->ToString());
  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root,
              op::DynamicUpdateSlice(
                  _, op::Slice(op::AllGather(op::GetTupleElement(op::While()))),
                  op::GetTupleElement(), op::Constant(), op::Constant()));
}

TEST_F(CollectivePipelinerTest, TransformWithAgInsertCustomCall) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(0)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  constant.2561 = s32[] constant(0)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5, get-tuple-element.394, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  rs.1 = bf16[1,1,128] reduce-scatter(mul), replica_groups={}, to_apply=add, channel_id=1, dimensions={1}
  ag.1 = bf16[1,8,128] all-gather(rs.1), replica_groups={}, channel_id=2, dimensions={1}
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ag.1, get-tuple-element.394, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(-8)
  p0 = bf16[3,8,128] parameter(0)
  cc = bf16[] constant(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(module.get(), /*last_run=*/false, 0,
                           /*pipeline_use_tree=*/false,
                           /*process_different_sized_ops=*/true,
                           CollectivePipeliner::PipeliningDirection::kForward,
                           IsAllGather)
                  .value());
  XLA_VLOG_LINES(1, module->ToString());
  RunOptimizer(module.get(), /*last_run=*/true, 1).value();
  XLA_VLOG_LINES(1, module->ToString());
  auto* root = module->entry_computation()->root_instruction();
  // Matching the pattern we expect for the output of the loop when an
  // all-gather is pipelined through the loop. We dynamic-slice the stacked
  // data, perform the all-gather and then put it in the stacked data again.
  EXPECT_THAT(root, op::DynamicUpdateSlice(
                        _, op::AllGather(op::GetTupleElement(op::While())),
                        op::GetTupleElement(), op::Constant(), op::Constant()));
}

TEST_F(CollectivePipelinerTest, PushAgOver) {
  constexpr absl::string_view hlo_string = R"(
HloModule module, entry_computation_layout={(bf16[3,8,128]{2,1,0})->bf16[3,8,128]{2,1,0}}

%add (lhs: bf16[], rhs: bf16[]) -> bf16[] {
  %lhs = bf16[] parameter(0)
  %rhs = bf16[] parameter(1)
  ROOT %add = bf16[] add(bf16[] %lhs, bf16[] %rhs)
}

%while_body.clone (loop_peel_param: (s32[], bf16[3,8,128], s32[])) -> (s32[], bf16[3,8,128], s32[]) {
  %loop_peel_param = (s32[], bf16[3,8,128]{2,1,0}, s32[]) parameter(0)
  %get-tuple-element.2 = s32[] get-tuple-element((s32[], bf16[3,8,128]{2,1,0}, s32[]) %loop_peel_param), index=0
  %constant.7 = s32[] constant(1)
  %add.4 = s32[] add(s32[] %get-tuple-element.2, s32[] %constant.7)
  %get-tuple-element.3 = bf16[3,8,128]{2,1,0} get-tuple-element((s32[], bf16[3,8,128]{2,1,0}, s32[]) %loop_peel_param), index=1
  %get-tuple-element.4 = s32[] get-tuple-element((s32[], bf16[3,8,128]{2,1,0}, s32[]) %loop_peel_param), index=2
  %constant.12 = s64[] constant(1)
  %custom-call = s32[] custom-call(s32[] %get-tuple-element.4, s64[] %constant.12), custom_call_target="InsertedByPreviousStep"
  %constant.13 = s32[] constant(0)
  %constant.10 = s32[] constant(0)
  %dynamic-slice.2 = bf16[1,8,128]{2,1,0} dynamic-slice(bf16[3,8,128]{2,1,0} %get-tuple-element.3, s32[] %custom-call, s32[] %constant.13, s32[] %constant.13), dynamic_slice_sizes={1,8,128}
  %ar.2 = bf16[1,1,128]{2,1,0} reduce-scatter(bf16[1,8,128]{2,1,0} %dynamic-slice.2), channel_id=2, replica_groups={}, to_apply=%add, dimensions={1}
  %ag.2 = bf16[1,8,128]{2,1,0} all-gather(bf16[1,1,128]{2,1,0} %ar.2), channel_id=32, replica_groups={}, dimensions={1}
  %dynamic-update-slice.2 = bf16[3,8,128]{2,1,0} dynamic-update-slice(bf16[3,8,128]{2,1,0} %get-tuple-element.3, bf16[1,8,128]{2,1,0} %ag.2, s32[] %custom-call, s32[] %constant.13, s32[] %constant.13)
  %dynamic-slice.1 = bf16[1,8,128]{2,1,0} dynamic-slice(bf16[3,8,128]{2,1,0} %get-tuple-element.3, s32[] %get-tuple-element.2, s32[] %constant.10, s32[] %constant.10), dynamic_slice_sizes={1,8,128}
  %mul.2 = bf16[1,8,128]{2,1,0} multiply(bf16[1,8,128]{2,1,0} %dynamic-slice.1, bf16[1,8,128]{2,1,0} %dynamic-slice.1)
  %constant.15 = s32[] constant(0)
  %dynamic-update-slice.4 = bf16[3,8,128]{2,1,0} dynamic-update-slice(bf16[3,8,128]{2,1,0} %dynamic-update-slice.2, bf16[1,8,128]{2,1,0} %mul.2, s32[] %get-tuple-element.2, s32[] %constant.15, s32[] %constant.15)
  ROOT %tuple.3 = (s32[], bf16[3,8,128]{2,1,0}, s32[]) tuple(s32[] %add.4, bf16[3,8,128]{2,1,0} %dynamic-update-slice.4, s32[] %get-tuple-element.2)
}

%while_cond.clone (loop_peel_cond_param: (s32[], bf16[3,8,128], s32[])) -> pred[] {
  %loop_peel_cond_param = (s32[], bf16[3,8,128]{2,1,0}, s32[]) parameter(0)
  %gte.1 = s32[] get-tuple-element((s32[], bf16[3,8,128]{2,1,0}, s32[]) %loop_peel_cond_param), index=0
  %constant.6 = s32[] constant(0)
  ROOT %cmp.1 = pred[] compare(s32[] %gte.1, s32[] %constant.6), direction=LT
}

ENTRY %entry (p0: bf16[3,8,128]) -> bf16[3,8,128] {
  %c0 = s32[] constant(-3)
  %p0 = bf16[3,8,128]{2,1,0} parameter(0)
  %tuple.1 = (s32[], bf16[3,8,128]{2,1,0}) tuple(s32[] %c0, bf16[3,8,128]{2,1,0} %p0)
  %get-tuple-element.0 = s32[] get-tuple-element((s32[], bf16[3,8,128]{2,1,0}) %tuple.1), index=0
  %constant.0 = s32[] constant(1)
  %constant.4 = s32[] constant(0)
  %add.1 = s32[] add(s32[] %get-tuple-element.0, s32[] %constant.0)
  %get-tuple-element.1 = bf16[3,8,128]{2,1,0} get-tuple-element((s32[], bf16[3,8,128]{2,1,0}) %tuple.1), index=1
  %dynamic-slice.0 = bf16[1,8,128]{2,1,0} dynamic-slice(bf16[3,8,128]{2,1,0} %get-tuple-element.1, s32[] %get-tuple-element.0, s32[] %constant.4, s32[] %constant.4), dynamic_slice_sizes={1,8,128}
  %mul.1 = bf16[1,8,128]{2,1,0} multiply(bf16[1,8,128]{2,1,0} %dynamic-slice.0, bf16[1,8,128]{2,1,0} %dynamic-slice.0)
  %dynamic-update-slice.0 = bf16[3,8,128]{2,1,0} dynamic-update-slice(bf16[3,8,128]{2,1,0} %get-tuple-element.1, bf16[1,8,128]{2,1,0} %mul.1, s32[] %get-tuple-element.0, s32[] %constant.4, s32[] %constant.4)
  %tuple.4 = (s32[], bf16[3,8,128]{2,1,0}, s32[]) tuple(s32[] %add.1, bf16[3,8,128]{2,1,0} %dynamic-update-slice.0, s32[] %get-tuple-element.0)
  %while.1 = (s32[], bf16[3,8,128]{2,1,0}, s32[]) while((s32[], bf16[3,8,128]{2,1,0}, s32[]) %tuple.4), condition=%while_cond.clone, body=%while_body.clone
  %get-tuple-element.6 = bf16[3,8,128]{2,1,0} get-tuple-element((s32[], bf16[3,8,128]{2,1,0}, s32[]) %while.1), index=1
  %get-tuple-element.5 = s32[] get-tuple-element((s32[], bf16[3,8,128]{2,1,0}, s32[]) %while.1), index=2
  %constant.14 = s32[] constant(0)
  %dynamic-slice.3 = bf16[1,8,128]{2,1,0} dynamic-slice(bf16[3,8,128]{2,1,0} %get-tuple-element.6, s32[] %get-tuple-element.5, s32[] %constant.14, s32[] %constant.14), dynamic_slice_sizes={1,8,128}
  %ar.3 = bf16[1,8,128]{2,1,0} all-reduce(bf16[1,8,128]{2,1,0} %dynamic-slice.3), channel_id=3, replica_groups={}, to_apply=%add
  ROOT %dynamic-update-slice.3 = bf16[3,8,128]{2,1,0} dynamic-update-slice(bf16[3,8,128]{2,1,0} %get-tuple-element.6, bf16[1,8,128]{2,1,0} %ar.3, s32[] %get-tuple-element.5, s32[] %constant.14, s32[] %constant.14)
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(module.get(), /*last_run=*/true, 1,
                           /*pipeline_use_tree=*/false,
                           /*process_different_sized_ops=*/true,
                           CollectivePipeliner::PipeliningDirection::kForward,
                           IsAllGather)
                  .value());
  XLA_VLOG_LINES(1, module->ToString());
  auto* root = module->entry_computation()->root_instruction();
  // Check that the all-gather can be pipelined after we had already a previous
  // round of pipelining performed previously for another op. (in this case
  // AllReduce).
  EXPECT_THAT(
      root,
      op::DynamicUpdateSlice(
          op::DynamicUpdateSlice(_, op::AllGather(), _, _, _),
          op::AllReduce(op::DynamicSlice(op::DynamicUpdateSlice(), _, _, _)),
          op::GetTupleElement(), op::Constant(), op::Constant()));
}

TEST_F(CollectivePipelinerTest, NoPushAgOverBecauseDifferentSize) {
  constexpr absl::string_view hlo_string = R"(
HloModule module, entry_computation_layout={(bf16[3,8,128]{2,1,0})->bf16[3,8,128]{2,1,0}}

%add (lhs: bf16[], rhs: bf16[]) -> bf16[] {
  %lhs = bf16[] parameter(0)
  %rhs = bf16[] parameter(1)
  ROOT %add = bf16[] add(bf16[] %lhs, bf16[] %rhs)
}

%while_body.clone (loop_peel_param: (s32[], bf16[3,8,128], s32[])) -> (s32[], bf16[3,8,128], s32[]) {
  %loop_peel_param = (s32[], bf16[3,8,128]{2,1,0}, s32[]) parameter(0)
  %get-tuple-element.2 = s32[] get-tuple-element((s32[], bf16[3,8,128]{2,1,0}, s32[]) %loop_peel_param), index=0
  %constant.7 = s32[] constant(1)
  %add.4 = s32[] add(s32[] %get-tuple-element.2, s32[] %constant.7)
  %get-tuple-element.3 = bf16[3,8,128]{2,1,0} get-tuple-element((s32[], bf16[3,8,128]{2,1,0}, s32[]) %loop_peel_param), index=1
  %get-tuple-element.4 = s32[] get-tuple-element((s32[], bf16[3,8,128]{2,1,0}, s32[]) %loop_peel_param), index=2
  %constant.12 = s64[] constant(1)
  %custom-call = s32[] custom-call(s32[] %get-tuple-element.4, s64[] %constant.12), custom_call_target="InsertedByPreviousStep"
  %constant.13 = s32[] constant(0)
  %constant.10 = s32[] constant(0)
  %dynamic-slice.2 = bf16[1,8,128]{2,1,0} dynamic-slice(bf16[3,8,128]{2,1,0} %get-tuple-element.3, s32[] %custom-call, s32[] %constant.13, s32[] %constant.13), dynamic_slice_sizes={1,8,128}
  %ar.2 = bf16[1,1,128]{2,1,0} reduce-scatter(bf16[1,8,128]{2,1,0} %dynamic-slice.2), channel_id=2, replica_groups={}, to_apply=%add, dimensions={1}
  %ag.2 = bf16[1,8,128]{2,1,0} all-gather(bf16[1,1,128]{2,1,0} %ar.2), channel_id=32, replica_groups={}, dimensions={1}
  %dynamic-update-slice.2 = bf16[3,8,128]{2,1,0} dynamic-update-slice(bf16[3,8,128]{2,1,0} %get-tuple-element.3, bf16[1,8,128]{2,1,0} %ag.2, s32[] %custom-call, s32[] %constant.13, s32[] %constant.13)
  %dynamic-slice.1 = bf16[1,8,128]{2,1,0} dynamic-slice(bf16[3,8,128]{2,1,0} %get-tuple-element.3, s32[] %get-tuple-element.2, s32[] %constant.10, s32[] %constant.10), dynamic_slice_sizes={1,8,128}
  %mul.2 = bf16[1,8,128]{2,1,0} multiply(bf16[1,8,128]{2,1,0} %dynamic-slice.1, bf16[1,8,128]{2,1,0} %dynamic-slice.1)
  %constant.15 = s32[] constant(0)
  %dynamic-update-slice.4 = bf16[3,8,128]{2,1,0} dynamic-update-slice(bf16[3,8,128]{2,1,0} %dynamic-update-slice.2, bf16[1,8,128]{2,1,0} %mul.2, s32[] %get-tuple-element.2, s32[] %constant.15, s32[] %constant.15)
  ROOT %tuple.3 = (s32[], bf16[3,8,128]{2,1,0}, s32[]) tuple(s32[] %add.4, bf16[3,8,128]{2,1,0} %dynamic-update-slice.4, s32[] %get-tuple-element.2)
}

%while_cond.clone (loop_peel_cond_param: (s32[], bf16[3,8,128], s32[])) -> pred[] {
  %loop_peel_cond_param = (s32[], bf16[3,8,128]{2,1,0}, s32[]) parameter(0)
  %gte.1 = s32[] get-tuple-element((s32[], bf16[3,8,128]{2,1,0}, s32[]) %loop_peel_cond_param), index=0
  %constant.6 = s32[] constant(0)
  ROOT %cmp.1 = pred[] compare(s32[] %gte.1, s32[] %constant.6), direction=LT
}

ENTRY %entry (p0: bf16[3,8,128]) -> bf16[3,8,128] {
  %c0 = s32[] constant(-3)
  %p0 = bf16[3,8,128]{2,1,0} parameter(0)
  %tuple.1 = (s32[], bf16[3,8,128]{2,1,0}) tuple(s32[] %c0, bf16[3,8,128]{2,1,0} %p0)
  %get-tuple-element.0 = s32[] get-tuple-element((s32[], bf16[3,8,128]{2,1,0}) %tuple.1), index=0
  %constant.0 = s32[] constant(1)
  %constant.4 = s32[] constant(0)
  %add.1 = s32[] add(s32[] %get-tuple-element.0, s32[] %constant.0)
  %get-tuple-element.1 = bf16[3,8,128]{2,1,0} get-tuple-element((s32[], bf16[3,8,128]{2,1,0}) %tuple.1), index=1
  %dynamic-slice.0 = bf16[1,8,128]{2,1,0} dynamic-slice(bf16[3,8,128]{2,1,0} %get-tuple-element.1, s32[] %get-tuple-element.0, s32[] %constant.4, s32[] %constant.4), dynamic_slice_sizes={1,8,128}
  %mul.1 = bf16[1,8,128]{2,1,0} multiply(bf16[1,8,128]{2,1,0} %dynamic-slice.0, bf16[1,8,128]{2,1,0} %dynamic-slice.0)
  %dynamic-update-slice.0 = bf16[3,8,128]{2,1,0} dynamic-update-slice(bf16[3,8,128]{2,1,0} %get-tuple-element.1, bf16[1,8,128]{2,1,0} %mul.1, s32[] %get-tuple-element.0, s32[] %constant.4, s32[] %constant.4)
  %tuple.4 = (s32[], bf16[3,8,128]{2,1,0}, s32[]) tuple(s32[] %add.1, bf16[3,8,128]{2,1,0} %dynamic-update-slice.0, s32[] %get-tuple-element.0)
  %while.1 = (s32[], bf16[3,8,128]{2,1,0}, s32[]) while((s32[], bf16[3,8,128]{2,1,0}, s32[]) %tuple.4), condition=%while_cond.clone, body=%while_body.clone
  %get-tuple-element.6 = bf16[3,8,128]{2,1,0} get-tuple-element((s32[], bf16[3,8,128]{2,1,0}, s32[]) %while.1), index=1
  %get-tuple-element.5 = s32[] get-tuple-element((s32[], bf16[3,8,128]{2,1,0}, s32[]) %while.1), index=2
  %constant.14 = s32[] constant(0)
  %dynamic-slice.3 = bf16[1,8,128]{2,1,0} dynamic-slice(bf16[3,8,128]{2,1,0} %get-tuple-element.6, s32[] %get-tuple-element.5, s32[] %constant.14, s32[] %constant.14), dynamic_slice_sizes={1,8,128}
  %ar.3 = bf16[1,8,128]{2,1,0} all-reduce(bf16[1,8,128]{2,1,0} %dynamic-slice.3), channel_id=3, replica_groups={}, to_apply=%add
  ROOT %dynamic-update-slice.3 = bf16[3,8,128]{2,1,0} dynamic-update-slice(bf16[3,8,128]{2,1,0} %get-tuple-element.6, bf16[1,8,128]{2,1,0} %ar.3, s32[] %get-tuple-element.5, s32[] %constant.14, s32[] %constant.14)
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_FALSE(RunOptimizer(module.get(), /*last_run=*/false, 1,
                            /*pipeline_use_tree=*/false,
                            /*process_different_sized_ops=*/false,
                            CollectivePipeliner::PipeliningDirection::kForward,
                            IsAllGather)
                   .value());
  XLA_VLOG_LINES(1, module->ToString());
}

TEST_F(CollectivePipelinerTest, TransformIncrementByTwoFormat) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,16,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,16,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.396 = bf16[3,16,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(2)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,16,128] dynamic-slice(get-tuple-element.396, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,16,128}
  mul = bf16[1,16,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,16,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  ds.1 = bf16[1,8,128] dynamic-slice(ar.1, constant.2561, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ds.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,16,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.396)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,16,128] parameter(0)
  c1 = bf16[] constant(0)
  b1 = bf16[3,8,128] broadcast(c1), dimensions={}
  tuple = (s32[], bf16[3,8,128], bf16[3,16,128]) tuple(c0, b1, p0)
  while = (s32[], bf16[3,8,128], bf16[3,16,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(module.get(), /*last_run=*/true).value());
  XLA_VLOG_LINES(1, module->ToString());
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      op::DynamicUpdateSlice(
          _, op::DynamicSlice(op::AllReduce(op::GetTupleElement()), _, _, _), _,
          _, _));
}

TEST_F(CollectivePipelinerTest, TransformIncrementByTwoFormatTranspose) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,16,128], bf16[3,16,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,16,128], bf16[3,16,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,16,128] get-tuple-element(param), index=1
  get-tuple-element.396 = bf16[3,16,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(2)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,16,128] dynamic-slice(get-tuple-element.396, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,16,128}
  mul = bf16[1,16,128] multiply(dynamic-slice.99, dynamic-slice.99)
  reshape.1 = bf16[2,16,64] reshape(mul)
  ar.1 = bf16[2,16,64] all-reduce(reshape.1), replica_groups={}, to_apply=add, channel_id=1
  transpose.1 = bf16[64,2,16] transpose(ar.1), dimensions={2,0,1}
  reshape.2 = bf16[1,16,128] reshape(transpose.1)
  dynamic-update-slice.35 = bf16[3,16,128] dynamic-update-slice(get-tuple-element.395, reshape.2, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,16,128], bf16[3,16,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.396)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,16,128] parameter(0)
  c1 = bf16[] constant(0)
  b1 = bf16[3,16,128] broadcast(c1), dimensions={}
  tuple.1 = (s32[], bf16[3,16,128], bf16[3,16,128]) tuple(c0, b1, p0)
  while = (s32[], bf16[3,16,128], bf16[3,16,128]) while(tuple.1), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,16,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(module.get(), /*last_run=*/true).value());
  XLA_VLOG_LINES(1, module->ToString());
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      op::DynamicUpdateSlice(
          _, op::Reshape(op::Transpose(op::AllReduce(op::GetTupleElement()))),
          _, _, _));
}

TEST_F(CollectivePipelinerTest, TransformIncrementIndexByOneBackwards) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,1,2,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,1,2,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.k = bf16[3,1,2,128] get-tuple-element(param), index=2
  constant.2561 = s32[] constant(0)
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.k = bf16[1,1,2,128] dynamic-slice(get-tuple-element.k, select.1348, constant.2561, constant.2561, constant.2561), dynamic_slice_sizes={1,1,2,128}
  r = bf16[1,2,128] reshape(dynamic-slice.k)
  a = bf16[1,2,128] add(r, r)
  ag = bf16[1,8,128] all-gather(a), dimensions={1}, replica_groups={}
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.395, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, ag)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ar.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,1,2,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.k)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  p1 = bf16[3,1,2,128] parameter(1)
  tuple = (s32[], bf16[3,8,128], bf16[3,1,2,128]) tuple(c0, p0, p1)
  while = (s32[], bf16[3,8,128], bf16[3,1,2,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(module.get(), /*last_run=*/true, 0,
                           /*pipeline_use_tree=*/false,
                           /*process_different_sized_ops=*/false,
                           CollectivePipeliner::PipeliningDirection::kBackward,
                           IsAllGather)
                  .value());
  XLA_VLOG_LINES(1, module->ToString());
  const int64_t while_count = absl::c_count_if(
      module->entry_computation()->instructions(),
      [](const HloInstruction* instruction) {
        return HloPredicateIsOp<HloOpcode::kWhile>(instruction);
      });
  EXPECT_EQ(while_count, 1);
}

TEST_F(CollectivePipelinerTest,
       TransformIncrementIndexByOneBackwardsModifyOut) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,1,2,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,1,2,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.k = bf16[3,1,2,128] get-tuple-element(param), index=2
  constant.2561 = s32[] constant(0)
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.k = bf16[1,1,2,128] dynamic-slice(get-tuple-element.k, select.1348, constant.2561, constant.2561, constant.2561), dynamic_slice_sizes={1,1,2,128}
  r = bf16[1,2,128] reshape(dynamic-slice.k)
  a = bf16[1,2,128] add(r, r)
  ag = bf16[1,8,128] all-gather(a), dimensions={1}, replica_groups={}
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.395, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, ag)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ar.1, select.1348, constant.2561, constant.2561)
  constant.10 = bf16[] constant(0)
  b = bf16[3,1,2,128] broadcast(constant.10), dimensions={}
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,1,2,128]) tuple(add.230, dynamic-update-slice.35, b)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  p1 = bf16[3,1,2,128] parameter(1)
  tuple = (s32[], bf16[3,8,128], bf16[3,1,2,128]) tuple(c0, p0, p1)
  while = (s32[], bf16[3,8,128], bf16[3,1,2,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_FALSE(RunOptimizer(module.get(), /*last_run=*/true, 0,
                            /*pipeline_use_tree=*/false,
                            /*process_different_sized_ops=*/false,
                            CollectivePipeliner::PipeliningDirection::kBackward,
                            IsAllGather)
                   .value());
  XLA_VLOG_LINES(1, module->ToString());
}

TEST_F(CollectivePipelinerTest,
       TransformIncrementIndexByOneBackwardsPlusForward) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,1,2,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,1,2,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.k = bf16[3,1,2,128] get-tuple-element(param), index=2
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=3
  constant.2561 = s32[] constant(0)
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.k = bf16[1,1,2,128] dynamic-slice(get-tuple-element.k, select.1348, constant.2561, constant.2561, constant.2561), dynamic_slice_sizes={1,1,2,128}
  r = bf16[1,2,128] reshape(dynamic-slice.k)
  a = bf16[1,2,128] add(r, r)
  ag = bf16[1,8,128] all-gather(a), dimensions={1}, replica_groups={}
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, ag)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ar.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,1,2,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.k, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  p1 = bf16[3,1,2,128] parameter(1)
  tuple = (s32[], bf16[3,8,128], bf16[3,1,2,128], bf16[3,8,128]) tuple(c0, p0, p1, p0)
  while = (s32[], bf16[3,8,128], bf16[3,1,2,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(module.get(), /*last_run=*/false, 0,
                           /*pipeline_use_tree=*/false,
                           /*process_different_sized_ops=*/true,
                           CollectivePipeliner::PipeliningDirection::kBackward,
                           IsAllGather)
                  .value());
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(RunOptimizer(module.get(), /*last_run=*/true, 0,
                           /*pipeline_use_tree=*/false,
                           /*process_different_sized_ops=*/true,
                           CollectivePipeliner::PipeliningDirection::kForward)
                  .value());
  XLA_VLOG_LINES(1, module->ToString());
}

TEST_F(CollectivePipelinerTest,
       TransformIncrementIndexByOneBackwardsPlusForwardConvertOutput) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], f32[3,8,128], bf16[3,1,2,128], f32[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], f32[3,8,128], bf16[3,1,2,128], f32[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = f32[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.k = bf16[3,1,2,128] get-tuple-element(param), index=2
  get-tuple-element.5 = f32[3,8,128] get-tuple-element(param), index=3
  constant.2561 = s32[] constant(0)
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.k = bf16[1,1,2,128] dynamic-slice(get-tuple-element.k, select.1348, constant.2561, constant.2561, constant.2561), dynamic_slice_sizes={1,1,2,128}
  r = bf16[1,2,128] reshape(dynamic-slice.k)
  a = bf16[1,2,128] add(r, r)
  ag = bf16[1,8,128] all-gather(a), dimensions={1}, replica_groups={}
  dynamic-slice.99 = f32[1,8,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  cvt.0 = bf16[1,8,128] convert(dynamic-slice.99)
  mul = bf16[1,8,128] multiply(cvt.0, ag)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  cvt.1 = f32[1,8,128] convert(ar.1)
  dynamic-update-slice.35 = f32[3,8,128] dynamic-update-slice(get-tuple-element.395, cvt.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], f32[3,8,128], bf16[3,1,2,128], f32[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.k, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = f32[3,8,128] parameter(0)
  p1 = bf16[3,1,2,128] parameter(1)
  tuple = (s32[], f32[3,8,128], bf16[3,1,2,128], f32[3,8,128]) tuple(c0, p0, p1, p0)
  while = (s32[], f32[3,8,128], bf16[3,1,2,128], f32[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = f32[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(module.get(), /*last_run=*/false, 0,
                           /*pipeline_use_tree=*/false,
                           /*process_different_sized_ops=*/true,
                           CollectivePipeliner::PipeliningDirection::kBackward,
                           IsAllGather)
                  .value());
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_TRUE(RunOptimizer(module.get(), /*last_run=*/true, 0,
                           /*pipeline_use_tree=*/false,
                           /*process_different_sized_ops=*/true,
                           CollectivePipeliner::PipeliningDirection::kForward)
                  .value());
  XLA_VLOG_LINES(1, module->ToString());
}

TEST_F(CollectivePipelinerTest, MultiUsesElementwise) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  c2 = bf16[] constant(2.0)
  bc = bf16[1,8,128] broadcast(c2)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  mul2 = bf16[1,8,128] multiply(ar.1, bc)
  mul3 = bf16[1,8,128] multiply(mul2, ar.1)
  mul4 = bf16[1,8,128] multiply(mul3, mul)
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, mul4, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(module.get(), /*last_run=*/true, 0,
                           /*pipeline_use_tree=*/true,
                           /*process_different_sized_ops=*/true,
                           CollectivePipeliner::PipeliningDirection::kForward)
                  .value());
  XLA_VLOG_LINES(1, module->ToString());
}

TEST_F(CollectivePipelinerTest, ElementWiseUser) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  mul2 = bf16[1,8,128] multiply(ar.1, mul)
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, mul2, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(module.get(), /*last_run=*/true, 0,
                           /*pipeline_use_tree=*/true,
                           /*process_different_sized_ops=*/true,
                           CollectivePipeliner::PipeliningDirection::kForward)
                  .value());
  XLA_VLOG_LINES(1, module->ToString());
}

TEST_F(CollectivePipelinerTest, TransformIncrementIndexByOneNotFirstIdxSink) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.35 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.35, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  %c = bf16[] custom-call(), custom_call_target="Boh"
  %b = bf16[1,8,128] broadcast(c), dimensions={}
  %a = bf16[1,8,128] add(ar.1, b)
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, a, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.35)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(module.get(), /*last_run=*/true,
                           /*level_to_operate_on=*/0,
                           /*pipeline_use_tree=*/true,
                           /*process_different_sized_ops=*/true,
                           CollectivePipeliner::kForwardSink)
                  .value());
  XLA_VLOG_LINES(0, module->ToString());
}

TEST_F(CollectivePipelinerTest,
       TransformIncrementIndexByOneNotFirstIdxSinkCustomCall) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.35 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.35, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  %c = bf16[] custom-call(), custom_call_target="Boh"
  %b = bf16[1,8,128] broadcast(c), dimensions={}
  %a = bf16[1,8,128] add(ar.1, b)
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, a, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.35)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(module.get(), /*last_run=*/false,
                           /*level_to_operate_on=*/0,
                           /*pipeline_use_tree=*/true,
                           /*process_different_sized_ops=*/true,
                           CollectivePipeliner::kForwardSink)
                  .value());
  XLA_VLOG_LINES(1, module->ToString());
  const HloInstruction* all_reduce = module->entry_computation()
                                         ->root_instruction()
                                         ->operand(0)
                                         ->operand(1)
                                         ->operand(0)
                                         ->operand(0);
  EXPECT_EQ(all_reduce->opcode(), HloOpcode::kAllReduce);
  EXPECT_EQ(all_reduce->shape().dimensions(0), 3);
}

// Checks that we shouldn't pipeline Send/Recv by accident while pipelining
// other collective, such as all-gather. In the test, the chain leading to
// all-gather contains Recv/Recv-done, which prevents us from pipelining the
// all-gather backward.
TEST_F(CollectivePipelinerTest, NotTransformAllGatherWithRecvInChainBackwards) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,1,2,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,1,2,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.k = bf16[3,1,2,128] get-tuple-element(param), index=2
  constant.2561 = s32[] constant(0)
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)

  after-all = token[] after-all()
  recv = (bf16[1,1,2,128], u32[], token[]) recv(after-all), channel_id=2, frontend_attributes={
      _xla_send_recv_source_target_pairs="{{0, 1}, {1, 2}, {2, 3}, {3, 4}}"
    }
  send = (bf16[1,1,2,128], u32[], token[]) send(get-tuple-element.k, after-all), channel_id=2, frontend_attributes={
      _xla_send_recv_source_target_pairs="{{0, 1}, {1, 2}, {2, 3}, {3, 4}}"
    }
  send-done = token[] send-done(send), channel_id=2
  recv-done = (bf16[1,1,2,128], token[]) recv-done(recv), channel_id=2
  recv-data = bf16[1,1,2,128] get-tuple-element(recv-done), index=0

  dynamic-slice.k = bf16[1,1,2,128] dynamic-slice(recv-data, select.1348, constant.2561, constant.2561, constant.2561), dynamic_slice_sizes={1,1,2,128}
  r = bf16[1,2,128] reshape(dynamic-slice.k)
  a = bf16[1,2,128] add(r, r)
  ag = bf16[1,8,128] all-gather(a), dimensions={1}, replica_groups={}
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.395, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, ag)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, ar.1, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,1,2,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.k)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  p1 = bf16[3,1,2,128] parameter(1)
  tuple = (s32[], bf16[3,8,128], bf16[3,1,2,128]) tuple(c0, p0, p1)
  while = (s32[], bf16[3,8,128], bf16[3,1,2,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_FALSE(RunOptimizer(module.get(), /*last_run=*/true, 0,
                            /*pipeline_use_tree=*/false,
                            /*process_different_sized_ops=*/false,
                            CollectivePipeliner::PipeliningDirection::kBackward,
                            IsAllGather)
                   .value());
}

TEST_F(CollectivePipelinerTest, TransformRecvSendBackwards) {
  constexpr absl::string_view hlo_string = R"(
  HloModule module
  cond {
    param = (u32[], f32[1, 1024, 1024]) parameter(0)
    count = get-tuple-element(%param), index=0
    ub = u32[] constant(25)
    ROOT result = pred[] compare(count, ub), direction=LT
  }

  body {
    param = (u32[], f32[1, 1024, 1024]) parameter(0)
    count = get-tuple-element(%param), index=0
    p = get-tuple-element(%param), index=1
    c1 = u32[] constant(1)
    new_count = u32[] add(count, c1)

    after-all = token[] after-all()
    recv = (f32[1, 1024, 1024], u32[], token[]) recv(after-all), channel_id=1, frontend_attributes={
      _xla_send_recv_source_target_pairs="{{0, 1}, {1, 2}, {2, 3}, {3, 4}}"
    }
    send = (f32[1, 1024, 1024], u32[], token[]) send(p, after-all), channel_id=1, frontend_attributes={
      _xla_send_recv_source_target_pairs="{{0, 1}, {1, 2}, {2, 3}, {3, 4}}"
    }
    recv-done = (f32[1, 1024, 1024], token[]) recv-done(recv), channel_id=1
    recv-data = f32[1, 1024, 1024] get-tuple-element(recv-done), index=0

    replica = u32[] replica-id()
    c10 = u32[] constant(10)
    sum = u32[] add(replica, c10)
    sum2 = u32[] add(sum, count)
    conv = f32[] convert(sum2)
    b = f32[1, 1024, 1024] add(p, recv-data)
    c = f32[1, 1024, 1024] multiply(b, b)
    d = f32[1, 1024, 1024] tan(c)
    s = f32[1, 1024, 1024] dot(c, d), lhs_batch_dims={0}, lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1}

    send-done = token[] send-done(send), channel_id=1
    ROOT result = (u32[], f32[1, 1024, 1024]) tuple(new_count, s)
  }

  ENTRY test_computation {
    c0 = u32[] constant(0)
    f0 = f32[] constant(0.0)
    init = f32[1, 1024, 1024] broadcast(f0), dimensions={}
    while_init = (u32[], f32[1, 1024, 1024]) tuple(c0, init)
    while_result = (u32[], f32[1, 1024, 1024]) while(while_init), body=body, condition=cond, backend_config="{\"known_trip_count\":{\"n\":\"25\"}}"
    ROOT result = f32[1, 1024, 1024] get-tuple-element(while_result), index=1
  }
  )";

  auto should_pipeline = [](const HloInstruction* instruction) {
    if (!HloPredicateIsOp<HloOpcode::kRecvDone>(instruction)) return false;
    const HloRecvDoneInstruction* recv_done =
        dynamic_cast<const HloRecvDoneInstruction*>(instruction);
    if (recv_done->is_host_transfer()) return false;
    // Check that the recv-done is used for non-trivial computation, which can
    // also help avoid repeatedly pipelining a loop.
    return (recv_done->user_count() == 1 && recv_done->parent() != nullptr &&
            recv_done->users()[0] != recv_done->parent()->root_instruction());
  };
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(module.get(), /*last_run=*/true, 0,
                           /*pipeline_use_tree=*/false,
                           /*process_different_sized_ops=*/false,
                           CollectivePipeliner::PipeliningDirection::kBackward,
                           should_pipeline)
                  .value());
  XLA_VLOG_LINES(10, module->ToString());
  auto recv1 =
      DynCast<HloRecvInstruction>(FindInstruction(module.get(), "recv.1"));
  EXPECT_NE(recv1, nullptr);
  auto recv2 =
      DynCast<HloRecvInstruction>(FindInstruction(module.get(), "recv.2"));
  EXPECT_NE(recv2, nullptr);
  EXPECT_EQ(recv1->channel_id(), recv2->channel_id());

  auto send1 =
      DynCast<HloSendInstruction>(FindInstruction(module.get(), "send.1"));
  EXPECT_NE(send1, nullptr);
  auto send2 =
      DynCast<HloSendInstruction>(FindInstruction(module.get(), "send.2"));
  EXPECT_NE(send2, nullptr);
  EXPECT_EQ(send1->channel_id(), send2->channel_id());

  EXPECT_EQ(recv1->channel_id(), send1->channel_id());
}

TEST_F(CollectivePipelinerTest, MultiUsesElementwiseMerge) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  c2 = bf16[] constant(2.0)
  bc = bf16[1,8,128] broadcast(c2)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  ar.2 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  mul2 = bf16[1,8,128] multiply(ar.1, bc)
  mul3 = bf16[1,8,128] multiply(mul2, ar.2)
  mul4 = bf16[1,8,128] multiply(mul3, mul)
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, mul4, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(module.get(), /*last_run=*/true, 0,
                           /*pipeline_use_tree=*/true,
                           /*process_different_sized_ops=*/true,
                           CollectivePipeliner::PipeliningDirection::kForward)
                  .value());
  XLA_VLOG_LINES(1, module->ToString());
}

TEST_F(CollectivePipelinerTest, MultiUsesElementwiseFeedTwo) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  gte = s32[] get-tuple-element(param), index=0
  constant.1 = s32[] constant(3)
  ROOT cmp = pred[] compare(gte, constant.1), direction=LT
}

while_body {
  param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
  get-tuple-element.394 = s32[] get-tuple-element(param), index=0
  get-tuple-element.395 = bf16[3,8,128] get-tuple-element(param), index=1
  get-tuple-element.5 = bf16[3,8,128] get-tuple-element(param), index=2
  constant.2557 = s32[] constant(1)
  add.230 = s32[] add(get-tuple-element.394, constant.2557)
  constant.2559 = s32[] constant(3)
  subtract.139 = s32[] subtract(constant.2559, get-tuple-element.394)
  constant.2560 = s32[] constant(-1)
  add.231 = s32[] add(subtract.139, constant.2560)
  constant.2561 = s32[] constant(0)
  compare.747 = pred[] compare(add.231, constant.2561), direction=LT
  constant.2562 = s32[] constant(2)
  add.232 = s32[] add(subtract.139, constant.2562)
  select.1348 = s32[] select(compare.747, add.232, add.231)
  dynamic-slice.99 = bf16[1,8,128] dynamic-slice(get-tuple-element.5, select.1348, constant.2561, constant.2561), dynamic_slice_sizes={1,8,128}
  mul = bf16[1,8,128] multiply(dynamic-slice.99, dynamic-slice.99)
  c2 = bf16[] constant(2.0)
  bc = bf16[1,8,128] broadcast(c2)
  ar.1 = bf16[1,8,128] all-reduce(mul), replica_groups={}, to_apply=add, channel_id=1
  ar.2 = bf16[1,8,128] all-reduce(ar.1), replica_groups={}, to_apply=add, channel_id=1
  mul2 = bf16[1,8,128] multiply(ar.1, bc)
  mul3 = bf16[1,8,128] multiply(mul2, ar.2)
  mul4 = bf16[1,8,128] multiply(mul3, mul)
  dynamic-update-slice.35 = bf16[3,8,128] dynamic-update-slice(get-tuple-element.395, mul4, select.1348, constant.2561, constant.2561)
  ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(add.230, dynamic-update-slice.35, get-tuple-element.5)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[3,8,128] parameter(0)
  tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
  while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple), condition=while_cond, body=while_body
  ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
}
)";
  auto module = ParseAndReturnUnverifiedModule(hlo_string, config_).value();
  EXPECT_TRUE(RunOptimizer(module.get(), /*last_run=*/true, 0,
                           /*pipeline_use_tree=*/true,
                           /*process_different_sized_ops=*/true,
                           CollectivePipeliner::PipeliningDirection::kForward)
                  .value());
  XLA_VLOG_LINES(1, module->ToString());
}

bool ConvIsLowerable(HloInstruction* conv) {
  return true;
}

TEST_F(CollectivePipelinerTest,
       h100bug) {
  constexpr absl::string_view hlo_string = R"(
HloModule pjit__wrapped_step_fn, input_output_alias={ {0}: (0, {}, may-alias), {1}: (1, {}, may-alias), {2}: (2, {}, may-alias), {3}: (3, {}, may-alias), {4}: (4, {}, may-alias), {5}: (9, {}, may-alias), {6}: (6, {}, may-alias), {8}: (8, {}, may-alias), {9}: (10, {}, may-alias), {10}: (11, {}, may-alias), {11}: (12, {}, may-alias), {12}: (36, {}, may-alias), {14}: (14, {}, may-alias), {15}: (37, {}, may-alias), {16}: (16, {}, may-alias), {17}: (17, {}, may-alias), {18}: (18, {}, may-alias), {19}: (19, {}, may-alias), {20}: (20, {}, may-alias), {21}: (21, {}, may-alias), {22}: (22, {}, may-alias), {23}: (23, {}, may-alias), {24}: (24, {}, may-alias), {25}: (25, {}, may-alias), {26}: (26, {}, may-alias), {27}: (27, {}, may-alias), {28}: (28, {}, may-alias), {29}: (29, {}, may-alias), {30}: (30, {}, may-alias), {31}: (31, {}, may-alias), {32}: (38, {}, may-alias), {33}: (33, {}, may-alias), {35}: (35, {}, may-alias), {36}: (39, {}, may-alias), {37}: (48, {}, may-alias), {38}: (49, {}, may-alias), {39}: (50, {}, may-alias), {41}: (41, {}, may-alias), {42}: (51, {}, may-alias), {43}: (43, {}, may-alias), {45}: (45, {}, may-alias), {47}: (47, {}, may-alias), {53}: (53, {}, may-alias), {55}: (55, {}, may-alias), {56}: (56, {}, may-alias) }, entry_computation_layout={(u32[], f32[8192]{0}, f32[8192]{0}, f32[256,8192]{1,0}, f32[1024,50304]{1,0}, /*index=5*/bf16[6,8192]{1,0}, f32[6,8192,1024]{2,1,0}, bf16[6,3,8192]{2,1,0}, f32[6,1024,3,8192]{3,2,1,0}, f32[6,8192]{1,0}, /*index=10*/f32[6,8192]{1,0}, f32[6,8192]{1,0}, f32[6,8192]{1,0}, bf16[6,1,32768]{2,1,0}, f32[6,1024,1,32768]{3,2,1,0}, /*index=15*/bf16[6,8192]{1,0}, f32[6,32768,1024]{2,1,0}, s32[], s32[], s32[], /*index=20*/f32[8192]{0}, f32[8192]{0}, f32[256,8192]{1,0}, f32[1024,50304]{1,0}, f32[8192]{0}, /*index=25*/f32[8192]{0}, f32[256,8192]{1,0}, f32[1024,50304]{1,0}, s32[], s32[6]{0}, /*index=30*/s32[6]{0}, s32[6]{0}, bf16[6,8192]{1,0}, f32[6,8192,1024]{2,1,0}, bf16[6,3,8192]{2,1,0}, /*index=35*/f32[6,1024,3,8192]{3,2,1,0}, f32[6,8192]{1,0}, f32[6,8192]{1,0}, f32[6,8192]{1,0}, f32[6,8192]{1,0}, /*index=40*/bf16[6,1,32768]{2,1,0}, f32[6,1024,1,32768]{3,2,1,0}, bf16[6,8192]{1,0}, f32[6,32768,1024]{2,1,0}, bf16[6,8192]{1,0}, /*index=45*/f32[6,8192,1024]{2,1,0}, bf16[6,3,8192]{2,1,0}, f32[6,1024,3,8192]{3,2,1,0}, f32[6,8192]{1,0}, f32[6,8192]{1,0}, /*index=50*/f32[6,8192]{1,0}, f32[6,8192]{1,0}, bf16[6,1,32768]{2,1,0}, f32[6,1024,1,32768]{3,2,1,0}, bf16[6,8192]{1,0}, /*index=55*/f32[6,32768,1024]{2,1,0}, s32[6]{0}, u32[4]{0}, f32[1]{0}, s32[1,2048]{1,0}, /*index=60*/s32[1,2048]{1,0}, f32[1,2048]{1,0}, s32[1,2048]{1,0}, s32[1,2048]{1,0}, f32[1,2048]{1,0})->(u32[], f32[8192]{0}, f32[8192]{0}, f32[256,8192]{1,0}, f32[1024,50304]{1,0}, /*index=5*/f32[6,8192]{1,0}, f32[6,8192,1024]{2,1,0}, f32[6,3,8192]{2,1,0}, f32[6,1024,3,8192]{3,2,1,0}, f32[6,8192]{1,0}, /*index=10*/f32[6,8192]{1,0}, f32[6,8192]{1,0}, f32[6,8192]{1,0}, f32[6,1,32768]{2,1,0}, f32[6,1024,1,32768]{3,2,1,0}, /*index=15*/f32[6,8192]{1,0}, f32[6,32768,1024]{2,1,0}, s32[], s32[], s32[], /*index=20*/f32[8192]{0}, f32[8192]{0}, f32[256,8192]{1,0}, f32[1024,50304]{1,0}, f32[8192]{0}, /*index=25*/f32[8192]{0}, f32[256,8192]{1,0}, f32[1024,50304]{1,0}, s32[], s32[6]{0}, /*index=30*/s32[6]{0}, s32[6]{0}, f32[6,8192]{1,0}, f32[6,8192,1024]{2,1,0}, f32[6,3,8192]{2,1,0}, /*index=35*/f32[6,1024,3,8192]{3,2,1,0}, f32[6,8192]{1,0}, f32[6,8192]{1,0}, f32[6,8192]{1,0}, f32[6,8192]{1,0}, /*index=40*/f32[6,1,32768]{2,1,0}, f32[6,1024,1,32768]{3,2,1,0}, f32[6,8192]{1,0}, f32[6,32768,1024]{2,1,0}, f32[6,8192]{1,0}, /*index=45*/f32[6,8192,1024]{2,1,0}, f32[6,3,8192]{2,1,0}, f32[6,1024,3,8192]{3,2,1,0}, f32[6,8192]{1,0}, f32[6,8192]{1,0}, /*index=50*/f32[6,8192]{1,0}, f32[6,8192]{1,0}, f32[6,1,32768]{2,1,0}, f32[6,1024,1,32768]{3,2,1,0}, f32[6,8192]{1,0}, /*index=55*/f32[6,32768,1024]{2,1,0}, s32[6]{0}, f32[], bf16[], bf16[], /*index=60*/f32[], f32[], f32[], f32[], f32[], /*index=65*/f32[], f32[], f32[], f32[], f32[], /*index=70*/bf16[8]{0}, s32[8,2048]{1,0}, f32[8]{0})}, allow_spmd_sharding_propagation_to_output={false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false}

region_10.798 {
  Arg_0.799 = f32[] parameter(0)
  Arg_1.800 = f32[] parameter(1)
  ROOT add.801 = f32[] add(Arg_0.799, Arg_1.800), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/reduce_sum[axes=(0, 1, 2)]" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=336}
}

region_0.369 {
  Arg_0.370 = s32[] parameter(0)
  Arg_1.371 = s32[] parameter(1)
  ROOT add.372 = s32[] add(Arg_0.370, Arg_1.371), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/reduce_sum[axes=(2, 1)]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
}

region_8.764 {
  Arg_0.765 = f32[] parameter(0)
  Arg_2.767 = f32[] parameter(2)
  compare.769 = pred[] compare(Arg_0.765, Arg_2.767), direction=GT, metadata={op_name="/gt" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=328}
  compare.770 = pred[] compare(Arg_0.765, Arg_0.765), direction=NE, metadata={op_name="/ne" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=328}
  or.771 = pred[] or(compare.769, compare.770), metadata={op_name="/or" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=328}
  select.776 = f32[] select(or.771, Arg_0.765, Arg_2.767), metadata={op_name="/select_n" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=328}
  compare.772 = pred[] compare(Arg_0.765, Arg_2.767), direction=EQ, metadata={op_name="/eq" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=328}
  Arg_1.766 = s32[] parameter(1)
  Arg_3.768 = s32[] parameter(3)
  compare.773 = pred[] compare(Arg_1.766, Arg_3.768), direction=LT, metadata={op_name="/lt" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=328}
  and.774 = pred[] and(compare.772, compare.773), metadata={op_name="/and" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=328}
  or.775 = pred[] or(or.771, and.774), metadata={op_name="/or" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=328}
  select.777 = s32[] select(or.775, Arg_1.766, Arg_3.768), metadata={op_name="/select_n" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=328}
  ROOT tuple.778 = (f32[], s32[]) tuple(select.776, select.777)
} // region_8.764

region_16.868 {
  Arg_0.869 = bf16[] parameter(0)
  Arg_1.870 = bf16[] parameter(1)
  ROOT add.871 = bf16[] add(Arg_0.869, Arg_1.870), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/reduce_sum[axes=(0, 1)]" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=381}
}

region_1.396.clone_spmd.1 {
  param.69 = (s32[], bf16[1,2048,8192]{2,1,0}, bf16[6,1,2048,3,8192]{4,3,2,1,0}, bf16[6,1,2048,8192]{3,2,1,0}, bf16[6,1,2048,1,32768]{4,3,2,1,0}, /*index=5*/bf16[6,1,2048,8192]{3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,8192,1024]{2,1,0}, bf16[6,3,8192]{2,1,0}, bf16[6,1024,3,8192]{3,2,1,0}, /*index=10*/bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,1,32768]{2,1,0}, /*index=15*/bf16[6,1024,1,32768]{3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,32768,1024]{2,1,0}, s32[1,1,2]{2,1,0}, u32[2]{0}, /*index=20*/bf16[8192,3,8192]{2,1,0}, bf16[8192,8192]{1,0}, bf16[8192,32768]{1,0}, bf16[32768,8192]{1,0}, s32[]) parameter(0)
  get-tuple-element.171 = bf16[6,1024,3,8192]{3,2,1,0} get-tuple-element(param.69), index=9
  get-tuple-element.166 = s32[] get-tuple-element(param.69), index=0
  constant.1128 = s32[] constant(0)
  compare.211 = pred[] compare(get-tuple-element.166, constant.1128), direction=LT, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/lt" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  constant.1129 = s32[] constant(6)
  add.391 = s32[] add(get-tuple-element.166, constant.1129), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/add" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  select.389 = s32[] select(compare.211, add.391, get-tuple-element.166), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/select_n" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-slice.59 = bf16[1,1024,3,8192]{3,2,1,0} dynamic-slice(get-tuple-element.171, select.389, constant.1128, constant.1128, constant.1128), dynamic_slice_sizes={1,1024,3,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192, 3, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.989 = bf16[1024,3,8192]{2,1,0} reshape(dynamic-slice.59), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.178 = bf16[6,8192,1024]{2,1,0} get-tuple-element(param.69), index=7
  dynamic-slice.62 = bf16[1,8192,1024]{2,1,0} dynamic-slice(get-tuple-element.178, select.389, constant.1128, constant.1128), dynamic_slice_sizes={1,8192,1024}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.997 = bf16[8192,1024]{1,0} reshape(dynamic-slice.62), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.185 = bf16[6,1024,1,32768]{3,2,1,0} get-tuple-element(param.69), index=15
  dynamic-slice.67 = bf16[1,1024,1,32768]{3,2,1,0} dynamic-slice(get-tuple-element.185, select.389, constant.1128, constant.1128, constant.1128), dynamic_slice_sizes={1,1024,1,32768}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192, 1, 32768)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1004 = bf16[1024,32768]{1,0} reshape(dynamic-slice.67)
  get-tuple-element.189 = bf16[6,32768,1024]{2,1,0} get-tuple-element(param.69), index=17
  dynamic-slice.70 = bf16[1,32768,1024]{2,1,0} dynamic-slice(get-tuple-element.189, select.389, constant.1128, constant.1128), dynamic_slice_sizes={1,32768,1024}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 32768, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1009 = bf16[32768,1024]{1,0} reshape(dynamic-slice.70), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  constant.1127 = s32[] constant(1)
  add.390 = s32[] add(get-tuple-element.166, constant.1127), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/add" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.167 = bf16[1,2048,8192]{2,1,0} get-tuple-element(param.69), index=1, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/xmap(<lambda>)/squeeze[dimensions=(0,)]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1186}
  get-tuple-element.168 = bf16[6,8192]{1,0} get-tuple-element(param.69), index=11
  dynamic-slice.57 = bf16[1,8192]{1,0} dynamic-slice(get-tuple-element.168, select.389, constant.1128), dynamic_slice_sizes={1,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.986 = bf16[8192]{0} reshape(dynamic-slice.57), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.169 = bf16[6,8192]{1,0} get-tuple-element(param.69), index=10
  dynamic-slice.58 = bf16[1,8192]{1,0} dynamic-slice(get-tuple-element.169, select.389, constant.1128), dynamic_slice_sizes={1,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.987 = bf16[8192]{0} reshape(dynamic-slice.58), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  custom-call.51 = (bf16[1,2048,8192]{2,1,0}, f32[2048]{0}, f32[2048]{0}) custom-call(get-tuple-element.167, reshape.986, reshape.987), custom_call_target="te_layernorm_forward", operand_layout_constraints={bf16[1,2048,8192]{2,1,0}, bf16[8192]{0}, bf16[8192]{0}}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/xmap(<lambda>)/te_layernorm_forward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}, backend_config="\000\010\000\000\000\000\000\000\000 \000\000\000\000\000\000\005\000\000\000\005\000\000\000\001O5X\254\305\'7"
  get-tuple-element.170 = bf16[1,2048,8192]{2,1,0} get-tuple-element(custom-call.51), index=0, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/xmap(<lambda>)/te_layernorm_forward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  reshape.988 = bf16[2048,8192]{1,0} reshape(get-tuple-element.170), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/reshape[new_sizes=(8, 2048, 8192) dimensions=None]" source_file="/opt/transformer-engine/transformer_engine/jax/layernorm.py" source_line=94}
  get-tuple-element.173 = bf16[8192,3,8192]{2,1,0} get-tuple-element(param.69), index=20
  reshape.991 = bf16[8192,24576]{1,0} reshape(get-tuple-element.173), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/dot_general[dimension_numbers=(((2,), (0,)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=652}
  dot.53 = bf16[2048,24576]{1,0} dot(reshape.988, reshape.991), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/dot_general[dimension_numbers=(((2,), (0,)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=652}
  reshape.992 = bf16[1,2048,3,8192]{3,2,1,0} reshape(dot.53), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/dot_general[dimension_numbers=(((2,), (0,)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=652}
  get-tuple-element.174 = bf16[6,3,8192]{2,1,0} get-tuple-element(param.69), index=8
  dynamic-slice.61 = bf16[1,3,8192]{2,1,0} dynamic-slice(get-tuple-element.174, select.389, constant.1128, constant.1128), dynamic_slice_sizes={1,3,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 3, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.993 = bf16[3,8192]{1,0} reshape(dynamic-slice.61), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  broadcast.934 = bf16[1,2048,3,8192]{3,2,1,0} broadcast(reshape.993), dimensions={2,3}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=664}
  add.393 = bf16[1,2048,3,8192]{3,2,1,0} add(reshape.992, broadcast.934), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=664}
  reshape.994 = bf16[1,2048,3,64,128]{4,3,2,1,0} reshape(add.393), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/squeeze[dimensions=(0,)]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1186}
  constant.1132 = bf16[0]{0} constant({}), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/full_to_shard[axes=OrderedDict() mesh=Mesh(device_ids=array([[[0],\n        [1],\n        [2],\n        [3],\n        [4],\n        [5],\n        [6],\n        [7]]]), axis_names=(\'replica\', \'data\', \'mdl\')) manual_axes=(\'mdl\', \'replica\', \'data\')]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1186}
  get-tuple-element.175 = s32[1,1,2]{2,1,0} get-tuple-element(param.69), index=18, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/full_to_shard[axes=OrderedDict([(\'replica\', 0), (\'data\', 1)]) mesh=Mesh(device_ids=array([[[0],\n        [1],\n        [2],\n        [3],\n        [4],\n        [5],\n        [6],\n        [7]]]), axis_names=(\'replica\', \'data\', \'mdl\')) manual_axes=(\'mdl\', \'replica\', \'data\')]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1186}
  reshape.995 = s32[2]{0} reshape(get-tuple-element.175), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/squeeze[dimensions=(0,)]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1186}
  get-tuple-element.176 = u32[2]{0} get-tuple-element(param.69), index=19, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/full_to_shard[axes=OrderedDict() mesh=Mesh(device_ids=array([[[0],\n        [1],\n        [2],\n        [3],\n        [4],\n        [5],\n        [6],\n        [7]]]), axis_names=(\'replica\', \'data\', \'mdl\')) manual_axes=(\'mdl\', \'replica\', \'data\')]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1186}
  custom-call.52 = (bf16[1,2048,64,128]{3,2,1,0}, f32[1,64,2048,1]{3,2,1,0}, u32[4]{0}) custom-call(reshape.994, constant.1132, reshape.995, get-tuple-element.176), custom_call_target="te_self_fused_attn_forward", operand_layout_constraints={bf16[1,2048,3,64,128]{4,3,2,1,0}, bf16[0]{0}, s32[2]{0}, u32[2]{0}}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/te_self_fused_attn_forward[attn_bias_type=NVTE_Bias_Type.NVTE_NO_BIAS attn_mask_type=NVTE_Mask_Type.NVTE_CAUSAL_MASK scaling_factor=0.08838834764831843 dropout_probability=0.0 is_training=True]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}, backend_config="\001\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\000\010\000\000\000\000\000\000\000\010\000\000\000\000\000\000\200\000\000\000\000\000\000\000\363\004\265=\000\000\000\000\000\000\000\000\002\000\000\000\005\000\000\000\001\177\000\000"
  get-tuple-element.177 = bf16[1,2048,64,128]{3,2,1,0} get-tuple-element(custom-call.52), index=0, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/te_self_fused_attn_forward[attn_bias_type=NVTE_Bias_Type.NVTE_NO_BIAS attn_mask_type=NVTE_Mask_Type.NVTE_CAUSAL_MASK scaling_factor=0.08838834764831843 dropout_probability=0.0 is_training=True]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  reshape.996 = bf16[2048,8192]{1,0} reshape(get-tuple-element.177), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/sharding_constraint[sharding=GSPMDSharding({devices=[8,1,1]<=[8]}) resource_env=ResourceEnv(Mesh(device_ids=array([[[0],\n        [1],\n        [2],\n        [3],\n        [4],\n        [5],\n        [6],\n        [7]]]), axis_names=(\'replica\', \'data\', \'mdl\')), ()) unconstrained_dims=set()]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=40}
  get-tuple-element.180 = bf16[8192,8192]{1,0} get-tuple-element(param.69), index=21
  dot.54 = bf16[2048,8192]{1,0} dot(reshape.996, get-tuple-element.180), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/out/dot_general[dimension_numbers=(((2,), (0,)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=451}
  get-tuple-element.181 = bf16[6,8192]{1,0} get-tuple-element(param.69), index=6
  dynamic-slice.64 = bf16[1,8192]{1,0} dynamic-slice(get-tuple-element.181, select.389, constant.1128), dynamic_slice_sizes={1,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.999 = bf16[8192]{0} reshape(dynamic-slice.64), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  broadcast.935 = bf16[2048,8192]{1,0} broadcast(reshape.999), dimensions={1}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/out/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=455}
  add.395 = bf16[2048,8192]{1,0} add(dot.54, broadcast.935), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/out/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=455}
  reshape.1000 = bf16[1,2048,8192]{2,1,0} reshape(add.395), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/out/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=455}
  add.396 = bf16[1,2048,8192]{2,1,0} add(reshape.1000, get-tuple-element.167), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/transformer.py" source_line=1124}
  get-tuple-element.182 = bf16[6,8192]{1,0} get-tuple-element(param.69), index=13
  dynamic-slice.65 = bf16[1,8192]{1,0} dynamic-slice(get-tuple-element.182, select.389, constant.1128), dynamic_slice_sizes={1,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1001 = bf16[8192]{0} reshape(dynamic-slice.65), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.183 = bf16[6,8192]{1,0} get-tuple-element(param.69), index=12
  dynamic-slice.66 = bf16[1,8192]{1,0} dynamic-slice(get-tuple-element.183, select.389, constant.1128), dynamic_slice_sizes={1,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1002 = bf16[8192]{0} reshape(dynamic-slice.66), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  custom-call.53 = (bf16[1,2048,8192]{2,1,0}, f32[2048]{0}, f32[2048]{0}) custom-call(add.396, reshape.1001, reshape.1002), custom_call_target="te_layernorm_forward", operand_layout_constraints={bf16[1,2048,8192]{2,1,0}, bf16[8192]{0}, bf16[8192]{0}}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/xmap(<lambda>)/te_layernorm_forward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}, backend_config="\000\010\000\000\000\000\000\000\000 \000\000\000\000\000\000\005\000\000\000\005\000\000\000\001O5X\254\305\'7"
  get-tuple-element.184 = bf16[1,2048,8192]{2,1,0} get-tuple-element(custom-call.53), index=0, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/xmap(<lambda>)/te_layernorm_forward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  reshape.1003 = bf16[2048,8192]{1,0} reshape(get-tuple-element.184), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/reshape[new_sizes=(8, 2048, 8192) dimensions=None]" source_file="/opt/transformer-engine/transformer_engine/jax/layernorm.py" source_line=94}
  get-tuple-element.187 = bf16[8192,32768]{1,0} get-tuple-element(param.69), index=22
  dot.55 = bf16[2048,32768]{1,0} dot(reshape.1003, get-tuple-element.187), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/dot_general[dimension_numbers=(((2,), (0,)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=962}
  get-tuple-element.188 = bf16[6,1,32768]{2,1,0} get-tuple-element(param.69), index=14
  dynamic-slice.69 = bf16[1,1,32768]{2,1,0} dynamic-slice(get-tuple-element.188, select.389, constant.1128, constant.1128), dynamic_slice_sizes={1,1,32768}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 1, 32768)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1006 = bf16[32768]{0} reshape(dynamic-slice.69)
  broadcast.936 = bf16[2048,32768]{1,0} broadcast(reshape.1006), dimensions={1}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=972}
  add.398 = bf16[2048,32768]{1,0} add(dot.55, broadcast.936), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=972}
  reshape.1007 = bf16[1,2048,1,32768]{3,2,1,0} reshape(add.398), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=972}
  multiply.278 = bf16[1,2048,1,32768]{3,2,1,0} multiply(reshape.1007, reshape.1007), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  multiply.279 = bf16[1,2048,1,32768]{3,2,1,0} multiply(reshape.1007, multiply.278), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  constant.1137 = bf16[] constant(0.04468)
  broadcast.937 = bf16[1,2048,1,32768]{3,2,1,0} broadcast(constant.1137), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  multiply.280 = bf16[1,2048,1,32768]{3,2,1,0} multiply(multiply.279, broadcast.937), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  add.399 = bf16[1,2048,1,32768]{3,2,1,0} add(reshape.1007, multiply.280), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  constant.1138 = bf16[] constant(0.7969)
  broadcast.938 = bf16[1,2048,1,32768]{3,2,1,0} broadcast(constant.1138), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  multiply.281 = bf16[1,2048,1,32768]{3,2,1,0} multiply(add.399, broadcast.938), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  tanh.3 = bf16[1,2048,1,32768]{3,2,1,0} tanh(multiply.281), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/tanh" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  constant.1139 = bf16[] constant(1)
  broadcast.939 = bf16[1,2048,1,32768]{3,2,1,0} broadcast(constant.1139), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  add.400 = bf16[1,2048,1,32768]{3,2,1,0} add(tanh.3, broadcast.939), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  constant.1140 = bf16[] constant(0.5)
  broadcast.940 = bf16[1,2048,1,32768]{3,2,1,0} broadcast(constant.1140), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  multiply.282 = bf16[1,2048,1,32768]{3,2,1,0} multiply(add.400, broadcast.940), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  multiply.283 = bf16[1,2048,1,32768]{3,2,1,0} multiply(reshape.1007, multiply.282), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  reshape.1008 = bf16[2048,32768]{1,0} reshape(multiply.283), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/reshape[new_sizes=(8, 2048, 32768) dimensions=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=985}
  get-tuple-element.191 = bf16[32768,8192]{1,0} get-tuple-element(param.69), index=23
  dot.56 = bf16[2048,8192]{1,0} dot(reshape.1008, get-tuple-element.191), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/dot_general[dimension_numbers=(((2,), (0,)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=1022}
  get-tuple-element.192 = bf16[6,8192]{1,0} get-tuple-element(param.69), index=16
  dynamic-slice.72 = bf16[1,8192]{1,0} dynamic-slice(get-tuple-element.192, select.389, constant.1128), dynamic_slice_sizes={1,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1011 = bf16[8192]{0} reshape(dynamic-slice.72), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  broadcast.941 = bf16[2048,8192]{1,0} broadcast(reshape.1011), dimensions={1}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=1030}
  add.402 = bf16[2048,8192]{1,0} add(dot.56, broadcast.941), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=1030}
  reshape.1012 = bf16[1,2048,8192]{2,1,0} reshape(add.402), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=1030}
  add.403 = bf16[1,2048,8192]{2,1,0} add(reshape.1012, add.396), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/transformer.py" source_line=1194}
  get-tuple-element.193 = bf16[6,1,2048,3,8192]{4,3,2,1,0} get-tuple-element(param.69), index=2
  reshape.1013 = bf16[1,1,2048,3,8192]{4,3,2,1,0} reshape(dot.53), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/broadcast_in_dim[shape=(1, 8, 2048, 3, 8192) broadcast_dimensions=(1, 2, 3, 4)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-update-slice.20 = bf16[6,1,2048,3,8192]{4,3,2,1,0} dynamic-update-slice(get-tuple-element.193, reshape.1013, select.389, constant.1128, constant.1128, /*index=5*/constant.1128, constant.1128), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_update_slice" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.194 = bf16[6,1,2048,8192]{3,2,1,0} get-tuple-element(param.69), index=3
  reshape.1014 = bf16[1,1,2048,8192]{3,2,1,0} reshape(dot.54), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/broadcast_in_dim[shape=(1, 8, 2048, 8192) broadcast_dimensions=(1, 2, 3)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-update-slice.21 = bf16[6,1,2048,8192]{3,2,1,0} dynamic-update-slice(get-tuple-element.194, reshape.1014, select.389, constant.1128, constant.1128, /*index=5*/constant.1128), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_update_slice" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.195 = bf16[6,1,2048,1,32768]{4,3,2,1,0} get-tuple-element(param.69), index=4
  reshape.1015 = bf16[1,1,2048,1,32768]{4,3,2,1,0} reshape(dot.55), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/broadcast_in_dim[shape=(1, 8, 2048, 1, 32768) broadcast_dimensions=(1, 2, 3, 4)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-update-slice.22 = bf16[6,1,2048,1,32768]{4,3,2,1,0} dynamic-update-slice(get-tuple-element.195, reshape.1015, select.389, constant.1128, constant.1128, /*index=5*/constant.1128, constant.1128), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_update_slice" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.196 = bf16[6,1,2048,8192]{3,2,1,0} get-tuple-element(param.69), index=5
  reshape.1016 = bf16[1,1,2048,8192]{3,2,1,0} reshape(get-tuple-element.167), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/broadcast_in_dim[shape=(1, 8, 2048, 8192) broadcast_dimensions=(1, 2, 3)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-update-slice.23 = bf16[6,1,2048,8192]{3,2,1,0} dynamic-update-slice(get-tuple-element.196, reshape.1016, select.389, constant.1128, constant.1128, /*index=5*/constant.1128), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_update_slice" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.172 = bf16[6,1024,3,8192]{3,2,1,0} get-tuple-element(param.69), index=9
  get-tuple-element.165 = s32[] get-tuple-element(param.69), index=24
  constant.1130 = s32[] constant(0)
  compare.212 = pred[] compare(get-tuple-element.165, constant.1130), direction=LT, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/lt" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  constant.1131 = s32[] constant(6)
  add.392 = s32[] add(get-tuple-element.165, constant.1131), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/add" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  select.390 = s32[] select(compare.212, add.392, get-tuple-element.165), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/select_n" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-slice.60 = bf16[1,1024,3,8192]{3,2,1,0} dynamic-slice(get-tuple-element.172, select.390, constant.1130, constant.1130, constant.1130), dynamic_slice_sizes={1,1024,3,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192, 3, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.990 = bf16[1024,3,8192]{2,1,0} reshape(dynamic-slice.60), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  all-gather.18 = bf16[8192,3,8192]{2,1,0} all-gather(reshape.990), channel_id=61, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={0}, use_global_device_ids=true, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/dot_general[dimension_numbers=(((2,), (0,)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=652}
  get-tuple-element.179 = bf16[6,8192,1024]{2,1,0} get-tuple-element(param.69), index=7
  constant.1133 = s32[] constant(0)
  compare.213 = pred[] compare(get-tuple-element.165, constant.1133), direction=LT, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/lt" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  constant.1134 = s32[] constant(6)
  add.394 = s32[] add(get-tuple-element.165, constant.1134), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/add" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  select.391 = s32[] select(compare.213, add.394, get-tuple-element.165), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/select_n" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-slice.63 = bf16[1,8192,1024]{2,1,0} dynamic-slice(get-tuple-element.179, select.391, constant.1133, constant.1133), dynamic_slice_sizes={1,8192,1024}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.998 = bf16[8192,1024]{1,0} reshape(dynamic-slice.63), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  all-gather.19 = bf16[8192,8192]{1,0} all-gather(reshape.998), channel_id=62, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={1}, use_global_device_ids=true, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/out/dot_general[dimension_numbers=(((2,), (0,)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=451}
  get-tuple-element.186 = bf16[6,1024,1,32768]{3,2,1,0} get-tuple-element(param.69), index=15
  constant.1135 = s32[] constant(0)
  compare.214 = pred[] compare(get-tuple-element.165, constant.1135), direction=LT, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/lt" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  constant.1136 = s32[] constant(6)
  add.397 = s32[] add(get-tuple-element.165, constant.1136), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/add" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  select.392 = s32[] select(compare.214, add.397, get-tuple-element.165), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/select_n" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-slice.68 = bf16[1,1024,1,32768]{3,2,1,0} dynamic-slice(get-tuple-element.186, select.392, constant.1135, constant.1135, constant.1135), dynamic_slice_sizes={1,1024,1,32768}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192, 1, 32768)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1005 = bf16[1024,32768]{1,0} reshape(dynamic-slice.68)
  all-gather.20 = bf16[8192,32768]{1,0} all-gather(reshape.1005), channel_id=63, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={0}, use_global_device_ids=true, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/dot_general[dimension_numbers=(((2,), (0,)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=962}
  get-tuple-element.190 = bf16[6,32768,1024]{2,1,0} get-tuple-element(param.69), index=17
  constant.1141 = s32[] constant(0)
  compare.215 = pred[] compare(get-tuple-element.165, constant.1141), direction=LT, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/lt" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  constant.1142 = s32[] constant(6)
  add.401 = s32[] add(get-tuple-element.165, constant.1142), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/add" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  select.393 = s32[] select(compare.215, add.401, get-tuple-element.165), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/select_n" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-slice.71 = bf16[1,32768,1024]{2,1,0} dynamic-slice(get-tuple-element.190, select.393, constant.1141, constant.1141), dynamic_slice_sizes={1,32768,1024}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 32768, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1010 = bf16[32768,1024]{1,0} reshape(dynamic-slice.71), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  all-gather.21 = bf16[32768,8192]{1,0} all-gather(reshape.1010), channel_id=64, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={1}, use_global_device_ids=true, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/dot_general[dimension_numbers=(((2,), (0,)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=1022}
  constant.1143 = s32[] constant(1)
  add.404 = s32[] add(add.390, constant.1143)
  ROOT tuple.9 = (s32[], bf16[1,2048,8192]{2,1,0}, bf16[6,1,2048,3,8192]{4,3,2,1,0}, bf16[6,1,2048,8192]{3,2,1,0}, bf16[6,1,2048,1,32768]{4,3,2,1,0}, /*index=5*/bf16[6,1,2048,8192]{3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,8192,1024]{2,1,0}, bf16[6,3,8192]{2,1,0}, bf16[6,1024,3,8192]{3,2,1,0}, /*index=10*/bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,1,32768]{2,1,0}, /*index=15*/bf16[6,1024,1,32768]{3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,32768,1024]{2,1,0}, s32[1,1,2]{2,1,0}, u32[2]{0}, /*index=20*/bf16[8192,3,8192]{2,1,0}, bf16[8192,8192]{1,0}, bf16[8192,32768]{1,0}, bf16[32768,8192]{1,0}, s32[]) tuple(add.390, add.403, dynamic-update-slice.20, dynamic-update-slice.21, dynamic-update-slice.22, /*index=5*/dynamic-update-slice.23, get-tuple-element.181, get-tuple-element.178, get-tuple-element.174, get-tuple-element.171, /*index=10*/get-tuple-element.169, get-tuple-element.168, get-tuple-element.183, get-tuple-element.182, get-tuple-element.188, /*index=15*/get-tuple-element.185, get-tuple-element.192, get-tuple-element.189, get-tuple-element.175, get-tuple-element.176, /*index=20*/all-gather.18, all-gather.19, all-gather.20, all-gather.21, add.404)
} // region_1.396.clone_spmd.1

region_2.614.clone_spmd.1 {
  cond_param = (s32[], bf16[1,2048,8192]{2,1,0}, bf16[6,1,2048,3,8192]{4,3,2,1,0}, bf16[6,1,2048,8192]{3,2,1,0}, bf16[6,1,2048,1,32768]{4,3,2,1,0}, /*index=5*/bf16[6,1,2048,8192]{3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,8192,1024]{2,1,0}, bf16[6,3,8192]{2,1,0}, bf16[6,1024,3,8192]{3,2,1,0}, /*index=10*/bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,1,32768]{2,1,0}, /*index=15*/bf16[6,1024,1,32768]{3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,32768,1024]{2,1,0}, s32[1,1,2]{2,1,0}, u32[2]{0}, /*index=20*/bf16[8192,3,8192]{2,1,0}, bf16[8192,8192]{1,0}, bf16[8192,32768]{1,0}, bf16[32768,8192]{1,0}, s32[]) parameter(0)
  get-tuple-element.197 = s32[] get-tuple-element(cond_param), index=0
  constant.1144 = s32[] constant(5)
  ROOT compare.216 = pred[] compare(get-tuple-element.197, constant.1144), direction=LT
}

region_25.1008_spmd.1 {
  param.70 = (s32[], bf16[1,2048,8192]{2,1,0}, bf16[6,8192]{1,0}, bf16[6,8192,1024]{2,1,0}, bf16[6,3,8192]{2,1,0}, /*index=5*/bf16[6,1024,3,8192]{3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, /*index=10*/bf16[6,1,32768]{2,1,0}, bf16[6,1024,1,32768]{3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,32768,1024]{2,1,0}, bf16[6,1,2048,3,8192]{4,3,2,1,0}, /*index=15*/bf16[6,1,2048,8192]{3,2,1,0}, bf16[6,1,2048,1,32768]{4,3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,8192,1024]{2,1,0}, bf16[6,3,8192]{2,1,0}, /*index=20*/bf16[6,1024,3,8192]{3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, /*index=25*/bf16[6,1,32768]{2,1,0}, bf16[6,1024,1,32768]{3,2,1,0}, bf16[6,32768,1024]{2,1,0}, bf16[6,1,2048,8192]{3,2,1,0}, bf16[1,1,2048,2048]{3,2,1,0}, /*index=30*/bf16[32768,8192]{1,0}, bf16[8192,32768]{1,0}, bf16[8192,8192]{1,0}, bf16[8192,3,8192]{2,1,0}, s32[]) parameter(0)
  get-tuple-element.234 = bf16[6,32768,1024]{2,1,0} get-tuple-element(param.70), index=27
  constant.1167 = s32[] constant(5)
  get-tuple-element.230 = s32[] get-tuple-element(param.70), index=0
  subtract.170 = s32[] subtract(constant.1167, get-tuple-element.230), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  constant.1168 = s32[] constant(0), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/reduce_sum[axes=(2, 1)]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  compare.222 = pred[] compare(subtract.170, constant.1168), direction=LT, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/lt" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  constant.1169 = s32[] constant(11)
  subtract.171 = s32[] subtract(constant.1169, get-tuple-element.230), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/add" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  select.399 = s32[] select(compare.222, subtract.171, subtract.170), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/select_n" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-slice.91 = bf16[1,32768,1024]{2,1,0} dynamic-slice(get-tuple-element.234, select.399, constant.1168, constant.1168), dynamic_slice_sizes={1,32768,1024}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 32768, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1051 = bf16[32768,1024]{1,0} reshape(dynamic-slice.91), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.237 = bf16[6,1024,1,32768]{3,2,1,0} get-tuple-element(param.70), index=26
  dynamic-slice.93 = bf16[1,1024,1,32768]{3,2,1,0} dynamic-slice(get-tuple-element.237, select.399, constant.1168, constant.1168, constant.1168), dynamic_slice_sizes={1,1024,1,32768}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192, 1, 32768)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1055 = bf16[1024,32768]{1,0} reshape(dynamic-slice.93)
  get-tuple-element.256 = bf16[6,8192,1024]{2,1,0} get-tuple-element(param.70), index=18
  dynamic-slice.102 = bf16[1,8192,1024]{2,1,0} dynamic-slice(get-tuple-element.256, select.399, constant.1168, constant.1168), dynamic_slice_sizes={1,8192,1024}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1066 = bf16[8192,1024]{1,0} reshape(dynamic-slice.102), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.260 = bf16[6,1024,3,8192]{3,2,1,0} get-tuple-element(param.70), index=20
  dynamic-slice.104 = bf16[1,1024,3,8192]{3,2,1,0} dynamic-slice(get-tuple-element.260, select.399, constant.1168, constant.1168, constant.1168), dynamic_slice_sizes={1,1024,3,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192, 3, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1070 = bf16[1024,3,8192]{2,1,0} reshape(dynamic-slice.104), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  partition-id.5 = u32[] partition-id(), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/out/transpose[permutation=(1, 0)]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=451}
  convert.176 = s32[] convert(partition-id.5), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/out/transpose[permutation=(1, 0)]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=451}
  constant.1194 = s32[] constant(1024), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/out/transpose[permutation=(1, 0)]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=451}
  multiply.305 = s32[] multiply(convert.176, constant.1194), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/out/transpose[permutation=(1, 0)]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=451}
  constant.1166 = s32[] constant(1)
  add.415 = s32[] add(get-tuple-element.230, constant.1166), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/add" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.231 = bf16[1,2048,8192]{2,1,0} get-tuple-element(param.70), index=1
  get-tuple-element.232 = bf16[6,1,2048,1,32768]{4,3,2,1,0} get-tuple-element(param.70), index=16
  dynamic-slice.89 = bf16[1,1,2048,1,32768]{4,3,2,1,0} dynamic-slice(get-tuple-element.232, select.399, constant.1168, constant.1168, constant.1168, /*index=5*/constant.1168), dynamic_slice_sizes={1,1,2048,1,32768}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8, 2048, 1, 32768)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.233 = bf16[6,1,32768]{2,1,0} get-tuple-element(param.70), index=25
  dynamic-slice.90 = bf16[1,1,32768]{2,1,0} dynamic-slice(get-tuple-element.233, select.399, constant.1168, constant.1168), dynamic_slice_sizes={1,1,32768}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 1, 32768)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1048 = bf16[32768]{0} reshape(dynamic-slice.90)
  broadcast.951 = bf16[1,1,2048,1,32768]{4,3,2,1,0} broadcast(reshape.1048), dimensions={4}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=972}
  add.416 = bf16[1,1,2048,1,32768]{4,3,2,1,0} add(dynamic-slice.89, broadcast.951), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=972}
  reshape.1049 = bf16[1,2048,1,32768]{3,2,1,0} reshape(add.416), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=972}
  reshape.1050 = bf16[2048,8192]{1,0} reshape(get-tuple-element.231)
  get-tuple-element.236 = bf16[32768,8192]{1,0} get-tuple-element(param.70), index=30
  dot.61 = bf16[2048,32768]{1,0} dot(reshape.1050, get-tuple-element.236), lhs_contracting_dims={1}, rhs_contracting_dims={1}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/dot_general[dimension_numbers=(((2,), (1,)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=1022}
  reshape.1053 = bf16[1,2048,1,32768]{3,2,1,0} reshape(dot.61), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/reshape[new_sizes=(8, 2048, 1, 32768) dimensions=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=985}
  multiply.290 = bf16[1,2048,1,32768]{3,2,1,0} multiply(reshape.1049, reshape.1053), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  constant.1173 = bf16[] constant(0.5)
  broadcast.952 = bf16[1,2048,1,32768]{3,2,1,0} broadcast(constant.1173), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  multiply.291 = bf16[1,2048,1,32768]{3,2,1,0} multiply(multiply.290, broadcast.952), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  constant.1174 = bf16[] constant(1)
  broadcast.953 = bf16[1,2048,1,32768]{3,2,1,0} broadcast(constant.1174), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  multiply.292 = bf16[1,2048,1,32768]{3,2,1,0} multiply(reshape.1049, reshape.1049), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  multiply.293 = bf16[1,2048,1,32768]{3,2,1,0} multiply(reshape.1049, multiply.292), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  constant.1175 = bf16[] constant(0.04468)
  broadcast.954 = bf16[1,2048,1,32768]{3,2,1,0} broadcast(constant.1175), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  multiply.294 = bf16[1,2048,1,32768]{3,2,1,0} multiply(multiply.293, broadcast.954), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  add.417 = bf16[1,2048,1,32768]{3,2,1,0} add(reshape.1049, multiply.294), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  constant.1176 = bf16[] constant(0.7969)
  broadcast.955 = bf16[1,2048,1,32768]{3,2,1,0} broadcast(constant.1176), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  multiply.295 = bf16[1,2048,1,32768]{3,2,1,0} multiply(add.417, broadcast.955), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  tanh.5 = bf16[1,2048,1,32768]{3,2,1,0} tanh(multiply.295), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/tanh" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  subtract.174 = bf16[1,2048,1,32768]{3,2,1,0} subtract(broadcast.953, tanh.5), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/sub" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  multiply.296 = bf16[1,2048,1,32768]{3,2,1,0} multiply(multiply.291, subtract.174), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  multiply.297 = bf16[1,2048,1,32768]{3,2,1,0} multiply(multiply.296, tanh.5), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  add.418 = bf16[1,2048,1,32768]{3,2,1,0} add(multiply.296, multiply.297), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add_any" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  multiply.298 = bf16[1,2048,1,32768]{3,2,1,0} multiply(add.418, broadcast.955), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  constant.1177 = bf16[] constant(0.03564)
  broadcast.956 = bf16[1,2048,1,32768]{3,2,1,0} broadcast(constant.1177), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  multiply.299 = bf16[1,2048,1,32768]{3,2,1,0} multiply(add.418, broadcast.956), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  constant.1178 = bf16[] constant(3)
  broadcast.957 = bf16[1,2048,1,32768]{3,2,1,0} broadcast(constant.1178), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  multiply.300 = bf16[1,2048,1,32768]{3,2,1,0} multiply(multiply.292, broadcast.957), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  multiply.301 = bf16[1,2048,1,32768]{3,2,1,0} multiply(multiply.299, multiply.300), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  add.419 = bf16[1,2048,1,32768]{3,2,1,0} add(multiply.298, multiply.301), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add_any" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  add.420 = bf16[1,2048,1,32768]{3,2,1,0} add(tanh.5, broadcast.953), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  multiply.302 = bf16[1,2048,1,32768]{3,2,1,0} multiply(add.420, broadcast.952), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  multiply.303 = bf16[1,2048,1,32768]{3,2,1,0} multiply(reshape.1053, multiply.302), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  add.421 = bf16[1,2048,1,32768]{3,2,1,0} add(add.419, multiply.303), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add_any" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  reshape.1054 = bf16[2048,32768]{1,0} reshape(add.421)
  get-tuple-element.239 = bf16[8192,32768]{1,0} get-tuple-element(param.70), index=31
  dot.62 = bf16[2048,8192]{1,0} dot(reshape.1054, get-tuple-element.239), lhs_contracting_dims={1}, rhs_contracting_dims={1}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/dot_general[dimension_numbers=(((2, 3), (1, 2)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=962}
  reshape.1057 = bf16[1,2048,8192]{2,1,0} reshape(dot.62), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/xmap(transpose(<lambda>))/squeeze[dimensions=(0,)]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1186}
  get-tuple-element.240 = bf16[6,1,2048,8192]{3,2,1,0} get-tuple-element(param.70), index=15
  dynamic-slice.95 = bf16[1,1,2048,8192]{3,2,1,0} dynamic-slice(get-tuple-element.240, select.399, constant.1168, constant.1168, constant.1168), dynamic_slice_sizes={1,1,2048,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8, 2048, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.241 = bf16[6,8192]{1,0} get-tuple-element(param.70), index=17
  dynamic-slice.96 = bf16[1,8192]{1,0} dynamic-slice(get-tuple-element.241, select.399, constant.1168), dynamic_slice_sizes={1,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1058 = bf16[8192]{0} reshape(dynamic-slice.96), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  broadcast.958 = bf16[1,1,2048,8192]{3,2,1,0} broadcast(reshape.1058), dimensions={3}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/out/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=455}
  add.422 = bf16[1,1,2048,8192]{3,2,1,0} add(dynamic-slice.95, broadcast.958), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/out/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=455}
  get-tuple-element.242 = bf16[6,1,2048,8192]{3,2,1,0} get-tuple-element(param.70), index=28
  dynamic-slice.97 = bf16[1,1,2048,8192]{3,2,1,0} dynamic-slice(get-tuple-element.242, select.399, constant.1168, constant.1168, constant.1168), dynamic_slice_sizes={1,1,2048,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8, 2048, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  add.423 = bf16[1,1,2048,8192]{3,2,1,0} add(add.422, dynamic-slice.97), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/transformer.py" source_line=1124}
  reshape.1059 = bf16[1,2048,8192]{2,1,0} reshape(add.423), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/xmap(<lambda>)/squeeze[dimensions=(0,)]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1186}
  get-tuple-element.243 = bf16[6,8192]{1,0} get-tuple-element(param.70), index=24
  dynamic-slice.98 = bf16[1,8192]{1,0} dynamic-slice(get-tuple-element.243, select.399, constant.1168), dynamic_slice_sizes={1,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1060 = bf16[8192]{0} reshape(dynamic-slice.98), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.244 = bf16[6,8192]{1,0} get-tuple-element(param.70), index=23
  dynamic-slice.99 = bf16[1,8192]{1,0} dynamic-slice(get-tuple-element.244, select.399, constant.1168), dynamic_slice_sizes={1,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1061 = bf16[8192]{0} reshape(dynamic-slice.99), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  custom-call.57 = (bf16[1,2048,8192]{2,1,0}, f32[2048]{0}, f32[2048]{0}) custom-call(reshape.1059, reshape.1060, reshape.1061), custom_call_target="te_layernorm_forward", operand_layout_constraints={bf16[1,2048,8192]{2,1,0}, bf16[8192]{0}, bf16[8192]{0}}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/xmap(<lambda>)/te_layernorm_forward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}, backend_config="\000\010\000\000\000\000\000\000\000 \000\000\000\000\000\000\005\000\000\000\005\000\000\000\001O5X\254\305\'7"
  get-tuple-element.245 = f32[2048]{0} get-tuple-element(custom-call.57), index=1, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/xmap(<lambda>)/te_layernorm_forward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  get-tuple-element.246 = f32[2048]{0} get-tuple-element(custom-call.57), index=2, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/xmap(<lambda>)/te_layernorm_forward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  custom-call.58 = (bf16[1,2048,8192]{2,1,0}, bf16[8192]{0}, bf16[8192]{0}) custom-call(reshape.1057, get-tuple-element.245, get-tuple-element.246, reshape.1059, reshape.1060), custom_call_target="te_layernorm_backward", operand_layout_constraints={bf16[1,2048,8192]{2,1,0}, f32[2048]{0}, f32[2048]{0}, bf16[1,2048,8192]{2,1,0}, bf16[8192]{0}}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/xmap(transpose(<lambda>))/te_layernorm_backward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}, backend_config="\000\010\000\000\000\000\000\000\000 \000\000\000\000\000\000\005\000\000\000\005\000\000\000\001O5X\254\305\'7"
  get-tuple-element.247 = bf16[1,2048,8192]{2,1,0} get-tuple-element(custom-call.58), index=0, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/xmap(transpose(<lambda>))/te_layernorm_backward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  add.424 = bf16[1,2048,8192]{2,1,0} add(get-tuple-element.231, get-tuple-element.247), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add_any" source_file="/opt/transformer-engine/transformer_engine/jax/layernorm.py" source_line=73}
  get-tuple-element.248 = bf16[6,1,2048,3,8192]{4,3,2,1,0} get-tuple-element(param.70), index=14
  dynamic-slice.100 = bf16[1,1,2048,3,8192]{4,3,2,1,0} dynamic-slice(get-tuple-element.248, select.399, constant.1168, constant.1168, constant.1168, /*index=5*/constant.1168), dynamic_slice_sizes={1,1,2048,3,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8, 2048, 3, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.249 = bf16[6,3,8192]{2,1,0} get-tuple-element(param.70), index=19
  dynamic-slice.101 = bf16[1,3,8192]{2,1,0} dynamic-slice(get-tuple-element.249, select.399, constant.1168, constant.1168), dynamic_slice_sizes={1,3,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 3, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1062 = bf16[3,8192]{1,0} reshape(dynamic-slice.101), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  broadcast.959 = bf16[1,1,2048,3,8192]{4,3,2,1,0} broadcast(reshape.1062), dimensions={3,4}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=664}
  add.425 = bf16[1,1,2048,3,8192]{4,3,2,1,0} add(dynamic-slice.100, broadcast.959), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=664}
  reshape.1063 = bf16[1,2048,3,64,128]{4,3,2,1,0} reshape(add.425), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(transpose(<lambda>))/squeeze[dimensions=(0,)]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1186}
  constant.1182 = bf16[0]{0} constant({})
  constant.1183 = s32[1]{0} constant({0}), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/concatenate[dimension=0]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  broadcast.960 = s32[1,1,2048,2048]{3,2,1,0} broadcast(constant.1166), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub.body_fn/sub/sub" source_file="/opt/paxml/paxml/contrib/gpu/scripts_gpu/te_helper.py" source_line=121}
  get-tuple-element.252 = bf16[1,1,2048,2048]{3,2,1,0} get-tuple-element(param.70), index=29
  iota.52 = s32[2048,2048]{1,0} iota(), iota_dimension=0, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/broadcast_in_dim[shape=(1, 2048, 2048, 1) broadcast_dimensions=(0, 1, 3)]" source_file="/opt/praxis/praxis/layers/attentions.py" source_line=103}
  iota.53 = s32[2048,2048]{1,0} iota(), iota_dimension=1, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/broadcast_in_dim[shape=(2048, 1, 1, 2048) broadcast_dimensions=(1, 2, 3)]" source_file="/opt/praxis/praxis/layers/attentions.py" source_line=102}
  compare.225 = pred[2048,2048]{1,0} compare(iota.52, iota.53), direction=LT, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/lt" source_file="/opt/praxis/praxis/layers/attentions.py" source_line=104}
  constant.1184 = bf16[] constant(-2.366e+38)
  broadcast.961 = bf16[2048,2048]{1,0} broadcast(constant.1184), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/mul" source_file="/opt/praxis/praxis/layers/attentions.py" source_line=104}
  constant.1185 = bf16[] constant(0)
  broadcast.962 = bf16[2048,2048]{1,0} broadcast(constant.1185), dimensions={}
  select.402 = bf16[2048,2048]{1,0} select(compare.225, broadcast.961, broadcast.962), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/mul" source_file="/opt/praxis/praxis/layers/attentions.py" source_line=104}
  reshape.1064 = bf16[1,1,2048,2048]{3,2,1,0} reshape(select.402), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/min" source_file="/opt/praxis/praxis/layers/attentions.py" source_line=170}
  minimum.5 = bf16[1,1,2048,2048]{3,2,1,0} minimum(get-tuple-element.252, reshape.1064), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/min" source_file="/opt/praxis/praxis/layers/attentions.py" source_line=170}
  broadcast.963 = bf16[1,1,2048,2048]{3,2,1,0} broadcast(constant.1185), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub.body_fn/sub/eq" source_file="/opt/paxml/paxml/contrib/gpu/scripts_gpu/te_helper.py" source_line=121}
  compare.226 = pred[1,1,2048,2048]{3,2,1,0} compare(minimum.5, broadcast.963), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/eq" source_file="/opt/paxml/paxml/contrib/gpu/scripts_gpu/te_helper.py" source_line=121}
  convert.173 = s32[1,1,2048,2048]{3,2,1,0} convert(compare.226), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/convert_element_type[new_dtype=int32 weak_type=True]" source_file="/opt/paxml/paxml/contrib/gpu/scripts_gpu/te_helper.py" source_line=121}
  subtract.177 = s32[1,1,2048,2048]{3,2,1,0} subtract(broadcast.960, convert.173), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/sub" source_file="/opt/paxml/paxml/contrib/gpu/scripts_gpu/te_helper.py" source_line=121}
  convert.174 = u8[1,1,2048,2048]{3,2,1,0} convert(subtract.177), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/convert_element_type[new_dtype=uint8 weak_type=False]" source_file="/opt/paxml/paxml/contrib/gpu/scripts_gpu/te_helper.py" source_line=122}
  slice.4 = u8[1,1,2048,1]{3,2,1,0} slice(convert.174), slice={[0:1], [0:1], [0:2048], [0:1]}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/slice[start_indices=(0, 0, 0, 0) limit_indices=(1, 1, 2048, 1) strides=None]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  constant.1186 = u8[] constant(0), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/eq" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  broadcast.964 = u8[1,1,2048,1]{3,2,1,0} broadcast(constant.1186), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/eq" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  compare.227 = pred[1,1,2048,1]{3,2,1,0} compare(slice.4, broadcast.964), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/eq" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  convert.175 = s32[1,1,2048,1]{3,2,1,0} convert(compare.227), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/convert_element_type[new_dtype=int32 weak_type=False]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  reduce.68 = s32[1]{0} reduce(convert.175, constant.1168), dimensions={1,2,3}, to_apply=region_0.369, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/reduce_sum[axes=(2, 1)]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  concatenate.3 = s32[2]{0} concatenate(constant.1183, reduce.68), dimensions={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/concatenate[dimension=0]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  constant.1187 = u32[] constant(0)
  broadcast.965 = u32[2]{0} broadcast(constant.1187), dimensions={}
  custom-call.59 = (bf16[1,2048,64,128]{3,2,1,0}, f32[1,64,2048,1]{3,2,1,0}, u32[4]{0}) custom-call(reshape.1063, constant.1182, concatenate.3, broadcast.965), custom_call_target="te_self_fused_attn_forward", operand_layout_constraints={bf16[1,2048,3,64,128]{4,3,2,1,0}, bf16[0]{0}, s32[2]{0}, u32[2]{0}}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/te_self_fused_attn_forward[attn_bias_type=NVTE_Bias_Type.NVTE_NO_BIAS attn_mask_type=NVTE_Mask_Type.NVTE_CAUSAL_MASK scaling_factor=0.08838834764831843 dropout_probability=0.0 is_training=True]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}, backend_config="\001\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\000\010\000\000\000\000\000\000\000\010\000\000\000\000\000\000\200\000\000\000\000\000\000\000\363\004\265=\000\000\000\000\000\000\000\000\002\000\000\000\005\000\000\000\001\177\000\000"
  get-tuple-element.253 = f32[1,64,2048,1]{3,2,1,0} get-tuple-element(custom-call.59), index=1, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/te_self_fused_attn_forward[attn_bias_type=NVTE_Bias_Type.NVTE_NO_BIAS attn_mask_type=NVTE_Mask_Type.NVTE_CAUSAL_MASK scaling_factor=0.08838834764831843 dropout_probability=0.0 is_training=True]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  get-tuple-element.254 = u32[4]{0} get-tuple-element(custom-call.59), index=2, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/te_self_fused_attn_forward[attn_bias_type=NVTE_Bias_Type.NVTE_NO_BIAS attn_mask_type=NVTE_Mask_Type.NVTE_CAUSAL_MASK scaling_factor=0.08838834764831843 dropout_probability=0.0 is_training=True]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  get-tuple-element.255 = bf16[1,2048,64,128]{3,2,1,0} get-tuple-element(custom-call.59), index=0, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/te_self_fused_attn_forward[attn_bias_type=NVTE_Bias_Type.NVTE_NO_BIAS attn_mask_type=NVTE_Mask_Type.NVTE_CAUSAL_MASK scaling_factor=0.08838834764831843 dropout_probability=0.0 is_training=True]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  reshape.1065 = bf16[2048,8192]{1,0} reshape(add.424), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add_any" source_file="/opt/transformer-engine/transformer_engine/jax/layernorm.py" source_line=73}
  get-tuple-element.258 = bf16[8192,8192]{1,0} get-tuple-element(param.70), index=32
  dot.63 = bf16[2048,8192]{1,0} dot(reshape.1065, get-tuple-element.258), lhs_contracting_dims={1}, rhs_contracting_dims={1}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/out/dot_general[dimension_numbers=(((2,), (1,)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=451}
  reshape.1068 = bf16[1,2048,64,128]{3,2,1,0} reshape(dot.63), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(transpose(<lambda>))/squeeze[dimensions=(0,)]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1186}
  custom-call.60 = (bf16[1,2048,3,64,128]{4,3,2,1,0}, bf16[0]{0}) custom-call(reshape.1063, get-tuple-element.253, get-tuple-element.254, get-tuple-element.255, reshape.1068, /*index=5*/concatenate.3), custom_call_target="te_self_fused_attn_backward", operand_layout_constraints={bf16[1,2048,3,64,128]{4,3,2,1,0}, f32[1,64,2048,1]{3,2,1,0}, u32[4]{0}, bf16[1,2048,64,128]{3,2,1,0}, bf16[1,2048,64,128]{3,2,1,0}, s32[2]{0}}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(transpose(<lambda>))/te_self_fused_attn_backward[attn_bias_type=NVTE_Bias_Type.NVTE_NO_BIAS attn_mask_type=NVTE_Mask_Type.NVTE_CAUSAL_MASK scaling_factor=0.08838834764831843 dropout_probability=0.0 is_training=True]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}, backend_config="\001\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\000\010\000\000\000\000\000\000\000\010\000\000\000\000\000\000\200\000\000\000\000\000\000\000\363\004\265=\000\000\000\000\000\000\000\000\002\000\000\000\005\000\000\000\001\177\000\000"
  get-tuple-element.259 = bf16[1,2048,3,64,128]{4,3,2,1,0} get-tuple-element(custom-call.60), index=0, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(transpose(<lambda>))/te_self_fused_attn_backward[attn_bias_type=NVTE_Bias_Type.NVTE_NO_BIAS attn_mask_type=NVTE_Mask_Type.NVTE_CAUSAL_MASK scaling_factor=0.08838834764831843 dropout_probability=0.0 is_training=True]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  reshape.1069 = bf16[2048,24576]{1,0} reshape(get-tuple-element.259), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/reshape[new_sizes=(8, 2048, 3, 8192) dimensions=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/transformer.py" source_line=654}
  get-tuple-element.262 = bf16[8192,3,8192]{2,1,0} get-tuple-element(param.70), index=33
  transpose.41 = bf16[3,8192,8192]{1,0,2} transpose(get-tuple-element.262), dimensions={1,2,0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/dot_general[dimension_numbers=(((2, 3), (1, 2)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=652}
  reshape.1073 = bf16[24576,8192]{1,0} reshape(transpose.41), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/dot_general[dimension_numbers=(((2, 3), (1, 2)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=652}
  dot.64 = bf16[2048,8192]{1,0} dot(reshape.1069, reshape.1073), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/dot_general[dimension_numbers=(((2, 3), (1, 2)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=652}
  reshape.1074 = bf16[1,2048,8192]{2,1,0} reshape(dot.64), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/xmap(transpose(<lambda>))/squeeze[dimensions=(0,)]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1186}
  reshape.1075 = bf16[1,2048,8192]{2,1,0} reshape(dynamic-slice.97), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/xmap(<lambda>)/squeeze[dimensions=(0,)]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1186}
  get-tuple-element.263 = bf16[6,8192]{1,0} get-tuple-element(param.70), index=22
  dynamic-slice.106 = bf16[1,8192]{1,0} dynamic-slice(get-tuple-element.263, select.399, constant.1168), dynamic_slice_sizes={1,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1077 = bf16[8192]{0} reshape(dynamic-slice.106), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.264 = bf16[6,8192]{1,0} get-tuple-element(param.70), index=21
  dynamic-slice.107 = bf16[1,8192]{1,0} dynamic-slice(get-tuple-element.264, select.399, constant.1168), dynamic_slice_sizes={1,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1078 = bf16[8192]{0} reshape(dynamic-slice.107), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  custom-call.61 = (bf16[1,2048,8192]{2,1,0}, f32[2048]{0}, f32[2048]{0}) custom-call(reshape.1075, reshape.1077, reshape.1078), custom_call_target="te_layernorm_forward", operand_layout_constraints={bf16[1,2048,8192]{2,1,0}, bf16[8192]{0}, bf16[8192]{0}}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/xmap(<lambda>)/te_layernorm_forward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}, backend_config="\000\010\000\000\000\000\000\000\000 \000\000\000\000\000\000\005\000\000\000\005\000\000\000\001O5X\254\305\'7"
  get-tuple-element.265 = f32[2048]{0} get-tuple-element(custom-call.61), index=1, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/xmap(<lambda>)/te_layernorm_forward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  get-tuple-element.266 = f32[2048]{0} get-tuple-element(custom-call.61), index=2, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/xmap(<lambda>)/te_layernorm_forward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  custom-call.62 = (bf16[1,2048,8192]{2,1,0}, bf16[8192]{0}, bf16[8192]{0}) custom-call(reshape.1074, get-tuple-element.265, get-tuple-element.266, reshape.1075, reshape.1077), custom_call_target="te_layernorm_backward", operand_layout_constraints={bf16[1,2048,8192]{2,1,0}, f32[2048]{0}, f32[2048]{0}, bf16[1,2048,8192]{2,1,0}, bf16[8192]{0}}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/xmap(transpose(<lambda>))/te_layernorm_backward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}, backend_config="\000\010\000\000\000\000\000\000\000 \000\000\000\000\000\000\005\000\000\000\005\000\000\000\001O5X\254\305\'7"
  get-tuple-element.267 = bf16[1,2048,8192]{2,1,0} get-tuple-element(custom-call.62), index=0, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/xmap(transpose(<lambda>))/te_layernorm_backward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  add.426 = bf16[1,2048,8192]{2,1,0} add(add.424, get-tuple-element.267), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/add_any" source_file="/opt/transformer-engine/transformer_engine/jax/layernorm.py" source_line=73}
  get-tuple-element.268 = bf16[6,8192]{1,0} get-tuple-element(param.70), index=2
  reduce.69 = bf16[8192]{0} reduce(add.424, constant.1185), dimensions={0,1}, to_apply=region_16.868, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/out/reduce_sum[axes=(0, 1)]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=455}
  all-reduce.34 = bf16[8192]{0} all-reduce(reduce.69), channel_id=73, replica_groups={{0}}, to_apply=region_16.868, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/out/reduce_sum[axes=(0, 1)]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=455}
  reshape.1079 = bf16[1,8192]{1,0} reshape(all-reduce.34), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/broadcast_in_dim[shape=(1, 8192) broadcast_dimensions=(1,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-update-slice.28 = bf16[6,8192]{1,0} dynamic-update-slice(get-tuple-element.268, reshape.1079, select.399, constant.1168), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_update_slice" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.269 = bf16[6,8192,1024]{2,1,0} get-tuple-element(param.70), index=3
  reshape.1080 = bf16[1,2048,8192]{2,1,0} reshape(get-tuple-element.255), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/reshape[new_sizes=(8, 2048, 8192) dimensions=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/transformer.py" source_line=720}
  transpose.42 = bf16[8192,1,2048]{0,2,1} transpose(reshape.1080), dimensions={2,0,1}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/sharding_constraint[sharding=GSPMDSharding({devices=[8,1,1]<=[8]}) resource_env=ResourceEnv(Mesh(device_ids=array([[[0],\n        [1],\n        [2],\n        [3],\n        [4],\n        [5],\n        [6],\n        [7]]]), axis_names=(\'replica\', \'data\', \'mdl\')), ()) unconstrained_dims=set()]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=40}
  reshape.1082 = bf16[8192,2048]{1,0} reshape(transpose.42), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/sharding_constraint[sharding=GSPMDSharding({devices=[8,1,1]<=[8]}) resource_env=ResourceEnv(Mesh(device_ids=array([[[0],\n        [1],\n        [2],\n        [3],\n        [4],\n        [5],\n        [6],\n        [7]]]), axis_names=(\'replica\', \'data\', \'mdl\')), ()) unconstrained_dims=set()]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=40}
  dot.65 = bf16[8192,8192]{1,0} dot(reshape.1082, reshape.1065), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/out/transpose[permutation=(1, 0)]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=451}
  reduce-scatter.8 = bf16[8192,1024]{1,0} reduce-scatter(dot.65), channel_id=74, replica_groups={{0}}, dimensions={1}, to_apply=region_16.868
  reshape.1083 = bf16[1,8192,1024]{2,1,0} reshape(reduce-scatter.8), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/broadcast_in_dim[shape=(1, 8192, 8192) broadcast_dimensions=(1, 2)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-update-slice.29 = bf16[6,8192,1024]{2,1,0} dynamic-update-slice(get-tuple-element.269, reshape.1083, select.399, constant.1168, constant.1168), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_update_slice" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.270 = bf16[6,3,8192]{2,1,0} get-tuple-element(param.70), index=4
  reduce.70 = bf16[3,64,128]{2,1,0} reduce(get-tuple-element.259, constant.1185), dimensions={0,1}, to_apply=region_16.868, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/reduce_sum[axes=(0, 1)]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=664}
  all-reduce.35 = bf16[3,64,128]{2,1,0} all-reduce(reduce.70), channel_id=75, replica_groups={{0}}, to_apply=region_16.868, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/reduce_sum[axes=(0, 1)]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=664}
  reshape.1084 = bf16[1,3,8192]{2,1,0} reshape(all-reduce.35), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/broadcast_in_dim[shape=(1, 3, 8192) broadcast_dimensions=(1, 2)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-update-slice.30 = bf16[6,3,8192]{2,1,0} dynamic-update-slice(get-tuple-element.270, reshape.1084, select.399, constant.1168, constant.1168), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_update_slice" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.271 = bf16[6,1024,3,8192]{3,2,1,0} get-tuple-element(param.70), index=5
  reshape.1085 = bf16[1,2048,3,8192]{3,2,1,0} reshape(get-tuple-element.259), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/reshape[new_sizes=(8, 2048, 3, 8192) dimensions=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/transformer.py" source_line=654}
  transpose.43 = bf16[3,8192,1,2048]{1,0,3,2} transpose(reshape.1085), dimensions={2,3,0,1}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/reshape[new_sizes=(8, 2048, 3, 8192) dimensions=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/transformer.py" source_line=654}
  reshape.1087 = bf16[24576,2048]{1,0} reshape(transpose.43), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/reshape[new_sizes=(8, 2048, 3, 8192) dimensions=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/transformer.py" source_line=654}
  get-tuple-element.272 = bf16[1,2048,8192]{2,1,0} get-tuple-element(custom-call.61), index=0, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/xmap(<lambda>)/te_layernorm_forward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  reshape.1088 = bf16[2048,8192]{1,0} reshape(get-tuple-element.272), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/reshape[new_sizes=(8, 2048, 8192) dimensions=None]" source_file="/opt/transformer-engine/transformer_engine/jax/layernorm.py" source_line=94}
  dot.66 = bf16[24576,8192]{1,0} dot(reshape.1087, reshape.1088), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/dot_general[dimension_numbers=(((0, 1), (0, 1)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=652}
  reshape.1089 = bf16[3,8192,8192]{2,1,0} reshape(dot.66), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/dot_general[dimension_numbers=(((0, 1), (0, 1)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=652}
  reduce-scatter.9 = bf16[3,8192,1024]{2,1,0} reduce-scatter(reshape.1089), channel_id=76, replica_groups={{0}}, dimensions={2}, to_apply=region_16.868
  transpose.44 = bf16[1024,3,8192]{0,2,1} transpose(reduce-scatter.9), dimensions={2,0,1}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/transpose[permutation=(2, 0, 1)]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=652}
  reshape.1090 = bf16[1,1024,3,8192]{3,2,1,0} reshape(transpose.44), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/broadcast_in_dim[shape=(1, 8192, 3, 8192) broadcast_dimensions=(1, 2, 3)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-update-slice.31 = bf16[6,1024,3,8192]{3,2,1,0} dynamic-update-slice(get-tuple-element.271, reshape.1090, select.399, constant.1168, constant.1168, /*index=5*/constant.1168), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_update_slice" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.273 = bf16[6,8192]{1,0} get-tuple-element(param.70), index=6
  get-tuple-element.274 = bf16[8192]{0} get-tuple-element(custom-call.62), index=2, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/xmap(transpose(<lambda>))/te_layernorm_backward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  all-reduce.36 = bf16[8192]{0} all-reduce(get-tuple-element.274), channel_id=77, replica_groups={{0,1,2,3,4,5,6,7}}, use_global_device_ids=true, to_apply=region_16.868
  reshape.1092 = bf16[1,8192]{1,0} reshape(all-reduce.36), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/broadcast_in_dim[shape=(1, 8192) broadcast_dimensions=(1,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-update-slice.32 = bf16[6,8192]{1,0} dynamic-update-slice(get-tuple-element.273, reshape.1092, select.399, constant.1168), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_update_slice" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.275 = bf16[6,8192]{1,0} get-tuple-element(param.70), index=7
  get-tuple-element.276 = bf16[8192]{0} get-tuple-element(custom-call.62), index=1, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/xmap(transpose(<lambda>))/te_layernorm_backward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  all-reduce.37 = bf16[8192]{0} all-reduce(get-tuple-element.276), channel_id=78, replica_groups={{0,1,2,3,4,5,6,7}}, use_global_device_ids=true, to_apply=region_16.868
  reshape.1093 = bf16[1,8192]{1,0} reshape(all-reduce.37), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/broadcast_in_dim[shape=(1, 8192) broadcast_dimensions=(1,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-update-slice.33 = bf16[6,8192]{1,0} dynamic-update-slice(get-tuple-element.275, reshape.1093, select.399, constant.1168), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_update_slice" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.277 = bf16[6,8192]{1,0} get-tuple-element(param.70), index=8
  get-tuple-element.278 = bf16[8192]{0} get-tuple-element(custom-call.58), index=2, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/xmap(transpose(<lambda>))/te_layernorm_backward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  all-reduce.38 = bf16[8192]{0} all-reduce(get-tuple-element.278), channel_id=79, replica_groups={{0,1,2,3,4,5,6,7}}, use_global_device_ids=true, to_apply=region_16.868
  reshape.1094 = bf16[1,8192]{1,0} reshape(all-reduce.38), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/broadcast_in_dim[shape=(1, 8192) broadcast_dimensions=(1,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-update-slice.34 = bf16[6,8192]{1,0} dynamic-update-slice(get-tuple-element.277, reshape.1094, select.399, constant.1168), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_update_slice" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.279 = bf16[6,8192]{1,0} get-tuple-element(param.70), index=9
  get-tuple-element.280 = bf16[8192]{0} get-tuple-element(custom-call.58), index=1, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/xmap(transpose(<lambda>))/te_layernorm_backward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  all-reduce.39 = bf16[8192]{0} all-reduce(get-tuple-element.280), channel_id=80, replica_groups={{0,1,2,3,4,5,6,7}}, use_global_device_ids=true, to_apply=region_16.868
  reshape.1095 = bf16[1,8192]{1,0} reshape(all-reduce.39), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/broadcast_in_dim[shape=(1, 8192) broadcast_dimensions=(1,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-update-slice.35 = bf16[6,8192]{1,0} dynamic-update-slice(get-tuple-element.279, reshape.1095, select.399, constant.1168), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_update_slice" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.281 = bf16[6,1,32768]{2,1,0} get-tuple-element(param.70), index=10
  reduce.71 = bf16[1,32768]{1,0} reduce(add.421, constant.1185), dimensions={0,1}, to_apply=region_16.868, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/reduce_sum[axes=(0, 1)]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=972}
  all-reduce.40 = bf16[1,32768]{1,0} all-reduce(reduce.71), channel_id=81, replica_groups={{0}}, to_apply=region_16.868, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/reduce_sum[axes=(0, 1)]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=972}
  reshape.1097 = bf16[1,1,32768]{2,1,0} reshape(all-reduce.40), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/broadcast_in_dim[shape=(1, 1, 32768) broadcast_dimensions=(1, 2)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-update-slice.36 = bf16[6,1,32768]{2,1,0} dynamic-update-slice(get-tuple-element.281, reshape.1097, select.399, constant.1168, constant.1168), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_update_slice" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.282 = bf16[6,1024,1,32768]{3,2,1,0} get-tuple-element(param.70), index=11
  reshape.1098 = bf16[1,2048,32768]{2,1,0} reshape(add.421)
  transpose.45 = bf16[32768,1,2048]{0,2,1} transpose(reshape.1098), dimensions={2,0,1}
  reshape.1099 = bf16[32768,2048]{1,0} reshape(transpose.45)
  get-tuple-element.283 = bf16[1,2048,8192]{2,1,0} get-tuple-element(custom-call.57), index=0, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/xmap(<lambda>)/te_layernorm_forward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  reshape.1100 = bf16[2048,8192]{1,0} reshape(get-tuple-element.283), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/reshape[new_sizes=(8, 2048, 8192) dimensions=None]" source_file="/opt/transformer-engine/transformer_engine/jax/layernorm.py" source_line=94}
  dot.67 = bf16[32768,8192]{1,0} dot(reshape.1099, reshape.1100), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/dot_general[dimension_numbers=(((0, 1), (0, 1)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=962}
  reduce-scatter.10 = bf16[32768,1024]{1,0} reduce-scatter(dot.67), channel_id=82, replica_groups={{0}}, dimensions={1}, to_apply=region_16.868
  reshape.1102 = bf16[1,32768,1024]{2,1,0} reshape(reduce-scatter.10), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/dot_general[dimension_numbers=(((0, 1), (0, 1)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=962}
  transpose.46 = bf16[1024,1,32768]{0,2,1} transpose(reshape.1102), dimensions={2,0,1}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/transpose[permutation=(2, 0, 1)]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=962}
  reshape.1103 = bf16[1,1024,1,32768]{3,2,1,0} reshape(transpose.46), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/broadcast_in_dim[shape=(1, 8192, 1, 32768) broadcast_dimensions=(1, 2, 3)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-update-slice.37 = bf16[6,1024,1,32768]{3,2,1,0} dynamic-update-slice(get-tuple-element.282, reshape.1103, select.399, constant.1168, constant.1168, /*index=5*/constant.1168), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_update_slice" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.284 = bf16[6,8192]{1,0} get-tuple-element(param.70), index=12
  reduce.72 = bf16[8192]{0} reduce(get-tuple-element.231, constant.1185), dimensions={0,1}, to_apply=region_16.868, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/reduce_sum[axes=(0, 1)]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=1030}
  all-reduce.41 = bf16[8192]{0} all-reduce(reduce.72), channel_id=83, replica_groups={{0}}, to_apply=region_16.868, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/reduce_sum[axes=(0, 1)]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=1030}
  reshape.1104 = bf16[1,8192]{1,0} reshape(all-reduce.41), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/broadcast_in_dim[shape=(1, 8192) broadcast_dimensions=(1,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-update-slice.38 = bf16[6,8192]{1,0} dynamic-update-slice(get-tuple-element.284, reshape.1104, select.399, constant.1168), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_update_slice" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.285 = bf16[6,32768,1024]{2,1,0} get-tuple-element(param.70), index=13
  multiply.304 = bf16[1,2048,1,32768]{3,2,1,0} multiply(reshape.1049, multiply.302), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  reshape.1105 = bf16[1,2048,32768]{2,1,0} reshape(multiply.304), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/reshape[new_sizes=(8, 2048, 32768) dimensions=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=985}
  transpose.47 = bf16[32768,1,2048]{0,2,1} transpose(reshape.1105), dimensions={2,0,1}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/reshape[new_sizes=(8, 2048, 32768) dimensions=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=985}
  reshape.1107 = bf16[32768,2048]{1,0} reshape(transpose.47), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/reshape[new_sizes=(8, 2048, 32768) dimensions=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=985}
  dot.68 = bf16[32768,8192]{1,0} dot(reshape.1107, reshape.1050), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/transpose[permutation=(1, 0)]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=1022}
  reduce-scatter.11 = bf16[32768,1024]{1,0} reduce-scatter(dot.68), channel_id=84, replica_groups={{0}}, dimensions={1}, to_apply=region_16.868
  reshape.1108 = bf16[1,32768,1024]{2,1,0} reshape(reduce-scatter.11), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/broadcast_in_dim[shape=(1, 32768, 8192) broadcast_dimensions=(1, 2)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-update-slice.39 = bf16[6,32768,1024]{2,1,0} dynamic-update-slice(get-tuple-element.285, reshape.1108, select.399, constant.1168, constant.1168), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_update_slice" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.235 = bf16[6,32768,1024]{2,1,0} get-tuple-element(param.70), index=27
  constant.1170 = s32[] constant(5)
  get-tuple-element.229 = s32[] get-tuple-element(param.70), index=34
  subtract.172 = s32[] subtract(constant.1170, get-tuple-element.229), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  constant.1171 = s32[] constant(0), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/reduce_sum[axes=(2, 1)]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  compare.223 = pred[] compare(subtract.172, constant.1171), direction=LT, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/lt" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  constant.1172 = s32[] constant(11)
  subtract.173 = s32[] subtract(constant.1172, get-tuple-element.229), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/add" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  select.400 = s32[] select(compare.223, subtract.173, subtract.172), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/select_n" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-slice.92 = bf16[1,32768,1024]{2,1,0} dynamic-slice(get-tuple-element.235, select.400, constant.1171, constant.1171), dynamic_slice_sizes={1,32768,1024}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 32768, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1052 = bf16[32768,1024]{1,0} reshape(dynamic-slice.92), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  all-gather.26 = bf16[32768,8192]{1,0} all-gather(reshape.1052), channel_id=69, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={1}, use_global_device_ids=true, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/dot_general[dimension_numbers=(((2,), (1,)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=1022}
  get-tuple-element.238 = bf16[6,1024,1,32768]{3,2,1,0} get-tuple-element(param.70), index=26
  constant.1179 = s32[] constant(5)
  subtract.175 = s32[] subtract(constant.1179, get-tuple-element.229), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  constant.1180 = s32[] constant(0), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/reduce_sum[axes=(2, 1)]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  compare.224 = pred[] compare(subtract.175, constant.1180), direction=LT, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/lt" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  constant.1181 = s32[] constant(11)
  subtract.176 = s32[] subtract(constant.1181, get-tuple-element.229), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/add" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  select.401 = s32[] select(compare.224, subtract.176, subtract.175), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/select_n" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-slice.94 = bf16[1,1024,1,32768]{3,2,1,0} dynamic-slice(get-tuple-element.238, select.401, constant.1180, constant.1180, constant.1180), dynamic_slice_sizes={1,1024,1,32768}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192, 1, 32768)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1056 = bf16[1024,32768]{1,0} reshape(dynamic-slice.94)
  all-gather.27 = bf16[8192,32768]{1,0} all-gather(reshape.1056), channel_id=70, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={0}, use_global_device_ids=true, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/dot_general[dimension_numbers=(((2, 3), (1, 2)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=962}
  get-tuple-element.257 = bf16[6,8192,1024]{2,1,0} get-tuple-element(param.70), index=18
  constant.1188 = s32[] constant(5)
  subtract.178 = s32[] subtract(constant.1188, get-tuple-element.229), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  constant.1189 = s32[] constant(0), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/reduce_sum[axes=(2, 1)]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  compare.228 = pred[] compare(subtract.178, constant.1189), direction=LT, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/lt" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  constant.1190 = s32[] constant(11)
  subtract.179 = s32[] subtract(constant.1190, get-tuple-element.229), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/add" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  select.403 = s32[] select(compare.228, subtract.179, subtract.178), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/select_n" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-slice.103 = bf16[1,8192,1024]{2,1,0} dynamic-slice(get-tuple-element.257, select.403, constant.1189, constant.1189), dynamic_slice_sizes={1,8192,1024}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1067 = bf16[8192,1024]{1,0} reshape(dynamic-slice.103), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  all-gather.28 = bf16[8192,8192]{1,0} all-gather(reshape.1067), channel_id=71, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={1}, use_global_device_ids=true, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/out/dot_general[dimension_numbers=(((2,), (1,)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=451}
  get-tuple-element.261 = bf16[6,1024,3,8192]{3,2,1,0} get-tuple-element(param.70), index=20
  constant.1191 = s32[] constant(5)
  subtract.180 = s32[] subtract(constant.1191, get-tuple-element.229), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  constant.1192 = s32[] constant(0), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/reduce_sum[axes=(2, 1)]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  compare.229 = pred[] compare(subtract.180, constant.1192), direction=LT, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/lt" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  constant.1193 = s32[] constant(11)
  subtract.181 = s32[] subtract(constant.1193, get-tuple-element.229), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/add" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  select.404 = s32[] select(compare.229, subtract.181, subtract.180), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/select_n" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-slice.105 = bf16[1,1024,3,8192]{3,2,1,0} dynamic-slice(get-tuple-element.261, select.404, constant.1192, constant.1192, constant.1192), dynamic_slice_sizes={1,1024,3,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192, 3, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1072 = bf16[1024,3,8192]{2,1,0} reshape(dynamic-slice.105), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  all-gather.29 = bf16[8192,3,8192]{2,1,0} all-gather(reshape.1072), channel_id=72, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={0}, use_global_device_ids=true, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/dot_general[dimension_numbers=(((2, 3), (1, 2)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=652}
  constant.1195 = s32[] constant(1)
  add.427 = s32[] add(add.415, constant.1195)
  ROOT tuple.12 = (s32[], bf16[1,2048,8192]{2,1,0}, bf16[6,8192]{1,0}, bf16[6,8192,1024]{2,1,0}, bf16[6,3,8192]{2,1,0}, /*index=5*/bf16[6,1024,3,8192]{3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, /*index=10*/bf16[6,1,32768]{2,1,0}, bf16[6,1024,1,32768]{3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,32768,1024]{2,1,0}, bf16[6,1,2048,3,8192]{4,3,2,1,0}, /*index=15*/bf16[6,1,2048,8192]{3,2,1,0}, bf16[6,1,2048,1,32768]{4,3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,8192,1024]{2,1,0}, bf16[6,3,8192]{2,1,0}, /*index=20*/bf16[6,1024,3,8192]{3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, /*index=25*/bf16[6,1,32768]{2,1,0}, bf16[6,1024,1,32768]{3,2,1,0}, bf16[6,32768,1024]{2,1,0}, bf16[6,1,2048,8192]{3,2,1,0}, bf16[1,1,2048,2048]{3,2,1,0}, /*index=30*/bf16[32768,8192]{1,0}, bf16[8192,32768]{1,0}, bf16[8192,8192]{1,0}, bf16[8192,3,8192]{2,1,0}, s32[]) tuple(add.415, add.426, dynamic-update-slice.28, dynamic-update-slice.29, dynamic-update-slice.30, /*index=5*/dynamic-update-slice.31, dynamic-update-slice.32, dynamic-update-slice.33, dynamic-update-slice.34, dynamic-update-slice.35, /*index=10*/dynamic-update-slice.36, dynamic-update-slice.37, dynamic-update-slice.38, dynamic-update-slice.39, get-tuple-element.248, /*index=15*/get-tuple-element.240, get-tuple-element.232, get-tuple-element.241, get-tuple-element.256, get-tuple-element.249, /*index=20*/get-tuple-element.260, get-tuple-element.264, get-tuple-element.263, get-tuple-element.244, get-tuple-element.243, /*index=25*/get-tuple-element.233, get-tuple-element.237, get-tuple-element.234, get-tuple-element.242, get-tuple-element.252, /*index=30*/all-gather.26, all-gather.27, all-gather.28, all-gather.29, add.427)
} // region_25.1008_spmd.1

region_39.1489_spmd.1 {
  cond_param.1 = (s32[], bf16[1,2048,8192]{2,1,0}, bf16[6,8192]{1,0}, bf16[6,8192,1024]{2,1,0}, bf16[6,3,8192]{2,1,0}, /*index=5*/bf16[6,1024,3,8192]{3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, /*index=10*/bf16[6,1,32768]{2,1,0}, bf16[6,1024,1,32768]{3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,32768,1024]{2,1,0}, bf16[6,1,2048,3,8192]{4,3,2,1,0}, /*index=15*/bf16[6,1,2048,8192]{3,2,1,0}, bf16[6,1,2048,1,32768]{4,3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,8192,1024]{2,1,0}, bf16[6,3,8192]{2,1,0}, /*index=20*/bf16[6,1024,3,8192]{3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, /*index=25*/bf16[6,1,32768]{2,1,0}, bf16[6,1024,1,32768]{3,2,1,0}, bf16[6,32768,1024]{2,1,0}, bf16[6,1,2048,8192]{3,2,1,0}, bf16[1,1,2048,2048]{3,2,1,0}, /*index=30*/bf16[32768,8192]{1,0}, bf16[8192,32768]{1,0}, bf16[8192,8192]{1,0}, bf16[8192,3,8192]{2,1,0}, s32[]) parameter(0)
  get-tuple-element.286 = s32[] get-tuple-element(cond_param.1), index=0
  constant.1196 = s32[] constant(5)
  ROOT compare.230 = pred[] compare(get-tuple-element.286, constant.1196), direction=LT
}

ENTRY main.4025_spmd {
  param.4 = u32[4]{0} parameter(57), sharding={replicated}
  param.5 = s32[1,2048]{1,0} parameter(62), sharding={devices=[8,1]<=[8]}
  param.6 = s32[1,2048]{1,0} parameter(63), sharding={devices=[8,1]<=[8]}
  param.7 = u32[] parameter(0), sharding={replicated}
  constant.787 = u32[] constant(1)
  add.223 = u32[] add(param.7, constant.787), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/paxml/paxml/train_states.py" source_line=65}
  param.8 = f32[8192]{0} parameter(1), sharding={replicated}
  param.9 = s32[1,2048]{1,0} parameter(60), sharding={devices=[8,1]<=[8]}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sharding_constraint[sharding=GSPMDSharding({devices=[8,1]<=[8]}) resource_env=ResourceEnv(Mesh(device_ids=array([[[0],\n        [1],\n        [2],\n        [3],\n        [4],\n        [5],\n        [6],\n        [7]]]), axis_names=(\'replica\', \'data\', \'mdl\')), ()) unconstrained_dims=set()]" source_file="/opt/praxis/praxis/py_utils.py" source_line=479}
  reshape.935 = s32[2048]{0} reshape(param.9)
  broadcast.549 = s32[1,2048,50304]{2,1,0} broadcast(reshape.935), dimensions={1}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/jit(_one_hot)/eq" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=304}
  iota.40 = s32[1,2048,50304]{2,1,0} iota(), iota_dimension=2, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/jit(_one_hot)/eq" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=304}
  compare.134 = pred[1,2048,50304]{2,1,0} compare(broadcast.549, iota.40), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/jit(_one_hot)/eq" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=304}
  constant.788 = f32[] constant(1)
  param.10 = f32[1,2048]{1,0} parameter(64), sharding={devices=[8,1]<=[8]}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sharding_constraint[sharding=GSPMDSharding({devices=[8,1]<=[8]}) resource_env=ResourceEnv(Mesh(device_ids=array([[[0],\n        [1],\n        [2],\n        [3],\n        [4],\n        [5],\n        [6],\n        [7]]]), axis_names=(\'replica\', \'data\', \'mdl\')), ()) unconstrained_dims=set()]" source_file="/opt/praxis/praxis/py_utils.py" source_line=479}
  convert.39 = bf16[1,2048]{1,0} convert(param.10), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=bfloat16 weak_type=False]" source_file="/opt/paxml/paxml/trainer_lib.py" source_line=472}
  param.11 = f32[1]{0} parameter(58), sharding={devices=[8]<=[8]}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sharding_constraint[sharding=GSPMDSharding({devices=[8]<=[8]}) resource_env=ResourceEnv(Mesh(device_ids=array([[[0],\n        [1],\n        [2],\n        [3],\n        [4],\n        [5],\n        [6],\n        [7]]]), axis_names=(\'replica\', \'data\', \'mdl\')), ()) unconstrained_dims=set()]" source_file="/opt/praxis/praxis/py_utils.py" source_line=479}
  convert.40 = bf16[1]{0} convert(param.11), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=bfloat16 weak_type=False]" source_file="/opt/paxml/paxml/trainer_lib.py" source_line=472}
  reshape.936 = bf16[] reshape(convert.40)
  broadcast.550 = bf16[1,2048]{1,0} broadcast(reshape.936), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/xformer_lm._prepare_predict_data/reshape[new_sizes=(8, 2048) dimensions=None]" source_file="/opt/praxis/praxis/layers/models.py" source_line=77}
  multiply.64 = bf16[1,2048]{1,0} multiply(convert.39, broadcast.550), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/xformer_lm._prepare_predict_data/mul" source_file="/opt/praxis/praxis/layers/models.py" source_line=80}
  convert.41 = f32[1,2048]{1,0} convert(multiply.64), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=336}
  constant.789 = f32[] constant(0)
  reduce.35 = f32[] reduce(convert.41, constant.789), dimensions={0,1}, to_apply=region_10.798, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/reduce_sum[axes=(0, 1, 2)]" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=336}
  all-reduce.16 = f32[] all-reduce(reduce.35), channel_id=9, replica_groups={{0}}, to_apply=region_10.798, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/reduce_sum[axes=(0, 1, 2)]" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=336}
  constant.790 = f32[] constant(1e-06)
  add.224 = f32[] add(all-reduce.16, constant.790), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/add" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=354}
  divide.2 = f32[] divide(constant.788, add.224), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/div" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=354}
  broadcast.551 = f32[1,2048]{1,0} broadcast(divide.2), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/broadcast_in_dim[shape=(8, 2048, 1) broadcast_dimensions=()]" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=332}
  multiply.65 = f32[1,2048]{1,0} multiply(broadcast.551, convert.41), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/mul" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=333}
  negate.40 = f32[1,2048]{1,0} negate(multiply.65), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/neg" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=322}
  reshape.937 = f32[2048]{0} reshape(negate.40)
  broadcast.552 = f32[1,2048,50304]{2,1,0} broadcast(reshape.937), dimensions={1}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/broadcast_in_dim[shape=(8, 2048, 50304) broadcast_dimensions=(0, 1)]" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=322}
  broadcast.553 = f32[1,2048,50304]{2,1,0} broadcast(constant.789), dimensions={}
  select.259 = f32[1,2048,50304]{2,1,0} select(compare.134, broadcast.552, broadcast.553), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/mul" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=323}
  negate.41 = f32[1,2048,50304]{2,1,0} negate(select.259), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/jit(log_softmax)/neg" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=301}
  reduce.36 = f32[1,2048]{1,0} reduce(negate.41, constant.789), dimensions={2}, to_apply=region_10.798, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/jit(log_softmax)/reduce_sum[axes=(2,)]" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=301}
  constant.793 = s32[] constant(0), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/reduce_sum[axes=(2, 1)]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  param.12 = s32[1,2048]{1,0} parameter(59), sharding={devices=[8,1]<=[8]}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sharding_constraint[sharding=GSPMDSharding({devices=[8,1]<=[8]}) resource_env=ResourceEnv(Mesh(device_ids=array([[[0],\n        [1],\n        [2],\n        [3],\n        [4],\n        [5],\n        [6],\n        [7]]]), axis_names=(\'replica\', \'data\', \'mdl\')), ()) unconstrained_dims=set()]" source_file="/opt/praxis/praxis/py_utils.py" source_line=479}
  reshape.938 = s32[2048]{0} reshape(param.12)
  broadcast.554 = s32[1,2048,50304]{2,1,0} broadcast(reshape.938), dimensions={1}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm._prepare_input/softmax.emb_lookup/jit(_one_hot)/eq" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=404}
  iota.41 = s32[1,2048,50304]{2,1,0} iota(), iota_dimension=2, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm._prepare_input/softmax.emb_lookup/jit(_one_hot)/eq" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=404}
  compare.135 = pred[1,2048,50304]{2,1,0} compare(broadcast.554, iota.41), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm._prepare_input/softmax.emb_lookup/jit(_one_hot)/eq" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=404}
  reshape.979 = pred[2048,50304]{1,0} reshape(compare.135), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm._prepare_input/softmax.emb_lookup/jit(_one_hot)/convert_element_type[new_dtype=bfloat16 weak_type=False]" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=404}
  convert.170 = bf16[2048,50304]{1,0} convert(reshape.979), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm._prepare_input/softmax.emb_lookup/jit(_one_hot)/convert_element_type[new_dtype=bfloat16 weak_type=False]" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=404}
  param.13 = f32[1024,50304]{1,0} parameter(4), sharding={devices=[8,1]<=[8]}
  convert.44 = bf16[1024,50304]{1,0} convert(param.13), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=bfloat16 weak_type=False]" source_file="/opt/paxml/paxml/trainer_lib.py" source_line=472}
  transpose.2 = bf16[50304,1024]{0,1} transpose(convert.44), dimensions={1,0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm._prepare_input/softmax.emb_lookup/transpose[permutation=(1, 0)]" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=399}
  all-gather.8 = bf16[50304,8192]{0,1} all-gather(transpose.2), channel_id=10, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={1}, use_global_device_ids=true, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm._prepare_input/softmax.emb_lookup/einsum/...y,yz->...z/dot_general[dimension_numbers=(((2,), (0,)), ((), ())) precision=None preferred_element_type=bfloat16]" source_file="/opt/praxis/praxis/layers/base_ops.py" source_line=42}
  dot.42 = bf16[2048,8192]{1,0} dot(convert.170, all-gather.8), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm._prepare_input/softmax.emb_lookup/einsum/...y,yz->...z/dot_general[dimension_numbers=(((2,), (0,)), ((), ())) precision=None preferred_element_type=bfloat16]" source_file="/opt/praxis/praxis/layers/base_ops.py" source_line=42}
  iota.42 = s32[1,2048,2048]{2,1,0} iota(), iota_dimension=1, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm._prepare_input/position_emb/jit(_one_hot)/eq" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=1146}
  iota.43 = s32[1,2048,2048]{2,1,0} iota(), iota_dimension=2, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm._prepare_input/position_emb/jit(_one_hot)/eq" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=1146}
  compare.136 = pred[1,2048,2048]{2,1,0} compare(iota.42, iota.43), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm._prepare_input/position_emb/jit(_one_hot)/eq" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=1146}
  convert.45 = bf16[1,2048,2048]{2,1,0} convert(compare.136), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm._prepare_input/position_emb/jit(_one_hot)/convert_element_type[new_dtype=bfloat16 weak_type=False]" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=1146}
  reshape.854 = bf16[2048,2048]{1,0} reshape(convert.45), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm._prepare_input/position_emb/jit(_one_hot)/convert_element_type[new_dtype=bfloat16 weak_type=False]" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=1146}
  param.14 = f32[256,8192]{1,0} parameter(3), sharding={devices=[8,1]<=[8]}
  convert.46 = bf16[256,8192]{1,0} convert(param.14), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=bfloat16 weak_type=False]" source_file="/opt/paxml/paxml/trainer_lib.py" source_line=472}
  all-gather.9 = bf16[2048,8192]{1,0} all-gather(convert.46), channel_id=11, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={0}, use_global_device_ids=true, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm._prepare_input/position_emb/einsum/...y,yz->...z/dot_general[dimension_numbers=(((2,), (0,)), ((), ())) precision=None preferred_element_type=bfloat16]" source_file="/opt/praxis/praxis/layers/base_ops.py" source_line=42}
  dot.43 = bf16[2048,8192]{1,0} dot(reshape.854, all-gather.9), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm._prepare_input/position_emb/einsum/...y,yz->...z/dot_general[dimension_numbers=(((2,), (0,)), ((), ())) precision=None preferred_element_type=bfloat16]" source_file="/opt/praxis/praxis/layers/base_ops.py" source_line=42}
  add.384 = bf16[2048,8192]{1,0} add(dot.42, dot.43), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm._prepare_input/add" source_file="/opt/praxis/praxis/layers/transformer_models.py" source_line=720}
  reshape.968 = bf16[1,2048,8192]{2,1,0} reshape(add.384), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm._prepare_input/add" source_file="/opt/praxis/praxis/layers/transformer_models.py" source_line=720}
  constant.794 = bf16[] constant(0)
  broadcast.555 = bf16[6,1,2048,3,8192]{4,3,2,1,0} broadcast(constant.794), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/broadcast_in_dim[shape=(6, 8, 2048, 3, 8192) broadcast_dimensions=()]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  broadcast.556 = bf16[6,1,2048,8192]{3,2,1,0} broadcast(constant.794), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/broadcast_in_dim[shape=(6, 8, 2048, 8192) broadcast_dimensions=()]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  broadcast.557 = bf16[6,1,2048,1,32768]{4,3,2,1,0} broadcast(constant.794), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/broadcast_in_dim[shape=(6, 8, 2048, 1, 32768) broadcast_dimensions=()]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  param.15 = bf16[6,8192]{1,0} parameter(5), sharding={replicated}
  param.16 = f32[6,8192,1024]{2,1,0} parameter(6), sharding={devices=[1,1,8]<=[8]}
  convert.47 = bf16[6,8192,1024]{2,1,0} convert(param.16), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=bfloat16 weak_type=False]" source_file="/opt/paxml/paxml/trainer_lib.py" source_line=472}
  param.17 = bf16[6,3,8192]{2,1,0} parameter(7), sharding={replicated}
  param.18 = f32[6,1024,3,8192]{3,2,1,0} parameter(8), sharding={devices=[1,8,1,1]<=[8]}
  convert.48 = bf16[6,1024,3,8192]{3,2,1,0} convert(param.18), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=bfloat16 weak_type=False]" source_file="/opt/paxml/paxml/trainer_lib.py" source_line=472}
  param.19 = f32[6,8192]{1,0} parameter(9), sharding={replicated}
  convert.49 = bf16[6,8192]{1,0} convert(param.19), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=bfloat16 weak_type=False]" source_file="/opt/paxml/paxml/trainer_lib.py" source_line=472}
  param.20 = f32[6,8192]{1,0} parameter(10), sharding={replicated}
  convert.50 = bf16[6,8192]{1,0} convert(param.20), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=bfloat16 weak_type=False]" source_file="/opt/paxml/paxml/trainer_lib.py" source_line=472}
  param.21 = f32[6,8192]{1,0} parameter(11), sharding={replicated}
  convert.51 = bf16[6,8192]{1,0} convert(param.21), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=bfloat16 weak_type=False]" source_file="/opt/paxml/paxml/trainer_lib.py" source_line=472}
  param.22 = f32[6,8192]{1,0} parameter(12), sharding={replicated}
  convert.52 = bf16[6,8192]{1,0} convert(param.22), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=bfloat16 weak_type=False]" source_file="/opt/paxml/paxml/trainer_lib.py" source_line=472}
  param.23 = bf16[6,1,32768]{2,1,0} parameter(13), sharding={replicated}
  param.24 = f32[6,1024,1,32768]{3,2,1,0} parameter(14), sharding={devices=[1,8,1,1]<=[8]}
  convert.53 = bf16[6,1024,1,32768]{3,2,1,0} convert(param.24), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=bfloat16 weak_type=False]" source_file="/opt/paxml/paxml/trainer_lib.py" source_line=472}
  param.25 = bf16[6,8192]{1,0} parameter(15), sharding={replicated}
  param.26 = f32[6,32768,1024]{2,1,0} parameter(16), sharding={devices=[1,1,8]<=[8]}
  convert.54 = bf16[6,32768,1024]{2,1,0} convert(param.26), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=bfloat16 weak_type=False]" source_file="/opt/paxml/paxml/trainer_lib.py" source_line=472}
  constant.797 = s32[1]{0} constant({0}), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/concatenate[dimension=0]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  constant.798 = s32[] constant(1)
  broadcast.558 = s32[1,1,2048,2048]{3,2,1,0} broadcast(constant.798), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub.body_fn/sub/sub" source_file="/opt/paxml/paxml/contrib/gpu/scripts_gpu/te_helper.py" source_line=121}
  constant.799 = bf16[] constant(1)
  broadcast.559 = bf16[1,2048]{1,0} broadcast(constant.799), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/sub" source_file="/opt/praxis/praxis/layers/transformer_models.py" source_line=780}
  param.27 = f32[1,2048]{1,0} parameter(61), sharding={devices=[8,1]<=[8]}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sharding_constraint[sharding=GSPMDSharding({devices=[8,1]<=[8]}) resource_env=ResourceEnv(Mesh(device_ids=array([[[0],\n        [1],\n        [2],\n        [3],\n        [4],\n        [5],\n        [6],\n        [7]]]), axis_names=(\'replica\', \'data\', \'mdl\')), ()) unconstrained_dims=set()]" source_file="/opt/praxis/praxis/py_utils.py" source_line=479}
  convert.55 = bf16[1,2048]{1,0} convert(param.27), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=bfloat16 weak_type=False]" source_file="/opt/paxml/paxml/trainer_lib.py" source_line=472}
  subtract.60 = bf16[1,2048]{1,0} subtract(broadcast.559, convert.55), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/sub" source_file="/opt/praxis/praxis/layers/transformer_models.py" source_line=780}
  reshape.980 = bf16[2048]{0} reshape(subtract.60)
  convert.171 = s32[2048]{0} convert(reshape.980)
  broadcast.560 = s32[1,2048,2048]{2,1,0} broadcast(convert.171), dimensions={1}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/ne" source_file="/opt/praxis/praxis/layers/attentions.py" source_line=133}
  broadcast.561 = s32[1,2048,2048]{2,1,0} broadcast(convert.171), dimensions={2}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/ne" source_file="/opt/praxis/praxis/layers/attentions.py" source_line=133}
  compare.137 = pred[1,2048,2048]{2,1,0} compare(broadcast.560, broadcast.561), direction=NE, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/ne" source_file="/opt/praxis/praxis/layers/attentions.py" source_line=133}
  constant.800 = bf16[] constant(-2.366e+38)
  broadcast.562 = bf16[1,2048,2048]{2,1,0} broadcast(constant.800), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/mul" source_file="/opt/praxis/praxis/layers/attentions.py" source_line=135}
  broadcast.563 = bf16[1,2048,2048]{2,1,0} broadcast(constant.794), dimensions={}
  select.260 = bf16[1,2048,2048]{2,1,0} select(compare.137, broadcast.562, broadcast.563), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/mul" source_file="/opt/praxis/praxis/layers/attentions.py" source_line=135}
  iota.44 = s32[2048,2048]{1,0} iota(), iota_dimension=0, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/broadcast_in_dim[shape=(1, 2048, 2048, 1) broadcast_dimensions=(0, 1, 3)]" source_file="/opt/praxis/praxis/layers/attentions.py" source_line=103}
  iota.45 = s32[2048,2048]{1,0} iota(), iota_dimension=1, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/broadcast_in_dim[shape=(2048, 1, 1, 2048) broadcast_dimensions=(1, 2, 3)]" source_file="/opt/praxis/praxis/layers/attentions.py" source_line=102}
  compare.138 = pred[2048,2048]{1,0} compare(iota.44, iota.45), direction=LT, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lt" source_file="/opt/praxis/praxis/layers/attentions.py" source_line=104}
  broadcast.564 = bf16[2048,2048]{1,0} broadcast(constant.800), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/mul" source_file="/opt/praxis/praxis/layers/attentions.py" source_line=104}
  broadcast.565 = bf16[2048,2048]{1,0} broadcast(constant.794), dimensions={}
  select.261 = bf16[2048,2048]{1,0} select(compare.138, broadcast.564, broadcast.565), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/mul" source_file="/opt/praxis/praxis/layers/attentions.py" source_line=104}
  reshape.941 = bf16[1,2048,2048]{2,1,0} reshape(select.261), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/min" source_file="/opt/praxis/praxis/layers/attentions.py" source_line=170}
  minimum.2 = bf16[1,2048,2048]{2,1,0} minimum(select.260, reshape.941), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/min" source_file="/opt/praxis/praxis/layers/attentions.py" source_line=170}
  reshape.801 = bf16[1,1,2048,2048]{3,2,1,0} reshape(minimum.2), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/min" source_file="/opt/praxis/praxis/layers/attentions.py" source_line=170}
  reshape.942 = bf16[1,1,2048,2048]{3,2,1,0} reshape(select.261), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub.body_fn/sub/min" source_file="/opt/praxis/praxis/layers/attentions.py" source_line=170}
  minimum.3 = bf16[1,1,2048,2048]{3,2,1,0} minimum(reshape.801, reshape.942), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub.body_fn/sub/min" source_file="/opt/praxis/praxis/layers/attentions.py" source_line=170}
  broadcast.570 = bf16[1,1,2048,2048]{3,2,1,0} broadcast(constant.794), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub.body_fn/sub/eq" source_file="/opt/paxml/paxml/contrib/gpu/scripts_gpu/te_helper.py" source_line=121}
  compare.140 = pred[1,1,2048,2048]{3,2,1,0} compare(minimum.3, broadcast.570), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub.body_fn/sub/eq" source_file="/opt/paxml/paxml/contrib/gpu/scripts_gpu/te_helper.py" source_line=121}
  convert.57 = s32[1,1,2048,2048]{3,2,1,0} convert(compare.140), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub.body_fn/sub/convert_element_type[new_dtype=int32 weak_type=True]" source_file="/opt/paxml/paxml/contrib/gpu/scripts_gpu/te_helper.py" source_line=121}
  subtract.61 = s32[1,1,2048,2048]{3,2,1,0} subtract(broadcast.558, convert.57), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub.body_fn/sub/sub" source_file="/opt/paxml/paxml/contrib/gpu/scripts_gpu/te_helper.py" source_line=121}
  convert.58 = u8[1,1,2048,2048]{3,2,1,0} convert(subtract.61), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub.body_fn/sub/convert_element_type[new_dtype=uint8 weak_type=False]" source_file="/opt/paxml/paxml/contrib/gpu/scripts_gpu/te_helper.py" source_line=122}
  slice.3 = u8[1,1,2048,1]{3,2,1,0} slice(convert.58), slice={[0:1], [0:1], [0:2048], [0:1]}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/slice[start_indices=(0, 0, 0, 0) limit_indices=(1, 1, 2048, 1) strides=None]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  constant.807 = u8[] constant(0), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/eq" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  broadcast.572 = u8[1,1,2048,1]{3,2,1,0} broadcast(constant.807), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/eq" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  compare.141 = pred[1,1,2048,1]{3,2,1,0} compare(slice.3, broadcast.572), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/eq" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  convert.59 = s32[1,1,2048,1]{3,2,1,0} convert(compare.141), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/convert_element_type[new_dtype=int32 weak_type=False]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  reduce.37 = s32[1]{0} reduce(convert.59, constant.793), dimensions={1,2,3}, to_apply=region_0.369, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/reduce_sum[axes=(2, 1)]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  concatenate.2 = s32[2]{0} concatenate(constant.797, reduce.37), dimensions={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/concatenate[dimension=0]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  reshape.804 = s32[1,1,2]{2,1,0} reshape(concatenate.2), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/broadcast_in_dim[shape=(1, 1, 2) broadcast_dimensions=(2,)]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1186}
  constant.809 = u32[] constant(0)
  broadcast.573 = u32[2]{0} broadcast(constant.809), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/shard_to_full[axes=OrderedDict() mesh=Mesh(device_ids=array([[[0],\n        [1],\n        [2],\n        [3],\n        [4],\n        [5],\n        [6],\n        [7]]]), axis_names=(\'replica\', \'data\', \'mdl\')) manual_axes=(\'mdl\', \'replica\', \'data\')]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1186}
  tuple.6 = (s32[], bf16[1,2048,8192]{2,1,0}, bf16[6,1,2048,3,8192]{4,3,2,1,0}, bf16[6,1,2048,8192]{3,2,1,0}, bf16[6,1,2048,1,32768]{4,3,2,1,0}, /*index=5*/bf16[6,1,2048,8192]{3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,8192,1024]{2,1,0}, bf16[6,3,8192]{2,1,0}, bf16[6,1024,3,8192]{3,2,1,0}, /*index=10*/bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,1,32768]{2,1,0}, /*index=15*/bf16[6,1024,1,32768]{3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,32768,1024]{2,1,0}, s32[1,1,2]{2,1,0}, u32[2]{0}) tuple(constant.793, reshape.968, broadcast.555, broadcast.556, broadcast.557, /*index=5*/broadcast.556, param.15, convert.47, param.17, convert.48, /*index=10*/convert.49, convert.50, convert.51, convert.52, param.23, /*index=15*/convert.53, param.25, convert.54, reshape.804, broadcast.573)
  get-tuple-element.161 = bf16[6,1024,3,8192]{3,2,1,0} get-tuple-element(tuple.6), index=9
  constant.1118 = s32[] constant(0)
  compare.207 = pred[] compare(constant.793, constant.1118), direction=LT, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/lt" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  constant.1119 = s32[] constant(6)
  add.386 = s32[] add(constant.793, constant.1119), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/add" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  select.385 = s32[] select(compare.207, add.386, constant.793), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/select_n" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-slice.53 = bf16[1,1024,3,8192]{3,2,1,0} dynamic-slice(get-tuple-element.161, select.385, constant.1118, constant.1118, constant.1118), dynamic_slice_sizes={1,1024,3,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192, 3, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.982 = bf16[1024,3,8192]{2,1,0} reshape(dynamic-slice.53), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  all-gather.14 = bf16[8192,3,8192]{2,1,0} all-gather(reshape.982), channel_id=57, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={0}, use_global_device_ids=true, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/dot_general[dimension_numbers=(((2,), (0,)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=652}
  get-tuple-element.162 = bf16[6,8192,1024]{2,1,0} get-tuple-element(tuple.6), index=7
  constant.1120 = s32[] constant(0)
  compare.208 = pred[] compare(constant.793, constant.1120), direction=LT, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/lt" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  constant.1121 = s32[] constant(6)
  add.387 = s32[] add(constant.793, constant.1121), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/add" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  select.386 = s32[] select(compare.208, add.387, constant.793), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/select_n" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-slice.54 = bf16[1,8192,1024]{2,1,0} dynamic-slice(get-tuple-element.162, select.386, constant.1120, constant.1120), dynamic_slice_sizes={1,8192,1024}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.983 = bf16[8192,1024]{1,0} reshape(dynamic-slice.54), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  all-gather.15 = bf16[8192,8192]{1,0} all-gather(reshape.983), channel_id=58, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={1}, use_global_device_ids=true, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/out/dot_general[dimension_numbers=(((2,), (0,)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=451}
  get-tuple-element.163 = bf16[6,1024,1,32768]{3,2,1,0} get-tuple-element(tuple.6), index=15
  constant.1122 = s32[] constant(0)
  compare.209 = pred[] compare(constant.793, constant.1122), direction=LT, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/lt" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  constant.1123 = s32[] constant(6)
  add.388 = s32[] add(constant.793, constant.1123), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/add" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  select.387 = s32[] select(compare.209, add.388, constant.793), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/select_n" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-slice.55 = bf16[1,1024,1,32768]{3,2,1,0} dynamic-slice(get-tuple-element.163, select.387, constant.1122, constant.1122, constant.1122), dynamic_slice_sizes={1,1024,1,32768}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192, 1, 32768)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.984 = bf16[1024,32768]{1,0} reshape(dynamic-slice.55)
  all-gather.16 = bf16[8192,32768]{1,0} all-gather(reshape.984), channel_id=59, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={0}, use_global_device_ids=true, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/dot_general[dimension_numbers=(((2,), (0,)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=962}
  get-tuple-element.164 = bf16[6,32768,1024]{2,1,0} get-tuple-element(tuple.6), index=17
  constant.1124 = s32[] constant(0)
  compare.210 = pred[] compare(constant.793, constant.1124), direction=LT, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/lt" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  constant.1125 = s32[] constant(6)
  add.389 = s32[] add(constant.793, constant.1125), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/add" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  select.388 = s32[] select(compare.210, add.389, constant.793), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/select_n" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-slice.56 = bf16[1,32768,1024]{2,1,0} dynamic-slice(get-tuple-element.164, select.388, constant.1124, constant.1124), dynamic_slice_sizes={1,32768,1024}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 32768, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.985 = bf16[32768,1024]{1,0} reshape(dynamic-slice.56), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  all-gather.17 = bf16[32768,8192]{1,0} all-gather(reshape.985), channel_id=60, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={1}, use_global_device_ids=true, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/dot_general[dimension_numbers=(((2,), (0,)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=1022}
  constant.1126 = s32[] constant(1)
  tuple.10 = (s32[], bf16[1,2048,8192]{2,1,0}, bf16[6,1,2048,3,8192]{4,3,2,1,0}, bf16[6,1,2048,8192]{3,2,1,0}, bf16[6,1,2048,1,32768]{4,3,2,1,0}, /*index=5*/bf16[6,1,2048,8192]{3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,8192,1024]{2,1,0}, bf16[6,3,8192]{2,1,0}, bf16[6,1024,3,8192]{3,2,1,0}, /*index=10*/bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,1,32768]{2,1,0}, /*index=15*/bf16[6,1024,1,32768]{3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,32768,1024]{2,1,0}, s32[1,1,2]{2,1,0}, u32[2]{0}, /*index=20*/bf16[8192,3,8192]{2,1,0}, bf16[8192,8192]{1,0}, bf16[8192,32768]{1,0}, bf16[32768,8192]{1,0}, s32[]) tuple(constant.793, reshape.968, broadcast.555, broadcast.556, broadcast.557, /*index=5*/broadcast.556, param.15, convert.47, param.17, convert.48, /*index=10*/convert.49, convert.50, convert.51, convert.52, param.23, /*index=15*/convert.53, param.25, convert.54, reshape.804, broadcast.573, /*index=20*/all-gather.14, all-gather.15, all-gather.16, all-gather.17, constant.1126)
  while.3 = (s32[], bf16[1,2048,8192]{2,1,0}, bf16[6,1,2048,3,8192]{4,3,2,1,0}, bf16[6,1,2048,8192]{3,2,1,0}, bf16[6,1,2048,1,32768]{4,3,2,1,0}, /*index=5*/bf16[6,1,2048,8192]{3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,8192,1024]{2,1,0}, bf16[6,3,8192]{2,1,0}, bf16[6,1024,3,8192]{3,2,1,0}, /*index=10*/bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,1,32768]{2,1,0}, /*index=15*/bf16[6,1024,1,32768]{3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,32768,1024]{2,1,0}, s32[1,1,2]{2,1,0}, u32[2]{0}, /*index=20*/bf16[8192,3,8192]{2,1,0}, bf16[8192,8192]{1,0}, bf16[8192,32768]{1,0}, bf16[32768,8192]{1,0}, s32[]) while(tuple.10), condition=region_2.614.clone_spmd.1, body=region_1.396.clone_spmd.1
  get-tuple-element.198 = s32[] get-tuple-element(while.3), index=0
  constant.1145 = s32[] constant(1)
  add.405 = s32[] add(get-tuple-element.198, constant.1145), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/add" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.199 = bf16[1,2048,8192]{2,1,0} get-tuple-element(while.3), index=1, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/xmap(<lambda>)/squeeze[dimensions=(0,)]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1186}
  get-tuple-element.200 = bf16[6,8192]{1,0} get-tuple-element(while.3), index=11
  constant.1146 = s32[] constant(0)
  compare.217 = pred[] compare(get-tuple-element.198, constant.1146), direction=LT, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/lt" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  constant.1147 = s32[] constant(6)
  add.406 = s32[] add(get-tuple-element.198, constant.1147), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/add" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  select.394 = s32[] select(compare.217, add.406, get-tuple-element.198), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/select_n" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-slice.73 = bf16[1,8192]{1,0} dynamic-slice(get-tuple-element.200, select.394, constant.1146), dynamic_slice_sizes={1,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1017 = bf16[8192]{0} reshape(dynamic-slice.73), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.201 = bf16[6,8192]{1,0} get-tuple-element(while.3), index=10
  dynamic-slice.74 = bf16[1,8192]{1,0} dynamic-slice(get-tuple-element.201, select.394, constant.1146), dynamic_slice_sizes={1,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1018 = bf16[8192]{0} reshape(dynamic-slice.74), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  custom-call.54 = (bf16[1,2048,8192]{2,1,0}, f32[2048]{0}, f32[2048]{0}) custom-call(get-tuple-element.199, reshape.1017, reshape.1018), custom_call_target="te_layernorm_forward", operand_layout_constraints={bf16[1,2048,8192]{2,1,0}, bf16[8192]{0}, bf16[8192]{0}}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/xmap(<lambda>)/te_layernorm_forward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}, backend_config="\000\010\000\000\000\000\000\000\000 \000\000\000\000\000\000\005\000\000\000\005\000\000\000\001O5X\254\305\'7"
  get-tuple-element.202 = bf16[1,2048,8192]{2,1,0} get-tuple-element(custom-call.54), index=0, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/xmap(<lambda>)/te_layernorm_forward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  reshape.1019 = bf16[2048,8192]{1,0} reshape(get-tuple-element.202), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/reshape[new_sizes=(8, 2048, 8192) dimensions=None]" source_file="/opt/transformer-engine/transformer_engine/jax/layernorm.py" source_line=94}
  get-tuple-element.204 = bf16[8192,3,8192]{2,1,0} get-tuple-element(while.3), index=20
  reshape.1021 = bf16[8192,24576]{1,0} reshape(get-tuple-element.204), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/dot_general[dimension_numbers=(((2,), (0,)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=652}
  dot.57 = bf16[2048,24576]{1,0} dot(reshape.1019, reshape.1021), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/dot_general[dimension_numbers=(((2,), (0,)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=652}
  reshape.1022 = bf16[1,2048,3,8192]{3,2,1,0} reshape(dot.57), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/dot_general[dimension_numbers=(((2,), (0,)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=652}
  get-tuple-element.205 = bf16[6,3,8192]{2,1,0} get-tuple-element(while.3), index=8
  dynamic-slice.76 = bf16[1,3,8192]{2,1,0} dynamic-slice(get-tuple-element.205, select.394, constant.1146, constant.1146), dynamic_slice_sizes={1,3,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 3, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1023 = bf16[3,8192]{1,0} reshape(dynamic-slice.76), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  broadcast.943 = bf16[1,2048,3,8192]{3,2,1,0} broadcast(reshape.1023), dimensions={2,3}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=664}
  add.407 = bf16[1,2048,3,8192]{3,2,1,0} add(reshape.1022, broadcast.943), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=664}
  reshape.1024 = bf16[1,2048,3,64,128]{4,3,2,1,0} reshape(add.407), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/squeeze[dimensions=(0,)]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1186}
  constant.1148 = bf16[0]{0} constant({}), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/full_to_shard[axes=OrderedDict() mesh=Mesh(device_ids=array([[[0],\n        [1],\n        [2],\n        [3],\n        [4],\n        [5],\n        [6],\n        [7]]]), axis_names=(\'replica\', \'data\', \'mdl\')) manual_axes=(\'mdl\', \'replica\', \'data\')]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1186}
  get-tuple-element.206 = s32[1,1,2]{2,1,0} get-tuple-element(while.3), index=18, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/full_to_shard[axes=OrderedDict([(\'replica\', 0), (\'data\', 1)]) mesh=Mesh(device_ids=array([[[0],\n        [1],\n        [2],\n        [3],\n        [4],\n        [5],\n        [6],\n        [7]]]), axis_names=(\'replica\', \'data\', \'mdl\')) manual_axes=(\'mdl\', \'replica\', \'data\')]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1186}
  reshape.1025 = s32[2]{0} reshape(get-tuple-element.206), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/squeeze[dimensions=(0,)]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1186}
  get-tuple-element.207 = u32[2]{0} get-tuple-element(while.3), index=19, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/full_to_shard[axes=OrderedDict() mesh=Mesh(device_ids=array([[[0],\n        [1],\n        [2],\n        [3],\n        [4],\n        [5],\n        [6],\n        [7]]]), axis_names=(\'replica\', \'data\', \'mdl\')) manual_axes=(\'mdl\', \'replica\', \'data\')]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1186}
  custom-call.55 = (bf16[1,2048,64,128]{3,2,1,0}, f32[1,64,2048,1]{3,2,1,0}, u32[4]{0}) custom-call(reshape.1024, constant.1148, reshape.1025, get-tuple-element.207), custom_call_target="te_self_fused_attn_forward", operand_layout_constraints={bf16[1,2048,3,64,128]{4,3,2,1,0}, bf16[0]{0}, s32[2]{0}, u32[2]{0}}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/te_self_fused_attn_forward[attn_bias_type=NVTE_Bias_Type.NVTE_NO_BIAS attn_mask_type=NVTE_Mask_Type.NVTE_CAUSAL_MASK scaling_factor=0.08838834764831843 dropout_probability=0.0 is_training=True]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}, backend_config="\001\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\000\010\000\000\000\000\000\000\000\010\000\000\000\000\000\000\200\000\000\000\000\000\000\000\363\004\265=\000\000\000\000\000\000\000\000\002\000\000\000\005\000\000\000\001\177\000\000"
  get-tuple-element.208 = bf16[1,2048,64,128]{3,2,1,0} get-tuple-element(custom-call.55), index=0, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/te_self_fused_attn_forward[attn_bias_type=NVTE_Bias_Type.NVTE_NO_BIAS attn_mask_type=NVTE_Mask_Type.NVTE_CAUSAL_MASK scaling_factor=0.08838834764831843 dropout_probability=0.0 is_training=True]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  reshape.1026 = bf16[2048,8192]{1,0} reshape(get-tuple-element.208), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/sharding_constraint[sharding=GSPMDSharding({devices=[8,1,1]<=[8]}) resource_env=ResourceEnv(Mesh(device_ids=array([[[0],\n        [1],\n        [2],\n        [3],\n        [4],\n        [5],\n        [6],\n        [7]]]), axis_names=(\'replica\', \'data\', \'mdl\')), ()) unconstrained_dims=set()]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=40}
  get-tuple-element.210 = bf16[8192,8192]{1,0} get-tuple-element(while.3), index=21
  dot.58 = bf16[2048,8192]{1,0} dot(reshape.1026, get-tuple-element.210), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/out/dot_general[dimension_numbers=(((2,), (0,)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=451}
  get-tuple-element.211 = bf16[6,8192]{1,0} get-tuple-element(while.3), index=6
  dynamic-slice.78 = bf16[1,8192]{1,0} dynamic-slice(get-tuple-element.211, select.394, constant.1146), dynamic_slice_sizes={1,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1028 = bf16[8192]{0} reshape(dynamic-slice.78), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  broadcast.944 = bf16[2048,8192]{1,0} broadcast(reshape.1028), dimensions={1}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/out/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=455}
  add.408 = bf16[2048,8192]{1,0} add(dot.58, broadcast.944), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/out/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=455}
  reshape.1029 = bf16[1,2048,8192]{2,1,0} reshape(add.408), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/out/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=455}
  add.409 = bf16[1,2048,8192]{2,1,0} add(reshape.1029, get-tuple-element.199), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/transformer.py" source_line=1124}
  get-tuple-element.212 = bf16[6,8192]{1,0} get-tuple-element(while.3), index=13
  dynamic-slice.79 = bf16[1,8192]{1,0} dynamic-slice(get-tuple-element.212, select.394, constant.1146), dynamic_slice_sizes={1,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1030 = bf16[8192]{0} reshape(dynamic-slice.79), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.213 = bf16[6,8192]{1,0} get-tuple-element(while.3), index=12
  dynamic-slice.80 = bf16[1,8192]{1,0} dynamic-slice(get-tuple-element.213, select.394, constant.1146), dynamic_slice_sizes={1,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1031 = bf16[8192]{0} reshape(dynamic-slice.80), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  custom-call.56 = (bf16[1,2048,8192]{2,1,0}, f32[2048]{0}, f32[2048]{0}) custom-call(add.409, reshape.1030, reshape.1031), custom_call_target="te_layernorm_forward", operand_layout_constraints={bf16[1,2048,8192]{2,1,0}, bf16[8192]{0}, bf16[8192]{0}}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/xmap(<lambda>)/te_layernorm_forward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}, backend_config="\000\010\000\000\000\000\000\000\000 \000\000\000\000\000\000\005\000\000\000\005\000\000\000\001O5X\254\305\'7"
  get-tuple-element.214 = bf16[1,2048,8192]{2,1,0} get-tuple-element(custom-call.56), index=0, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/xmap(<lambda>)/te_layernorm_forward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  reshape.1032 = bf16[2048,8192]{1,0} reshape(get-tuple-element.214), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/reshape[new_sizes=(8, 2048, 8192) dimensions=None]" source_file="/opt/transformer-engine/transformer_engine/jax/layernorm.py" source_line=94}
  get-tuple-element.216 = bf16[8192,32768]{1,0} get-tuple-element(while.3), index=22
  dot.59 = bf16[2048,32768]{1,0} dot(reshape.1032, get-tuple-element.216), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/dot_general[dimension_numbers=(((2,), (0,)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=962}
  get-tuple-element.217 = bf16[6,1,32768]{2,1,0} get-tuple-element(while.3), index=14
  dynamic-slice.82 = bf16[1,1,32768]{2,1,0} dynamic-slice(get-tuple-element.217, select.394, constant.1146, constant.1146), dynamic_slice_sizes={1,1,32768}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 1, 32768)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1034 = bf16[32768]{0} reshape(dynamic-slice.82)
  broadcast.945 = bf16[2048,32768]{1,0} broadcast(reshape.1034), dimensions={1}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=972}
  add.410 = bf16[2048,32768]{1,0} add(dot.59, broadcast.945), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=972}
  reshape.1035 = bf16[1,2048,1,32768]{3,2,1,0} reshape(add.410), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=972}
  multiply.284 = bf16[1,2048,1,32768]{3,2,1,0} multiply(reshape.1035, reshape.1035), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  multiply.285 = bf16[1,2048,1,32768]{3,2,1,0} multiply(reshape.1035, multiply.284), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  constant.1149 = bf16[] constant(0.04468)
  broadcast.946 = bf16[1,2048,1,32768]{3,2,1,0} broadcast(constant.1149), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  multiply.286 = bf16[1,2048,1,32768]{3,2,1,0} multiply(multiply.285, broadcast.946), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  add.411 = bf16[1,2048,1,32768]{3,2,1,0} add(reshape.1035, multiply.286), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  constant.1150 = bf16[] constant(0.7969)
  broadcast.947 = bf16[1,2048,1,32768]{3,2,1,0} broadcast(constant.1150), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  multiply.287 = bf16[1,2048,1,32768]{3,2,1,0} multiply(add.411, broadcast.947), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  tanh.4 = bf16[1,2048,1,32768]{3,2,1,0} tanh(multiply.287), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/tanh" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  constant.1151 = bf16[] constant(1)
  broadcast.948 = bf16[1,2048,1,32768]{3,2,1,0} broadcast(constant.1151), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  add.412 = bf16[1,2048,1,32768]{3,2,1,0} add(tanh.4, broadcast.948), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  constant.1152 = bf16[] constant(0.5)
  broadcast.949 = bf16[1,2048,1,32768]{3,2,1,0} broadcast(constant.1152), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  multiply.288 = bf16[1,2048,1,32768]{3,2,1,0} multiply(add.412, broadcast.949), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  multiply.289 = bf16[1,2048,1,32768]{3,2,1,0} multiply(reshape.1035, multiply.288), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  reshape.1036 = bf16[2048,32768]{1,0} reshape(multiply.289), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/reshape[new_sizes=(8, 2048, 32768) dimensions=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=985}
  get-tuple-element.219 = bf16[32768,8192]{1,0} get-tuple-element(while.3), index=23
  dot.60 = bf16[2048,8192]{1,0} dot(reshape.1036, get-tuple-element.219), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/dot_general[dimension_numbers=(((2,), (0,)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=1022}
  get-tuple-element.220 = bf16[6,8192]{1,0} get-tuple-element(while.3), index=16
  dynamic-slice.84 = bf16[1,8192]{1,0} dynamic-slice(get-tuple-element.220, select.394, constant.1146), dynamic_slice_sizes={1,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1038 = bf16[8192]{0} reshape(dynamic-slice.84), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  broadcast.950 = bf16[2048,8192]{1,0} broadcast(reshape.1038), dimensions={1}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=1030}
  add.413 = bf16[2048,8192]{1,0} add(dot.60, broadcast.950), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=1030}
  reshape.1039 = bf16[1,2048,8192]{2,1,0} reshape(add.413), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=1030}
  add.414 = bf16[1,2048,8192]{2,1,0} add(reshape.1039, add.409), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/transformer.py" source_line=1194}
  get-tuple-element.221 = bf16[6,1,2048,3,8192]{4,3,2,1,0} get-tuple-element(while.3), index=2
  reshape.1040 = bf16[1,1,2048,3,8192]{4,3,2,1,0} reshape(dot.57), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/broadcast_in_dim[shape=(1, 8, 2048, 3, 8192) broadcast_dimensions=(1, 2, 3, 4)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-update-slice.24 = bf16[6,1,2048,3,8192]{4,3,2,1,0} dynamic-update-slice(get-tuple-element.221, reshape.1040, select.394, constant.1146, constant.1146, /*index=5*/constant.1146, constant.1146), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_update_slice" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.222 = bf16[6,1,2048,8192]{3,2,1,0} get-tuple-element(while.3), index=3
  reshape.1041 = bf16[1,1,2048,8192]{3,2,1,0} reshape(dot.58), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/broadcast_in_dim[shape=(1, 8, 2048, 8192) broadcast_dimensions=(1, 2, 3)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-update-slice.25 = bf16[6,1,2048,8192]{3,2,1,0} dynamic-update-slice(get-tuple-element.222, reshape.1041, select.394, constant.1146, constant.1146, /*index=5*/constant.1146), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_update_slice" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.223 = bf16[6,1,2048,1,32768]{4,3,2,1,0} get-tuple-element(while.3), index=4
  reshape.1042 = bf16[1,1,2048,1,32768]{4,3,2,1,0} reshape(dot.59), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/broadcast_in_dim[shape=(1, 8, 2048, 1, 32768) broadcast_dimensions=(1, 2, 3, 4)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-update-slice.26 = bf16[6,1,2048,1,32768]{4,3,2,1,0} dynamic-update-slice(get-tuple-element.223, reshape.1042, select.394, constant.1146, constant.1146, /*index=5*/constant.1146, constant.1146), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_update_slice" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.224 = bf16[6,1,2048,8192]{3,2,1,0} get-tuple-element(while.3), index=5
  reshape.1043 = bf16[1,1,2048,8192]{3,2,1,0} reshape(get-tuple-element.199), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/broadcast_in_dim[shape=(1, 8, 2048, 8192) broadcast_dimensions=(1, 2, 3)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-update-slice.27 = bf16[6,1,2048,8192]{3,2,1,0} dynamic-update-slice(get-tuple-element.224, reshape.1043, select.394, constant.1146, constant.1146, /*index=5*/constant.1146), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_update_slice" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.209 = bf16[6,8192,1024]{2,1,0} get-tuple-element(while.3), index=7
  get-tuple-element.203 = bf16[6,1024,3,8192]{3,2,1,0} get-tuple-element(while.3), index=9
  get-tuple-element.215 = bf16[6,1024,1,32768]{3,2,1,0} get-tuple-element(while.3), index=15
  get-tuple-element.218 = bf16[6,32768,1024]{2,1,0} get-tuple-element(while.3), index=17
  tuple.11 = (s32[], bf16[1,2048,8192]{2,1,0}, bf16[6,1,2048,3,8192]{4,3,2,1,0}, bf16[6,1,2048,8192]{3,2,1,0}, bf16[6,1,2048,1,32768]{4,3,2,1,0}, /*index=5*/bf16[6,1,2048,8192]{3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,8192,1024]{2,1,0}, bf16[6,3,8192]{2,1,0}, bf16[6,1024,3,8192]{3,2,1,0}, /*index=10*/bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,1,32768]{2,1,0}, /*index=15*/bf16[6,1024,1,32768]{3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,32768,1024]{2,1,0}, s32[1,1,2]{2,1,0}, u32[2]{0}) tuple(add.405, add.414, dynamic-update-slice.24, dynamic-update-slice.25, dynamic-update-slice.26, /*index=5*/dynamic-update-slice.27, get-tuple-element.211, get-tuple-element.209, get-tuple-element.205, get-tuple-element.203, /*index=10*/get-tuple-element.201, get-tuple-element.200, get-tuple-element.213, get-tuple-element.212, get-tuple-element.217, /*index=15*/get-tuple-element.215, get-tuple-element.220, get-tuple-element.218, get-tuple-element.206, get-tuple-element.207)
  get-tuple-element.141 = bf16[1,2048,8192]{2,1,0} get-tuple-element(tuple.11), index=1, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while[cond_nconsts=0 body_nconsts=16]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  convert.60 = f32[1,2048,8192]{2,1,0} convert(get-tuple-element.141), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=372}
  reduce.38 = f32[1,2048]{1,0} reduce(convert.60, constant.789), dimensions={2}, to_apply=region_10.798, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/reduce_sum[axes=(2,)]" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=372}
  constant.811 = f32[] constant(0.000122070312)
  broadcast.574 = f32[1,2048]{1,0} broadcast(constant.811), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/div" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=372}
  multiply.66 = f32[1,2048]{1,0} multiply(reduce.38, broadcast.574), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/div" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=372}
  convert.61 = bf16[1,2048]{1,0} convert(multiply.66), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/convert_element_type[new_dtype=bfloat16 weak_type=False]" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=372}
  reshape.944 = bf16[2048]{0} reshape(convert.61)
  broadcast.575 = bf16[1,2048,8192]{2,1,0} broadcast(reshape.944), dimensions={1}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/sub" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=374}
  subtract.62 = bf16[1,2048,8192]{2,1,0} subtract(get-tuple-element.141, broadcast.575), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/sub" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=374}
  multiply.67 = bf16[1,2048,8192]{2,1,0} multiply(subtract.62, subtract.62), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/mul" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=373}
  convert.62 = f32[1,2048,8192]{2,1,0} convert(multiply.67), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=373}
  reduce.39 = f32[1,2048]{1,0} reduce(convert.62, constant.789), dimensions={2}, to_apply=region_10.798, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/reduce_sum[axes=(2,)]" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=373}
  multiply.68 = f32[1,2048]{1,0} multiply(reduce.39, broadcast.574), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/div" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=373}
  convert.63 = bf16[1,2048]{1,0} convert(multiply.68), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/convert_element_type[new_dtype=bfloat16 weak_type=False]" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=373}
  constant.814 = bf16[] constant(1.001e-05)
  broadcast.578 = bf16[1,2048]{1,0} broadcast(constant.814), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/add" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=374}
  add.226 = bf16[1,2048]{1,0} add(convert.63, broadcast.578), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/add" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=374}
  reshape.805 = bf16[1,2048,1]{2,1,0} reshape(add.226), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/add" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=374}
  rsqrt.0 = bf16[1,2048,1]{2,1,0} rsqrt(reshape.805), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/rsqrt" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=374}
  reshape.945 = bf16[2048]{0} reshape(rsqrt.0)
  broadcast.579 = bf16[1,2048,8192]{2,1,0} broadcast(reshape.945), dimensions={1}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/mul" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=374}
  multiply.69 = bf16[1,2048,8192]{2,1,0} multiply(subtract.62, broadcast.579), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/mul" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=374}
  param.28 = f32[8192]{0} parameter(2), sharding={replicated}
  convert.64 = bf16[8192]{0} convert(param.28), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=bfloat16 weak_type=False]" source_file="/opt/paxml/paxml/trainer_lib.py" source_line=472}
  broadcast.580 = bf16[8192]{0} broadcast(constant.799), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/add" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=378}
  add.227 = bf16[8192]{0} add(convert.64, broadcast.580), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/add" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=378}
  broadcast.581 = bf16[1,2048,8192]{2,1,0} broadcast(add.227), dimensions={2}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/mul" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=379}
  multiply.70 = bf16[1,2048,8192]{2,1,0} multiply(multiply.69, broadcast.581), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/mul" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=379}
  convert.65 = bf16[8192]{0} convert(param.8), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=bfloat16 weak_type=False]" source_file="/opt/paxml/paxml/trainer_lib.py" source_line=472}
  broadcast.582 = bf16[1,2048,8192]{2,1,0} broadcast(convert.65), dimensions={2}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/add" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=381}
  add.228 = bf16[1,2048,8192]{2,1,0} add(multiply.70, broadcast.582), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/add" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=381}
  reshape.858 = bf16[2048,8192]{1,0} reshape(add.228), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/add" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=381}
  all-gather.10 = bf16[8192,50304]{1,0} all-gather(convert.44), channel_id=16, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={0}, use_global_device_ids=true, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/softmax.get_logits/logits_ffn/linear/einsum/...y,yz->...z/dot_general[dimension_numbers=(((2,), (0,)), ((), ())) precision=None preferred_element_type=bfloat16]" source_file="/opt/praxis/praxis/layers/base_ops.py" source_line=42}
  dot.44 = bf16[2048,50304]{1,0} dot(reshape.858, all-gather.10), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/softmax.get_logits/logits_ffn/linear/einsum/...y,yz->...z/dot_general[dimension_numbers=(((2,), (0,)), ((), ())) precision=None preferred_element_type=bfloat16]" source_file="/opt/praxis/praxis/layers/base_ops.py" source_line=42}
  reshape.981 = bf16[1,2048,50304]{2,1,0} reshape(dot.44), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=300}
  convert.172 = f32[1,2048,50304]{2,1,0} convert(reshape.981), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=300}
  iota.48 = s32[1,2048,50304]{2,1,0} iota(), iota_dimension=2, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/iota[dtype=int32 shape=(8, 2048, 50304) dimension=2]" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=328}
  constant.816 = f32[] constant(-inf)
  reduce.40 = (f32[1,2048]{1,0}, s32[1,2048]{1,0}) reduce(convert.172, iota.48, constant.816, constant.793), dimensions={2}, to_apply=region_8.764, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/reduce[computation=<function _compute_argminmax.<locals>.reducer_fn at 0x7ffbf96b15a0> consts=() dimensions=(2,)]" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=328}
  get-tuple-element.142 = f32[1,2048]{1,0} get-tuple-element(reduce.40), index=0, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/jit(log_softmax)/reduce_max[axes=(2,)]" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=301}
  reshape.946 = f32[2048]{0} reshape(get-tuple-element.142)
  broadcast.583 = f32[1,2048,50304]{2,1,0} broadcast(reshape.946), dimensions={1}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/jit(log_softmax)/sub" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=301}
  subtract.64 = f32[1,2048,50304]{2,1,0} subtract(convert.172, broadcast.583), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/jit(log_softmax)/sub" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=301}
  exponential.1 = f32[1,2048,50304]{2,1,0} exponential(subtract.64), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/jit(log_softmax)/exp" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=301}
  reduce.41 = f32[1,2048]{1,0} reduce(exponential.1, constant.789), dimensions={2}, to_apply=region_10.798, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/jit(log_softmax)/reduce_sum[axes=(2,)]" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=301}
  divide.3 = f32[1,2048]{1,0} divide(reduce.36, reduce.41), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/jit(log_softmax)/div" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=301}
  reshape.947 = f32[2048]{0} reshape(divide.3)
  broadcast.584 = f32[1,2048,50304]{2,1,0} broadcast(reshape.947), dimensions={1}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/jit(log_softmax)/broadcast_in_dim[shape=(8, 2048, 50304) broadcast_dimensions=(0, 1)]" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=301}
  multiply.71 = f32[1,2048,50304]{2,1,0} multiply(broadcast.584, exponential.1), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/jit(log_softmax)/mul" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=301}
  add.229 = f32[1,2048,50304]{2,1,0} add(select.259, multiply.71), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/jit(log_softmax)/add_any" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=301}
  convert.67 = bf16[1,2048,50304]{2,1,0} convert(add.229), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/convert_element_type[new_dtype=bfloat16 weak_type=False]" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=300}
  reshape.861 = bf16[2048,50304]{1,0} reshape(convert.67), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/softmax.get_logits/logits_ffn/linear/sharding_constraint[sharding=GSPMDSharding({devices=[8,1,1]<=[8]}) resource_env=ResourceEnv(Mesh(device_ids=array([[[0],\n        [1],\n        [2],\n        [3],\n        [4],\n        [5],\n        [6],\n        [7]]]), axis_names=(\'replica\', \'data\', \'mdl\')), ()) unconstrained_dims=set()]" source_file="/opt/praxis/praxis/py_utils.py" source_line=479}
  dot.45 = bf16[2048,8192]{1,0} dot(reshape.861, all-gather.10), lhs_contracting_dims={1}, rhs_contracting_dims={1}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/softmax.get_logits/logits_ffn/linear/einsum/...y,yz->...z/dot_general[dimension_numbers=(((2,), (1,)), ((), ())) precision=None preferred_element_type=bfloat16]" source_file="/opt/praxis/praxis/layers/base_ops.py" source_line=42}
  reduce.42 = bf16[8192]{0} reduce(dot.45, constant.794), dimensions={0}, to_apply=region_16.868, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/reduce_sum[axes=(0, 1)]" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=381}
  all-reduce.17 = bf16[8192]{0} all-reduce(reduce.42), channel_id=17, replica_groups={{0}}, to_apply=region_16.868, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/reduce_sum[axes=(0, 1)]" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=381}
  convert.68 = f32[8192]{0} convert(all-reduce.17), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/paxml/paxml/trainer_lib.py" source_line=472}
  multiply.72 = f32[8192]{0} multiply(convert.68, convert.68), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  reduce.43 = f32[] reduce(multiply.72, constant.789), dimensions={0}, to_apply=region_10.798, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/reduce_sum[axes=(0,)]" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  reshape.863 = bf16[1,2048,8192]{2,1,0} reshape(dot.45), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/softmax.get_logits/logits_ffn/linear/einsum/...y,yz->...z/dot_general[dimension_numbers=(((2,), (1,)), ((), ())) precision=None preferred_element_type=bfloat16]" source_file="/opt/praxis/praxis/layers/base_ops.py" source_line=42}
  multiply.73 = bf16[1,2048,8192]{2,1,0} multiply(multiply.69, reshape.863), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/mul" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=379}
  reduce.44 = bf16[8192]{0} reduce(multiply.73, constant.794), dimensions={0,1}, to_apply=region_16.868, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/reduce_sum[axes=(0, 1)]" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=379}
  all-reduce.18 = bf16[8192]{0} all-reduce(reduce.44), channel_id=18, replica_groups={{0}}, to_apply=region_16.868, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/reduce_sum[axes=(0, 1)]" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=379}
  convert.69 = f32[8192]{0} convert(all-reduce.18), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/paxml/paxml/trainer_lib.py" source_line=472}
  multiply.74 = f32[8192]{0} multiply(convert.69, convert.69), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  reduce.45 = f32[] reduce(multiply.74, constant.789), dimensions={0}, to_apply=region_10.798, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/reduce_sum[axes=(0,)]" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  add.230 = f32[] add(reduce.43, reduce.45), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/reduce_sum[axes=(0,)]" source_file="/opt/paxml/paxml/learners.py" source_line=51}
  transpose.35 = bf16[2048,1,2048]{0,2,1} transpose(convert.45), dimensions={2,0,1}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm._prepare_input/position_emb/jit(_one_hot)/convert_element_type[new_dtype=bfloat16 weak_type=False]" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=1146}
  reshape.864 = bf16[2048,2048]{1,0} reshape(transpose.35), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm._prepare_input/position_emb/jit(_one_hot)/convert_element_type[new_dtype=bfloat16 weak_type=False]" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=1146}
  multiply.75 = bf16[1,2048,8192]{2,1,0} multiply(reshape.863, broadcast.581), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/mul" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=379}
  multiply.76 = bf16[1,2048,8192]{2,1,0} multiply(subtract.62, multiply.75), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/mul" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=374}
  reduce.46 = bf16[1,2048]{1,0} reduce(multiply.76, constant.794), dimensions={2}, to_apply=region_16.868, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/reduce_sum[axes=(2,)]" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=374}
  reshape.807 = bf16[1,2048,1]{2,1,0} reshape(reduce.46), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/reshape[new_sizes=(8, 2048, 1) dimensions=None]" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=374}
  divide.4 = bf16[1,2048,1]{2,1,0} divide(rsqrt.0, reshape.805), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/div" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=374}
  constant.825 = bf16[] constant(-0.5)
  broadcast.586 = bf16[1,2048,1]{2,1,0} broadcast(constant.825), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/mul" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=374}
  multiply.77 = bf16[1,2048,1]{2,1,0} multiply(divide.4, broadcast.586), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/mul" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=374}
  multiply.78 = bf16[1,2048,1]{2,1,0} multiply(reshape.807, multiply.77), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/mul" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=374}
  convert.70 = f32[1,2048,1]{2,1,0} convert(multiply.78), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=373}
  broadcast.588 = f32[1,2048,1]{2,1,0} broadcast(constant.811), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/div" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=372}
  multiply.79 = f32[1,2048,1]{2,1,0} multiply(convert.70, broadcast.588), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/div" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=373}
  convert.71 = bf16[1,2048,1]{2,1,0} convert(multiply.79), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/convert_element_type[new_dtype=bfloat16 weak_type=False]" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=373}
  constant.827 = bf16[] constant(2)
  broadcast.590 = bf16[1,2048,1]{2,1,0} broadcast(constant.827), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/mul" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=373}
  multiply.80 = bf16[1,2048,1]{2,1,0} multiply(convert.71, broadcast.590), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/mul" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=373}
  reshape.948 = bf16[2048]{0} reshape(multiply.80)
  broadcast.591 = bf16[1,2048,8192]{2,1,0} broadcast(reshape.948), dimensions={1}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/mul" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=373}
  multiply.81 = bf16[1,2048,8192]{2,1,0} multiply(subtract.62, broadcast.591), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/mul" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=373}
  multiply.82 = bf16[1,2048,8192]{2,1,0} multiply(multiply.75, broadcast.579), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/mul" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=374}
  add.231 = bf16[1,2048,8192]{2,1,0} add(multiply.81, multiply.82), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/add_any" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=374}
  negate.42 = bf16[1,2048,8192]{2,1,0} negate(multiply.81), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/neg" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=373}
  reduce.47 = bf16[1,2048]{1,0} reduce(negate.42, constant.794), dimensions={2}, to_apply=region_16.868, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/reduce_sum[axes=(2,)]" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=373}
  negate.43 = bf16[1,2048,8192]{2,1,0} negate(multiply.82), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/neg" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=374}
  reduce.48 = bf16[1,2048]{1,0} reduce(negate.43, constant.794), dimensions={2}, to_apply=region_16.868, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/reduce_sum[axes=(2,)]" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=374}
  add.232 = bf16[1,2048]{1,0} add(reduce.47, reduce.48), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/add_any" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=374}
  convert.72 = f32[1,2048]{1,0} convert(add.232), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=372}
  multiply.83 = f32[1,2048]{1,0} multiply(convert.72, broadcast.574), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/div" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=372}
  convert.73 = bf16[1,2048]{1,0} convert(multiply.83), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/convert_element_type[new_dtype=bfloat16 weak_type=False]" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=372}
  reshape.950 = bf16[2048]{0} reshape(convert.73)
  broadcast.594 = bf16[1,2048,8192]{2,1,0} broadcast(reshape.950), dimensions={1}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/broadcast_in_dim[shape=(8, 2048, 8192) broadcast_dimensions=(0, 1)]" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=372}
  add.233 = bf16[1,2048,8192]{2,1,0} add(add.231, broadcast.594), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/add_any" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=372}
  broadcast.595 = bf16[6,8192]{1,0} broadcast(constant.794), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/broadcast_in_dim[shape=(6, 8192) broadcast_dimensions=()]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  broadcast.596 = bf16[6,8192,1024]{2,1,0} broadcast(constant.794), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/broadcast_in_dim[shape=(6, 8192, 8192) broadcast_dimensions=()]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  broadcast.597 = bf16[6,3,8192]{2,1,0} broadcast(constant.794), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/broadcast_in_dim[shape=(6, 3, 8192) broadcast_dimensions=()]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  broadcast.598 = bf16[6,1024,3,8192]{3,2,1,0} broadcast(constant.794), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/broadcast_in_dim[shape=(6, 8192, 3, 8192) broadcast_dimensions=()]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  broadcast.599 = bf16[6,1,32768]{2,1,0} broadcast(constant.794), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/broadcast_in_dim[shape=(6, 1, 32768) broadcast_dimensions=()]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  broadcast.600 = bf16[6,1024,1,32768]{3,2,1,0} broadcast(constant.794), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/broadcast_in_dim[shape=(6, 8192, 1, 32768) broadcast_dimensions=()]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  broadcast.601 = bf16[6,32768,1024]{2,1,0} broadcast(constant.794), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/broadcast_in_dim[shape=(6, 32768, 8192) broadcast_dimensions=()]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.143 = bf16[6,1,2048,3,8192]{4,3,2,1,0} get-tuple-element(tuple.11), index=2, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while[cond_nconsts=0 body_nconsts=16]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.144 = bf16[6,1,2048,8192]{3,2,1,0} get-tuple-element(tuple.11), index=3, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while[cond_nconsts=0 body_nconsts=16]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.145 = bf16[6,1,2048,1,32768]{4,3,2,1,0} get-tuple-element(tuple.11), index=4, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while[cond_nconsts=0 body_nconsts=16]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.146 = bf16[6,1,2048,8192]{3,2,1,0} get-tuple-element(tuple.11), index=5, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while[cond_nconsts=0 body_nconsts=16]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  tuple.7 = (s32[], bf16[1,2048,8192]{2,1,0}, bf16[6,8192]{1,0}, bf16[6,8192,1024]{2,1,0}, bf16[6,3,8192]{2,1,0}, /*index=5*/bf16[6,1024,3,8192]{3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, /*index=10*/bf16[6,1,32768]{2,1,0}, bf16[6,1024,1,32768]{3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,32768,1024]{2,1,0}, bf16[6,1,2048,3,8192]{4,3,2,1,0}, /*index=15*/bf16[6,1,2048,8192]{3,2,1,0}, bf16[6,1,2048,1,32768]{4,3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,8192,1024]{2,1,0}, bf16[6,3,8192]{2,1,0}, /*index=20*/bf16[6,1024,3,8192]{3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, /*index=25*/bf16[6,1,32768]{2,1,0}, bf16[6,1024,1,32768]{3,2,1,0}, bf16[6,32768,1024]{2,1,0}, bf16[6,1,2048,8192]{3,2,1,0}, bf16[1,1,2048,2048]{3,2,1,0}) tuple(constant.793, add.233, broadcast.595, broadcast.596, broadcast.597, /*index=5*/broadcast.598, broadcast.595, broadcast.595, broadcast.595, broadcast.595, /*index=10*/broadcast.599, broadcast.600, broadcast.595, broadcast.601, get-tuple-element.143, /*index=15*/get-tuple-element.144, get-tuple-element.145, param.15, convert.47, param.17, /*index=20*/convert.48, convert.49, convert.50, convert.51, convert.52, /*index=25*/param.23, convert.53, convert.54, get-tuple-element.146, reshape.801), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while[cond_nconsts=0 body_nconsts=16]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.225 = bf16[6,32768,1024]{2,1,0} get-tuple-element(tuple.7), index=27
  constant.1153 = s32[] constant(5)
  subtract.162 = s32[] subtract(constant.1153, constant.793), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  constant.1154 = s32[] constant(0), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/reduce_sum[axes=(2, 1)]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  compare.218 = pred[] compare(subtract.162, constant.1154), direction=LT, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/lt" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  constant.1155 = s32[] constant(11)
  subtract.163 = s32[] subtract(constant.1155, constant.793), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/add" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  select.395 = s32[] select(compare.218, subtract.163, subtract.162), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/select_n" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-slice.85 = bf16[1,32768,1024]{2,1,0} dynamic-slice(get-tuple-element.225, select.395, constant.1154, constant.1154), dynamic_slice_sizes={1,32768,1024}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 32768, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1044 = bf16[32768,1024]{1,0} reshape(dynamic-slice.85), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  all-gather.22 = bf16[32768,8192]{1,0} all-gather(reshape.1044), channel_id=65, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={1}, use_global_device_ids=true, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/dot_general[dimension_numbers=(((2,), (1,)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=1022}
  get-tuple-element.226 = bf16[6,1024,1,32768]{3,2,1,0} get-tuple-element(tuple.7), index=26
  constant.1156 = s32[] constant(5)
  subtract.164 = s32[] subtract(constant.1156, constant.793), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  constant.1157 = s32[] constant(0), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/reduce_sum[axes=(2, 1)]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  compare.219 = pred[] compare(subtract.164, constant.1157), direction=LT, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/lt" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  constant.1158 = s32[] constant(11)
  subtract.165 = s32[] subtract(constant.1158, constant.793), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/add" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  select.396 = s32[] select(compare.219, subtract.165, subtract.164), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/select_n" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-slice.86 = bf16[1,1024,1,32768]{3,2,1,0} dynamic-slice(get-tuple-element.226, select.396, constant.1157, constant.1157, constant.1157), dynamic_slice_sizes={1,1024,1,32768}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192, 1, 32768)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1045 = bf16[1024,32768]{1,0} reshape(dynamic-slice.86)
  all-gather.23 = bf16[8192,32768]{1,0} all-gather(reshape.1045), channel_id=66, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={0}, use_global_device_ids=true, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/dot_general[dimension_numbers=(((2, 3), (1, 2)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=962}
  get-tuple-element.227 = bf16[6,8192,1024]{2,1,0} get-tuple-element(tuple.7), index=18
  constant.1159 = s32[] constant(5)
  subtract.166 = s32[] subtract(constant.1159, constant.793), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  constant.1160 = s32[] constant(0), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/reduce_sum[axes=(2, 1)]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  compare.220 = pred[] compare(subtract.166, constant.1160), direction=LT, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/lt" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  constant.1161 = s32[] constant(11)
  subtract.167 = s32[] subtract(constant.1161, constant.793), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/add" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  select.397 = s32[] select(compare.220, subtract.167, subtract.166), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/select_n" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-slice.87 = bf16[1,8192,1024]{2,1,0} dynamic-slice(get-tuple-element.227, select.397, constant.1160, constant.1160), dynamic_slice_sizes={1,8192,1024}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1046 = bf16[8192,1024]{1,0} reshape(dynamic-slice.87), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  all-gather.24 = bf16[8192,8192]{1,0} all-gather(reshape.1046), channel_id=67, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={1}, use_global_device_ids=true, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/out/dot_general[dimension_numbers=(((2,), (1,)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=451}
  get-tuple-element.228 = bf16[6,1024,3,8192]{3,2,1,0} get-tuple-element(tuple.7), index=20
  constant.1162 = s32[] constant(5)
  subtract.168 = s32[] subtract(constant.1162, constant.793), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  constant.1163 = s32[] constant(0), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/reduce_sum[axes=(2, 1)]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  compare.221 = pred[] compare(subtract.168, constant.1163), direction=LT, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/lt" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  constant.1164 = s32[] constant(11)
  subtract.169 = s32[] subtract(constant.1164, constant.793), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/add" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  select.398 = s32[] select(compare.221, subtract.169, subtract.168), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/select_n" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-slice.88 = bf16[1,1024,3,8192]{3,2,1,0} dynamic-slice(get-tuple-element.228, select.398, constant.1163, constant.1163, constant.1163), dynamic_slice_sizes={1,1024,3,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192, 3, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1047 = bf16[1024,3,8192]{2,1,0} reshape(dynamic-slice.88), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  all-gather.25 = bf16[8192,3,8192]{2,1,0} all-gather(reshape.1047), channel_id=68, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={0}, use_global_device_ids=true, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/dot_general[dimension_numbers=(((2, 3), (1, 2)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=652}
  constant.1165 = s32[] constant(1)
  tuple.13 = (s32[], bf16[1,2048,8192]{2,1,0}, bf16[6,8192]{1,0}, bf16[6,8192,1024]{2,1,0}, bf16[6,3,8192]{2,1,0}, /*index=5*/bf16[6,1024,3,8192]{3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, /*index=10*/bf16[6,1,32768]{2,1,0}, bf16[6,1024,1,32768]{3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,32768,1024]{2,1,0}, bf16[6,1,2048,3,8192]{4,3,2,1,0}, /*index=15*/bf16[6,1,2048,8192]{3,2,1,0}, bf16[6,1,2048,1,32768]{4,3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,8192,1024]{2,1,0}, bf16[6,3,8192]{2,1,0}, /*index=20*/bf16[6,1024,3,8192]{3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, /*index=25*/bf16[6,1,32768]{2,1,0}, bf16[6,1024,1,32768]{3,2,1,0}, bf16[6,32768,1024]{2,1,0}, bf16[6,1,2048,8192]{3,2,1,0}, bf16[1,1,2048,2048]{3,2,1,0}, /*index=30*/bf16[32768,8192]{1,0}, bf16[8192,32768]{1,0}, bf16[8192,8192]{1,0}, bf16[8192,3,8192]{2,1,0}, s32[]) tuple(constant.793, add.233, broadcast.595, broadcast.596, broadcast.597, /*index=5*/broadcast.598, broadcast.595, broadcast.595, broadcast.595, broadcast.595, /*index=10*/broadcast.599, broadcast.600, broadcast.595, broadcast.601, get-tuple-element.143, /*index=15*/get-tuple-element.144, get-tuple-element.145, param.15, convert.47, param.17, /*index=20*/convert.48, convert.49, convert.50, convert.51, convert.52, /*index=25*/param.23, convert.53, convert.54, get-tuple-element.146, reshape.801, /*index=30*/all-gather.22, all-gather.23, all-gather.24, all-gather.25, constant.1165)
  while.4 = (s32[], bf16[1,2048,8192]{2,1,0}, bf16[6,8192]{1,0}, bf16[6,8192,1024]{2,1,0}, bf16[6,3,8192]{2,1,0}, /*index=5*/bf16[6,1024,3,8192]{3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, /*index=10*/bf16[6,1,32768]{2,1,0}, bf16[6,1024,1,32768]{3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,32768,1024]{2,1,0}, bf16[6,1,2048,3,8192]{4,3,2,1,0}, /*index=15*/bf16[6,1,2048,8192]{3,2,1,0}, bf16[6,1,2048,1,32768]{4,3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,8192,1024]{2,1,0}, bf16[6,3,8192]{2,1,0}, /*index=20*/bf16[6,1024,3,8192]{3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, /*index=25*/bf16[6,1,32768]{2,1,0}, bf16[6,1024,1,32768]{3,2,1,0}, bf16[6,32768,1024]{2,1,0}, bf16[6,1,2048,8192]{3,2,1,0}, bf16[1,1,2048,2048]{3,2,1,0}, /*index=30*/bf16[32768,8192]{1,0}, bf16[8192,32768]{1,0}, bf16[8192,8192]{1,0}, bf16[8192,3,8192]{2,1,0}, s32[]) while(tuple.13), condition=region_39.1489_spmd.1, body=region_25.1008_spmd.1
  get-tuple-element.287 = s32[] get-tuple-element(while.4), index=0
  constant.1197 = s32[] constant(1)
  add.428 = s32[] add(get-tuple-element.287, constant.1197), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/add" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.288 = bf16[1,2048,8192]{2,1,0} get-tuple-element(while.4), index=1
  get-tuple-element.289 = bf16[6,1,2048,1,32768]{4,3,2,1,0} get-tuple-element(while.4), index=16
  constant.1198 = s32[] constant(5)
  subtract.182 = s32[] subtract(constant.1198, get-tuple-element.287), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  constant.1199 = s32[] constant(0), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/reduce_sum[axes=(2, 1)]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  compare.231 = pred[] compare(subtract.182, constant.1199), direction=LT, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/lt" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  constant.1200 = s32[] constant(11)
  subtract.183 = s32[] subtract(constant.1200, get-tuple-element.287), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/add" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  select.405 = s32[] select(compare.231, subtract.183, subtract.182), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/select_n" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-slice.108 = bf16[1,1,2048,1,32768]{4,3,2,1,0} dynamic-slice(get-tuple-element.289, select.405, constant.1199, constant.1199, constant.1199, /*index=5*/constant.1199), dynamic_slice_sizes={1,1,2048,1,32768}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8, 2048, 1, 32768)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.290 = bf16[6,1,32768]{2,1,0} get-tuple-element(while.4), index=25
  dynamic-slice.109 = bf16[1,1,32768]{2,1,0} dynamic-slice(get-tuple-element.290, select.405, constant.1199, constant.1199), dynamic_slice_sizes={1,1,32768}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 1, 32768)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1109 = bf16[32768]{0} reshape(dynamic-slice.109)
  broadcast.966 = bf16[1,1,2048,1,32768]{4,3,2,1,0} broadcast(reshape.1109), dimensions={4}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=972}
  add.429 = bf16[1,1,2048,1,32768]{4,3,2,1,0} add(dynamic-slice.108, broadcast.966), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=972}
  reshape.1110 = bf16[1,2048,1,32768]{3,2,1,0} reshape(add.429), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=972}
  reshape.1112 = bf16[2048,8192]{1,0} reshape(get-tuple-element.288)
  get-tuple-element.292 = bf16[32768,8192]{1,0} get-tuple-element(while.4), index=30
  dot.69 = bf16[2048,32768]{1,0} dot(reshape.1112, get-tuple-element.292), lhs_contracting_dims={1}, rhs_contracting_dims={1}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/dot_general[dimension_numbers=(((2,), (1,)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=1022}
  reshape.1114 = bf16[1,2048,1,32768]{3,2,1,0} reshape(dot.69), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/reshape[new_sizes=(8, 2048, 1, 32768) dimensions=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=985}
  multiply.306 = bf16[1,2048,1,32768]{3,2,1,0} multiply(reshape.1110, reshape.1114), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  constant.1201 = bf16[] constant(0.5)
  broadcast.967 = bf16[1,2048,1,32768]{3,2,1,0} broadcast(constant.1201), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  multiply.307 = bf16[1,2048,1,32768]{3,2,1,0} multiply(multiply.306, broadcast.967), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  constant.1202 = bf16[] constant(1)
  broadcast.968 = bf16[1,2048,1,32768]{3,2,1,0} broadcast(constant.1202), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  multiply.308 = bf16[1,2048,1,32768]{3,2,1,0} multiply(reshape.1110, reshape.1110), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  multiply.309 = bf16[1,2048,1,32768]{3,2,1,0} multiply(reshape.1110, multiply.308), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  constant.1203 = bf16[] constant(0.04468)
  broadcast.969 = bf16[1,2048,1,32768]{3,2,1,0} broadcast(constant.1203), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  multiply.310 = bf16[1,2048,1,32768]{3,2,1,0} multiply(multiply.309, broadcast.969), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  add.430 = bf16[1,2048,1,32768]{3,2,1,0} add(reshape.1110, multiply.310), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  constant.1204 = bf16[] constant(0.7969)
  broadcast.970 = bf16[1,2048,1,32768]{3,2,1,0} broadcast(constant.1204), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  multiply.311 = bf16[1,2048,1,32768]{3,2,1,0} multiply(add.430, broadcast.970), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  tanh.6 = bf16[1,2048,1,32768]{3,2,1,0} tanh(multiply.311), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/tanh" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  subtract.184 = bf16[1,2048,1,32768]{3,2,1,0} subtract(broadcast.968, tanh.6), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/sub" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  multiply.312 = bf16[1,2048,1,32768]{3,2,1,0} multiply(multiply.307, subtract.184), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  multiply.313 = bf16[1,2048,1,32768]{3,2,1,0} multiply(multiply.312, tanh.6), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  add.432 = bf16[1,2048,1,32768]{3,2,1,0} add(multiply.312, multiply.313), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add_any" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  multiply.314 = bf16[1,2048,1,32768]{3,2,1,0} multiply(add.432, broadcast.970), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  constant.1205 = bf16[] constant(0.03564)
  broadcast.971 = bf16[1,2048,1,32768]{3,2,1,0} broadcast(constant.1205), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  multiply.315 = bf16[1,2048,1,32768]{3,2,1,0} multiply(add.432, broadcast.971), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  constant.1206 = bf16[] constant(3)
  broadcast.972 = bf16[1,2048,1,32768]{3,2,1,0} broadcast(constant.1206), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  multiply.316 = bf16[1,2048,1,32768]{3,2,1,0} multiply(multiply.308, broadcast.972), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  multiply.317 = bf16[1,2048,1,32768]{3,2,1,0} multiply(multiply.315, multiply.316), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  add.433 = bf16[1,2048,1,32768]{3,2,1,0} add(multiply.314, multiply.317), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add_any" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  add.434 = bf16[1,2048,1,32768]{3,2,1,0} add(tanh.6, broadcast.968), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  multiply.318 = bf16[1,2048,1,32768]{3,2,1,0} multiply(add.434, broadcast.967), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  multiply.319 = bf16[1,2048,1,32768]{3,2,1,0} multiply(reshape.1114, multiply.318), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  add.435 = bf16[1,2048,1,32768]{3,2,1,0} add(add.433, multiply.319), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add_any" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  reshape.1115 = bf16[2048,32768]{1,0} reshape(add.435)
  get-tuple-element.294 = bf16[8192,32768]{1,0} get-tuple-element(while.4), index=31
  dot.70 = bf16[2048,8192]{1,0} dot(reshape.1115, get-tuple-element.294), lhs_contracting_dims={1}, rhs_contracting_dims={1}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/dot_general[dimension_numbers=(((2, 3), (1, 2)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=962}
  reshape.1118 = bf16[1,2048,8192]{2,1,0} reshape(dot.70), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/xmap(transpose(<lambda>))/squeeze[dimensions=(0,)]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1186}
  get-tuple-element.295 = bf16[6,1,2048,8192]{3,2,1,0} get-tuple-element(while.4), index=15
  dynamic-slice.112 = bf16[1,1,2048,8192]{3,2,1,0} dynamic-slice(get-tuple-element.295, select.405, constant.1199, constant.1199, constant.1199), dynamic_slice_sizes={1,1,2048,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8, 2048, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.296 = bf16[6,8192]{1,0} get-tuple-element(while.4), index=17
  dynamic-slice.113 = bf16[1,8192]{1,0} dynamic-slice(get-tuple-element.296, select.405, constant.1199), dynamic_slice_sizes={1,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1119 = bf16[8192]{0} reshape(dynamic-slice.113), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  broadcast.973 = bf16[1,1,2048,8192]{3,2,1,0} broadcast(reshape.1119), dimensions={3}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/out/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=455}
  add.437 = bf16[1,1,2048,8192]{3,2,1,0} add(dynamic-slice.112, broadcast.973), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/out/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=455}
  get-tuple-element.297 = bf16[6,1,2048,8192]{3,2,1,0} get-tuple-element(while.4), index=28
  dynamic-slice.114 = bf16[1,1,2048,8192]{3,2,1,0} dynamic-slice(get-tuple-element.297, select.405, constant.1199, constant.1199, constant.1199), dynamic_slice_sizes={1,1,2048,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8, 2048, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  add.438 = bf16[1,1,2048,8192]{3,2,1,0} add(add.437, dynamic-slice.114), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/transformer.py" source_line=1124}
  reshape.1120 = bf16[1,2048,8192]{2,1,0} reshape(add.438), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/xmap(<lambda>)/squeeze[dimensions=(0,)]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1186}
  get-tuple-element.298 = bf16[6,8192]{1,0} get-tuple-element(while.4), index=24
  dynamic-slice.115 = bf16[1,8192]{1,0} dynamic-slice(get-tuple-element.298, select.405, constant.1199), dynamic_slice_sizes={1,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1122 = bf16[8192]{0} reshape(dynamic-slice.115), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.299 = bf16[6,8192]{1,0} get-tuple-element(while.4), index=23
  dynamic-slice.116 = bf16[1,8192]{1,0} dynamic-slice(get-tuple-element.299, select.405, constant.1199), dynamic_slice_sizes={1,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1123 = bf16[8192]{0} reshape(dynamic-slice.116), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  custom-call.63 = (bf16[1,2048,8192]{2,1,0}, f32[2048]{0}, f32[2048]{0}) custom-call(reshape.1120, reshape.1122, reshape.1123), custom_call_target="te_layernorm_forward", operand_layout_constraints={bf16[1,2048,8192]{2,1,0}, bf16[8192]{0}, bf16[8192]{0}}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/xmap(<lambda>)/te_layernorm_forward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}, backend_config="\000\010\000\000\000\000\000\000\000 \000\000\000\000\000\000\005\000\000\000\005\000\000\000\001O5X\254\305\'7"
  get-tuple-element.300 = f32[2048]{0} get-tuple-element(custom-call.63), index=1, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/xmap(<lambda>)/te_layernorm_forward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  get-tuple-element.301 = f32[2048]{0} get-tuple-element(custom-call.63), index=2, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/xmap(<lambda>)/te_layernorm_forward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  custom-call.64 = (bf16[1,2048,8192]{2,1,0}, bf16[8192]{0}, bf16[8192]{0}) custom-call(reshape.1118, get-tuple-element.300, get-tuple-element.301, reshape.1120, reshape.1122), custom_call_target="te_layernorm_backward", operand_layout_constraints={bf16[1,2048,8192]{2,1,0}, f32[2048]{0}, f32[2048]{0}, bf16[1,2048,8192]{2,1,0}, bf16[8192]{0}}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/xmap(transpose(<lambda>))/te_layernorm_backward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}, backend_config="\000\010\000\000\000\000\000\000\000 \000\000\000\000\000\000\005\000\000\000\005\000\000\000\001O5X\254\305\'7"
  get-tuple-element.302 = bf16[1,2048,8192]{2,1,0} get-tuple-element(custom-call.64), index=0, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/xmap(transpose(<lambda>))/te_layernorm_backward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  add.439 = bf16[1,2048,8192]{2,1,0} add(get-tuple-element.288, get-tuple-element.302), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add_any" source_file="/opt/transformer-engine/transformer_engine/jax/layernorm.py" source_line=73}
  get-tuple-element.303 = bf16[6,1,2048,3,8192]{4,3,2,1,0} get-tuple-element(while.4), index=14
  dynamic-slice.117 = bf16[1,1,2048,3,8192]{4,3,2,1,0} dynamic-slice(get-tuple-element.303, select.405, constant.1199, constant.1199, constant.1199, /*index=5*/constant.1199), dynamic_slice_sizes={1,1,2048,3,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8, 2048, 3, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.304 = bf16[6,3,8192]{2,1,0} get-tuple-element(while.4), index=19
  dynamic-slice.118 = bf16[1,3,8192]{2,1,0} dynamic-slice(get-tuple-element.304, select.405, constant.1199, constant.1199), dynamic_slice_sizes={1,3,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 3, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1124 = bf16[3,8192]{1,0} reshape(dynamic-slice.118), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  broadcast.974 = bf16[1,1,2048,3,8192]{4,3,2,1,0} broadcast(reshape.1124), dimensions={3,4}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=664}
  add.440 = bf16[1,1,2048,3,8192]{4,3,2,1,0} add(dynamic-slice.117, broadcast.974), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/add" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=664}
  reshape.1125 = bf16[1,2048,3,64,128]{4,3,2,1,0} reshape(add.440), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(transpose(<lambda>))/squeeze[dimensions=(0,)]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1186}
  constant.1207 = bf16[0]{0} constant({})
  constant.1208 = s32[1]{0} constant({0}), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/concatenate[dimension=0]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  broadcast.975 = s32[1,1,2048,2048]{3,2,1,0} broadcast(constant.1197), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub.body_fn/sub/sub" source_file="/opt/paxml/paxml/contrib/gpu/scripts_gpu/te_helper.py" source_line=121}
  get-tuple-element.305 = bf16[1,1,2048,2048]{3,2,1,0} get-tuple-element(while.4), index=29
  iota.54 = s32[2048,2048]{1,0} iota(), iota_dimension=0, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/broadcast_in_dim[shape=(1, 2048, 2048, 1) broadcast_dimensions=(0, 1, 3)]" source_file="/opt/praxis/praxis/layers/attentions.py" source_line=103}
  iota.55 = s32[2048,2048]{1,0} iota(), iota_dimension=1, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/broadcast_in_dim[shape=(2048, 1, 1, 2048) broadcast_dimensions=(1, 2, 3)]" source_file="/opt/praxis/praxis/layers/attentions.py" source_line=102}
  compare.232 = pred[2048,2048]{1,0} compare(iota.54, iota.55), direction=LT, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/lt" source_file="/opt/praxis/praxis/layers/attentions.py" source_line=104}
  constant.1209 = bf16[] constant(-2.366e+38)
  broadcast.976 = bf16[2048,2048]{1,0} broadcast(constant.1209), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/mul" source_file="/opt/praxis/praxis/layers/attentions.py" source_line=104}
  constant.1210 = bf16[] constant(0)
  broadcast.977 = bf16[2048,2048]{1,0} broadcast(constant.1210), dimensions={}
  select.406 = bf16[2048,2048]{1,0} select(compare.232, broadcast.976, broadcast.977), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/mul" source_file="/opt/praxis/praxis/layers/attentions.py" source_line=104}
  reshape.1127 = bf16[1,1,2048,2048]{3,2,1,0} reshape(select.406), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/min" source_file="/opt/praxis/praxis/layers/attentions.py" source_line=170}
  minimum.6 = bf16[1,1,2048,2048]{3,2,1,0} minimum(get-tuple-element.305, reshape.1127), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/min" source_file="/opt/praxis/praxis/layers/attentions.py" source_line=170}
  broadcast.978 = bf16[1,1,2048,2048]{3,2,1,0} broadcast(constant.1210), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub.body_fn/sub/eq" source_file="/opt/paxml/paxml/contrib/gpu/scripts_gpu/te_helper.py" source_line=121}
  compare.233 = pred[1,1,2048,2048]{3,2,1,0} compare(minimum.6, broadcast.978), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/eq" source_file="/opt/paxml/paxml/contrib/gpu/scripts_gpu/te_helper.py" source_line=121}
  convert.177 = s32[1,1,2048,2048]{3,2,1,0} convert(compare.233), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/convert_element_type[new_dtype=int32 weak_type=True]" source_file="/opt/paxml/paxml/contrib/gpu/scripts_gpu/te_helper.py" source_line=121}
  subtract.185 = s32[1,1,2048,2048]{3,2,1,0} subtract(broadcast.975, convert.177), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/sub" source_file="/opt/paxml/paxml/contrib/gpu/scripts_gpu/te_helper.py" source_line=121}
  convert.178 = u8[1,1,2048,2048]{3,2,1,0} convert(subtract.185), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/convert_element_type[new_dtype=uint8 weak_type=False]" source_file="/opt/paxml/paxml/contrib/gpu/scripts_gpu/te_helper.py" source_line=122}
  slice.5 = u8[1,1,2048,1]{3,2,1,0} slice(convert.178), slice={[0:1], [0:1], [0:2048], [0:1]}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/slice[start_indices=(0, 0, 0, 0) limit_indices=(1, 1, 2048, 1) strides=None]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  constant.1211 = u8[] constant(0), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/eq" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  broadcast.979 = u8[1,1,2048,1]{3,2,1,0} broadcast(constant.1211), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/eq" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  compare.234 = pred[1,1,2048,1]{3,2,1,0} compare(slice.5, broadcast.979), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/eq" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  convert.179 = s32[1,1,2048,1]{3,2,1,0} convert(compare.234), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/convert_element_type[new_dtype=int32 weak_type=False]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  reduce.73 = s32[1]{0} reduce(convert.179, constant.1199), dimensions={1,2,3}, to_apply=region_0.369, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/reduce_sum[axes=(2, 1)]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  concatenate.4 = s32[2]{0} concatenate(constant.1208, reduce.73), dimensions={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/concatenate[dimension=0]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  constant.1212 = u32[] constant(0)
  broadcast.980 = u32[2]{0} broadcast(constant.1212), dimensions={}
  custom-call.65 = (bf16[1,2048,64,128]{3,2,1,0}, f32[1,64,2048,1]{3,2,1,0}, u32[4]{0}) custom-call(reshape.1125, constant.1207, concatenate.4, broadcast.980), custom_call_target="te_self_fused_attn_forward", operand_layout_constraints={bf16[1,2048,3,64,128]{4,3,2,1,0}, bf16[0]{0}, s32[2]{0}, u32[2]{0}}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/te_self_fused_attn_forward[attn_bias_type=NVTE_Bias_Type.NVTE_NO_BIAS attn_mask_type=NVTE_Mask_Type.NVTE_CAUSAL_MASK scaling_factor=0.08838834764831843 dropout_probability=0.0 is_training=True]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}, backend_config="\001\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\000\010\000\000\000\000\000\000\000\010\000\000\000\000\000\000\200\000\000\000\000\000\000\000\363\004\265=\000\000\000\000\000\000\000\000\002\000\000\000\005\000\000\000\001\177\000\000"
  get-tuple-element.306 = f32[1,64,2048,1]{3,2,1,0} get-tuple-element(custom-call.65), index=1, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/te_self_fused_attn_forward[attn_bias_type=NVTE_Bias_Type.NVTE_NO_BIAS attn_mask_type=NVTE_Mask_Type.NVTE_CAUSAL_MASK scaling_factor=0.08838834764831843 dropout_probability=0.0 is_training=True]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  get-tuple-element.307 = u32[4]{0} get-tuple-element(custom-call.65), index=2, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/te_self_fused_attn_forward[attn_bias_type=NVTE_Bias_Type.NVTE_NO_BIAS attn_mask_type=NVTE_Mask_Type.NVTE_CAUSAL_MASK scaling_factor=0.08838834764831843 dropout_probability=0.0 is_training=True]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  get-tuple-element.308 = bf16[1,2048,64,128]{3,2,1,0} get-tuple-element(custom-call.65), index=0, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(<lambda>)/te_self_fused_attn_forward[attn_bias_type=NVTE_Bias_Type.NVTE_NO_BIAS attn_mask_type=NVTE_Mask_Type.NVTE_CAUSAL_MASK scaling_factor=0.08838834764831843 dropout_probability=0.0 is_training=True]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  reshape.1128 = bf16[2048,8192]{1,0} reshape(add.439), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/add_any" source_file="/opt/transformer-engine/transformer_engine/jax/layernorm.py" source_line=73}
  get-tuple-element.310 = bf16[8192,8192]{1,0} get-tuple-element(while.4), index=32
  dot.71 = bf16[2048,8192]{1,0} dot(reshape.1128, get-tuple-element.310), lhs_contracting_dims={1}, rhs_contracting_dims={1}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/out/dot_general[dimension_numbers=(((2,), (1,)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=451}
  reshape.1130 = bf16[1,2048,64,128]{3,2,1,0} reshape(dot.71), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(transpose(<lambda>))/squeeze[dimensions=(0,)]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1186}
  custom-call.66 = (bf16[1,2048,3,64,128]{4,3,2,1,0}, bf16[0]{0}) custom-call(reshape.1125, get-tuple-element.306, get-tuple-element.307, get-tuple-element.308, reshape.1130, /*index=5*/concatenate.4), custom_call_target="te_self_fused_attn_backward", operand_layout_constraints={bf16[1,2048,3,64,128]{4,3,2,1,0}, f32[1,64,2048,1]{3,2,1,0}, u32[4]{0}, bf16[1,2048,64,128]{3,2,1,0}, bf16[1,2048,64,128]{3,2,1,0}, s32[2]{0}}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(transpose(<lambda>))/te_self_fused_attn_backward[attn_bias_type=NVTE_Bias_Type.NVTE_NO_BIAS attn_mask_type=NVTE_Mask_Type.NVTE_CAUSAL_MASK scaling_factor=0.08838834764831843 dropout_probability=0.0 is_training=True]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}, backend_config="\001\000\000\000\000\000\000\000@\000\000\000\000\000\000\000\000\010\000\000\000\000\000\000\000\010\000\000\000\000\000\000\200\000\000\000\000\000\000\000\363\004\265=\000\000\000\000\000\000\000\000\002\000\000\000\005\000\000\000\001\177\000\000"
  get-tuple-element.311 = bf16[1,2048,3,64,128]{4,3,2,1,0} get-tuple-element(custom-call.66), index=0, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/xmap(transpose(<lambda>))/te_self_fused_attn_backward[attn_bias_type=NVTE_Bias_Type.NVTE_NO_BIAS attn_mask_type=NVTE_Mask_Type.NVTE_CAUSAL_MASK scaling_factor=0.08838834764831843 dropout_probability=0.0 is_training=True]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  reshape.1132 = bf16[2048,24576]{1,0} reshape(get-tuple-element.311), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/reshape[new_sizes=(8, 2048, 3, 8192) dimensions=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/transformer.py" source_line=654}
  get-tuple-element.313 = bf16[8192,3,8192]{2,1,0} get-tuple-element(while.4), index=33
  transpose.48 = bf16[3,8192,8192]{1,0,2} transpose(get-tuple-element.313), dimensions={1,2,0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/dot_general[dimension_numbers=(((2, 3), (1, 2)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=652}
  reshape.1134 = bf16[24576,8192]{1,0} reshape(transpose.48), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/dot_general[dimension_numbers=(((2, 3), (1, 2)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=652}
  dot.72 = bf16[2048,8192]{1,0} dot(reshape.1132, reshape.1134), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/dot_general[dimension_numbers=(((2, 3), (1, 2)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=652}
  reshape.1135 = bf16[1,2048,8192]{2,1,0} reshape(dot.72), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/xmap(transpose(<lambda>))/squeeze[dimensions=(0,)]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1186}
  reshape.1137 = bf16[1,2048,8192]{2,1,0} reshape(dynamic-slice.114), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/xmap(<lambda>)/squeeze[dimensions=(0,)]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1186}
  get-tuple-element.314 = bf16[6,8192]{1,0} get-tuple-element(while.4), index=22
  dynamic-slice.121 = bf16[1,8192]{1,0} dynamic-slice(get-tuple-element.314, select.405, constant.1199), dynamic_slice_sizes={1,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1138 = bf16[8192]{0} reshape(dynamic-slice.121), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.315 = bf16[6,8192]{1,0} get-tuple-element(while.4), index=21
  dynamic-slice.122 = bf16[1,8192]{1,0} dynamic-slice(get-tuple-element.315, select.405, constant.1199), dynamic_slice_sizes={1,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1139 = bf16[8192]{0} reshape(dynamic-slice.122), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  custom-call.67 = (bf16[1,2048,8192]{2,1,0}, f32[2048]{0}, f32[2048]{0}) custom-call(reshape.1137, reshape.1138, reshape.1139), custom_call_target="te_layernorm_forward", operand_layout_constraints={bf16[1,2048,8192]{2,1,0}, bf16[8192]{0}, bf16[8192]{0}}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/xmap(<lambda>)/te_layernorm_forward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}, backend_config="\000\010\000\000\000\000\000\000\000 \000\000\000\000\000\000\005\000\000\000\005\000\000\000\001O5X\254\305\'7"
  get-tuple-element.316 = f32[2048]{0} get-tuple-element(custom-call.67), index=1, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/xmap(<lambda>)/te_layernorm_forward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  get-tuple-element.317 = f32[2048]{0} get-tuple-element(custom-call.67), index=2, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/xmap(<lambda>)/te_layernorm_forward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  custom-call.68 = (bf16[1,2048,8192]{2,1,0}, bf16[8192]{0}, bf16[8192]{0}) custom-call(reshape.1135, get-tuple-element.316, get-tuple-element.317, reshape.1137, reshape.1138), custom_call_target="te_layernorm_backward", operand_layout_constraints={bf16[1,2048,8192]{2,1,0}, f32[2048]{0}, f32[2048]{0}, bf16[1,2048,8192]{2,1,0}, bf16[8192]{0}}, api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/xmap(transpose(<lambda>))/te_layernorm_backward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}, backend_config="\000\010\000\000\000\000\000\000\000 \000\000\000\000\000\000\005\000\000\000\005\000\000\000\001O5X\254\305\'7"
  get-tuple-element.318 = bf16[1,2048,8192]{2,1,0} get-tuple-element(custom-call.68), index=0, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/xmap(transpose(<lambda>))/te_layernorm_backward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  add.442 = bf16[1,2048,8192]{2,1,0} add(add.439, get-tuple-element.318), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/add_any" source_file="/opt/transformer-engine/transformer_engine/jax/layernorm.py" source_line=73}
  get-tuple-element.319 = bf16[6,8192]{1,0} get-tuple-element(while.4), index=2
  reduce.74 = bf16[8192]{0} reduce(add.439, constant.1210), dimensions={0,1}, to_apply=region_16.868, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/out/reduce_sum[axes=(0, 1)]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=455}
  all-reduce.42 = bf16[8192]{0} all-reduce(reduce.74), channel_id=85, replica_groups={{0}}, to_apply=region_16.868, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/out/reduce_sum[axes=(0, 1)]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=455}
  reshape.1140 = bf16[1,8192]{1,0} reshape(all-reduce.42), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/broadcast_in_dim[shape=(1, 8192) broadcast_dimensions=(1,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-update-slice.40 = bf16[6,8192]{1,0} dynamic-update-slice(get-tuple-element.319, reshape.1140, select.405, constant.1199), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_update_slice" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.320 = bf16[6,8192,1024]{2,1,0} get-tuple-element(while.4), index=3
  reshape.1142 = bf16[1,2048,8192]{2,1,0} reshape(get-tuple-element.308), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/reshape[new_sizes=(8, 2048, 8192) dimensions=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/transformer.py" source_line=720}
  transpose.49 = bf16[8192,1,2048]{0,2,1} transpose(reshape.1142), dimensions={2,0,1}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/sharding_constraint[sharding=GSPMDSharding({devices=[8,1,1]<=[8]}) resource_env=ResourceEnv(Mesh(device_ids=array([[[0],\n        [1],\n        [2],\n        [3],\n        [4],\n        [5],\n        [6],\n        [7]]]), axis_names=(\'replica\', \'data\', \'mdl\')), ()) unconstrained_dims=set()]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=40}
  reshape.1144 = bf16[8192,2048]{1,0} reshape(transpose.49), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/sharding_constraint[sharding=GSPMDSharding({devices=[8,1,1]<=[8]}) resource_env=ResourceEnv(Mesh(device_ids=array([[[0],\n        [1],\n        [2],\n        [3],\n        [4],\n        [5],\n        [6],\n        [7]]]), axis_names=(\'replica\', \'data\', \'mdl\')), ()) unconstrained_dims=set()]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=40}
  dot.73 = bf16[8192,8192]{1,0} dot(reshape.1144, reshape.1128), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/out/transpose[permutation=(1, 0)]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=451}
  reduce-scatter.12 = bf16[8192,1024]{1,0} reduce-scatter(dot.73), channel_id=86, replica_groups={{0}}, dimensions={1}, to_apply=region_16.868
  reshape.1146 = bf16[1,8192,1024]{2,1,0} reshape(reduce-scatter.12), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/broadcast_in_dim[shape=(1, 8192, 8192) broadcast_dimensions=(1, 2)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-update-slice.41 = bf16[6,8192,1024]{2,1,0} dynamic-update-slice(get-tuple-element.320, reshape.1146, select.405, constant.1199, constant.1199), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_update_slice" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.321 = bf16[6,3,8192]{2,1,0} get-tuple-element(while.4), index=4
  reduce.75 = bf16[3,64,128]{2,1,0} reduce(get-tuple-element.311, constant.1210), dimensions={0,1}, to_apply=region_16.868, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/reduce_sum[axes=(0, 1)]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=664}
  all-reduce.43 = bf16[3,64,128]{2,1,0} all-reduce(reduce.75), channel_id=87, replica_groups={{0}}, to_apply=region_16.868, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/reduce_sum[axes=(0, 1)]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=664}
  reshape.1148 = bf16[1,3,8192]{2,1,0} reshape(all-reduce.43), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/broadcast_in_dim[shape=(1, 3, 8192) broadcast_dimensions=(1, 2)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-update-slice.42 = bf16[6,3,8192]{2,1,0} dynamic-update-slice(get-tuple-element.321, reshape.1148, select.405, constant.1199, constant.1199), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_update_slice" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.322 = bf16[6,1024,3,8192]{3,2,1,0} get-tuple-element(while.4), index=5
  reshape.1150 = bf16[1,2048,3,8192]{3,2,1,0} reshape(get-tuple-element.311), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/reshape[new_sizes=(8, 2048, 3, 8192) dimensions=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/transformer.py" source_line=654}
  transpose.50 = bf16[3,8192,1,2048]{1,0,3,2} transpose(reshape.1150), dimensions={2,3,0,1}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/reshape[new_sizes=(8, 2048, 3, 8192) dimensions=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/transformer.py" source_line=654}
  reshape.1151 = bf16[24576,2048]{1,0} reshape(transpose.50), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/reshape[new_sizes=(8, 2048, 3, 8192) dimensions=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/transformer.py" source_line=654}
  get-tuple-element.323 = bf16[1,2048,8192]{2,1,0} get-tuple-element(custom-call.67), index=0, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/xmap(<lambda>)/te_layernorm_forward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  reshape.1152 = bf16[2048,8192]{1,0} reshape(get-tuple-element.323), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/reshape[new_sizes=(8, 2048, 8192) dimensions=None]" source_file="/opt/transformer-engine/transformer_engine/jax/layernorm.py" source_line=94}
  dot.74 = bf16[24576,8192]{1,0} dot(reshape.1151, reshape.1152), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/dot_general[dimension_numbers=(((0, 1), (0, 1)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=652}
  reshape.1154 = bf16[3,8192,8192]{2,1,0} reshape(dot.74), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/dot_general[dimension_numbers=(((0, 1), (0, 1)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=652}
  reduce-scatter.13 = bf16[3,8192,1024]{2,1,0} reduce-scatter(reshape.1154), channel_id=88, replica_groups={{0}}, dimensions={2}, to_apply=region_16.868
  transpose.51 = bf16[1024,3,8192]{0,2,1} transpose(reduce-scatter.13), dimensions={2,0,1}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/transpose[permutation=(2, 0, 1)]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=652}
  reshape.1156 = bf16[1,1024,3,8192]{3,2,1,0} reshape(transpose.51), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/broadcast_in_dim[shape=(1, 8192, 3, 8192) broadcast_dimensions=(1, 2, 3)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-update-slice.43 = bf16[6,1024,3,8192]{3,2,1,0} dynamic-update-slice(get-tuple-element.322, reshape.1156, select.405, constant.1199, constant.1199, /*index=5*/constant.1199), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_update_slice" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.324 = bf16[6,8192]{1,0} get-tuple-element(while.4), index=6
  get-tuple-element.325 = bf16[8192]{0} get-tuple-element(custom-call.68), index=2, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/xmap(transpose(<lambda>))/te_layernorm_backward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  all-reduce.44 = bf16[8192]{0} all-reduce(get-tuple-element.325), channel_id=89, replica_groups={{0,1,2,3,4,5,6,7}}, use_global_device_ids=true, to_apply=region_16.868
  reshape.1157 = bf16[1,8192]{1,0} reshape(all-reduce.44), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/broadcast_in_dim[shape=(1, 8192) broadcast_dimensions=(1,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-update-slice.44 = bf16[6,8192]{1,0} dynamic-update-slice(get-tuple-element.324, reshape.1157, select.405, constant.1199), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_update_slice" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.326 = bf16[6,8192]{1,0} get-tuple-element(while.4), index=7
  get-tuple-element.327 = bf16[8192]{0} get-tuple-element(custom-call.68), index=1, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/qkv/xmap(transpose(<lambda>))/te_layernorm_backward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  all-reduce.45 = bf16[8192]{0} all-reduce(get-tuple-element.327), channel_id=90, replica_groups={{0,1,2,3,4,5,6,7}}, use_global_device_ids=true, to_apply=region_16.868
  reshape.1158 = bf16[1,8192]{1,0} reshape(all-reduce.45), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/broadcast_in_dim[shape=(1, 8192) broadcast_dimensions=(1,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-update-slice.45 = bf16[6,8192]{1,0} dynamic-update-slice(get-tuple-element.326, reshape.1158, select.405, constant.1199), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_update_slice" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.328 = bf16[6,8192]{1,0} get-tuple-element(while.4), index=8
  get-tuple-element.329 = bf16[8192]{0} get-tuple-element(custom-call.64), index=2, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/xmap(transpose(<lambda>))/te_layernorm_backward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  all-reduce.46 = bf16[8192]{0} all-reduce(get-tuple-element.329), channel_id=91, replica_groups={{0,1,2,3,4,5,6,7}}, use_global_device_ids=true, to_apply=region_16.868
  reshape.1159 = bf16[1,8192]{1,0} reshape(all-reduce.46), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/broadcast_in_dim[shape=(1, 8192) broadcast_dimensions=(1,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-update-slice.46 = bf16[6,8192]{1,0} dynamic-update-slice(get-tuple-element.328, reshape.1159, select.405, constant.1199), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_update_slice" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.330 = bf16[6,8192]{1,0} get-tuple-element(while.4), index=9
  get-tuple-element.331 = bf16[8192]{0} get-tuple-element(custom-call.64), index=1, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/xmap(transpose(<lambda>))/te_layernorm_backward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  all-reduce.47 = bf16[8192]{0} all-reduce(get-tuple-element.331), channel_id=92, replica_groups={{0,1,2,3,4,5,6,7}}, use_global_device_ids=true, to_apply=region_16.868
  reshape.1160 = bf16[1,8192]{1,0} reshape(all-reduce.47), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/broadcast_in_dim[shape=(1, 8192) broadcast_dimensions=(1,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-update-slice.47 = bf16[6,8192]{1,0} dynamic-update-slice(get-tuple-element.330, reshape.1160, select.405, constant.1199), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_update_slice" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.332 = bf16[6,1,32768]{2,1,0} get-tuple-element(while.4), index=10
  reduce.76 = bf16[1,32768]{1,0} reduce(add.435, constant.1210), dimensions={0,1}, to_apply=region_16.868, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/reduce_sum[axes=(0, 1)]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=972}
  all-reduce.48 = bf16[1,32768]{1,0} all-reduce(reduce.76), channel_id=93, replica_groups={{0}}, to_apply=region_16.868, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/reduce_sum[axes=(0, 1)]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=972}
  reshape.1161 = bf16[1,1,32768]{2,1,0} reshape(all-reduce.48), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/broadcast_in_dim[shape=(1, 1, 32768) broadcast_dimensions=(1, 2)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-update-slice.48 = bf16[6,1,32768]{2,1,0} dynamic-update-slice(get-tuple-element.332, reshape.1161, select.405, constant.1199, constant.1199), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_update_slice" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.333 = bf16[6,1024,1,32768]{3,2,1,0} get-tuple-element(while.4), index=11
  reshape.1162 = bf16[1,2048,32768]{2,1,0} reshape(add.435)
  transpose.52 = bf16[32768,1,2048]{0,2,1} transpose(reshape.1162), dimensions={2,0,1}
  reshape.1163 = bf16[32768,2048]{1,0} reshape(transpose.52)
  get-tuple-element.334 = bf16[1,2048,8192]{2,1,0} get-tuple-element(custom-call.63), index=0, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/xmap(<lambda>)/te_layernorm_forward[zero_centered_gamma=True epsilon=1e-05]" source_file="/opt/transformer-engine/transformer_engine/jax/sharding.py" source_line=1179}
  reshape.1165 = bf16[2048,8192]{1,0} reshape(get-tuple-element.334), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/reshape[new_sizes=(8, 2048, 8192) dimensions=None]" source_file="/opt/transformer-engine/transformer_engine/jax/layernorm.py" source_line=94}
  dot.75 = bf16[32768,8192]{1,0} dot(reshape.1163, reshape.1165), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/dot_general[dimension_numbers=(((0, 1), (0, 1)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=962}
  reduce-scatter.14 = bf16[32768,1024]{1,0} reduce-scatter(dot.75), channel_id=94, replica_groups={{0}}, dimensions={1}, to_apply=region_16.868
  reshape.1166 = bf16[1,32768,1024]{2,1,0} reshape(reduce-scatter.14), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/dot_general[dimension_numbers=(((0, 1), (0, 1)), ((), ())) precision=None preferred_element_type=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=962}
  transpose.53 = bf16[1024,1,32768]{0,2,1} transpose(reshape.1166), dimensions={2,0,1}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/transpose[permutation=(2, 0, 1)]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=962}
  reshape.1167 = bf16[1,1024,1,32768]{3,2,1,0} reshape(transpose.53), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/broadcast_in_dim[shape=(1, 8192, 1, 32768) broadcast_dimensions=(1, 2, 3)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-update-slice.49 = bf16[6,1024,1,32768]{3,2,1,0} dynamic-update-slice(get-tuple-element.333, reshape.1167, select.405, constant.1199, constant.1199, /*index=5*/constant.1199), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_update_slice" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.335 = bf16[6,8192]{1,0} get-tuple-element(while.4), index=12
  reduce.77 = bf16[8192]{0} reduce(get-tuple-element.288, constant.1210), dimensions={0,1}, to_apply=region_16.868, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/reduce_sum[axes=(0, 1)]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=1030}
  all-reduce.49 = bf16[8192]{0} all-reduce(reduce.77), channel_id=95, replica_groups={{0}}, to_apply=region_16.868, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/reduce_sum[axes=(0, 1)]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=1030}
  reshape.1168 = bf16[1,8192]{1,0} reshape(all-reduce.49), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/broadcast_in_dim[shape=(1, 8192) broadcast_dimensions=(1,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-update-slice.50 = bf16[6,8192]{1,0} dynamic-update-slice(get-tuple-element.335, reshape.1168, select.405, constant.1199), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_update_slice" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.336 = bf16[6,32768,1024]{2,1,0} get-tuple-element(while.4), index=13
  multiply.320 = bf16[1,2048,1,32768]{3,2,1,0} multiply(reshape.1110, multiply.318), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/mul" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=982}
  reshape.1169 = bf16[1,2048,32768]{2,1,0} reshape(multiply.320), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/reshape[new_sizes=(8, 2048, 32768) dimensions=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=985}
  transpose.54 = bf16[32768,1,2048]{0,2,1} transpose(reshape.1169), dimensions={2,0,1}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/reshape[new_sizes=(8, 2048, 32768) dimensions=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=985}
  reshape.1170 = bf16[32768,2048]{1,0} reshape(transpose.54), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/reshape[new_sizes=(8, 2048, 32768) dimensions=None]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=985}
  dot.76 = bf16[32768,8192]{1,0} dot(reshape.1170, reshape.1112), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/mlp/transpose[permutation=(1, 0)]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=1022}
  reduce-scatter.15 = bf16[32768,1024]{1,0} reduce-scatter(dot.76), channel_id=96, replica_groups={{0}}, dimensions={1}, to_apply=region_16.868
  reshape.1172 = bf16[1,32768,1024]{2,1,0} reshape(reduce-scatter.15), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/broadcast_in_dim[shape=(1, 32768, 8192) broadcast_dimensions=(1, 2)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-update-slice.51 = bf16[6,32768,1024]{2,1,0} dynamic-update-slice(get-tuple-element.336, reshape.1172, select.405, constant.1199, constant.1199), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_update_slice" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  get-tuple-element.309 = bf16[6,8192,1024]{2,1,0} get-tuple-element(while.4), index=18
  get-tuple-element.312 = bf16[6,1024,3,8192]{3,2,1,0} get-tuple-element(while.4), index=20
  get-tuple-element.293 = bf16[6,1024,1,32768]{3,2,1,0} get-tuple-element(while.4), index=26
  get-tuple-element.291 = bf16[6,32768,1024]{2,1,0} get-tuple-element(while.4), index=27
  tuple.14 = (s32[], bf16[1,2048,8192]{2,1,0}, bf16[6,8192]{1,0}, bf16[6,8192,1024]{2,1,0}, bf16[6,3,8192]{2,1,0}, /*index=5*/bf16[6,1024,3,8192]{3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, /*index=10*/bf16[6,1,32768]{2,1,0}, bf16[6,1024,1,32768]{3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,32768,1024]{2,1,0}, bf16[6,1,2048,3,8192]{4,3,2,1,0}, /*index=15*/bf16[6,1,2048,8192]{3,2,1,0}, bf16[6,1,2048,1,32768]{4,3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,8192,1024]{2,1,0}, bf16[6,3,8192]{2,1,0}, /*index=20*/bf16[6,1024,3,8192]{3,2,1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, bf16[6,8192]{1,0}, /*index=25*/bf16[6,1,32768]{2,1,0}, bf16[6,1024,1,32768]{3,2,1,0}, bf16[6,32768,1024]{2,1,0}, bf16[6,1,2048,8192]{3,2,1,0}, bf16[1,1,2048,2048]{3,2,1,0}) tuple(add.428, add.442, dynamic-update-slice.40, dynamic-update-slice.41, dynamic-update-slice.42, /*index=5*/dynamic-update-slice.43, dynamic-update-slice.44, dynamic-update-slice.45, dynamic-update-slice.46, dynamic-update-slice.47, /*index=10*/dynamic-update-slice.48, dynamic-update-slice.49, dynamic-update-slice.50, dynamic-update-slice.51, get-tuple-element.303, /*index=15*/get-tuple-element.295, get-tuple-element.289, get-tuple-element.296, get-tuple-element.309, get-tuple-element.304, /*index=20*/get-tuple-element.312, get-tuple-element.315, get-tuple-element.314, get-tuple-element.299, get-tuple-element.298, /*index=25*/get-tuple-element.290, get-tuple-element.293, get-tuple-element.291, get-tuple-element.297, get-tuple-element.305)
  get-tuple-element.147 = bf16[1,2048,8192]{2,1,0} get-tuple-element(tuple.14), index=1, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while[cond_nconsts=0 body_nconsts=16]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.865 = bf16[2048,8192]{1,0} reshape(get-tuple-element.147), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/lm._prepare_input/position_emb/sharding_constraint[sharding=GSPMDSharding({devices=[8,1,1]<=[8]}) resource_env=ResourceEnv(Mesh(device_ids=array([[[0],\n        [1],\n        [2],\n        [3],\n        [4],\n        [5],\n        [6],\n        [7]]]), axis_names=(\'replica\', \'data\', \'mdl\')), ()) unconstrained_dims=set()]" source_file="/opt/praxis/praxis/py_utils.py" source_line=479}
  dot.46 = bf16[2048,8192]{1,0} dot(reshape.864, reshape.865), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/lm._prepare_input/position_emb/einsum/...y,yz->...z/transpose[permutation=(1, 0)]" source_file="/opt/praxis/praxis/layers/base_ops.py" source_line=42}
  reduce-scatter.4 = bf16[256,8192]{1,0} reduce-scatter(dot.46), channel_id=53, replica_groups={{0}}, dimensions={0}, to_apply=region_16.868
  convert.74 = f32[256,8192]{0,1} convert(reduce-scatter.4), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/paxml/paxml/trainer_lib.py" source_line=472}
  multiply.84 = f32[256,8192]{0,1} multiply(convert.74, convert.74), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  reduce.49 = f32[] reduce(multiply.84, constant.789), dimensions={0,1}, to_apply=region_10.798, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/reduce_sum[axes=(0, 1)]" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  all-reduce.20 = f32[] all-reduce(reduce.49), channel_id=32, replica_groups={{0}}, to_apply=region_10.798, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/reduce_sum[axes=(0, 1)]" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  add.234 = f32[] add(add.230, all-reduce.20), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/reduce_sum[axes=(0,)]" source_file="/opt/paxml/paxml/learners.py" source_line=51}
  transpose.37 = bf16[8192,1,2048]{0,2,1} transpose(add.228), dimensions={2,0,1}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/add" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=381}
  reshape.867 = bf16[8192,2048]{1,0} reshape(transpose.37), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/final_ln/add" source_file="/opt/praxis/praxis/layers/normalizations.py" source_line=381}
  dot.47 = bf16[8192,50304]{1,0} dot(reshape.867, reshape.861), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/softmax.get_logits/logits_ffn/linear/einsum/...y,yz->...z/transpose[permutation=(1, 0)]" source_file="/opt/praxis/praxis/layers/base_ops.py" source_line=42}
  transpose.39 = bf16[8192,1,2048]{0,2,1} transpose(get-tuple-element.147), dimensions={2,0,1}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/lm._prepare_input/position_emb/sharding_constraint[sharding=GSPMDSharding({devices=[8,1,1]<=[8]}) resource_env=ResourceEnv(Mesh(device_ids=array([[[0],\n        [1],\n        [2],\n        [3],\n        [4],\n        [5],\n        [6],\n        [7]]]), axis_names=(\'replica\', \'data\', \'mdl\')), ()) unconstrained_dims=set()]" source_file="/opt/praxis/praxis/py_utils.py" source_line=479}
  reshape.870 = bf16[8192,2048]{1,0} reshape(transpose.39), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/lm._prepare_input/position_emb/sharding_constraint[sharding=GSPMDSharding({devices=[8,1,1]<=[8]}) resource_env=ResourceEnv(Mesh(device_ids=array([[[0],\n        [1],\n        [2],\n        [3],\n        [4],\n        [5],\n        [6],\n        [7]]]), axis_names=(\'replica\', \'data\', \'mdl\')), ()) unconstrained_dims=set()]" source_file="/opt/praxis/praxis/py_utils.py" source_line=479}
  dot.48 = bf16[8192,50304]{1,0} dot(reshape.870, convert.170), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/lm._prepare_input/softmax.emb_lookup/einsum/...y,yz->...z/dot_general[dimension_numbers=(((0, 1), (0, 1)), ((), ())) precision=None preferred_element_type=bfloat16]" source_file="/opt/praxis/praxis/layers/base_ops.py" source_line=42}
  add.385 = bf16[8192,50304]{1,0} add(dot.47, dot.48), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/lm._prepare_input/softmax.emb_lookup/add_any" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=399}
  reduce-scatter.7 = bf16[1024,50304]{0,1} reduce-scatter(add.385), channel_id=56, replica_groups={{0}}, dimensions={0}, to_apply=region_16.868
  convert.75 = f32[1024,50304]{0,1} convert(reduce-scatter.7), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/paxml/paxml/trainer_lib.py" source_line=472}
  multiply.85 = f32[1024,50304]{0,1} multiply(convert.75, convert.75), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  reduce.50 = f32[] reduce(multiply.85, constant.789), dimensions={0,1}, to_apply=region_10.798, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/reduce_sum[axes=(0, 1)]" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  all-reduce.23 = f32[] all-reduce(reduce.50), channel_id=35, replica_groups={{0}}, to_apply=region_10.798, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/reduce_sum[axes=(0, 1)]" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  add.236 = f32[] add(add.234, all-reduce.23), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/reduce_sum[axes=(0,)]" source_file="/opt/paxml/paxml/learners.py" source_line=51}
  get-tuple-element.148 = bf16[6,8192]{1,0} get-tuple-element(tuple.14), index=2, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while[cond_nconsts=0 body_nconsts=16]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  multiply.86 = bf16[6,8192]{1,0} multiply(get-tuple-element.148, get-tuple-element.148), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  convert.76 = f32[6,8192]{1,0} convert(multiply.86), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  reduce.51 = f32[] reduce(convert.76, constant.789), dimensions={0,1}, to_apply=region_10.798, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/reduce_sum[axes=(0, 1)]" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  convert.77 = bf16[] convert(reduce.51), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=bfloat16 weak_type=False]" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  convert.78 = f32[] convert(convert.77), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/paxml/paxml/learners.py" source_line=51}
  add.237 = f32[] add(add.236, convert.78), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/reduce_sum[axes=(0,)]" source_file="/opt/paxml/paxml/learners.py" source_line=51}
  get-tuple-element.149 = bf16[6,8192,1024]{2,1,0} get-tuple-element(tuple.14), index=3, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while[cond_nconsts=0 body_nconsts=16]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  convert.79 = f32[6,8192,1024]{2,1,0} convert(get-tuple-element.149), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/paxml/paxml/trainer_lib.py" source_line=472}
  multiply.87 = f32[6,8192,1024]{2,1,0} multiply(convert.79, convert.79), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  reduce.52 = f32[] reduce(multiply.87, constant.789), dimensions={0,1,2}, to_apply=region_10.798, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/reduce_sum[axes=(0, 1, 2)]" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  all-reduce.24 = f32[] all-reduce(reduce.52), channel_id=36, replica_groups={{0}}, to_apply=region_10.798, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/reduce_sum[axes=(0, 1, 2)]" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  add.238 = f32[] add(add.237, all-reduce.24), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/reduce_sum[axes=(0,)]" source_file="/opt/paxml/paxml/learners.py" source_line=51}
  get-tuple-element.150 = bf16[6,3,8192]{2,1,0} get-tuple-element(tuple.14), index=4, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while[cond_nconsts=0 body_nconsts=16]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  multiply.88 = bf16[6,3,8192]{2,1,0} multiply(get-tuple-element.150, get-tuple-element.150), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  convert.80 = f32[6,3,8192]{2,1,0} convert(multiply.88), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  reduce.53 = f32[] reduce(convert.80, constant.789), dimensions={0,1,2}, to_apply=region_10.798, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/reduce_sum[axes=(0, 1, 2)]" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  convert.81 = bf16[] convert(reduce.53), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=bfloat16 weak_type=False]" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  convert.82 = f32[] convert(convert.81), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/paxml/paxml/learners.py" source_line=51}
  add.239 = f32[] add(add.238, convert.82), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/reduce_sum[axes=(0,)]" source_file="/opt/paxml/paxml/learners.py" source_line=51}
  get-tuple-element.151 = bf16[6,1024,3,8192]{3,2,1,0} get-tuple-element(tuple.14), index=5, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while[cond_nconsts=0 body_nconsts=16]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  convert.83 = f32[6,1024,3,8192]{3,2,1,0} convert(get-tuple-element.151), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/paxml/paxml/trainer_lib.py" source_line=472}
  multiply.89 = f32[6,1024,3,8192]{3,2,1,0} multiply(convert.83, convert.83), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  reduce.54 = f32[] reduce(multiply.89, constant.789), dimensions={0,1,2,3}, to_apply=region_10.798, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/reduce_sum[axes=(0, 1, 2, 3)]" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  all-reduce.25 = f32[] all-reduce(reduce.54), channel_id=37, replica_groups={{0}}, to_apply=region_10.798, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/reduce_sum[axes=(0, 1, 2, 3)]" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  add.240 = f32[] add(add.239, all-reduce.25), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/reduce_sum[axes=(0,)]" source_file="/opt/paxml/paxml/learners.py" source_line=51}
  get-tuple-element.152 = bf16[6,8192]{1,0} get-tuple-element(tuple.14), index=6, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while[cond_nconsts=0 body_nconsts=16]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  convert.84 = f32[6,8192]{1,0} convert(get-tuple-element.152), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/paxml/paxml/trainer_lib.py" source_line=472}
  multiply.90 = f32[6,8192]{1,0} multiply(convert.84, convert.84), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  reduce.55 = f32[] reduce(multiply.90, constant.789), dimensions={0,1}, to_apply=region_10.798, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/reduce_sum[axes=(0, 1)]" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  add.241 = f32[] add(add.240, reduce.55), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/reduce_sum[axes=(0,)]" source_file="/opt/paxml/paxml/learners.py" source_line=51}
  get-tuple-element.153 = bf16[6,8192]{1,0} get-tuple-element(tuple.14), index=7, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while[cond_nconsts=0 body_nconsts=16]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  convert.85 = f32[6,8192]{1,0} convert(get-tuple-element.153), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/paxml/paxml/trainer_lib.py" source_line=472}
  multiply.91 = f32[6,8192]{1,0} multiply(convert.85, convert.85), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  reduce.56 = f32[] reduce(multiply.91, constant.789), dimensions={0,1}, to_apply=region_10.798, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/reduce_sum[axes=(0, 1)]" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  add.242 = f32[] add(add.241, reduce.56), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/reduce_sum[axes=(0,)]" source_file="/opt/paxml/paxml/learners.py" source_line=51}
  get-tuple-element.154 = bf16[6,8192]{1,0} get-tuple-element(tuple.14), index=8, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while[cond_nconsts=0 body_nconsts=16]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  convert.86 = f32[6,8192]{1,0} convert(get-tuple-element.154), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/paxml/paxml/trainer_lib.py" source_line=472}
  multiply.92 = f32[6,8192]{1,0} multiply(convert.86, convert.86), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  reduce.57 = f32[] reduce(multiply.92, constant.789), dimensions={0,1}, to_apply=region_10.798, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/reduce_sum[axes=(0, 1)]" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  add.243 = f32[] add(add.242, reduce.57), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/reduce_sum[axes=(0,)]" source_file="/opt/paxml/paxml/learners.py" source_line=51}
  get-tuple-element.155 = bf16[6,8192]{1,0} get-tuple-element(tuple.14), index=9, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while[cond_nconsts=0 body_nconsts=16]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  convert.87 = f32[6,8192]{1,0} convert(get-tuple-element.155), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/paxml/paxml/trainer_lib.py" source_line=472}
  multiply.93 = f32[6,8192]{1,0} multiply(convert.87, convert.87), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  reduce.58 = f32[] reduce(multiply.93, constant.789), dimensions={0,1}, to_apply=region_10.798, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/reduce_sum[axes=(0, 1)]" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  add.244 = f32[] add(add.243, reduce.58), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/reduce_sum[axes=(0,)]" source_file="/opt/paxml/paxml/learners.py" source_line=51}
  get-tuple-element.156 = bf16[6,1,32768]{2,1,0} get-tuple-element(tuple.14), index=10, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while[cond_nconsts=0 body_nconsts=16]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  multiply.94 = bf16[6,1,32768]{2,1,0} multiply(get-tuple-element.156, get-tuple-element.156), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  convert.88 = f32[6,1,32768]{2,1,0} convert(multiply.94), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  reduce.59 = f32[] reduce(convert.88, constant.789), dimensions={0,1,2}, to_apply=region_10.798, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/reduce_sum[axes=(0, 1, 2)]" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  convert.89 = bf16[] convert(reduce.59), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=bfloat16 weak_type=False]" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  convert.90 = f32[] convert(convert.89), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/paxml/paxml/learners.py" source_line=51}
  add.245 = f32[] add(add.244, convert.90), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/reduce_sum[axes=(0,)]" source_file="/opt/paxml/paxml/learners.py" source_line=51}
  get-tuple-element.157 = bf16[6,1024,1,32768]{3,2,1,0} get-tuple-element(tuple.14), index=11, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while[cond_nconsts=0 body_nconsts=16]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  convert.91 = f32[6,1024,1,32768]{3,2,1,0} convert(get-tuple-element.157), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/paxml/paxml/trainer_lib.py" source_line=472}
  multiply.95 = f32[6,1024,1,32768]{3,2,1,0} multiply(convert.91, convert.91), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  reduce.60 = f32[] reduce(multiply.95, constant.789), dimensions={0,1,2,3}, to_apply=region_10.798, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/reduce_sum[axes=(0, 1, 2, 3)]" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  all-reduce.26 = f32[] all-reduce(reduce.60), channel_id=38, replica_groups={{0}}, to_apply=region_10.798, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/reduce_sum[axes=(0, 1, 2, 3)]" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  add.246 = f32[] add(add.245, all-reduce.26), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/reduce_sum[axes=(0,)]" source_file="/opt/paxml/paxml/learners.py" source_line=51}
  get-tuple-element.158 = bf16[6,8192]{1,0} get-tuple-element(tuple.14), index=12, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while[cond_nconsts=0 body_nconsts=16]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  multiply.96 = bf16[6,8192]{1,0} multiply(get-tuple-element.158, get-tuple-element.158), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  convert.92 = f32[6,8192]{1,0} convert(multiply.96), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  reduce.61 = f32[] reduce(convert.92, constant.789), dimensions={0,1}, to_apply=region_10.798, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/reduce_sum[axes=(0, 1)]" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  convert.93 = bf16[] convert(reduce.61), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=bfloat16 weak_type=False]" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  convert.94 = f32[] convert(convert.93), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/paxml/paxml/learners.py" source_line=51}
  add.247 = f32[] add(add.246, convert.94), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/reduce_sum[axes=(0,)]" source_file="/opt/paxml/paxml/learners.py" source_line=51}
  get-tuple-element.159 = bf16[6,32768,1024]{2,1,0} get-tuple-element(tuple.14), index=13, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while[cond_nconsts=0 body_nconsts=16]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  convert.95 = f32[6,32768,1024]{2,1,0} convert(get-tuple-element.159), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/paxml/paxml/trainer_lib.py" source_line=472}
  multiply.97 = f32[6,32768,1024]{2,1,0} multiply(convert.95, convert.95), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  reduce.62 = f32[] reduce(multiply.97, constant.789), dimensions={0,1,2}, to_apply=region_10.798, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/reduce_sum[axes=(0, 1, 2)]" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  all-reduce.27 = f32[] all-reduce(reduce.62), channel_id=39, replica_groups={{0}}, to_apply=region_10.798, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/reduce_sum[axes=(0, 1, 2)]" source_file="/opt/paxml/paxml/learners.py" source_line=49}
  add.248 = f32[] add(add.247, all-reduce.27), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/reduce_sum[axes=(0,)]" source_file="/opt/paxml/paxml/learners.py" source_line=51}
  sqrt.0 = f32[] sqrt(add.248), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sqrt" source_file="/opt/paxml/paxml/learners.py" source_line=51}
  is-finite.1 = pred[] is-finite(sqrt.0), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(isfinite)/is_finite" source_file="/opt/paxml/paxml/learners.py" source_line=270}
  broadcast.602 = pred[8192]{0} broadcast(is-finite.1), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/broadcast_in_dim[shape=(8192,) broadcast_dimensions=()]" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  param.29 = s32[] parameter(19), sharding={replicated}
  constant.865 = s32[] constant(116)
  compare.142 = pred[] compare(param.29, constant.865), direction=LT, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/lt" source_file="/usr/local/lib/python3.10/dist-packages/optax/_src/schedule.py" source_line=399}
  constant.866 = s32[] constant(115)
  compare.143 = pred[] compare(param.29, constant.866), direction=LT, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/lt" source_file="/usr/local/lib/python3.10/dist-packages/optax/_src/schedule.py" source_line=399}
  convert.96 = f32[] convert(param.29), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/praxis/praxis/schedules.py" source_line=96}
  compare.144 = pred[] compare(convert.96, constant.789), direction=LT, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/lt" source_file="/opt/praxis/praxis/schedules.py" source_line=105}
  constant.869 = f32[] constant(115)
  compare.145 = pred[] compare(convert.96, constant.869), direction=GE, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/ge" source_file="/opt/praxis/praxis/schedules.py" source_line=105}
  constant.871 = f32[] constant(0.00869565178)
  multiply.98 = f32[] multiply(convert.96, constant.871), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/div" source_file="/opt/praxis/praxis/schedules.py" source_line=99}
  select.263 = f32[] select(compare.145, constant.788, multiply.98), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/praxis/praxis/schedules.py" source_line=105}
  select.264 = f32[] select(compare.144, constant.789, select.263), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/praxis/praxis/schedules.py" source_line=105}
  constant.872 = s32[] constant(-115)
  add.249 = s32[] add(param.29, constant.872), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub" source_file="/usr/local/lib/python3.10/dist-packages/optax/_src/schedule.py" source_line=399}
  convert.97 = f32[] convert(add.249), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/praxis/praxis/schedules.py" source_line=96}
  compare.146 = pred[] compare(convert.97, constant.789), direction=LT, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/lt" source_file="/opt/praxis/praxis/schedules.py" source_line=105}
  compare.147 = pred[] compare(convert.97, constant.788), direction=GE, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/ge" source_file="/opt/praxis/praxis/schedules.py" source_line=105}
  multiply.99 = f32[] multiply(convert.97, constant.789), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/schedules.py" source_line=104}
  add.250 = f32[] add(multiply.99, constant.788), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/schedules.py" source_line=104}
  select.265 = f32[] select(compare.147, constant.788, add.250), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/praxis/praxis/schedules.py" source_line=105}
  select.266 = f32[] select(compare.146, constant.788, select.265), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/praxis/praxis/schedules.py" source_line=105}
  select.267 = f32[] select(compare.143, select.264, select.266), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/usr/local/lib/python3.10/dist-packages/optax/_src/schedule.py" source_line=399}
  constant.879 = s32[] constant(-116)
  add.251 = s32[] add(param.29, constant.879), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub" source_file="/usr/local/lib/python3.10/dist-packages/optax/_src/schedule.py" source_line=399}
  convert.98 = f32[] convert(add.251), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/praxis/praxis/schedules.py" source_line=96}
  compare.148 = pred[] compare(convert.98, constant.789), direction=LT, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/lt" source_file="/opt/praxis/praxis/schedules.py" source_line=105}
  constant.882 = f32[] constant(62384)
  compare.149 = pred[] compare(convert.98, constant.882), direction=GE, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/ge" source_file="/opt/praxis/praxis/schedules.py" source_line=105}
  constant.883 = f32[] constant(3.14159274)
  constant.884 = f32[] constant(5.03589508e-05)
  multiply.100 = f32[] multiply(convert.98, constant.884), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/schedules.py" source_line=104}
  select.268 = f32[] select(compare.149, constant.883, multiply.100), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/praxis/praxis/schedules.py" source_line=105}
  select.269 = f32[] select(compare.148, constant.789, select.268), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/praxis/praxis/schedules.py" source_line=105}
  cosine.0 = f32[] cosine(select.269), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/cos" source_file="/opt/praxis/praxis/schedules.py" source_line=204}
  add.252 = f32[] add(cosine.0, constant.788), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/schedules.py" source_line=204}
  constant.886 = f32[] constant(0.45)
  multiply.101 = f32[] multiply(add.252, constant.886), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/schedules.py" source_line=203}
  constant.887 = f32[] constant(0.1)
  add.253 = f32[] add(multiply.101, constant.887), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/schedules.py" source_line=203}
  select.270 = f32[] select(compare.142, select.267, add.253), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/usr/local/lib/python3.10/dist-packages/optax/_src/schedule.py" source_line=399}
  constant.888 = f32[] constant(-0.00016)
  multiply.102 = f32[] multiply(select.270, constant.888), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=710}
  broadcast.603 = f32[8192]{0} broadcast(multiply.102), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=712}
  constant.891 = f32[] constant(0.9)
  power.0 = f32[] power(constant.891, convert.96), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/pow" source_file="/opt/praxis/praxis/optimizers.py" source_line=336}
  subtract.65 = f32[] subtract(constant.788, power.0), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub" source_file="/opt/praxis/praxis/optimizers.py" source_line=336}
  multiply.103 = f32[] multiply(subtract.65, constant.891), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=336}
  add.254 = f32[] add(convert.96, constant.788), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=335}
  power.1 = f32[] power(constant.891, add.254), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/pow" source_file="/opt/praxis/praxis/optimizers.py" source_line=336}
  subtract.66 = f32[] subtract(constant.788, power.1), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub" source_file="/opt/praxis/praxis/optimizers.py" source_line=336}
  divide.5 = f32[] divide(multiply.103, subtract.66), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/div" source_file="/opt/praxis/praxis/optimizers.py" source_line=336}
  subtract.67 = f32[] subtract(constant.788, divide.5), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  broadcast.604 = f32[8192]{0} broadcast(subtract.67), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  divide.6 = f32[] divide(constant.788, sqrt.0), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/div" source_file="/opt/paxml/paxml/learners.py" source_line=278}
  minimum.4 = f32[] minimum(divide.6, constant.788), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/min" source_file="/opt/paxml/paxml/learners.py" source_line=276}
  broadcast.605 = f32[8192]{0} broadcast(minimum.4), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=280}
  multiply.104 = f32[8192]{0} multiply(convert.68, broadcast.605), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=280}
  compare.150 = pred[8192]{0} compare(multiply.104, multiply.104), direction=NE, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(nan_to_num)/jit(isnan)/ne" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  constant.898 = f32[] constant(nan)
  broadcast.606 = f32[8192]{0} broadcast(constant.898), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(nan_to_num)/jit(_where)/broadcast_in_dim[shape=(8192,) broadcast_dimensions=()]" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.271 = f32[8192]{0} select(compare.150, broadcast.606, multiply.104), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(nan_to_num)/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  constant.899 = f32[] constant(inf)
  broadcast.607 = f32[8192]{0} broadcast(constant.899), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(nan_to_num)/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  compare.151 = pred[8192]{0} compare(select.271, broadcast.607), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(nan_to_num)/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.272 = f32[8192]{0} select(compare.151, broadcast.606, select.271), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(nan_to_num)/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  broadcast.609 = f32[8192]{0} broadcast(constant.816), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(nan_to_num)/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  compare.152 = pred[8192]{0} compare(select.272, broadcast.609), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(nan_to_num)/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.273 = f32[8192]{0} select(compare.152, broadcast.606, select.272), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(nan_to_num)/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  multiply.105 = f32[8192]{0} multiply(broadcast.604, select.273), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  broadcast.611 = f32[8192]{0} broadcast(divide.5), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  param.30 = f32[8192]{0} parameter(20), sharding={replicated}
  multiply.106 = f32[8192]{0} multiply(broadcast.611, param.30), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  add.255 = f32[8192]{0} add(multiply.105, multiply.106), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  constant.905 = f32[] constant(0.95)
  power.2 = f32[] power(constant.905, convert.96), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/pow" source_file="/opt/praxis/praxis/optimizers.py" source_line=336}
  subtract.68 = f32[] subtract(constant.788, power.2), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub" source_file="/opt/praxis/praxis/optimizers.py" source_line=336}
  multiply.107 = f32[] multiply(subtract.68, constant.905), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=336}
  power.3 = f32[] power(constant.905, add.254), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/pow" source_file="/opt/praxis/praxis/optimizers.py" source_line=336}
  subtract.69 = f32[] subtract(constant.788, power.3), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub" source_file="/opt/praxis/praxis/optimizers.py" source_line=336}
  divide.7 = f32[] divide(multiply.107, subtract.69), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/div" source_file="/opt/praxis/praxis/optimizers.py" source_line=336}
  subtract.70 = f32[] subtract(constant.788, divide.7), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  broadcast.612 = f32[8192]{0} broadcast(subtract.70), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  multiply.108 = f32[8192]{0} multiply(select.273, select.273), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  multiply.109 = f32[8192]{0} multiply(broadcast.612, multiply.108), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  broadcast.613 = f32[8192]{0} broadcast(divide.7), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  param.31 = f32[8192]{0} parameter(24), sharding={replicated}
  multiply.110 = f32[8192]{0} multiply(broadcast.613, param.31), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  add.257 = f32[8192]{0} add(multiply.109, multiply.110), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  sqrt.1 = f32[8192]{0} sqrt(add.257), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sqrt" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  constant.910 = f32[] constant(1e-08)
  broadcast.614 = f32[8192]{0} broadcast(constant.910), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  add.258 = f32[8192]{0} add(sqrt.1, broadcast.614), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  divide.8 = f32[8192]{0} divide(add.255, add.258), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/div" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  broadcast.615 = f32[8192]{0} broadcast(constant.887), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  multiply.111 = f32[8192]{0} multiply(param.8, broadcast.615), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  add.259 = f32[8192]{0} add(divide.8, multiply.111), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  multiply.112 = f32[8192]{0} multiply(broadcast.603, add.259), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=712}
  broadcast.616 = f32[8192]{0} broadcast(constant.789), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/broadcast_in_dim[shape=(8192,) broadcast_dimensions=()]" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  select.274 = f32[8192]{0} select(broadcast.602, multiply.112, broadcast.616), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  add.260 = f32[8192]{0} add(param.8, select.274), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/paxml/paxml/learners.py" source_line=428}
  multiply.114 = f32[8192]{0} multiply(convert.69, broadcast.605), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=280}
  compare.153 = pred[8192]{0} compare(multiply.114, multiply.114), direction=NE, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(nan_to_num)/jit(isnan)/ne" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.275 = f32[8192]{0} select(compare.153, broadcast.606, multiply.114), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(nan_to_num)/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  compare.154 = pred[8192]{0} compare(select.275, broadcast.607), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(nan_to_num)/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.276 = f32[8192]{0} select(compare.154, broadcast.606, select.275), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(nan_to_num)/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  compare.155 = pred[8192]{0} compare(select.276, broadcast.609), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(nan_to_num)/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.277 = f32[8192]{0} select(compare.155, broadcast.606, select.276), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(nan_to_num)/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  multiply.115 = f32[8192]{0} multiply(broadcast.604, select.277), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  param.32 = f32[8192]{0} parameter(21), sharding={replicated}
  multiply.116 = f32[8192]{0} multiply(broadcast.611, param.32), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  add.262 = f32[8192]{0} add(multiply.115, multiply.116), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  multiply.118 = f32[8192]{0} multiply(select.277, select.277), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  multiply.119 = f32[8192]{0} multiply(broadcast.612, multiply.118), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  param.33 = f32[8192]{0} parameter(25), sharding={replicated}
  multiply.120 = f32[8192]{0} multiply(broadcast.613, param.33), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  add.264 = f32[8192]{0} add(multiply.119, multiply.120), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  sqrt.2 = f32[8192]{0} sqrt(add.264), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sqrt" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  add.265 = f32[8192]{0} add(sqrt.2, broadcast.614), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  divide.11 = f32[8192]{0} divide(add.262, add.265), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/div" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  multiply.121 = f32[8192]{0} multiply(param.28, broadcast.615), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  add.266 = f32[8192]{0} add(divide.11, multiply.121), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  multiply.122 = f32[8192]{0} multiply(broadcast.603, add.266), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=712}
  select.278 = f32[8192]{0} select(broadcast.602, multiply.122, broadcast.616), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  add.267 = f32[8192]{0} add(param.28, select.278), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/paxml/paxml/learners.py" source_line=428}
  broadcast.629 = pred[256,8192]{1,0} broadcast(is-finite.1), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/broadcast_in_dim[shape=(2048, 8192) broadcast_dimensions=()]" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  broadcast.630 = f32[256,8192]{1,0} broadcast(multiply.102), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=712}
  broadcast.631 = f32[256,8192]{1,0} broadcast(subtract.67), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  broadcast.632 = f32[256,8192]{1,0} broadcast(minimum.4), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=280}
  multiply.124 = f32[256,8192]{0,1} multiply(convert.74, broadcast.632), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=280}
  compare.156 = pred[256,8192]{1,0} compare(multiply.124, multiply.124), direction=NE, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(nan_to_num)/jit(isnan)/ne" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  broadcast.633 = f32[256,8192]{1,0} broadcast(constant.898), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(nan_to_num)/jit(_where)/broadcast_in_dim[shape=(2048, 8192) broadcast_dimensions=()]" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.279 = f32[256,8192]{1,0} select(compare.156, broadcast.633, multiply.124), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(nan_to_num)/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  broadcast.634 = f32[256,8192]{1,0} broadcast(constant.899), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(nan_to_num)/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  compare.157 = pred[256,8192]{1,0} compare(select.279, broadcast.634), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(nan_to_num)/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.280 = f32[256,8192]{1,0} select(compare.157, broadcast.633, select.279), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(nan_to_num)/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  broadcast.636 = f32[256,8192]{1,0} broadcast(constant.816), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(nan_to_num)/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  compare.158 = pred[256,8192]{1,0} compare(select.280, broadcast.636), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(nan_to_num)/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.281 = f32[256,8192]{1,0} select(compare.158, broadcast.633, select.280), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(nan_to_num)/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  multiply.125 = f32[256,8192]{1,0} multiply(broadcast.631, select.281), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  broadcast.638 = f32[256,8192]{1,0} broadcast(divide.5), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  param.34 = f32[256,8192]{1,0} parameter(22), sharding={devices=[8,1]<=[8]}
  multiply.126 = f32[256,8192]{1,0} multiply(broadcast.638, param.34), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  add.269 = f32[256,8192]{1,0} add(multiply.125, multiply.126), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  broadcast.639 = f32[256,8192]{1,0} broadcast(subtract.70), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  multiply.128 = f32[256,8192]{1,0} multiply(select.281, select.281), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  multiply.129 = f32[256,8192]{1,0} multiply(broadcast.639, multiply.128), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  broadcast.640 = f32[256,8192]{1,0} broadcast(divide.7), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  param.35 = f32[256,8192]{1,0} parameter(26), sharding={devices=[8,1]<=[8]}
  multiply.130 = f32[256,8192]{1,0} multiply(broadcast.640, param.35), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  add.271 = f32[256,8192]{1,0} add(multiply.129, multiply.130), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  sqrt.3 = f32[256,8192]{1,0} sqrt(add.271), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sqrt" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  broadcast.641 = f32[256,8192]{1,0} broadcast(constant.910), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  add.272 = f32[256,8192]{1,0} add(sqrt.3, broadcast.641), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  divide.14 = f32[256,8192]{1,0} divide(add.269, add.272), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/div" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  broadcast.642 = f32[256,8192]{1,0} broadcast(constant.887), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  multiply.131 = f32[256,8192]{1,0} multiply(param.14, broadcast.642), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  add.273 = f32[256,8192]{1,0} add(divide.14, multiply.131), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  multiply.132 = f32[256,8192]{1,0} multiply(broadcast.630, add.273), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=712}
  broadcast.643 = f32[256,8192]{1,0} broadcast(constant.789), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/broadcast_in_dim[shape=(2048, 8192) broadcast_dimensions=()]" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  select.282 = f32[256,8192]{1,0} select(broadcast.629, multiply.132, broadcast.643), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  add.274 = f32[256,8192]{1,0} add(param.14, select.282), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/paxml/paxml/learners.py" source_line=428}
  broadcast.644 = pred[1024,50304]{1,0} broadcast(is-finite.1), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/broadcast_in_dim[shape=(8192, 50304) broadcast_dimensions=()]" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  broadcast.645 = f32[1024,50304]{1,0} broadcast(multiply.102), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=712}
  broadcast.646 = f32[1024,50304]{1,0} broadcast(subtract.67), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  broadcast.647 = f32[1024,50304]{1,0} broadcast(minimum.4), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=280}
  multiply.134 = f32[1024,50304]{0,1} multiply(convert.75, broadcast.647), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=280}
  compare.159 = pred[1024,50304]{1,0} compare(multiply.134, multiply.134), direction=NE, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(nan_to_num)/jit(isnan)/ne" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  broadcast.648 = f32[1024,50304]{1,0} broadcast(constant.898), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(nan_to_num)/jit(_where)/broadcast_in_dim[shape=(8192, 50304) broadcast_dimensions=()]" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.283 = f32[1024,50304]{1,0} select(compare.159, broadcast.648, multiply.134), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(nan_to_num)/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  broadcast.649 = f32[1024,50304]{1,0} broadcast(constant.899), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(nan_to_num)/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  compare.160 = pred[1024,50304]{1,0} compare(select.283, broadcast.649), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(nan_to_num)/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.284 = f32[1024,50304]{1,0} select(compare.160, broadcast.648, select.283), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(nan_to_num)/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  broadcast.651 = f32[1024,50304]{1,0} broadcast(constant.816), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(nan_to_num)/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  compare.161 = pred[1024,50304]{1,0} compare(select.284, broadcast.651), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(nan_to_num)/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.285 = f32[1024,50304]{1,0} select(compare.161, broadcast.648, select.284), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(nan_to_num)/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  multiply.135 = f32[1024,50304]{1,0} multiply(broadcast.646, select.285), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  broadcast.653 = f32[1024,50304]{1,0} broadcast(divide.5), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  param.36 = f32[1024,50304]{1,0} parameter(23), sharding={devices=[8,1]<=[8]}
  multiply.136 = f32[1024,50304]{1,0} multiply(broadcast.653, param.36), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  add.276 = f32[1024,50304]{1,0} add(multiply.135, multiply.136), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  broadcast.654 = f32[1024,50304]{1,0} broadcast(subtract.70), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  multiply.138 = f32[1024,50304]{1,0} multiply(select.285, select.285), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  multiply.139 = f32[1024,50304]{1,0} multiply(broadcast.654, multiply.138), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  broadcast.655 = f32[1024,50304]{1,0} broadcast(divide.7), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  param.37 = f32[1024,50304]{1,0} parameter(27), sharding={devices=[8,1]<=[8]}
  multiply.140 = f32[1024,50304]{1,0} multiply(broadcast.655, param.37), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  add.278 = f32[1024,50304]{1,0} add(multiply.139, multiply.140), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  sqrt.4 = f32[1024,50304]{1,0} sqrt(add.278), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sqrt" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  broadcast.656 = f32[1024,50304]{1,0} broadcast(constant.910), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  add.279 = f32[1024,50304]{1,0} add(sqrt.4, broadcast.656), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  divide.17 = f32[1024,50304]{1,0} divide(add.276, add.279), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/div" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  broadcast.657 = f32[1024,50304]{1,0} broadcast(constant.887), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  multiply.141 = f32[1024,50304]{1,0} multiply(param.13, broadcast.657), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  add.280 = f32[1024,50304]{1,0} add(divide.17, multiply.141), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  multiply.142 = f32[1024,50304]{1,0} multiply(broadcast.645, add.280), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=712}
  broadcast.658 = f32[1024,50304]{1,0} broadcast(constant.789), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/broadcast_in_dim[shape=(8192, 50304) broadcast_dimensions=()]" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  select.286 = f32[1024,50304]{1,0} select(broadcast.644, multiply.142, broadcast.658), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  add.281 = f32[1024,50304]{1,0} add(param.13, select.286), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/paxml/paxml/learners.py" source_line=428}
  convert.107 = f32[6,8192]{1,0} convert(param.15), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/paxml/paxml/learners.py" source_line=428}
  broadcast.659 = pred[6,8192]{1,0} broadcast(is-finite.1), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/broadcast_in_dim[shape=(6, 8192) broadcast_dimensions=()]" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  param.38 = s32[6]{0} parameter(31), sharding={replicated}
  broadcast.660 = s32[6]{0} broadcast(constant.865), dimensions={}
  compare.162 = pred[6]{0} compare(param.38, broadcast.660), direction=LT, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/lt" source_file="/usr/local/lib/python3.10/dist-packages/optax/_src/schedule.py" source_line=399}
  broadcast.661 = s32[6]{0} broadcast(constant.866), dimensions={}
  compare.163 = pred[6]{0} compare(param.38, broadcast.661), direction=LT, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/lt" source_file="/usr/local/lib/python3.10/dist-packages/optax/_src/schedule.py" source_line=399}
  convert.108 = f32[6]{0} convert(param.38), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/praxis/praxis/schedules.py" source_line=96}
  broadcast.662 = f32[6]{0} broadcast(constant.789), dimensions={}
  compare.164 = pred[6]{0} compare(convert.108, broadcast.662), direction=LT, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/lt" source_file="/opt/praxis/praxis/schedules.py" source_line=105}
  broadcast.664 = f32[6]{0} broadcast(constant.869), dimensions={}
  compare.165 = pred[6]{0} compare(convert.108, broadcast.664), direction=GE, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/ge" source_file="/opt/praxis/praxis/schedules.py" source_line=105}
  broadcast.665 = f32[6]{0} broadcast(constant.788), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(_where))/broadcast_in_dim[shape=(6,) broadcast_dimensions=()]" source_file="/opt/praxis/praxis/schedules.py" source_line=105}
  broadcast.666 = f32[6]{0} broadcast(constant.871), dimensions={}
  multiply.143 = f32[6]{0} multiply(convert.108, broadcast.666), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/div" source_file="/opt/praxis/praxis/schedules.py" source_line=99}
  select.287 = f32[6]{0} select(compare.165, broadcast.665, multiply.143), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(_where))/select_n" source_file="/opt/praxis/praxis/schedules.py" source_line=105}
  select.288 = f32[6]{0} select(compare.164, broadcast.662, select.287), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(_where))/select_n" source_file="/opt/praxis/praxis/schedules.py" source_line=105}
  broadcast.667 = s32[6]{0} broadcast(constant.872), dimensions={}
  add.282 = s32[6]{0} add(param.38, broadcast.667), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub" source_file="/usr/local/lib/python3.10/dist-packages/optax/_src/schedule.py" source_line=399}
  convert.109 = f32[6]{0} convert(add.282), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/praxis/praxis/schedules.py" source_line=96}
  compare.166 = pred[6]{0} compare(convert.109, broadcast.662), direction=LT, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/lt" source_file="/opt/praxis/praxis/schedules.py" source_line=105}
  compare.167 = pred[6]{0} compare(convert.109, broadcast.665), direction=GE, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/ge" source_file="/opt/praxis/praxis/schedules.py" source_line=105}
  multiply.144 = f32[6]{0} multiply(convert.109, broadcast.662), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/schedules.py" source_line=104}
  add.283 = f32[6]{0} add(multiply.144, broadcast.665), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/schedules.py" source_line=104}
  select.289 = f32[6]{0} select(compare.167, broadcast.665, add.283), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(_where))/select_n" source_file="/opt/praxis/praxis/schedules.py" source_line=105}
  select.290 = f32[6]{0} select(compare.166, broadcast.665, select.289), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(_where))/select_n" source_file="/opt/praxis/praxis/schedules.py" source_line=105}
  select.291 = f32[6]{0} select(compare.163, select.288, select.290), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(_where))/select_n" source_file="/usr/local/lib/python3.10/dist-packages/optax/_src/schedule.py" source_line=399}
  broadcast.673 = s32[6]{0} broadcast(constant.879), dimensions={}
  add.284 = s32[6]{0} add(param.38, broadcast.673), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub" source_file="/usr/local/lib/python3.10/dist-packages/optax/_src/schedule.py" source_line=399}
  convert.110 = f32[6]{0} convert(add.284), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/praxis/praxis/schedules.py" source_line=96}
  compare.168 = pred[6]{0} compare(convert.110, broadcast.662), direction=LT, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/lt" source_file="/opt/praxis/praxis/schedules.py" source_line=105}
  broadcast.675 = f32[6]{0} broadcast(constant.882), dimensions={}
  compare.169 = pred[6]{0} compare(convert.110, broadcast.675), direction=GE, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/ge" source_file="/opt/praxis/praxis/schedules.py" source_line=105}
  broadcast.676 = f32[6]{0} broadcast(constant.883), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(_where))/broadcast_in_dim[shape=(6,) broadcast_dimensions=()]" source_file="/opt/praxis/praxis/schedules.py" source_line=105}
  broadcast.677 = f32[6]{0} broadcast(constant.884), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/schedules.py" source_line=104}
  multiply.145 = f32[6]{0} multiply(convert.110, broadcast.677), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/schedules.py" source_line=104}
  select.292 = f32[6]{0} select(compare.169, broadcast.676, multiply.145), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(_where))/select_n" source_file="/opt/praxis/praxis/schedules.py" source_line=105}
  select.293 = f32[6]{0} select(compare.168, broadcast.662, select.292), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(_where))/select_n" source_file="/opt/praxis/praxis/schedules.py" source_line=105}
  cosine.1 = f32[6]{0} cosine(select.293), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/cos" source_file="/opt/praxis/praxis/schedules.py" source_line=204}
  add.285 = f32[6]{0} add(cosine.1, broadcast.665), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/schedules.py" source_line=204}
  broadcast.678 = f32[6]{0} broadcast(constant.886), dimensions={}
  multiply.146 = f32[6]{0} multiply(add.285, broadcast.678), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/schedules.py" source_line=203}
  broadcast.679 = f32[6]{0} broadcast(constant.887), dimensions={}
  add.286 = f32[6]{0} add(multiply.146, broadcast.679), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/schedules.py" source_line=203}
  select.294 = f32[6]{0} select(compare.162, select.291, add.286), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(_where))/select_n" source_file="/usr/local/lib/python3.10/dist-packages/optax/_src/schedule.py" source_line=399}
  broadcast.680 = f32[6]{0} broadcast(constant.888), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=710}
  multiply.147 = f32[6]{0} multiply(select.294, broadcast.680), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=710}
  broadcast.681 = f32[6,8192]{1,0} broadcast(multiply.147), dimensions={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=712}
  broadcast.682 = f32[6]{0} broadcast(constant.891), dimensions={}
  power.16 = f32[6]{0} power(broadcast.682, convert.108), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/pow" source_file="/opt/praxis/praxis/optimizers.py" source_line=336}
  subtract.89 = f32[6]{0} subtract(broadcast.665, power.16), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub" source_file="/opt/praxis/praxis/optimizers.py" source_line=336}
  multiply.148 = f32[6]{0} multiply(subtract.89, broadcast.682), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=336}
  add.287 = f32[6]{0} add(convert.108, broadcast.665), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=335}
  power.17 = f32[6]{0} power(broadcast.682, add.287), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/pow" source_file="/opt/praxis/praxis/optimizers.py" source_line=336}
  subtract.90 = f32[6]{0} subtract(broadcast.665, power.17), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub" source_file="/opt/praxis/praxis/optimizers.py" source_line=336}
  divide.18 = f32[6]{0} divide(multiply.148, subtract.90), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/div" source_file="/opt/praxis/praxis/optimizers.py" source_line=336}
  subtract.91 = f32[6]{0} subtract(broadcast.665, divide.18), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  broadcast.683 = f32[6,8192]{1,0} broadcast(subtract.91), dimensions={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  convert.112 = f32[6,8192]{1,0} convert(get-tuple-element.148), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/paxml/paxml/learners.py" source_line=280}
  broadcast.684 = f32[6,8192]{1,0} broadcast(minimum.4), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=280}
  multiply.149 = f32[6,8192]{1,0} multiply(convert.112, broadcast.684), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=280}
  compare.170 = pred[6,8192]{1,0} compare(multiply.149, multiply.149), direction=NE, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(isnan)/ne" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  broadcast.686 = f32[6,8192]{1,0} broadcast(constant.898), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/broadcast_in_dim[shape=(6, 8192) broadcast_dimensions=(1,)]" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.295 = f32[6,8192]{1,0} select(compare.170, broadcast.686, multiply.149), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  broadcast.688 = f32[6,8192]{1,0} broadcast(constant.899), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  compare.171 = pred[6,8192]{1,0} compare(select.295, broadcast.688), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.296 = f32[6,8192]{1,0} select(compare.171, broadcast.686, select.295), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  broadcast.690 = f32[6,8192]{1,0} broadcast(constant.816), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  compare.172 = pred[6,8192]{1,0} compare(select.296, broadcast.690), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.297 = f32[6,8192]{1,0} select(compare.172, broadcast.686, select.296), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  multiply.150 = f32[6,8192]{1,0} multiply(broadcast.683, select.297), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  broadcast.692 = f32[6,8192]{1,0} broadcast(divide.18), dimensions={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  param.39 = bf16[6,8192]{1,0} parameter(32), sharding={replicated}
  convert.113 = f32[6,8192]{1,0} convert(param.39), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  multiply.151 = f32[6,8192]{1,0} multiply(broadcast.692, convert.113), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  add.288 = f32[6,8192]{1,0} add(multiply.150, multiply.151), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  broadcast.694 = f32[6]{0} broadcast(constant.905), dimensions={}
  power.18 = f32[6]{0} power(broadcast.694, convert.108), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/pow" source_file="/opt/praxis/praxis/optimizers.py" source_line=336}
  subtract.92 = f32[6]{0} subtract(broadcast.665, power.18), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub" source_file="/opt/praxis/praxis/optimizers.py" source_line=336}
  multiply.152 = f32[6]{0} multiply(subtract.92, broadcast.694), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=336}
  power.19 = f32[6]{0} power(broadcast.694, add.287), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/pow" source_file="/opt/praxis/praxis/optimizers.py" source_line=336}
  subtract.93 = f32[6]{0} subtract(broadcast.665, power.19), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub" source_file="/opt/praxis/praxis/optimizers.py" source_line=336}
  divide.19 = f32[6]{0} divide(multiply.152, subtract.93), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/div" source_file="/opt/praxis/praxis/optimizers.py" source_line=336}
  subtract.94 = f32[6]{0} subtract(broadcast.665, divide.19), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sub" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  broadcast.696 = f32[6,8192]{1,0} broadcast(subtract.94), dimensions={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  multiply.153 = f32[6,8192]{1,0} multiply(select.297, select.297), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  multiply.154 = f32[6,8192]{1,0} multiply(broadcast.696, multiply.153), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  broadcast.697 = f32[6,8192]{1,0} broadcast(divide.19), dimensions={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  param.40 = bf16[6,8192]{1,0} parameter(44), sharding={replicated}
  convert.115 = f32[6,8192]{1,0} convert(param.40), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  multiply.155 = f32[6,8192]{1,0} multiply(broadcast.697, convert.115), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  add.290 = f32[6,8192]{1,0} add(multiply.154, multiply.155), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  sqrt.5 = f32[6,8192]{1,0} sqrt(add.290), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sqrt" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  broadcast.698 = f32[6,8192]{1,0} broadcast(constant.910), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  add.291 = f32[6,8192]{1,0} add(sqrt.5, broadcast.698), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  divide.20 = f32[6,8192]{1,0} divide(add.288, add.291), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/div" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  constant.1003 = bf16[] constant(0.1001)
  broadcast.700 = bf16[6,8192]{1,0} broadcast(constant.1003), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  multiply.156 = bf16[6,8192]{1,0} multiply(param.15, broadcast.700), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  convert.116 = f32[6,8192]{1,0} convert(multiply.156), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  add.292 = f32[6,8192]{1,0} add(divide.20, convert.116), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  multiply.157 = f32[6,8192]{1,0} multiply(broadcast.681, add.292), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=712}
  broadcast.702 = f32[6,8192]{1,0} broadcast(constant.789), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/broadcast_in_dim[shape=(6, 8192) broadcast_dimensions=()]" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  select.298 = f32[6,8192]{1,0} select(broadcast.659, multiply.157, broadcast.702), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  add.293 = f32[6,8192]{1,0} add(convert.107, select.298), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/paxml/paxml/learners.py" source_line=428}
  broadcast.703 = pred[6,8192,1024]{2,1,0} broadcast(is-finite.1), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/broadcast_in_dim[shape=(6, 8192, 8192) broadcast_dimensions=()]" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  broadcast.705 = f32[6,8192,1024]{2,1,0} broadcast(multiply.147), dimensions={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=712}
  broadcast.707 = f32[6,8192,1024]{2,1,0} broadcast(subtract.91), dimensions={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  broadcast.708 = f32[6,8192,1024]{2,1,0} broadcast(minimum.4), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=280}
  multiply.159 = f32[6,8192,1024]{2,1,0} multiply(convert.79, broadcast.708), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=280}
  compare.173 = pred[6,8192,1024]{2,1,0} compare(multiply.159, multiply.159), direction=NE, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(isnan)/ne" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  broadcast.709 = f32[6,8192,1024]{2,1,0} broadcast(constant.898), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/broadcast_in_dim[shape=(6, 8192, 8192) broadcast_dimensions=(1, 2)]" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.299 = f32[6,8192,1024]{2,1,0} select(compare.173, broadcast.709, multiply.159), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  broadcast.710 = f32[6,8192,1024]{2,1,0} broadcast(constant.899), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  compare.174 = pred[6,8192,1024]{2,1,0} compare(select.299, broadcast.710), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.300 = f32[6,8192,1024]{2,1,0} select(compare.174, broadcast.709, select.299), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  broadcast.712 = f32[6,8192,1024]{2,1,0} broadcast(constant.816), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  compare.175 = pred[6,8192,1024]{2,1,0} compare(select.300, broadcast.712), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.301 = f32[6,8192,1024]{2,1,0} select(compare.175, broadcast.709, select.300), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  multiply.160 = f32[6,8192,1024]{2,1,0} multiply(broadcast.707, select.301), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  broadcast.714 = f32[6,8192,1024]{2,1,0} broadcast(divide.18), dimensions={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  param.41 = f32[6,8192,1024]{2,1,0} parameter(33), sharding={devices=[1,1,8]<=[8]}
  multiply.161 = f32[6,8192,1024]{2,1,0} multiply(broadcast.714, param.41), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  add.295 = f32[6,8192,1024]{2,1,0} add(multiply.160, multiply.161), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  broadcast.715 = f32[6,8192,1024]{2,1,0} broadcast(subtract.94), dimensions={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  multiply.163 = f32[6,8192,1024]{2,1,0} multiply(select.301, select.301), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  multiply.164 = f32[6,8192,1024]{2,1,0} multiply(broadcast.715, multiply.163), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  broadcast.716 = f32[6,8192,1024]{2,1,0} broadcast(divide.19), dimensions={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  param.42 = f32[6,8192,1024]{2,1,0} parameter(45), sharding={devices=[1,1,8]<=[8]}
  multiply.165 = f32[6,8192,1024]{2,1,0} multiply(broadcast.716, param.42), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  add.297 = f32[6,8192,1024]{2,1,0} add(multiply.164, multiply.165), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  sqrt.6 = f32[6,8192,1024]{2,1,0} sqrt(add.297), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sqrt" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  broadcast.717 = f32[6,8192,1024]{2,1,0} broadcast(constant.910), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  add.298 = f32[6,8192,1024]{2,1,0} add(sqrt.6, broadcast.717), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  divide.23 = f32[6,8192,1024]{2,1,0} divide(add.295, add.298), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/div" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  broadcast.718 = f32[6,8192,1024]{2,1,0} broadcast(constant.887), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  multiply.166 = f32[6,8192,1024]{2,1,0} multiply(param.16, broadcast.718), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  add.299 = f32[6,8192,1024]{2,1,0} add(divide.23, multiply.166), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  multiply.167 = f32[6,8192,1024]{2,1,0} multiply(broadcast.705, add.299), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=712}
  broadcast.719 = f32[6,8192,1024]{2,1,0} broadcast(constant.789), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/broadcast_in_dim[shape=(6, 8192, 8192) broadcast_dimensions=()]" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  select.302 = f32[6,8192,1024]{2,1,0} select(broadcast.703, multiply.167, broadcast.719), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  add.300 = f32[6,8192,1024]{2,1,0} add(param.16, select.302), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/paxml/paxml/learners.py" source_line=428}
  convert.119 = f32[6,3,8192]{2,1,0} convert(param.17), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/paxml/paxml/learners.py" source_line=428}
  broadcast.720 = pred[6,3,8192]{2,1,0} broadcast(is-finite.1), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/broadcast_in_dim[shape=(6, 3, 8192) broadcast_dimensions=()]" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  broadcast.721 = f32[6,3,8192]{2,1,0} broadcast(multiply.147), dimensions={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=712}
  broadcast.722 = f32[6,3,8192]{2,1,0} broadcast(subtract.91), dimensions={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  convert.121 = f32[6,3,8192]{2,1,0} convert(get-tuple-element.150), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/paxml/paxml/learners.py" source_line=280}
  broadcast.723 = f32[6,3,8192]{2,1,0} broadcast(minimum.4), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=280}
  multiply.169 = f32[6,3,8192]{2,1,0} multiply(convert.121, broadcast.723), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=280}
  compare.176 = pred[6,3,8192]{2,1,0} compare(multiply.169, multiply.169), direction=NE, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(isnan)/ne" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  broadcast.724 = f32[6,3,8192]{2,1,0} broadcast(constant.898), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/broadcast_in_dim[shape=(6, 3, 8192) broadcast_dimensions=(1, 2)]" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.303 = f32[6,3,8192]{2,1,0} select(compare.176, broadcast.724, multiply.169), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  broadcast.725 = f32[6,3,8192]{2,1,0} broadcast(constant.899), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  compare.177 = pred[6,3,8192]{2,1,0} compare(select.303, broadcast.725), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.304 = f32[6,3,8192]{2,1,0} select(compare.177, broadcast.724, select.303), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  broadcast.729 = f32[6,3,8192]{2,1,0} broadcast(constant.816), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  compare.178 = pred[6,3,8192]{2,1,0} compare(select.304, broadcast.729), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.305 = f32[6,3,8192]{2,1,0} select(compare.178, broadcast.724, select.304), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  multiply.170 = f32[6,3,8192]{2,1,0} multiply(broadcast.722, select.305), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  broadcast.731 = f32[6,3,8192]{2,1,0} broadcast(divide.18), dimensions={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  param.43 = bf16[6,3,8192]{2,1,0} parameter(34), sharding={replicated}
  convert.122 = f32[6,3,8192]{2,1,0} convert(param.43), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  multiply.171 = f32[6,3,8192]{2,1,0} multiply(broadcast.731, convert.122), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  add.302 = f32[6,3,8192]{2,1,0} add(multiply.170, multiply.171), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  broadcast.732 = f32[6,3,8192]{2,1,0} broadcast(subtract.94), dimensions={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  multiply.173 = f32[6,3,8192]{2,1,0} multiply(select.305, select.305), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  multiply.174 = f32[6,3,8192]{2,1,0} multiply(broadcast.732, multiply.173), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  broadcast.733 = f32[6,3,8192]{2,1,0} broadcast(divide.19), dimensions={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  param.44 = bf16[6,3,8192]{2,1,0} parameter(46), sharding={replicated}
  convert.124 = f32[6,3,8192]{2,1,0} convert(param.44), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  multiply.175 = f32[6,3,8192]{2,1,0} multiply(broadcast.733, convert.124), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  add.304 = f32[6,3,8192]{2,1,0} add(multiply.174, multiply.175), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  sqrt.7 = f32[6,3,8192]{2,1,0} sqrt(add.304), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sqrt" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  broadcast.735 = f32[6,3,8192]{2,1,0} broadcast(constant.910), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  add.305 = f32[6,3,8192]{2,1,0} add(sqrt.7, broadcast.735), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  divide.26 = f32[6,3,8192]{2,1,0} divide(add.302, add.305), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/div" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  broadcast.737 = bf16[6,3,8192]{2,1,0} broadcast(constant.1003), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  multiply.176 = bf16[6,3,8192]{2,1,0} multiply(param.17, broadcast.737), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  convert.125 = f32[6,3,8192]{2,1,0} convert(multiply.176), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  add.306 = f32[6,3,8192]{2,1,0} add(divide.26, convert.125), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  multiply.177 = f32[6,3,8192]{2,1,0} multiply(broadcast.721, add.306), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=712}
  broadcast.738 = f32[6,3,8192]{2,1,0} broadcast(constant.789), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/broadcast_in_dim[shape=(6, 3, 8192) broadcast_dimensions=()]" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  select.306 = f32[6,3,8192]{2,1,0} select(broadcast.720, multiply.177, broadcast.738), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  add.307 = f32[6,3,8192]{2,1,0} add(convert.119, select.306), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/paxml/paxml/learners.py" source_line=428}
  broadcast.739 = pred[6,1024,3,8192]{3,2,1,0} broadcast(is-finite.1), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/broadcast_in_dim[shape=(6, 8192, 3, 8192) broadcast_dimensions=()]" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  broadcast.740 = f32[6,1024,3,8192]{3,2,1,0} broadcast(multiply.147), dimensions={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=712}
  broadcast.741 = f32[6,1024,3,8192]{3,2,1,0} broadcast(subtract.91), dimensions={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  broadcast.742 = f32[6,1024,3,8192]{3,2,1,0} broadcast(minimum.4), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=280}
  multiply.179 = f32[6,1024,3,8192]{3,2,1,0} multiply(convert.83, broadcast.742), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=280}
  compare.179 = pred[6,1024,3,8192]{3,2,1,0} compare(multiply.179, multiply.179), direction=NE, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(isnan)/ne" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  broadcast.743 = f32[6,1024,3,8192]{3,2,1,0} broadcast(constant.898), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/broadcast_in_dim[shape=(6, 8192, 3, 8192) broadcast_dimensions=(1, 2, 3)]" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.307 = f32[6,1024,3,8192]{3,2,1,0} select(compare.179, broadcast.743, multiply.179), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  broadcast.744 = f32[6,1024,3,8192]{3,2,1,0} broadcast(constant.899), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  compare.180 = pred[6,1024,3,8192]{3,2,1,0} compare(select.307, broadcast.744), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.308 = f32[6,1024,3,8192]{3,2,1,0} select(compare.180, broadcast.743, select.307), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  broadcast.746 = f32[6,1024,3,8192]{3,2,1,0} broadcast(constant.816), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  compare.181 = pred[6,1024,3,8192]{3,2,1,0} compare(select.308, broadcast.746), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.309 = f32[6,1024,3,8192]{3,2,1,0} select(compare.181, broadcast.743, select.308), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  multiply.180 = f32[6,1024,3,8192]{3,2,1,0} multiply(broadcast.741, select.309), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  broadcast.749 = f32[6,1024,3,8192]{3,2,1,0} broadcast(divide.18), dimensions={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  param.45 = f32[6,1024,3,8192]{3,2,1,0} parameter(35), sharding={devices=[1,8,1,1]<=[8]}
  multiply.181 = f32[6,1024,3,8192]{3,2,1,0} multiply(broadcast.749, param.45), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  add.309 = f32[6,1024,3,8192]{3,2,1,0} add(multiply.180, multiply.181), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  broadcast.752 = f32[6,1024,3,8192]{3,2,1,0} broadcast(subtract.94), dimensions={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  multiply.183 = f32[6,1024,3,8192]{3,2,1,0} multiply(select.309, select.309), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  multiply.184 = f32[6,1024,3,8192]{3,2,1,0} multiply(broadcast.752, multiply.183), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  broadcast.754 = f32[6,1024,3,8192]{3,2,1,0} broadcast(divide.19), dimensions={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  param.46 = f32[6,1024,3,8192]{3,2,1,0} parameter(47), sharding={devices=[1,8,1,1]<=[8]}
  multiply.185 = f32[6,1024,3,8192]{3,2,1,0} multiply(broadcast.754, param.46), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  add.311 = f32[6,1024,3,8192]{3,2,1,0} add(multiply.184, multiply.185), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  sqrt.8 = f32[6,1024,3,8192]{3,2,1,0} sqrt(add.311), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sqrt" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  broadcast.755 = f32[6,1024,3,8192]{3,2,1,0} broadcast(constant.910), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  add.313 = f32[6,1024,3,8192]{3,2,1,0} add(sqrt.8, broadcast.755), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  divide.29 = f32[6,1024,3,8192]{3,2,1,0} divide(add.309, add.313), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/div" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  broadcast.756 = f32[6,1024,3,8192]{3,2,1,0} broadcast(constant.887), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  multiply.186 = f32[6,1024,3,8192]{3,2,1,0} multiply(param.18, broadcast.756), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  add.314 = f32[6,1024,3,8192]{3,2,1,0} add(divide.29, multiply.186), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  multiply.187 = f32[6,1024,3,8192]{3,2,1,0} multiply(broadcast.740, add.314), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=712}
  broadcast.757 = f32[6,1024,3,8192]{3,2,1,0} broadcast(constant.789), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/broadcast_in_dim[shape=(6, 8192, 3, 8192) broadcast_dimensions=()]" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  select.310 = f32[6,1024,3,8192]{3,2,1,0} select(broadcast.739, multiply.187, broadcast.757), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  add.315 = f32[6,1024,3,8192]{3,2,1,0} add(param.18, select.310), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/paxml/paxml/learners.py" source_line=428}
  multiply.189 = f32[6,8192]{1,0} multiply(convert.84, broadcast.684), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=280}
  compare.182 = pred[6,8192]{1,0} compare(multiply.189, multiply.189), direction=NE, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(isnan)/ne" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.311 = f32[6,8192]{1,0} select(compare.182, broadcast.686, multiply.189), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  compare.183 = pred[6,8192]{1,0} compare(select.311, broadcast.688), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.312 = f32[6,8192]{1,0} select(compare.183, broadcast.686, select.311), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  compare.184 = pred[6,8192]{1,0} compare(select.312, broadcast.690), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.313 = f32[6,8192]{1,0} select(compare.184, broadcast.686, select.312), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  multiply.190 = f32[6,8192]{1,0} multiply(broadcast.683, select.313), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  param.47 = f32[6,8192]{1,0} parameter(36), sharding={replicated}
  multiply.191 = f32[6,8192]{1,0} multiply(broadcast.692, param.47), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  add.317 = f32[6,8192]{1,0} add(multiply.190, multiply.191), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  multiply.193 = f32[6,8192]{1,0} multiply(select.313, select.313), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  multiply.194 = f32[6,8192]{1,0} multiply(broadcast.696, multiply.193), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  param.48 = f32[6,8192]{1,0} parameter(48), sharding={replicated}
  multiply.195 = f32[6,8192]{1,0} multiply(broadcast.697, param.48), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  add.319 = f32[6,8192]{1,0} add(multiply.194, multiply.195), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  sqrt.9 = f32[6,8192]{1,0} sqrt(add.319), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sqrt" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  add.320 = f32[6,8192]{1,0} add(sqrt.9, broadcast.698), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  divide.32 = f32[6,8192]{1,0} divide(add.317, add.320), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/div" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  broadcast.770 = f32[6,8192]{1,0} broadcast(constant.887), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  multiply.196 = f32[6,8192]{1,0} multiply(param.19, broadcast.770), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  add.321 = f32[6,8192]{1,0} add(divide.32, multiply.196), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  multiply.197 = f32[6,8192]{1,0} multiply(broadcast.681, add.321), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=712}
  select.314 = f32[6,8192]{1,0} select(broadcast.659, multiply.197, broadcast.702), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  add.322 = f32[6,8192]{1,0} add(param.19, select.314), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/paxml/paxml/learners.py" source_line=428}
  multiply.199 = f32[6,8192]{1,0} multiply(convert.85, broadcast.684), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=280}
  compare.185 = pred[6,8192]{1,0} compare(multiply.199, multiply.199), direction=NE, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(isnan)/ne" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.315 = f32[6,8192]{1,0} select(compare.185, broadcast.686, multiply.199), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  compare.186 = pred[6,8192]{1,0} compare(select.315, broadcast.688), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.316 = f32[6,8192]{1,0} select(compare.186, broadcast.686, select.315), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  compare.187 = pred[6,8192]{1,0} compare(select.316, broadcast.690), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.317 = f32[6,8192]{1,0} select(compare.187, broadcast.686, select.316), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  multiply.200 = f32[6,8192]{1,0} multiply(broadcast.683, select.317), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  param.49 = f32[6,8192]{1,0} parameter(37), sharding={replicated}
  multiply.201 = f32[6,8192]{1,0} multiply(broadcast.692, param.49), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  add.324 = f32[6,8192]{1,0} add(multiply.200, multiply.201), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  multiply.203 = f32[6,8192]{1,0} multiply(select.317, select.317), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  multiply.204 = f32[6,8192]{1,0} multiply(broadcast.696, multiply.203), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  param.50 = f32[6,8192]{1,0} parameter(49), sharding={replicated}
  multiply.205 = f32[6,8192]{1,0} multiply(broadcast.697, param.50), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  add.326 = f32[6,8192]{1,0} add(multiply.204, multiply.205), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  sqrt.10 = f32[6,8192]{1,0} sqrt(add.326), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sqrt" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  add.327 = f32[6,8192]{1,0} add(sqrt.10, broadcast.698), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  divide.35 = f32[6,8192]{1,0} divide(add.324, add.327), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/div" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  multiply.206 = f32[6,8192]{1,0} multiply(param.20, broadcast.770), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  add.328 = f32[6,8192]{1,0} add(divide.35, multiply.206), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  multiply.207 = f32[6,8192]{1,0} multiply(broadcast.681, add.328), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=712}
  select.318 = f32[6,8192]{1,0} select(broadcast.659, multiply.207, broadcast.702), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  add.329 = f32[6,8192]{1,0} add(param.20, select.318), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/paxml/paxml/learners.py" source_line=428}
  multiply.209 = f32[6,8192]{1,0} multiply(convert.86, broadcast.684), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=280}
  compare.188 = pred[6,8192]{1,0} compare(multiply.209, multiply.209), direction=NE, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(isnan)/ne" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.319 = f32[6,8192]{1,0} select(compare.188, broadcast.686, multiply.209), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  compare.189 = pred[6,8192]{1,0} compare(select.319, broadcast.688), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.320 = f32[6,8192]{1,0} select(compare.189, broadcast.686, select.319), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  compare.190 = pred[6,8192]{1,0} compare(select.320, broadcast.690), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.321 = f32[6,8192]{1,0} select(compare.190, broadcast.686, select.320), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  multiply.210 = f32[6,8192]{1,0} multiply(broadcast.683, select.321), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  param.51 = f32[6,8192]{1,0} parameter(38), sharding={replicated}
  multiply.211 = f32[6,8192]{1,0} multiply(broadcast.692, param.51), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  add.331 = f32[6,8192]{1,0} add(multiply.210, multiply.211), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  multiply.213 = f32[6,8192]{1,0} multiply(select.321, select.321), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  multiply.214 = f32[6,8192]{1,0} multiply(broadcast.696, multiply.213), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  param.52 = f32[6,8192]{1,0} parameter(50), sharding={replicated}
  multiply.215 = f32[6,8192]{1,0} multiply(broadcast.697, param.52), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  add.333 = f32[6,8192]{1,0} add(multiply.214, multiply.215), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  sqrt.11 = f32[6,8192]{1,0} sqrt(add.333), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sqrt" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  add.334 = f32[6,8192]{1,0} add(sqrt.11, broadcast.698), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  divide.38 = f32[6,8192]{1,0} divide(add.331, add.334), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/div" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  multiply.216 = f32[6,8192]{1,0} multiply(param.21, broadcast.770), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  add.335 = f32[6,8192]{1,0} add(divide.38, multiply.216), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  multiply.217 = f32[6,8192]{1,0} multiply(broadcast.681, add.335), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=712}
  select.322 = f32[6,8192]{1,0} select(broadcast.659, multiply.217, broadcast.702), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  add.336 = f32[6,8192]{1,0} add(param.21, select.322), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/paxml/paxml/learners.py" source_line=428}
  multiply.219 = f32[6,8192]{1,0} multiply(convert.87, broadcast.684), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=280}
  compare.191 = pred[6,8192]{1,0} compare(multiply.219, multiply.219), direction=NE, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(isnan)/ne" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.323 = f32[6,8192]{1,0} select(compare.191, broadcast.686, multiply.219), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  compare.192 = pred[6,8192]{1,0} compare(select.323, broadcast.688), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.324 = f32[6,8192]{1,0} select(compare.192, broadcast.686, select.323), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  compare.193 = pred[6,8192]{1,0} compare(select.324, broadcast.690), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.325 = f32[6,8192]{1,0} select(compare.193, broadcast.686, select.324), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  multiply.220 = f32[6,8192]{1,0} multiply(broadcast.683, select.325), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  param.53 = f32[6,8192]{1,0} parameter(39), sharding={replicated}
  multiply.221 = f32[6,8192]{1,0} multiply(broadcast.692, param.53), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  add.338 = f32[6,8192]{1,0} add(multiply.220, multiply.221), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  multiply.223 = f32[6,8192]{1,0} multiply(select.325, select.325), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  multiply.224 = f32[6,8192]{1,0} multiply(broadcast.696, multiply.223), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  param.54 = f32[6,8192]{1,0} parameter(51), sharding={replicated}
  multiply.225 = f32[6,8192]{1,0} multiply(broadcast.697, param.54), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  add.340 = f32[6,8192]{1,0} add(multiply.224, multiply.225), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  sqrt.12 = f32[6,8192]{1,0} sqrt(add.340), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sqrt" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  add.341 = f32[6,8192]{1,0} add(sqrt.12, broadcast.698), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  divide.41 = f32[6,8192]{1,0} divide(add.338, add.341), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/div" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  multiply.226 = f32[6,8192]{1,0} multiply(param.22, broadcast.770), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  add.342 = f32[6,8192]{1,0} add(divide.41, multiply.226), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  multiply.227 = f32[6,8192]{1,0} multiply(broadcast.681, add.342), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=712}
  select.326 = f32[6,8192]{1,0} select(broadcast.659, multiply.227, broadcast.702), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  add.343 = f32[6,8192]{1,0} add(param.22, select.326), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/paxml/paxml/learners.py" source_line=428}
  convert.136 = f32[6,1,32768]{2,1,0} convert(param.23), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/paxml/paxml/learners.py" source_line=428}
  broadcast.808 = pred[6,1,32768]{2,1,0} broadcast(is-finite.1), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/broadcast_in_dim[shape=(6, 1, 32768) broadcast_dimensions=()]" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  broadcast.809 = f32[6,1,32768]{2,1,0} broadcast(multiply.147), dimensions={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=712}
  broadcast.810 = f32[6,1,32768]{2,1,0} broadcast(subtract.91), dimensions={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  convert.138 = f32[6,1,32768]{2,1,0} convert(get-tuple-element.156), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/paxml/paxml/learners.py" source_line=280}
  broadcast.811 = f32[6,1,32768]{2,1,0} broadcast(minimum.4), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=280}
  multiply.229 = f32[6,1,32768]{2,1,0} multiply(convert.138, broadcast.811), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=280}
  compare.194 = pred[6,1,32768]{2,1,0} compare(multiply.229, multiply.229), direction=NE, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(isnan)/ne" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  broadcast.812 = f32[6,1,32768]{2,1,0} broadcast(constant.898), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/broadcast_in_dim[shape=(6, 1, 32768) broadcast_dimensions=(1, 2)]" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.327 = f32[6,1,32768]{2,1,0} select(compare.194, broadcast.812, multiply.229), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  broadcast.814 = f32[6,1,32768]{2,1,0} broadcast(constant.899), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  compare.195 = pred[6,1,32768]{2,1,0} compare(select.327, broadcast.814), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.328 = f32[6,1,32768]{2,1,0} select(compare.195, broadcast.812, select.327), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  broadcast.816 = f32[6,1,32768]{2,1,0} broadcast(constant.816), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  compare.196 = pred[6,1,32768]{2,1,0} compare(select.328, broadcast.816), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.329 = f32[6,1,32768]{2,1,0} select(compare.196, broadcast.812, select.328), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  multiply.230 = f32[6,1,32768]{2,1,0} multiply(broadcast.810, select.329), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  broadcast.818 = f32[6,1,32768]{2,1,0} broadcast(divide.18), dimensions={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  param.55 = bf16[6,1,32768]{2,1,0} parameter(40), sharding={replicated}
  convert.139 = f32[6,1,32768]{2,1,0} convert(param.55), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  multiply.231 = f32[6,1,32768]{2,1,0} multiply(broadcast.818, convert.139), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  add.345 = f32[6,1,32768]{2,1,0} add(multiply.230, multiply.231), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  broadcast.819 = f32[6,1,32768]{2,1,0} broadcast(subtract.94), dimensions={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  multiply.233 = f32[6,1,32768]{2,1,0} multiply(select.329, select.329), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  multiply.234 = f32[6,1,32768]{2,1,0} multiply(broadcast.819, multiply.233), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  broadcast.820 = f32[6,1,32768]{2,1,0} broadcast(divide.19), dimensions={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  param.56 = bf16[6,1,32768]{2,1,0} parameter(52), sharding={replicated}
  convert.141 = f32[6,1,32768]{2,1,0} convert(param.56), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  multiply.235 = f32[6,1,32768]{2,1,0} multiply(broadcast.820, convert.141), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  add.347 = f32[6,1,32768]{2,1,0} add(multiply.234, multiply.235), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  sqrt.13 = f32[6,1,32768]{2,1,0} sqrt(add.347), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sqrt" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  broadcast.821 = f32[6,1,32768]{2,1,0} broadcast(constant.910), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  add.348 = f32[6,1,32768]{2,1,0} add(sqrt.13, broadcast.821), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  divide.44 = f32[6,1,32768]{2,1,0} divide(add.345, add.348), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/div" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  broadcast.822 = bf16[6,1,32768]{2,1,0} broadcast(constant.1003), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  multiply.236 = bf16[6,1,32768]{2,1,0} multiply(param.23, broadcast.822), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  convert.142 = f32[6,1,32768]{2,1,0} convert(multiply.236), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  add.349 = f32[6,1,32768]{2,1,0} add(divide.44, convert.142), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  multiply.237 = f32[6,1,32768]{2,1,0} multiply(broadcast.809, add.349), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=712}
  broadcast.823 = f32[6,1,32768]{2,1,0} broadcast(constant.789), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/broadcast_in_dim[shape=(6, 1, 32768) broadcast_dimensions=()]" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  select.330 = f32[6,1,32768]{2,1,0} select(broadcast.808, multiply.237, broadcast.823), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  add.350 = f32[6,1,32768]{2,1,0} add(convert.136, select.330), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/paxml/paxml/learners.py" source_line=428}
  broadcast.824 = pred[6,1024,1,32768]{3,2,1,0} broadcast(is-finite.1), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/broadcast_in_dim[shape=(6, 8192, 1, 32768) broadcast_dimensions=()]" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  broadcast.825 = f32[6,1024,1,32768]{3,2,1,0} broadcast(multiply.147), dimensions={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=712}
  broadcast.826 = f32[6,1024,1,32768]{3,2,1,0} broadcast(subtract.91), dimensions={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  broadcast.827 = f32[6,1024,1,32768]{3,2,1,0} broadcast(minimum.4), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=280}
  multiply.239 = f32[6,1024,1,32768]{3,2,1,0} multiply(convert.91, broadcast.827), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=280}
  compare.197 = pred[6,1024,1,32768]{3,2,1,0} compare(multiply.239, multiply.239), direction=NE, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(isnan)/ne" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  broadcast.828 = f32[6,1024,1,32768]{3,2,1,0} broadcast(constant.898), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/broadcast_in_dim[shape=(6, 8192, 1, 32768) broadcast_dimensions=(1, 2, 3)]" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.331 = f32[6,1024,1,32768]{3,2,1,0} select(compare.197, broadcast.828, multiply.239), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  broadcast.829 = f32[6,1024,1,32768]{3,2,1,0} broadcast(constant.899), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  compare.198 = pred[6,1024,1,32768]{3,2,1,0} compare(select.331, broadcast.829), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.332 = f32[6,1024,1,32768]{3,2,1,0} select(compare.198, broadcast.828, select.331), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  broadcast.832 = f32[6,1024,1,32768]{3,2,1,0} broadcast(constant.816), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  compare.199 = pred[6,1024,1,32768]{3,2,1,0} compare(select.332, broadcast.832), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.333 = f32[6,1024,1,32768]{3,2,1,0} select(compare.199, broadcast.828, select.332), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  multiply.240 = f32[6,1024,1,32768]{3,2,1,0} multiply(broadcast.826, select.333), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  broadcast.834 = f32[6,1024,1,32768]{3,2,1,0} broadcast(divide.18), dimensions={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  param.57 = f32[6,1024,1,32768]{3,2,1,0} parameter(41), sharding={devices=[1,8,1,1]<=[8]}
  multiply.241 = f32[6,1024,1,32768]{3,2,1,0} multiply(broadcast.834, param.57), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  add.352 = f32[6,1024,1,32768]{3,2,1,0} add(multiply.240, multiply.241), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  broadcast.835 = f32[6,1024,1,32768]{3,2,1,0} broadcast(subtract.94), dimensions={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  multiply.243 = f32[6,1024,1,32768]{3,2,1,0} multiply(select.333, select.333), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  multiply.244 = f32[6,1024,1,32768]{3,2,1,0} multiply(broadcast.835, multiply.243), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  broadcast.836 = f32[6,1024,1,32768]{3,2,1,0} broadcast(divide.19), dimensions={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  param.58 = f32[6,1024,1,32768]{3,2,1,0} parameter(53), sharding={devices=[1,8,1,1]<=[8]}
  multiply.245 = f32[6,1024,1,32768]{3,2,1,0} multiply(broadcast.836, param.58), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  add.354 = f32[6,1024,1,32768]{3,2,1,0} add(multiply.244, multiply.245), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  sqrt.14 = f32[6,1024,1,32768]{3,2,1,0} sqrt(add.354), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sqrt" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  broadcast.837 = f32[6,1024,1,32768]{3,2,1,0} broadcast(constant.910), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  add.355 = f32[6,1024,1,32768]{3,2,1,0} add(sqrt.14, broadcast.837), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  divide.47 = f32[6,1024,1,32768]{3,2,1,0} divide(add.352, add.355), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/div" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  broadcast.838 = f32[6,1024,1,32768]{3,2,1,0} broadcast(constant.887), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  multiply.246 = f32[6,1024,1,32768]{3,2,1,0} multiply(param.24, broadcast.838), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  add.356 = f32[6,1024,1,32768]{3,2,1,0} add(divide.47, multiply.246), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  multiply.247 = f32[6,1024,1,32768]{3,2,1,0} multiply(broadcast.825, add.356), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=712}
  broadcast.840 = f32[6,1024,1,32768]{3,2,1,0} broadcast(constant.789), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/broadcast_in_dim[shape=(6, 8192, 1, 32768) broadcast_dimensions=()]" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  select.334 = f32[6,1024,1,32768]{3,2,1,0} select(broadcast.824, multiply.247, broadcast.840), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  add.357 = f32[6,1024,1,32768]{3,2,1,0} add(param.24, select.334), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/paxml/paxml/learners.py" source_line=428}
  convert.145 = f32[6,8192]{1,0} convert(param.25), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/paxml/paxml/learners.py" source_line=428}
  convert.147 = f32[6,8192]{1,0} convert(get-tuple-element.158), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/paxml/paxml/learners.py" source_line=280}
  multiply.249 = f32[6,8192]{1,0} multiply(convert.147, broadcast.684), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=280}
  compare.200 = pred[6,8192]{1,0} compare(multiply.249, multiply.249), direction=NE, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(isnan)/ne" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.335 = f32[6,8192]{1,0} select(compare.200, broadcast.686, multiply.249), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  compare.201 = pred[6,8192]{1,0} compare(select.335, broadcast.688), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.336 = f32[6,8192]{1,0} select(compare.201, broadcast.686, select.335), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  compare.202 = pred[6,8192]{1,0} compare(select.336, broadcast.690), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.337 = f32[6,8192]{1,0} select(compare.202, broadcast.686, select.336), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  multiply.250 = f32[6,8192]{1,0} multiply(broadcast.683, select.337), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  param.59 = bf16[6,8192]{1,0} parameter(42), sharding={replicated}
  convert.148 = f32[6,8192]{1,0} convert(param.59), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  multiply.251 = f32[6,8192]{1,0} multiply(broadcast.692, convert.148), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  add.359 = f32[6,8192]{1,0} add(multiply.250, multiply.251), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  multiply.253 = f32[6,8192]{1,0} multiply(select.337, select.337), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  multiply.254 = f32[6,8192]{1,0} multiply(broadcast.696, multiply.253), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  param.60 = bf16[6,8192]{1,0} parameter(54), sharding={replicated}
  convert.150 = f32[6,8192]{1,0} convert(param.60), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  multiply.255 = f32[6,8192]{1,0} multiply(broadcast.697, convert.150), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  add.361 = f32[6,8192]{1,0} add(multiply.254, multiply.255), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  sqrt.15 = f32[6,8192]{1,0} sqrt(add.361), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sqrt" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  add.362 = f32[6,8192]{1,0} add(sqrt.15, broadcast.698), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  divide.50 = f32[6,8192]{1,0} divide(add.359, add.362), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/div" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  multiply.256 = bf16[6,8192]{1,0} multiply(param.25, broadcast.700), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  convert.151 = f32[6,8192]{1,0} convert(multiply.256), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  add.363 = f32[6,8192]{1,0} add(divide.50, convert.151), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  multiply.257 = f32[6,8192]{1,0} multiply(broadcast.681, add.363), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=712}
  select.338 = f32[6,8192]{1,0} select(broadcast.659, multiply.257, broadcast.702), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  add.364 = f32[6,8192]{1,0} add(convert.145, select.338), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/paxml/paxml/learners.py" source_line=428}
  broadcast.853 = pred[6,32768,1024]{2,1,0} broadcast(is-finite.1), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/broadcast_in_dim[shape=(6, 32768, 8192) broadcast_dimensions=()]" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  broadcast.854 = f32[6,32768,1024]{2,1,0} broadcast(multiply.147), dimensions={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=712}
  broadcast.855 = f32[6,32768,1024]{2,1,0} broadcast(subtract.91), dimensions={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  broadcast.856 = f32[6,32768,1024]{2,1,0} broadcast(minimum.4), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=280}
  multiply.259 = f32[6,32768,1024]{2,1,0} multiply(convert.95, broadcast.856), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/paxml/paxml/learners.py" source_line=280}
  compare.203 = pred[6,32768,1024]{2,1,0} compare(multiply.259, multiply.259), direction=NE, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(isnan)/ne" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  broadcast.857 = f32[6,32768,1024]{2,1,0} broadcast(constant.898), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/broadcast_in_dim[shape=(6, 32768, 8192) broadcast_dimensions=(1, 2)]" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.339 = f32[6,32768,1024]{2,1,0} select(compare.203, broadcast.857, multiply.259), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  broadcast.858 = f32[6,32768,1024]{2,1,0} broadcast(constant.899), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  compare.204 = pred[6,32768,1024]{2,1,0} compare(select.339, broadcast.858), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.340 = f32[6,32768,1024]{2,1,0} select(compare.204, broadcast.857, select.339), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  broadcast.861 = f32[6,32768,1024]{2,1,0} broadcast(constant.816), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  compare.205 = pred[6,32768,1024]{2,1,0} compare(select.340, broadcast.861), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/eq" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  select.341 = f32[6,32768,1024]{2,1,0} select(compare.205, broadcast.857, select.340), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/vmap(jit(nan_to_num))/jit(_where)/select_n" source_file="/opt/praxis/praxis/optimizers.py" source_line=313}
  multiply.260 = f32[6,32768,1024]{2,1,0} multiply(broadcast.855, select.341), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  broadcast.863 = f32[6,32768,1024]{2,1,0} broadcast(divide.18), dimensions={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  param.61 = f32[6,32768,1024]{2,1,0} parameter(43), sharding={devices=[1,1,8]<=[8]}
  multiply.261 = f32[6,32768,1024]{2,1,0} multiply(broadcast.863, param.61), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  add.366 = f32[6,32768,1024]{2,1,0} add(multiply.260, multiply.261), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=344}
  broadcast.864 = f32[6,32768,1024]{2,1,0} broadcast(subtract.94), dimensions={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  multiply.263 = f32[6,32768,1024]{2,1,0} multiply(select.341, select.341), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  multiply.264 = f32[6,32768,1024]{2,1,0} multiply(broadcast.864, multiply.263), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  broadcast.865 = f32[6,32768,1024]{2,1,0} broadcast(divide.19), dimensions={0}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  param.62 = f32[6,32768,1024]{2,1,0} parameter(55), sharding={devices=[1,1,8]<=[8]}
  multiply.265 = f32[6,32768,1024]{2,1,0} multiply(broadcast.865, param.62), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  add.368 = f32[6,32768,1024]{2,1,0} add(multiply.264, multiply.265), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=345}
  sqrt.16 = f32[6,32768,1024]{2,1,0} sqrt(add.368), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/sqrt" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  broadcast.866 = f32[6,32768,1024]{2,1,0} broadcast(constant.910), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  add.369 = f32[6,32768,1024]{2,1,0} add(sqrt.16, broadcast.866), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  divide.53 = f32[6,32768,1024]{2,1,0} divide(add.366, add.369), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/div" source_file="/opt/praxis/praxis/optimizers.py" source_line=701}
  broadcast.867 = f32[6,32768,1024]{2,1,0} broadcast(constant.887), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  multiply.266 = f32[6,32768,1024]{2,1,0} multiply(param.26, broadcast.867), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  add.370 = f32[6,32768,1024]{2,1,0} add(divide.53, multiply.266), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=708}
  multiply.267 = f32[6,32768,1024]{2,1,0} multiply(broadcast.854, add.370), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/mul" source_file="/opt/praxis/praxis/optimizers.py" source_line=712}
  broadcast.868 = f32[6,32768,1024]{2,1,0} broadcast(constant.789), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/broadcast_in_dim[shape=(6, 32768, 8192) broadcast_dimensions=()]" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  select.342 = f32[6,32768,1024]{2,1,0} select(broadcast.853, multiply.267, broadcast.868), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  add.371 = f32[6,32768,1024]{2,1,0} add(param.26, select.342), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/paxml/paxml/learners.py" source_line=428}
  param.63 = s32[] parameter(17), sharding={replicated}
  add.373 = s32[] add(param.63, constant.798), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=554}
  select.343 = s32[] select(is-finite.1, add.373, param.63), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=362}
  param.64 = s32[] parameter(18), sharding={replicated}
  add.374 = s32[] add(param.64, constant.798), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=554}
  select.344 = s32[] select(is-finite.1, add.374, param.64), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=362}
  add.375 = s32[] add(param.29, constant.798), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=714}
  select.345 = s32[] select(is-finite.1, add.375, param.29), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=362}
  select.346 = f32[8192]{0} select(broadcast.602, add.255, param.30), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  select.347 = f32[8192]{0} select(broadcast.602, add.262, param.32), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  select.348 = f32[256,8192]{1,0} select(broadcast.629, add.269, param.34), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  select.349 = f32[1024,50304]{1,0} select(broadcast.644, add.276, param.36), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  select.350 = f32[8192]{0} select(broadcast.602, add.257, param.31), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  select.351 = f32[8192]{0} select(broadcast.602, add.264, param.33), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  select.352 = f32[256,8192]{1,0} select(broadcast.629, add.271, param.35), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  select.353 = f32[1024,50304]{1,0} select(broadcast.644, add.278, param.37), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  param.65 = s32[] parameter(28), sharding={replicated}
  add.376 = s32[] add(param.65, constant.798), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=616}
  select.354 = s32[] select(is-finite.1, add.376, param.65), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=362}
  broadcast.877 = pred[6]{0} broadcast(is-finite.1), dimensions={}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/broadcast_in_dim[shape=(6,) broadcast_dimensions=()]" source_file="/opt/paxml/paxml/learners.py" source_line=362}
  param.66 = s32[6]{0} parameter(29), sharding={replicated}
  broadcast.878 = s32[6]{0} broadcast(constant.798), dimensions={}
  add.377 = s32[6]{0} add(param.66, broadcast.878), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=554}
  select.355 = s32[6]{0} select(broadcast.877, add.377, param.66), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=362}
  param.67 = s32[6]{0} parameter(30), sharding={replicated}
  add.378 = s32[6]{0} add(param.67, broadcast.878), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=554}
  select.356 = s32[6]{0} select(broadcast.877, add.378, param.67), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=362}
  add.379 = s32[6]{0} add(param.38, broadcast.878), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=714}
  select.357 = s32[6]{0} select(broadcast.877, add.379, param.38), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=362}
  select.358 = f32[6,8192]{1,0} select(broadcast.659, add.288, convert.113), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=362}
  select.359 = f32[6,8192,1024]{2,1,0} select(broadcast.703, add.295, param.41), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  select.360 = f32[6,3,8192]{2,1,0} select(broadcast.720, add.302, convert.122), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=362}
  select.361 = f32[6,1024,3,8192]{3,2,1,0} select(broadcast.739, add.309, param.45), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  select.362 = f32[6,8192]{1,0} select(broadcast.659, add.317, param.47), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  select.363 = f32[6,8192]{1,0} select(broadcast.659, add.324, param.49), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  select.364 = f32[6,8192]{1,0} select(broadcast.659, add.331, param.51), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  select.365 = f32[6,8192]{1,0} select(broadcast.659, add.338, param.53), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  select.366 = f32[6,1,32768]{2,1,0} select(broadcast.808, add.345, convert.139), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=362}
  select.367 = f32[6,1024,1,32768]{3,2,1,0} select(broadcast.824, add.352, param.57), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  select.368 = f32[6,8192]{1,0} select(broadcast.659, add.359, convert.148), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=362}
  select.369 = f32[6,32768,1024]{2,1,0} select(broadcast.853, add.366, param.61), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  select.370 = f32[6,8192]{1,0} select(broadcast.659, add.290, convert.115), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=362}
  select.371 = f32[6,8192,1024]{2,1,0} select(broadcast.703, add.297, param.42), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  select.372 = f32[6,3,8192]{2,1,0} select(broadcast.720, add.304, convert.124), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=362}
  select.373 = f32[6,1024,3,8192]{3,2,1,0} select(broadcast.739, add.311, param.46), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  select.374 = f32[6,8192]{1,0} select(broadcast.659, add.319, param.48), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  select.375 = f32[6,8192]{1,0} select(broadcast.659, add.326, param.50), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  select.376 = f32[6,8192]{1,0} select(broadcast.659, add.333, param.52), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  select.377 = f32[6,8192]{1,0} select(broadcast.659, add.340, param.54), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  select.378 = f32[6,1,32768]{2,1,0} select(broadcast.808, add.347, convert.141), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=362}
  select.379 = f32[6,1024,1,32768]{3,2,1,0} select(broadcast.824, add.354, param.58), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  select.380 = f32[6,8192]{1,0} select(broadcast.659, add.361, convert.150), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=362}
  select.381 = f32[6,32768,1024]{2,1,0} select(broadcast.853, add.368, param.62), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=354}
  param.68 = s32[6]{0} parameter(56), sharding={replicated}
  add.380 = s32[6]{0} add(param.68, broadcast.878), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/add" source_file="/opt/praxis/praxis/optimizers.py" source_line=616}
  select.382 = s32[6]{0} select(broadcast.877, add.380, param.68), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jit(_where)/select_n" source_file="/opt/paxml/paxml/learners.py" source_line=362}
  log.2 = f32[1,2048]{1,0} log(reduce.41), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/jit(log_softmax)/log" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=301}
  reshape.955 = f32[2048]{0} reshape(log.2)
  broadcast.910 = f32[1,2048,50304]{2,1,0} broadcast(reshape.955), dimensions={1}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/jit(log_softmax)/sub" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=301}
  subtract.161 = f32[1,2048,50304]{2,1,0} subtract(subtract.64, broadcast.910), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/jit(log_softmax)/sub" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=301}
  select.383 = f32[1,2048,50304]{2,1,0} select(compare.134, subtract.161, broadcast.553), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/mul" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=323}
  reduce.63 = f32[1,2048]{1,0} reduce(select.383, constant.789), dimensions={2}, to_apply=region_10.798, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/reduce_sum[axes=(2,)]" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=322}
  negate.44 = f32[1,2048]{1,0} negate(reduce.63), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/neg" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=322}
  multiply.268 = f32[1,2048]{1,0} multiply(negate.44, convert.41), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/mul" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=333}
  reduce.64 = f32[] reduce(multiply.268, constant.789), dimensions={0,1}, to_apply=region_10.798, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/reduce_sum[axes=(0, 1, 2)]" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=332}
  all-reduce.28 = f32[] all-reduce(reduce.64), channel_id=40, replica_groups={{0}}, to_apply=region_10.798, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/reduce_sum[axes=(0, 1, 2)]" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=332}
  divide.54 = f32[] divide(all-reduce.28, add.224), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/div" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=354}
  get-tuple-element.160 = s32[1,2048]{1,0} get-tuple-element(reduce.40), index=1, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/reduce[computation=<function _compute_argminmax.<locals>.reducer_fn at 0x7ffbf96b15a0> consts=() dimensions=(2,)]" source_file="/opt/praxis/praxis/layers/embedding_softmax.py" source_line=328}
  compare.206 = pred[1,2048]{1,0} compare(param.9, get-tuple-element.160), direction=EQ, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_loss/eq" source_file="/opt/praxis/praxis/layers/models.py" source_line=131}
  broadcast.913 = bf16[1,2048]{1,0} broadcast(constant.794), dimensions={}
  select.384 = bf16[1,2048]{1,0} select(compare.206, multiply.64, broadcast.913), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_loss/mul" source_file="/opt/praxis/praxis/layers/models.py" source_line=131}
  convert.162 = f32[1,2048]{1,0} convert(select.384), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_loss/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/praxis/praxis/layers/models.py" source_line=131}
  reduce.65 = f32[] reduce(convert.162, constant.789), dimensions={0,1}, to_apply=region_10.798, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_loss/reduce_sum[axes=(0, 1)]" source_file="/opt/praxis/praxis/layers/models.py" source_line=131}
  all-reduce.29 = f32[] all-reduce(reduce.65), channel_id=41, replica_groups={{0}}, to_apply=region_10.798, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_loss/reduce_sum[axes=(0, 1)]" source_file="/opt/praxis/praxis/layers/models.py" source_line=131}
  convert.163 = bf16[] convert(all-reduce.29), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_loss/convert_element_type[new_dtype=bfloat16 weak_type=False]" source_file="/opt/praxis/praxis/layers/models.py" source_line=131}
  convert.164 = f32[] convert(convert.163), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_loss/convert_element_type[new_dtype=float32 weak_type=False]" source_file="/opt/praxis/praxis/layers/models.py" source_line=131}
  maximum.0 = f32[] maximum(all-reduce.16, constant.788), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_loss/max" source_file="/opt/praxis/praxis/layers/models.py" source_line=131}
  divide.55 = f32[] divide(convert.164, maximum.0), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_loss/div" source_file="/opt/praxis/praxis/layers/models.py" source_line=131}
  all-gather.11 = bf16[8]{0} all-gather(convert.40), channel_id=42, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={0}, use_global_device_ids=true
  all-gather.12 = s32[8,2048]{1,0} all-gather(param.9), channel_id=43, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={0}, use_global_device_ids=true
  reduce.66 = f32[1]{0} reduce(multiply.268, constant.789), dimensions={1}, to_apply=region_10.798, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/reduce_sum[axes=(1,)]" source_file="/opt/praxis/praxis/layers/transformer_models.py" source_line=616}
  negate.45 = f32[1]{0} negate(reduce.66), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_loss/neg" source_file="/opt/praxis/praxis/layers/models.py" source_line=172}
  all-gather.13 = f32[8]{0} all-gather(negate.45), channel_id=44, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={0}, use_global_device_ids=true
  ROOT tuple.8 = (u32[], f32[8192]{0}, f32[8192]{0}, f32[256,8192]{1,0}, f32[1024,50304]{1,0}, /*index=5*/f32[6,8192]{1,0}, f32[6,8192,1024]{2,1,0}, f32[6,3,8192]{2,1,0}, f32[6,1024,3,8192]{3,2,1,0}, f32[6,8192]{1,0}, /*index=10*/f32[6,8192]{1,0}, f32[6,8192]{1,0}, f32[6,8192]{1,0}, f32[6,1,32768]{2,1,0}, f32[6,1024,1,32768]{3,2,1,0}, /*index=15*/f32[6,8192]{1,0}, f32[6,32768,1024]{2,1,0}, s32[], s32[], s32[], /*index=20*/f32[8192]{0}, f32[8192]{0}, f32[256,8192]{1,0}, f32[1024,50304]{1,0}, f32[8192]{0}, /*index=25*/f32[8192]{0}, f32[256,8192]{1,0}, f32[1024,50304]{1,0}, s32[], s32[6]{0}, /*index=30*/s32[6]{0}, s32[6]{0}, f32[6,8192]{1,0}, f32[6,8192,1024]{2,1,0}, f32[6,3,8192]{2,1,0}, /*index=35*/f32[6,1024,3,8192]{3,2,1,0}, f32[6,8192]{1,0}, f32[6,8192]{1,0}, f32[6,8192]{1,0}, f32[6,8192]{1,0}, /*index=40*/f32[6,1,32768]{2,1,0}, f32[6,1024,1,32768]{3,2,1,0}, f32[6,8192]{1,0}, f32[6,32768,1024]{2,1,0}, f32[6,8192]{1,0}, /*index=45*/f32[6,8192,1024]{2,1,0}, f32[6,3,8192]{2,1,0}, f32[6,1024,3,8192]{3,2,1,0}, f32[6,8192]{1,0}, f32[6,8192]{1,0}, /*index=50*/f32[6,8192]{1,0}, f32[6,8192]{1,0}, f32[6,1,32768]{2,1,0}, f32[6,1024,1,32768]{3,2,1,0}, f32[6,8192]{1,0}, /*index=55*/f32[6,32768,1024]{2,1,0}, s32[6]{0}, f32[], bf16[], bf16[], /*index=60*/f32[], f32[], f32[], f32[], f32[], /*index=65*/f32[], f32[], f32[], f32[], f32[], /*index=70*/bf16[8]{0}, s32[8,2048]{1,0}, f32[8]{0}) tuple(add.223, add.260, add.267, add.274, add.281, /*index=5*/add.293, add.300, add.307, add.315, add.322, /*index=10*/add.329, add.336, add.343, add.350, add.357, /*index=15*/add.364, add.371, select.343, select.344, select.345, /*index=20*/select.346, select.347, select.348, select.349, select.350, /*index=25*/select.351, select.352, select.353, select.354, select.355, /*index=30*/select.356, select.357, select.358, select.359, select.360, /*index=35*/select.361, select.362, select.363, select.364, select.365, /*index=40*/select.366, select.367, select.368, select.369, select.370, /*index=45*/select.371, select.372, select.373, select.374, select.375, /*index=50*/select.376, select.377, select.378, select.379, select.380, /*index=55*/select.381, select.382, divide.54, constant.794, constant.799, /*index=60*/divide.54, all-reduce.16, divide.55, all-reduce.16, divide.54, /*index=65*/all-reduce.16, all-reduce.16, constant.788, divide.54, all-reduce.16, /*index=70*/all-gather.11, all-gather.12, all-gather.13)
  partition-id.4 = u32[] partition-id(), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/softmax.get_logits/logits_ffn/linear/einsum/...y,yz->...z/transpose[permutation=(1, 0)]" source_file="/opt/praxis/praxis/layers/base_ops.py" source_line=42}
  convert.167 = s32[] convert(partition-id.4), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/lm._prepare_input/position_emb/einsum/...y,yz->...z/transpose[permutation=(1, 0)]" source_file="/opt/praxis/praxis/layers/base_ops.py" source_line=42}
  constant.1112 = s32[] constant(256), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/lm._prepare_input/position_emb/einsum/...y,yz->...z/transpose[permutation=(1, 0)]" source_file="/opt/praxis/praxis/layers/base_ops.py" source_line=42}
  multiply.275 = s32[] multiply(convert.167, constant.1112), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/lm._prepare_input/position_emb/einsum/...y,yz->...z/transpose[permutation=(1, 0)]" source_file="/opt/praxis/praxis/layers/base_ops.py" source_line=42}
  constant.1115 = s32[] constant(1024), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/softmax.get_logits/logits_ffn/linear/einsum/...y,yz->...z/transpose[permutation=(1, 0)]" source_file="/opt/praxis/praxis/layers/base_ops.py" source_line=42}
  multiply.277 = s32[] multiply(convert.167, constant.1115), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/lm.compute_loss/softmax/softmax.get_logits/logits_ffn/linear/einsum/...y,yz->...z/transpose[permutation=(1, 0)]" source_file="/opt/praxis/praxis/layers/base_ops.py" source_line=42}
  dynamic-slice.75 = bf16[1,1024,3,8192]{3,2,1,0} dynamic-slice(get-tuple-element.203, select.394, constant.1146, constant.1146, constant.1146), dynamic_slice_sizes={1,1024,3,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192, 3, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1020 = bf16[1024,3,8192]{2,1,0} reshape(dynamic-slice.75), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-slice.77 = bf16[1,8192,1024]{2,1,0} dynamic-slice(get-tuple-element.209, select.394, constant.1146, constant.1146), dynamic_slice_sizes={1,8192,1024}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1027 = bf16[8192,1024]{1,0} reshape(dynamic-slice.77), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-slice.81 = bf16[1,1024,1,32768]{3,2,1,0} dynamic-slice(get-tuple-element.215, select.394, constant.1146, constant.1146, constant.1146), dynamic_slice_sizes={1,1024,1,32768}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192, 1, 32768)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1033 = bf16[1024,32768]{1,0} reshape(dynamic-slice.81)
  dynamic-slice.83 = bf16[1,32768,1024]{2,1,0} dynamic-slice(get-tuple-element.218, select.394, constant.1146, constant.1146), dynamic_slice_sizes={1,32768,1024}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 32768, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1037 = bf16[32768,1024]{1,0} reshape(dynamic-slice.83), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/jvp(xformer_lm.apply)/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-slice.110 = bf16[1,32768,1024]{2,1,0} dynamic-slice(get-tuple-element.291, select.405, constant.1199, constant.1199), dynamic_slice_sizes={1,32768,1024}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 32768, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1113 = bf16[32768,1024]{1,0} reshape(dynamic-slice.110), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-slice.111 = bf16[1,1024,1,32768]{3,2,1,0} dynamic-slice(get-tuple-element.293, select.405, constant.1199, constant.1199, constant.1199), dynamic_slice_sizes={1,1024,1,32768}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192, 1, 32768)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1117 = bf16[1024,32768]{1,0} reshape(dynamic-slice.111)
  dynamic-slice.119 = bf16[1,8192,1024]{2,1,0} dynamic-slice(get-tuple-element.309, select.405, constant.1199, constant.1199), dynamic_slice_sizes={1,8192,1024}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1129 = bf16[8192,1024]{1,0} reshape(dynamic-slice.119), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  dynamic-slice.120 = bf16[1,1024,3,8192]{3,2,1,0} dynamic-slice(get-tuple-element.312, select.405, constant.1199, constant.1199, constant.1199), dynamic_slice_sizes={1,1024,3,8192}, metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/dynamic_slice[slice_sizes=(1, 8192, 3, 8192)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  reshape.1133 = bf16[1024,3,8192]{2,1,0} reshape(dynamic-slice.120), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/squeeze[dimensions=(0,)]" source_file="/opt/flax/flax/core/axes_scan.py" source_line=163}
  partition-id.6 = u32[] partition-id(), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/out/transpose[permutation=(1, 0)]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=451}
  convert.180 = s32[] convert(partition-id.6), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/out/transpose[permutation=(1, 0)]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=451}
  constant.1213 = s32[] constant(1024), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/out/transpose[permutation=(1, 0)]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=451}
  multiply.321 = s32[] multiply(convert.180, constant.1213), metadata={op_name="pjit(_wrapped_step_fn)/jit(main)/transpose(jvp(xformer_lm.apply))/xformer_lm/xformer_lm.compute_predictions/lm/transformer/repeat/repeat.call_with_custom_method/while/body/remat/sub.body_fn/sub/x_layers_0/transformerlayer/transformerlayer._call_with_boxed_params_init/cld/attention/out/transpose[permutation=(1, 0)]" source_file="/opt/transformer-engine/transformer_engine/jax/flax/module.py" source_line=451}
} // main.4025_spmd

)";
  LOG(ERROR) << "#####parsing hlo";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  LOG(ERROR) << "#####parsed hlo";

  CollectivePipeliner::Config config{
      /*level_to_operate_on=*/0,
      /*max_pipelining_per_loop=*/INT64_MAX,
      /*last_run=*/true,
      /*pipeline_use_tree=*/false,
      /*process_different_sized_ops=*/true,
      /*pipelining_direction=*/
      CollectivePipeliner::PipeliningDirection::kForward,
      /*should_process=*/HloPredicateIsOp<HloOpcode::kReduceScatter>};
  CollectivePipeliner collect_pipe(config);
  ASSERT_IS_OK(collect_pipe.Run(module.get()).status());
  LOG(ERROR) << "###after pipeliner " << module->ToString();
  LOG(ERROR) << "###print done ";

  AlgebraicSimplifierOptions layout_insensitive_algsimp_opts({},
                                                             ConvIsLowerable);

  // GPU only supports canonical convolutions.
  layout_insensitive_algsimp_opts.set_supports_non_canonical_dots(false);

  // "slow" minmax means we propagate nan.
  layout_insensitive_algsimp_opts.set_minmax_propagate_nan(true);

  // Always simplify reduce(transpose(x)) and reduce(reshape(x)), even when
  // the transpose/reshape has multiple users.  This helps int8 models, which
  // tend to have lots of transpose+reshape's (converting between NCHW and
  // NCHW_VECT_C).  Without this, those reshape+transposes can get materialized
  // out, which is really bad for perf.
  layout_insensitive_algsimp_opts
      .set_unconditionally_simplify_reduce_of_transpose_or_reshape(true);

  layout_insensitive_algsimp_opts
      .set_enable_unconditional_reduce_of_concat_replacement(false);

  AlgebraicSimplifier alg_simp(layout_insensitive_algsimp_opts);
  HloDCE dce;
  LOG(ERROR) << "#####running dce";
  ASSERT_IS_OK(dce.Run(module.get()).status());

  ASSERT_IS_OK(alg_simp.Run(module.get()).status());
}

}  // namespace
}  // namespace xla
