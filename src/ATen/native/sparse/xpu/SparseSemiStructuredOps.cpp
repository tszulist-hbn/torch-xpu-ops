/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/ATen.h>
#include <ATen/native/sparse/SparseStubs.h>

namespace at::native {

std::tuple<Tensor, Tensor> _to_sparse_semi_structured_xpu(const Tensor& dense) {
  // MVP implementation: CPU fallback
  // Move to CPU, perform conversion using upstream CUDA path, move back to XPU
  Tensor dense_cpu = dense.cpu();

  // Call the CUDA implementation on CPU
  auto [sparse_cpu, meta_cpu] = at::_to_sparse_semi_structured(dense_cpu);

  // Move results back to XPU device
  Tensor sparse_xpu = sparse_cpu.to(dense.device());
  Tensor meta_xpu = meta_cpu.to(dense.device());

  return std::make_tuple(sparse_xpu, meta_xpu);
}

} // namespace at::native
