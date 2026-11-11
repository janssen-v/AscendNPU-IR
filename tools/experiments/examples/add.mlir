module {
  func.func @add(%a: tensor<64xf32>, %b: tensor<64xf32>, %out: memref<64xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %e = tensor.empty() : tensor<64xf32>
    %r = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%a, %b : tensor<64xf32>, tensor<64xf32>) outs(%e : tensor<64xf32>) -> tensor<64xf32>
    hivm.hir.store ins(%r : tensor<64xf32>) outs(%out : memref<64xf32>)
    return
  }
}

