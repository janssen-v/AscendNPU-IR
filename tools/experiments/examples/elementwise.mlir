module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.consumed}) {
    %vectorized = transform.hivm.vectorize_elementwise %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %vectorized {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.yield
  }
}
