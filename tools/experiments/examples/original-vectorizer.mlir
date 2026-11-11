module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.consumed}) {
    %hfusion = transform.apply_registered_pass "convert-hivm-to-hfusion" to %root : (!transform.any_op) -> !transform.any_op
    %vectors = transform.apply_registered_pass "hfusion-auto-vectorize" to %hfusion : (!transform.any_op) -> !transform.any_op
    %outlined = transform.apply_registered_pass "outline-vector-function" to %vectors : (!transform.any_op) -> !transform.any_op
    %canonical = transform.apply_registered_pass "canonicalize" to %outlined : (!transform.any_op) -> !transform.any_op
    %hivm = transform.apply_registered_pass "convert-hfusion-to-hivm" to %canonical : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
