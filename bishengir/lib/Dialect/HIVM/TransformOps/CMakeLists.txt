add_bishengir_dialect_library(BiShengIRHIVMTransformOps
  HIVMTransformOps.cpp
  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Dialect/HIVM/TransformOps

  DEPENDS
  BiShengIRHIVMTransformOpsIncGen

  LINK_LIBS PUBLIC
  BiShengIRHIVMDialect
  BiShengIRMathExtDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRTensorDialect
  MLIRTransformDialect
  MLIRTransformDialectUtils
  MLIRVectorDialect
  )

# Keep the transaction runtime independent of vectorization and recipe policy.
add_bishengir_library(BiShengIRHIVMTransactions
  Transaction.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRParser
  MLIRBytecodeReader
  MLIRBytecodeWriter
)
