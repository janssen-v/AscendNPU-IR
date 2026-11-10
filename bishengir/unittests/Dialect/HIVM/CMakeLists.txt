add_bishengir_unittest(BiShengIRHIVMTransactionTests
  TransactionTest.cpp
)

target_link_libraries(BiShengIRHIVMTransactionTests PRIVATE
  BiShengIRHIVMTransactions
  MLIRAsmParser
  MLIRFuncDialect
)

add_dependencies(check-bishengir run-BiShengIRHIVMTransactionTests)
