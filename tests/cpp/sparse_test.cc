#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/tvm.h>

TEST(Tensor, Basic) {
  using namespace tvm;
  Var m("m"), n("n"), l("l");

  Tensor A = placeholder({m, l}, Float(32), "A", kCSRStorage);
  Tensor B = placeholder({n, l}, Float(32), "B", kCSRStorage);
  CHECK(A->stype == tvm::StorageType(kCSRStorage));
  // CHECK((A->stype)->type_code_ == kCSRStorage);

  auto C = compute({m, n}, [&](Var i, Var j) {
      return A[i][j];
    }, "C");

  Tensor::Slice x = A[n];
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
