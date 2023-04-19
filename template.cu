#include <cstdio>
#include <cstdlib>
#include "helper.hpp"

static int eval() {
  REQUIRE(10 == 10);
  REQUIRE(3 == 5);
}

TEST_CASE("Group 10", "[gten]") {
  eval();
}
