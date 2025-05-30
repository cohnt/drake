#include "drake/solvers/csdp_solver.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/solvers/test/csdp_test_examples.h"
#include "drake/solvers/test/linear_program_examples.h"
#include "drake/solvers/test/second_order_cone_program_examples.h"
#include "drake/solvers/test/semidefinite_program_examples.h"
#include "drake/solvers/test/sos_examples.h"

namespace drake {
namespace solvers {
namespace test {
GTEST_TEST(TestSemidefiniteProgram, SolveSDPwithOverlappingVariables) {
  CsdpSolver solver;
  if (solver.available()) {
    test::SolveSDPwithOverlappingVariables(solver, 1E-7);
  }
}

TEST_F(CsdpDocExample, Solve) {
  CsdpSolver solver;
  if (solver.available()) {
    MathematicalProgramResult result;
    solver.Solve(*prog_, {}, {}, &result);
    EXPECT_TRUE(result.is_success());
    const double tol = 5E-7;
    EXPECT_NEAR(result.get_optimal_cost(), -2.75, tol);
    Eigen::Matrix2d X1_expected;
    X1_expected << 0.125, 0.125, 0.125, 0.125;
    EXPECT_TRUE(CompareMatrices(result.GetSolution(X1_), X1_expected, tol));
    Eigen::Matrix3d X2_expected;
    X2_expected.setZero();
    X2_expected(0, 0) = 2.0 / 3;
    EXPECT_TRUE(CompareMatrices(result.GetSolution(X2_), X2_expected, tol));
    EXPECT_TRUE(
        CompareMatrices(result.GetSolution(y_), Eigen::Vector2d::Zero(), tol));
    const CsdpSolverDetails& solver_details =
        result.get_solver_details<CsdpSolver>();
    EXPECT_EQ(solver_details.return_code, 0);
    EXPECT_NEAR(solver_details.primal_objective, 2.75, tol);
    EXPECT_NEAR(solver_details.dual_objective, 2.75, tol);
    EXPECT_TRUE(
        CompareMatrices(solver_details.y_val, Eigen::Vector2d(0.75, 1), tol));
    Eigen::Matrix<double, 7, 7> Z_expected;
    Z_expected.setZero();
    Z_expected(0, 0) = 0.25;
    Z_expected(0, 1) = -0.25;
    Z_expected(1, 0) = -0.25;
    Z_expected(1, 1) = 0.25;
    Z_expected(3, 3) = 2.0;
    Z_expected(4, 4) = 2.0;
    Z_expected(5, 5) = 0.75;
    Z_expected(6, 6) = 1.0;
    EXPECT_TRUE(CompareMatrices(Eigen::MatrixXd(solver_details.Z_val),
                                Z_expected, tol));
    // Now add an empty constraint to the program and solve it again, there
    // should be no error.
    prog_->AddLinearEqualityConstraint(Eigen::RowVector2d(0, 0), 0, y_);
    solver.Solve(*prog_, {}, {}, &result);
    EXPECT_TRUE(result.is_success());
    EXPECT_NEAR(result.get_optimal_cost(), -2.75, tol);
  }
}

TEST_F(TrivialSDP1, Solve) {
  CsdpSolver solver;
  if (solver.available()) {
    MathematicalProgramResult result;
    solver.Solve(*prog_, {}, {}, &result);
    EXPECT_TRUE(result.is_success());
    const double tol = 1E-7;
    EXPECT_NEAR(result.get_optimal_cost(), -2.0 / 3.0, tol);
    const Eigen::Matrix3d X1_expected = Eigen::Matrix3d::Constant(1.0 / 3);
    EXPECT_TRUE(CompareMatrices(result.GetSolution(X1_), X1_expected, tol));
  }
}

GTEST_TEST(TestSemidefiniteProgram, TestTrivial2x2SDP) {
  CsdpSolver solver;
  if (solver.available()) {
    TestTrivial2x2SDP(solver, 1E-5, /*check_dual=*/false, /*dual_tol=*/1E-5);
  }
}

GTEST_TEST(TestSemidefiniteProgram, Test1x1with3x3SDP) {
  CsdpSolver solver;
  if (solver.available()) {
    Test1x1with3x3SDP(solver, 1E-4, /*check_dual=*/false, /*dual_tol=*/1E-5);
  }
}

GTEST_TEST(TestSemidefiniteProgram, Test2x2with3x3SDP) {
  CsdpSolver solver;
  if (solver.available()) {
    Test2x2with3x3SDP(solver, 1E-2, /*check_dual=*/false, /*dual_tol=*/1E-5);
  }
}

GTEST_TEST(TestSemidefiniteProgram, TestTrivial1x1LMI) {
  CsdpSolver solver;
  if (solver.available()) {
    TestTrivial1x1LMI(solver, 1E-5, /*check_dual=*/false, /*dual_tol=*/1E-7);
  }
}

GTEST_TEST(TestSemidefiniteProgram, Test2X2LMI) {
  CsdpSolver solver;
  if (solver.available()) {
    Test2x2LMI(solver, 1E-7, /*check_dual=*/false, /*dual_tol=*/1E-7);
  }
}

std::vector<RemoveFreeVariableMethod> GetRemoveFreeVariableMethods() {
  return {RemoveFreeVariableMethod::kNullspace,
          RemoveFreeVariableMethod::kTwoSlackVariables,
          RemoveFreeVariableMethod::kLorentzConeSlack};
}

TEST_F(LinearProgramBoundingBox1, Solve) {
  for (auto method : GetRemoveFreeVariableMethods()) {
    SCOPED_TRACE(fmt::format("method = {}", method));
    CsdpSolver solver;
    if (solver.available()) {
      SolverOptions solver_options;
      solver_options.SetOption(solver.id(), "drake::RemoveFreeVariableMethod",
                               static_cast<int>(method));
      MathematicalProgramResult result;
      solver.Solve(*prog_, {}, solver_options, &result);
      EXPECT_TRUE(result.is_success());
      const double tol = 2E-7;
      EXPECT_NEAR(result.get_optimal_cost(), -43, 2 * tol);
      Eigen::Matrix<double, 7, 1> x_expected;
      x_expected << 0, 5, -1, 10, 5, 0, 1;
      EXPECT_TRUE(
          CompareMatrices(result.GetSolution(x_).head<7>(), x_expected, tol));
    }
  }
}

TEST_F(CsdpLinearProgram2, Solve) {
  for (auto method : GetRemoveFreeVariableMethods()) {
    SCOPED_TRACE(fmt::format("method = {}", method));
    CsdpSolver solver;
    if (solver.available()) {
      SolverOptions solver_options;
      solver_options.SetOption(solver.id(), "drake::RemoveFreeVariableMethod",
                               static_cast<int>(method));
      MathematicalProgramResult result;
      solver.Solve(*prog_, {}, solver_options, &result);
      EXPECT_TRUE(result.is_success());
      const double tol = 1E-7;
      EXPECT_NEAR(result.get_optimal_cost(), 28.0 / 13, tol);
      EXPECT_TRUE(
          CompareMatrices(result.GetSolution(x_),
                          Eigen::Vector3d(5.0 / 13, -2.0 / 13, 9.0 / 13), tol));
    }
  }
}

TEST_F(CsdpLinearProgram3, Solve) {
  for (auto method : GetRemoveFreeVariableMethods()) {
    SCOPED_TRACE(fmt::format("method = {}", method));
    CsdpSolver solver;
    if (solver.available()) {
      SolverOptions solver_options;
      solver_options.SetOption(solver.id(), "drake::RemoveFreeVariableMethod",
                               static_cast<int>(method));
      MathematicalProgramResult result;
      solver.Solve(*prog_, {}, solver_options, &result);
      EXPECT_TRUE(result.is_success());
      const double tol = 1E-7;
      EXPECT_NEAR(result.get_optimal_cost(), -121.0 / 9, tol);
      EXPECT_TRUE(CompareMatrices(result.GetSolution(x_),
                                  Eigen::Vector3d(10, -2.0 / 3, -17.0 / 9),
                                  tol));
    }
  }
}

TEST_F(DuplicatedVariableLinearProgramTest1, Test) {
  for (auto method : GetRemoveFreeVariableMethods()) {
    SCOPED_TRACE(fmt::format("method = {}", method));
    CsdpSolver solver;
    if (solver.is_available()) {
      SolverOptions solver_options;
      solver_options.SetOption(solver.id(), "drake::RemoveFreeVariableMethod",
                               static_cast<int>(method));
      CheckSolution(solver, solver_options);
    }
  }
}

TEST_F(TrivialSDP2, Solve) {
  for (auto method : GetRemoveFreeVariableMethods()) {
    SCOPED_TRACE(fmt::format("method = {}", method));
    CsdpSolver solver;
    if (solver.available()) {
      SolverOptions solver_options;
      solver_options.SetOption(solver.id(), "drake::RemoveFreeVariableMethod",
                               static_cast<int>(method));
      MathematicalProgramResult result;
      solver.Solve(*prog_, {}, solver_options, &result);
      EXPECT_TRUE(result.is_success());
      const double tol = 1E-7;
      EXPECT_NEAR(result.get_optimal_cost(), -1.0 / 3, tol);
      EXPECT_TRUE(CompareMatrices(result.GetSolution(X1_),
                                  Eigen::Matrix2d::Zero(), tol));
      EXPECT_NEAR(result.GetSolution(y_), 1.0 / 3, tol);
    }
  }
}

TEST_F(TrivialSOCP1, Solve) {
  for (auto method : GetRemoveFreeVariableMethods()) {
    SCOPED_TRACE(fmt::format("method = {}", method));
    CsdpSolver solver;
    if (solver.available()) {
      SolverOptions solver_options;
      solver_options.SetOption(solver.id(), "drake::RemoveFreeVariableMethod",
                               static_cast<int>(method));
      MathematicalProgramResult result;
      solver.Solve(*prog_, {}, solver_options, &result);
      EXPECT_TRUE(result.is_success());
      const double tol = 1E-7;
      EXPECT_NEAR(result.get_optimal_cost(), -10, tol);
      EXPECT_TRUE(CompareMatrices(result.GetSolution(x_),
                                  Eigen::Vector3d(10, 0, 0), tol));
    }
  }
}

TEST_F(TrivialSOCP2, Solve) {
  for (auto method : GetRemoveFreeVariableMethods()) {
    SCOPED_TRACE(fmt::format("method = {}", method));
    CsdpSolver solver;
    if (solver.available()) {
      SolverOptions solver_options;
      solver_options.SetOption(solver.id(), "drake::RemoveFreeVariableMethod",
                               static_cast<int>(method));
      MathematicalProgramResult result;
      solver.Solve(*prog_, {}, solver_options, &result);
      EXPECT_TRUE(result.is_success());
      // CSDP does not have high accuracy for solving an SOCP.
      const double tol = 3.8E-5;
      EXPECT_NEAR(result.get_optimal_cost(), -1, tol);
      EXPECT_TRUE(
          CompareMatrices(result.GetSolution(x_), Eigen::Vector2d(0, 1), tol));
    }
  }
}

TEST_F(TrivialSOCP3, Solve) {
  for (auto method : GetRemoveFreeVariableMethods()) {
    SCOPED_TRACE(fmt::format("method = {}", method));
    CsdpSolver solver;
    if (solver.available()) {
      SolverOptions solver_options;
      solver_options.SetOption(solver.id(), "drake::RemoveFreeVariableMethod",
                               static_cast<int>(method));
      MathematicalProgramResult result;
      solver.Solve(*prog_, {}, solver_options, &result);
      EXPECT_TRUE(result.is_success());
      // The accuracy for this test on the Mac machine is 5E-6.
      const double tol = 5.E-6;
      EXPECT_NEAR(result.get_optimal_cost(), 2 - std::sqrt(7.1), tol);
      EXPECT_TRUE(CompareMatrices(result.GetSolution(x_),
                                  Eigen::Vector2d(-0.1, 2 - std::sqrt(7.1)),
                                  tol));
    }
  }
}

GTEST_TEST(TestSOCP, TestSocpDuplicatedVariable1) {
  for (auto method : GetRemoveFreeVariableMethods()) {
    SCOPED_TRACE(fmt::format("method = {}", method));
    CsdpSolver solver;
    if (solver.available()) {
      SolverOptions solver_options;
      solver_options.SetOption(solver.id(), "drake::RemoveFreeVariableMethod",
                               static_cast<int>(method));
      TestSocpDuplicatedVariable1(solver, solver_options, 1E-6);
    }
  }
}

GTEST_TEST(TestSOCP, TestSocpDuplicatedVariable2) {
  for (auto method : GetRemoveFreeVariableMethods()) {
    SCOPED_TRACE(fmt::format("method = {}", method));
    CsdpSolver solver;
    if (solver.available()) {
      SolverOptions solver_options;
      solver_options.SetOption(solver.id(), "drake::RemoveFreeVariableMethod",
                               static_cast<int>(method));
      // Loosen the tolerances a bit (otherwise macOS is sad).
      solver_options.SetOption(solver.id(), "axtol", 1e-6);
      solver_options.SetOption(solver.id(), "objtol", 1e-6);
      TestSocpDuplicatedVariable2(solver, solver_options, 1E-5);
    }
  }
}

GTEST_TEST(TestSOCP, TestSocpDuplicatedVariable3) {
  for (auto method : GetRemoveFreeVariableMethods()) {
    SCOPED_TRACE(fmt::format("method = {}", method));
    CsdpSolver solver;
    if (solver.available()) {
      SolverOptions solver_options;
      solver_options.SetOption(solver.id(), "drake::RemoveFreeVariableMethod",
                               static_cast<int>(method));
      TestSocpDuplicatedVariable3(solver, solver_options, 1E-4);
    }
  }
}
}  // namespace test

namespace test {
TEST_F(InfeasibleLinearProgramTest0, TestInfeasible) {
  for (auto method : GetRemoveFreeVariableMethods()) {
    SCOPED_TRACE(fmt::format("method = {}", method));
    CsdpSolver solver;
    if (solver.available()) {
      SolverOptions solver_options;
      solver_options.SetOption(solver.id(), "drake::RemoveFreeVariableMethod",
                               static_cast<int>(method));
      auto result = solver.Solve(*prog_, {}, solver_options);
      EXPECT_EQ(result.get_solution_result(),
                SolutionResult::kInfeasibleConstraints);
      EXPECT_EQ(result.get_optimal_cost(),
                MathematicalProgram::kGlobalInfeasibleCost);
    }
  }
}

TEST_F(UnboundedLinearProgramTest0, TestUnbounded) {
  for (auto method : GetRemoveFreeVariableMethods()) {
    SCOPED_TRACE(fmt::format("method = {}", method));
    CsdpSolver solver;
    if (solver.available()) {
      SolverOptions solver_options;
      solver_options.SetOption(solver.id(), "drake::RemoveFreeVariableMethod",
                               static_cast<int>(method));
      auto result = solver.Solve(*prog_, {}, solver_options);
      EXPECT_EQ(result.get_solution_result(), SolutionResult::kDualInfeasible);
    }
  }
}

GTEST_TEST(TestSemidefiniteProgram, CommonLyapunov) {
  for (auto method : GetRemoveFreeVariableMethods()) {
    SCOPED_TRACE(fmt::format("method = {}", method));
    CsdpSolver solver;
    if (solver.available()) {
      SolverOptions solver_options;
      solver_options.SetOption(solver.id(), "drake::RemoveFreeVariableMethod",
                               static_cast<int>(method));
      FindCommonLyapunov(solver, solver_options, 1E-6);
    }
  }
}

GTEST_TEST(TestSemidefiniteProgram, OuterEllipsoid) {
  for (auto method : GetRemoveFreeVariableMethods()) {
    SCOPED_TRACE(fmt::format("method = {}", method));
    CsdpSolver solver;
    if (solver.available()) {
      SolverOptions solver_options;
      solver_options.SetOption(solver.id(), "drake::RemoveFreeVariableMethod",
                               static_cast<int>(method));
      FindOuterEllipsoid(solver, solver_options, 1E-6);
    }
  }
}

GTEST_TEST(TestSemidefiniteProgram, EigenvalueProblem) {
  for (auto method : GetRemoveFreeVariableMethods()) {
    SCOPED_TRACE(fmt::format("method = {}", method));
    CsdpSolver solver;
    if (solver.available()) {
      SolverOptions solver_options;
      solver_options.SetOption(solver.id(), "drake::RemoveFreeVariableMethod",
                               static_cast<int>(method));
      SolveEigenvalueProblem(solver, solver_options, 1E-6,
                             /*check_dual=*/false);
    }
  }
}

TEST_P(TestEllipsoidsSeparation, TestSOCP) {
  for (auto method : GetRemoveFreeVariableMethods()) {
    SCOPED_TRACE(fmt::format("method = {}", method));
    CsdpSolver solver;
    if (solver.available()) {
      SolverOptions solver_options;
      solver_options.SetOption(solver.id(), "drake::RemoveFreeVariableMethod",
                               static_cast<int>(method));
      SolveAndCheckSolution(solver, solver_options, 1E-6);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    CsdpTest, TestEllipsoidsSeparation,
    ::testing::ValuesIn(GetEllipsoidsSeparationProblems()));

TEST_P(TestFindSpringEquilibrium, TestSOCP) {
  for (auto method : GetRemoveFreeVariableMethods()) {
    SCOPED_TRACE(fmt::format("method = {}", method));
    CsdpSolver solver;
    if (solver.available()) {
      SolverOptions solver_options;
      solver_options.SetOption(solver.id(), "drake::RemoveFreeVariableMethod",
                               static_cast<int>(method));
      SolveAndCheckSolution(solver, solver_options, 7E-5);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    CsdpTest, TestFindSpringEquilibrium,
    ::testing::ValuesIn(GetFindSpringEquilibriumProblems()));

GTEST_TEST(TestSOCP, MaximizeGeometricMeanTrivialProblem1) {
  MaximizeGeometricMeanTrivialProblem1 prob;
  for (auto method : GetRemoveFreeVariableMethods()) {
    SCOPED_TRACE(fmt::format("method = {}", method));
    CsdpSolver solver;
    if (solver.available()) {
      SolverOptions solver_options;
      solver_options.SetOption(solver.id(), "drake::RemoveFreeVariableMethod",
                               static_cast<int>(method));
      const auto result = solver.Solve(prob.prog(), {}, solver_options);
      prob.CheckSolution(result, 1E-6);
    }
  }
}

GTEST_TEST(TestSOCP, MaximizeGeometricMeanTrivialProblem2) {
  MaximizeGeometricMeanTrivialProblem2 prob;
  for (auto method : GetRemoveFreeVariableMethods()) {
    SCOPED_TRACE(fmt::format("method = {}", method));
    CsdpSolver solver;
    if (solver.available()) {
      SolverOptions solver_options;
      solver_options.SetOption(solver.id(), "drake::RemoveFreeVariableMethod",
                               static_cast<int>(method));
      const auto result = solver.Solve(prob.prog(), {}, solver_options);
      prob.CheckSolution(result, 1E-6);
    }
  }
}

GTEST_TEST(TestSOCP, SmallestEllipsoidCoveringProblem) {
  for (auto method : GetRemoveFreeVariableMethods()) {
    SCOPED_TRACE(fmt::format("method = {}", method));
    CsdpSolver solver;
    SolverOptions solver_options;
    solver_options.SetOption(solver.id(), "drake::RemoveFreeVariableMethod",
                             static_cast<int>(method));
    SolveAndCheckSmallestEllipsoidCoveringProblems(solver, solver_options,
                                                   1E-6);
  }
}

GTEST_TEST(TestSOS, UnivariateQuarticSos) {
  UnivariateQuarticSos dut;
  for (auto method : GetRemoveFreeVariableMethods()) {
    SCOPED_TRACE(fmt::format("method = {}", method));
    CsdpSolver solver;
    if (solver.available()) {
      SolverOptions solver_options;
      solver_options.SetOption(solver.id(), "drake::RemoveFreeVariableMethod",
                               static_cast<int>(method));
      const auto result = solver.Solve(dut.prog(), {}, solver_options);
      dut.CheckResult(result, 1E-10);
    }
  }
}

GTEST_TEST(TestSOS, BivariateQuarticSos) {
  BivariateQuarticSos dut;
  for (auto method : GetRemoveFreeVariableMethods()) {
    SCOPED_TRACE(fmt::format("method = {}", method));
    CsdpSolver solver;
    if (solver.available()) {
      SolverOptions solver_options;
      solver_options.SetOption(solver.id(), "drake::RemoveFreeVariableMethod",
                               static_cast<int>(method));
      const auto result = solver.Solve(dut.prog(), {}, solver_options);
      dut.CheckResult(result, 1E-10);
    }
  }
}

GTEST_TEST(TestSOS, SimpleSos1) {
  SimpleSos1 dut;
  for (auto method : GetRemoveFreeVariableMethods()) {
    SCOPED_TRACE(fmt::format("method = {}", method));
    CsdpSolver solver;
    if (solver.available()) {
      SolverOptions solver_options;
      solver_options.SetOption(solver.id(), "drake::RemoveFreeVariableMethod",
                               static_cast<int>(method));
      const auto result = solver.Solve(dut.prog(), {}, solver_options);
      dut.CheckResult(result, 1E-10);
    }
  }
}

GTEST_TEST(TestSOS, MotzkinPolynomial) {
  MotzkinPolynomial dut;
  for (auto method : GetRemoveFreeVariableMethods()) {
    SCOPED_TRACE(fmt::format("method = {}", method));
    CsdpSolver solver;
    if (solver.available()) {
      SolverOptions solver_options;
      solver_options.SetOption(solver.id(), "drake::RemoveFreeVariableMethod",
                               static_cast<int>(method));
      const auto result = solver.Solve(dut.prog(), {}, solver_options);
      dut.CheckResult(result, 1.5E-8);
    }
  }
}

GTEST_TEST(TestSOS, UnivariateNonnegative1) {
  UnivariateNonnegative1 dut;
  for (auto method : GetRemoveFreeVariableMethods()) {
    SCOPED_TRACE(fmt::format("method = {}", method));
    CsdpSolver solver;
    if (solver.available()) {
      SolverOptions solver_options;
      solver_options.SetOption(solver.id(), "drake::RemoveFreeVariableMethod",
                               static_cast<int>(method));
      const auto result =
          solver.Solve(dut.prog(), std::nullopt, solver_options);
      dut.CheckResult(result, 6E-9);
    }
  }
}

// This is a code coverage test, not a functional test. If the code that
// handles verbosity options has a segfault or always throws an exception,
// then this would catch it.
TEST_F(TrivialSDP1, SolveVerbose) {
  SolverOptions options;
  options.SetOption(CommonSolverOption::kPrintToConsole, 1);
  CsdpSolver solver;
  if (solver.available()) {
    solver.Solve(*prog_, {}, options);
  }
}

// This is a code coverage test of reporting malformed options.
TEST_F(TrivialSDP1, UnknownRemoveFreeVariableMethod) {
  UnivariateNonnegative1 example;
  CsdpSolver solver;
  SolverOptions options;
  options.SetOption(solver.id(), "drake::RemoveFreeVariableMethod", 222);
  if (solver.available()) {
    DRAKE_EXPECT_THROWS_MESSAGE(solver.Solve(example.prog(), {}, options),
                                ".*Bad.*RemoveFreeVariableMethod.*");
  }
}

// Confirm that setting solver options has an effect on the solve.
TEST_F(TrivialSDP1, SolverOptionsPropagation) {
  CsdpSolver solver;
  if (!solver.available()) {
    return;
  }

  // Solving with default options works fine.
  {
    SolverOptions options;
    auto result = solver.Solve(*prog_, {}, options);
    EXPECT_TRUE(result.is_success());
  }

  // Setting an absurd `int` option causes a failure.
  {
    SolverOptions options;
    options.SetOption(CsdpSolver::id(), "maxiter", 0);
    auto result = solver.Solve(*prog_, {}, options);
    EXPECT_EQ(result.get_solution_result(), kIterationLimit);
  }

  // Setting an absurd `double` option causes a failure.
  {
    SolverOptions options;
    options.SetOption(CsdpSolver::id(), "axtol", 1e-100);
    auto result = solver.Solve(*prog_, {}, options);
    EXPECT_EQ(result.get_solution_result(), kSolverSpecificError);
  }
}

}  // namespace test
}  // namespace solvers
}  // namespace drake
