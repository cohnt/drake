#include "drake/planning/iris/iris_np2.h"

#include <iostream>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include "drake/common/fmt_eigen.h"
#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/common/test_utilities/maybe_pause_for_user.h"
#include "drake/common/yaml/yaml_io.h"
#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/geometry/optimization/hyperellipsoid.h"
#include "drake/planning/iris/iris_common.h"
#include "drake/planning/iris/iris_from_clique_cover.h"
#include "drake/planning/iris/test/iris_test_utilities.h"
#include "drake/solvers/equality_constrained_qp_solver.h"
#include "drake/solvers/ipopt_solver.h"
#include "drake/solvers/nlopt_solver.h"

namespace drake {
namespace planning {
namespace {

using common::MaybePauseForUser;
using Eigen::Vector2d;
using Eigen::VectorX;
using Eigen::VectorXd;
using geometry::Sphere;
using geometry::optimization::HPolyhedron;
using geometry::optimization::Hyperellipsoid;
using symbolic::Expression;
using symbolic::Variable;

TEST_F(BimanualIiwaParameterization, RegionTest) {
  IrisNp2Options options;
  auto scene_graph_checker =
      dynamic_cast<SceneGraphCollisionChecker*>(checker_.get());
  ASSERT_TRUE(scene_graph_checker != nullptr);

  meshcat_->Delete();
  meshcat_->ResetRenderMode();

  options.sampled_iris_options.verbose = true;

  auto parameterization_double = [this](const Eigen::VectorXd& q_and_psi) {
    return ParameterizationDouble(q_and_psi);
  };
  auto parameterization_autodiff =
      [this](const Eigen::VectorX<AutoDiffXd>& q_and_psi) {
        return ParameterizationAutodiff(q_and_psi);
      };

  options.parameterization = IrisParameterizationFunction(
      parameterization_double, parameterization_autodiff,
      /* parameterization_is_threadsafe */ true,
      /* parameterization_dimension */ 8);

  solvers::MathematicalProgram prog;
  auto q_and_psi = prog.NewContinuousVariables(8, "q");
  auto reachability_constraint = prog.AddConstraint(
      std::make_shared<internal::IiwaBimanualReachableConstraint>(true, true,
                                                                  true),
      q_and_psi);
  reachability_constraint.evaluator()->set_description(
      "Reachability Constraint");

  Eigen::VectorXd iiwa_lower = plant_ptr_->GetPositionLowerLimits().head(7);
  Eigen::VectorXd iiwa_upper = plant_ptr_->GetPositionUpperLimits().head(7);
  auto joint_limit_constraint = prog.AddConstraint(
      std::make_shared<internal::IiwaBimanualJointLimitConstraint>(
          iiwa_lower, iiwa_upper, /*shoulder_up = */ true,
          /*elbow_up = */ true, /*wrist_up = */ true),
      q_and_psi);
  joint_limit_constraint.evaluator()->set_description("Joint Limit Constraint");

  options.sampled_iris_options.prog_with_additional_constraints = &prog;

  // Find a valid seed point.
  auto is_valid = [&](const Eigen::VectorXd& q) {
    for (const auto& binding : prog.GetAllConstraints()) {
      Eigen::VectorXd y;
      binding.evaluator()->Eval(q, &y);
      const auto& lb = binding.evaluator()->lower_bound();
      const auto& ub = binding.evaluator()->upper_bound();
      if ((y.array() < lb.array()).any() || (y.array() > ub.array()).any()) {
        // std::cout << "Violated " << binding.evaluator()->get_description() <<
        // std::endl; std::cout << fmt::format("lb {}",
        // fmt_eigen(lb.transpose())) << std::endl; std::cout <<
        // fmt::format("val {}", fmt_eigen(y.transpose())) << std::endl;
        // std::cout << fmt::format("ub {}", fmt_eigen(ub.transpose())) <<
        // std::endl;
        return false;  // violated some constraint
      }
    }

    // 2. Check collision
    if (!scene_graph_checker->CheckConfigCollisionFree(
            parameterization_double(q))) {
      // std::cout << "In collision" << std::endl;
      return false;
    }

    return true;  // passes all constraints and collision-free
  };

  std::unique_ptr<systems::Context<double>> diagram_context =
      scene_graph_checker->model().CreateDefaultContext();
  auto& plant = scene_graph_checker->plant();
  systems::Context<double>* plant_context =
      &plant.GetMyMutableContextFromRoot(diagram_context.get());

  Eigen::VectorXd seed(8);
  const int kMaxTries = 500;
  const int kMaxRegions = 3;
  int num_regions = 0;
  bool found_valid = false;
  for (int attempts = 0; attempts < kMaxTries; ++attempts) {
    if (num_regions >= kMaxRegions) {
      break;
    }

    // std::cout << attempts << std::endl;
    seed.setRandom();
    seed.head(7) = ((seed.head(7).array() + 1.0) * 0.5 *
                        (iiwa_upper - iiwa_lower).array() +
                    iiwa_lower.array())
                       .matrix();
    seed.tail(1) = (seed.tail(1).array() * M_PI).matrix();

    if (!is_valid(seed)) {
      continue;
    }

    ++num_regions;
    found_valid = true;
    Eigen::VectorXd q = parameterization_double(seed);

    plant.SetPositions(plant_context, q);
    scene_graph_checker->model().ForcedPublish(*diagram_context);
    MaybePauseForUser(fmt::format(
        "Valid seed found on attempt {}: Press enter to continue...",
        attempts));

    Hyperellipsoid starting_ellipsoid =
        Hyperellipsoid::MakeHypersphere(1e-2, seed);

    Eigen::VectorXd parameterization_lb(8);
    Eigen::VectorXd parameterization_ub(8);
    parameterization_lb.head(7) = iiwa_lower;
    parameterization_ub.head(7) = iiwa_upper;
    parameterization_lb[7] = -1.0 * M_PI;
    parameterization_ub[7] = 1.0 * M_PI;

    HPolyhedron domain =
        HPolyhedron::MakeBox(parameterization_lb, parameterization_ub);

    // solvers::IpoptSolver solver;
    // options.solver = &solver;

    HPolyhedron region =
        IrisNp2(*scene_graph_checker, starting_ellipsoid, domain, options);

    std::vector<Eigen::VectorXd> samples;
    int kNumSamples = 100;
    int kMaxBad = 5;
    int num_bad = 0;
    RandomGenerator generator;
    for (int i = 0; i < kNumSamples; ++i) {
      Eigen::VectorXd sample = region.UniformSample(&generator, 1000);
      samples.push_back(sample);
      if (!is_valid(sample)) {
        ++num_bad;
      }
    }

    EXPECT_LE(num_bad, kMaxBad);
    std::cout << fmt::format("{} failures out of {} samples", num_bad,
                             kNumSamples);

    // for (int i = 0; i < ssize(samples); ++i) {
    //   Eigen::VectorXd config = parameterization_double(samples[i]);
    //   plant.SetPositions(plant_context, config);
    //   scene_graph_checker->model().ForcedPublish(*diagram_context);
    //   MaybePauseForUser(fmt::format("Region point {}", i));
    // }
  }
  EXPECT_TRUE(found_valid);
}

TEST_F(BimanualIiwaParameterization, CliqueCovers) {
  IrisNp2Options options;
  auto scene_graph_checker =
      dynamic_cast<SceneGraphCollisionChecker*>(checker_.get());
  ASSERT_TRUE(scene_graph_checker != nullptr);

  meshcat_->Delete();
  meshcat_->ResetRenderMode();

  options.sampled_iris_options.verbose = true;

  auto parameterization_double = [this](const Eigen::VectorXd& q_and_psi) {
    return ParameterizationDouble(q_and_psi);
  };
  auto parameterization_autodiff =
      [this](const Eigen::VectorX<AutoDiffXd>& q_and_psi) {
        return ParameterizationAutodiff(q_and_psi);
      };

  options.parameterization = IrisParameterizationFunction(
      parameterization_double, parameterization_autodiff,
      /* parameterization_is_threadsafe */ true,
      /* parameterization_dimension */ 8);

  solvers::MathematicalProgram prog;
  auto q_and_psi = prog.NewContinuousVariables(8, "q");
  auto reachability_constraint = prog.AddConstraint(
      std::make_shared<internal::IiwaBimanualReachableConstraint>(true, true,
                                                                  true),
      q_and_psi);
  reachability_constraint.evaluator()->set_description(
      "Reachability Constraint");

  Eigen::VectorXd iiwa_lower = plant_ptr_->GetPositionLowerLimits().head(7);
  Eigen::VectorXd iiwa_upper = plant_ptr_->GetPositionUpperLimits().head(7);
  auto joint_limit_constraint = prog.AddConstraint(
      std::make_shared<internal::IiwaBimanualJointLimitConstraint>(
          iiwa_lower, iiwa_upper, /*shoulder_up = */ true,
          /*elbow_up = */ true, /*wrist_up = */ true),
      q_and_psi);
  joint_limit_constraint.evaluator()->set_description("Joint Limit Constraint");

  options.sampled_iris_options.prog_with_additional_constraints = &prog;

  auto is_valid = [&](const Eigen::VectorXd& q) {
    for (const auto& binding : prog.GetAllConstraints()) {
      Eigen::VectorXd y;
      binding.evaluator()->Eval(q, &y);
      const auto& lb = binding.evaluator()->lower_bound();
      const auto& ub = binding.evaluator()->upper_bound();
      if ((y.array() < lb.array()).any() || (y.array() > ub.array()).any()) {
        // std::cout << "Violated " << binding.evaluator()->get_description() <<
        // std::endl; std::cout << fmt::format("lb {}",
        // fmt_eigen(lb.transpose())) << std::endl; std::cout <<
        // fmt::format("val {}", fmt_eigen(y.transpose())) << std::endl;
        // std::cout << fmt::format("ub {}", fmt_eigen(ub.transpose())) <<
        // std::endl;
        return false;  // violated some constraint
      }
    }

    // 2. Check collision
    if (!scene_graph_checker->CheckConfigCollisionFree(
            parameterization_double(q))) {
      // std::cout << "In collision" << std::endl;
      return false;
    }

    return true;  // passes all constraints and collision-free
  };

  std::unique_ptr<systems::Context<double>> diagram_context =
      scene_graph_checker->model().CreateDefaultContext();
  auto& plant = scene_graph_checker->plant();
  systems::Context<double>* plant_context =
      &plant.GetMyMutableContextFromRoot(diagram_context.get());

  Eigen::VectorXd parameterization_lb(8);
  Eigen::VectorXd parameterization_ub(8);
  parameterization_lb.head(7) = iiwa_lower;
  parameterization_ub.head(7) = iiwa_upper;
  parameterization_lb[7] = -1.0 * M_PI;
  parameterization_ub[7] = 1.0 * M_PI;

  Eigen::VectorXd domain_lb = parameterization_lb;
  Eigen::VectorXd domain_ub = parameterization_ub;

  domain_lb.head(7).setZero();
  domain_ub.head(7) *= 0.5;

  HPolyhedron domain = HPolyhedron::MakeBox(domain_lb, domain_ub);

  // solvers::IpoptSolver solver;
  // options.solver = &solver;

  scene_graph_checker->set_edge_step_size(0.1);

  IrisFromCliqueCoverOptions clique_cover_options;
  clique_cover_options.iris_options = options;
  clique_cover_options.num_points_per_visibility_round = 200;
  clique_cover_options.iteration_limit = 10;
  RandomGenerator generator;
  std::vector<HPolyhedron> sets;
  IrisInConfigurationSpaceFromCliqueCover(
      *scene_graph_checker, clique_cover_options, &generator, &sets,
      /* max_clique_solver= */ nullptr, /* provided_domain= */ &domain);

  int kNumSamples = 100;
  int kSamplesPerSet = 5;
  int kMaxBad = 5;
  std::vector<Eigen::VectorXd> samples;
  for (int i = 0; i < ssize(sets); ++i) {
    int num_bad = 0;
    for (int j = 0; j < kNumSamples; ++j) {
      Eigen::VectorXd sample = sets[i].UniformSample(&generator, 1000);
      if (!is_valid(sample)) {
        ++num_bad;
      } else if (ssize(samples) < (i + 1) * kSamplesPerSet) {
        samples.push_back(sample);
      }
    }
    EXPECT_LE(num_bad, kMaxBad);
  }

  for (int i = 0; i < ssize(samples); ++i) {
    Eigen::VectorXd q = parameterization_double(samples[i]);
    plant.SetPositions(plant_context, q);
    scene_graph_checker->model().ForcedPublish(*diagram_context);
    MaybePauseForUser(fmt::format("Region point {}", i));
  }
}

}  // namespace
}  // namespace planning
}  // namespace drake
