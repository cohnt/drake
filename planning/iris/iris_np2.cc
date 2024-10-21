#include "drake/planning/iris/iris_np2.h"

#include <iostream>

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <common_robotics_utilities/parallelism.hpp>

#include "drake/common/symbolic/expression.h"
#include "drake/geometry/optimization/affine_ball.h"
#include "drake/geometry/optimization/cartesian_product.h"
#include "drake/geometry/optimization/convex_set.h"
#include "drake/geometry/optimization/iris_internal.h"
#include "drake/geometry/optimization/minkowski_sum.h"
#include "drake/geometry/optimization/vpolytope.h"
#include "drake/geometry/shape_specification.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/joint.h"
#include "drake/multibody/tree/planar_joint.h"
#include "drake/multibody/tree/quaternion_floating_joint.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/multibody/tree/rpy_floating_joint.h"
#include "drake/planning/robot_diagram.h"
#include "drake/solvers/choose_best_solver.h"
#include "drake/solvers/ipopt_solver.h"
#include "drake/solvers/snopt_solver.h"
#include "drake/solvers/solve.h"
#include "drake/multibody/tree/quaternion_floating_joint.h"

namespace drake {
namespace planning {

using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using geometry::Role;
using geometry::SceneGraphInspector;
using geometry::optimization::ConvexSet;
using geometry::optimization::internal::ClosestCollisionProgram;
using geometry::optimization::internal::SamePointConstraint;
using multibody::MultibodyPlant;
using systems::Context;
using math::RigidTransform;
using multibody::QuaternionFloatingJoint;

namespace {
// Copied from iris.cc

using geometry::Box;
using geometry::Capsule;
using geometry::Convex;
using geometry::Cylinder;
using geometry::Ellipsoid;
using geometry::HalfSpace;
using geometry::Mesh;
using geometry::Sphere;

using geometry::FrameId;
using geometry::GeometryId;
using geometry::QueryObject;
using geometry::ShapeReifier;

using geometry::optimization::CartesianProduct;
using geometry::optimization::ConvexSet;
using geometry::optimization::HPolyhedron;
using geometry::optimization::Hyperellipsoid;
using geometry::optimization::MinkowskiSum;
using geometry::optimization::VPolytope;

// Constructs a ConvexSet for each supported Shape and adds it to the set.
class IrisConvexSetMaker final : public ShapeReifier {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(IrisConvexSetMaker);

  IrisConvexSetMaker(const QueryObject<double>& query,
                     std::optional<FrameId> reference_frame)
      : query_{query}, reference_frame_{reference_frame} {};

  void set_reference_frame(const FrameId& reference_frame) {
    DRAKE_DEMAND(reference_frame.is_valid());
    *reference_frame_ = reference_frame;
  }

  void set_geometry_id(const GeometryId& geom_id) { geom_id_ = geom_id; }

  using ShapeReifier::ImplementGeometry;

  void ImplementGeometry(const Box&, void* data) {
    DRAKE_DEMAND(geom_id_.is_valid());
    auto& set = *static_cast<copyable_unique_ptr<ConvexSet>*>(data);
    // Note: We choose HPolyhedron over VPolytope here, but the IRIS paper
    // discusses a significant performance improvement using a "least-distance
    // programming" instance from CVXGEN that exploited the VPolytope
    // representation.  So we may wish to revisit this.
    set = std::make_unique<HPolyhedron>(query_, geom_id_, reference_frame_);
  }

  void ImplementGeometry(const Capsule&, void* data) {
    DRAKE_DEMAND(geom_id_.is_valid());
    auto& set = *static_cast<copyable_unique_ptr<ConvexSet>*>(data);
    set = std::make_unique<MinkowskiSum>(query_, geom_id_, reference_frame_);
  }

  void ImplementGeometry(const Cylinder&, void* data) {
    DRAKE_DEMAND(geom_id_.is_valid());
    auto& set = *static_cast<copyable_unique_ptr<ConvexSet>*>(data);
    set =
        std::make_unique<CartesianProduct>(query_, geom_id_, reference_frame_);
  }

  void ImplementGeometry(const Ellipsoid&, void* data) {
    DRAKE_DEMAND(geom_id_.is_valid());
    auto& set = *static_cast<copyable_unique_ptr<ConvexSet>*>(data);
    set = std::make_unique<Hyperellipsoid>(query_, geom_id_, reference_frame_);
  }

  void ImplementGeometry(const HalfSpace&, void* data) {
    DRAKE_DEMAND(geom_id_.is_valid());
    auto& set = *static_cast<copyable_unique_ptr<ConvexSet>*>(data);
    set = std::make_unique<HPolyhedron>(query_, geom_id_, reference_frame_);
  }

  void ImplementGeometry(const Sphere&, void* data) {
    DRAKE_DEMAND(geom_id_.is_valid());
    auto& set = *static_cast<copyable_unique_ptr<ConvexSet>*>(data);
    set = std::make_unique<Hyperellipsoid>(query_, geom_id_, reference_frame_);
  }

  void ImplementGeometry(const Convex&, void* data) {
    DRAKE_DEMAND(geom_id_.is_valid());
    auto& set = *static_cast<copyable_unique_ptr<ConvexSet>*>(data);
    set = std::make_unique<VPolytope>(query_, geom_id_, reference_frame_);
  }

  void ImplementGeometry(const Mesh&, void* data) {
    DRAKE_DEMAND(geom_id_.is_valid());
    auto& set = *static_cast<copyable_unique_ptr<ConvexSet>*>(data);
    set = std::make_unique<VPolytope>(query_, geom_id_, reference_frame_);
  }

 private:
  const QueryObject<double>& query_{};
  std::optional<FrameId> reference_frame_{};
  GeometryId geom_id_{};
};

struct GeometryPairWithDistance {
  GeometryId geomA;
  GeometryId geomB;
  double distance;

  GeometryPairWithDistance(GeometryId gA, GeometryId gB, double dist)
      : geomA(gA), geomB(gB), distance(dist) {}

  bool operator<(const GeometryPairWithDistance& other) const {
    return distance < other.distance;
  }
};

// int unadaptive_test_samples(double p, double delta, double tau) {
//   return static_cast<int>(-2 * std::log(delta) / (tau * tau * p) + 0.5);
// }

// int FindCollisionPairIndex(
//     const MultibodyPlant<double>& plant, Context<double>* context,
//     const Eigen::VectorXd& configuration,
//     const std::vector<GeometryPairWithDistance>& sorted_pairs) {
//   // Call ComputeSignedDistancePairClosestPoints for each pair of collision
//   // geometries until finding a pair that is in collision and returning the
//   // corresponding index
//   int pair_in_collision = -1;
//   int i_pair = 0;
//   for (const auto& pair : sorted_pairs) {
//     plant.SetPositions(context, configuration);
//     auto query_object = plant.get_geometry_query_input_port()
//                             .template Eval<QueryObject<double>>(*context);
//     const double distance =
//         query_object
//             .ComputeSignedDistancePairClosestPoints(pair.geomA, pair.geomB)
//             .distance;
//     if (distance < 0.0) {
//       pair_in_collision = i_pair;
//       break;
//     }
//     ++i_pair;
//   }

//   return pair_in_collision;
// }

// Add the tangent to the (scaled) ellipsoid at @p point as a
// constraint.
void AddTangentToPolytope(
    const Hyperellipsoid& E, const Eigen::Ref<const Eigen::VectorXd>& point,
    double configuration_space_margin,
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>* A,
    Eigen::VectorXd* b, int* num_constraints) {
  while (*num_constraints >= A->rows()) {
    // Increase pre-allocated polytope size.
    A->conservativeResize(A->rows() * 2, A->cols());
    b->conservativeResize(b->rows() * 2);
  }

  A->row(*num_constraints) =
      (E.A().transpose() * E.A() * (point - E.center())).normalized();
  (*b)[*num_constraints] =
      A->row(*num_constraints) * point - configuration_space_margin;
  if (A->row(*num_constraints) * E.center() > (*b)[*num_constraints]) {
    throw std::logic_error(
        "The current center of the IRIS region is within "
        "options.configuration_space_margin of being infeasible.  Check your "
        "sample point and/or any additional constraints you've passed in via "
        "the options. The configuration space surrounding the sample point "
        "must have an interior.");
  }
  *num_constraints += 1;
}

void MakeGuessFeasible(const HPolyhedron& P, Eigen::VectorXd* guess) {
  const auto& A = P.A();
  const auto& b = P.b();
  const int M = A.rows();
  // Add kEps below because we want to be strictly on the correct side of the
  // inequality (and robust to floating point errors).
  const double kEps = 1e-14;
  // Try projecting the guess onto any violated constraints, one-by-one.
  for (int i = M - 1; i >= 0; --i) {
    if (A.row(i) * *guess - b[i] > 0) {
      // guess = argmin_x ||x - guess||^2 s.t. A.row(i) * x = b(i).
      *guess -=
          A.row(i).normalized().transpose() * (A.row(i) * *guess - b[i] + kEps);
    }
  }
  // If this causes a different constraint to be violated, then just return the
  // Chebyshev center.
  if (!P.PointInSet(*guess)) {
    // Note: This can throw if the set becomes empty. But it should not happen;
    // we check that the seed is feasible in AddTangentToPolytope.
    *guess = P.ChebyshevCenter();
  }
}

/* Given a joint, check if it is encompassed by the continuous revolute
framework. If so, return a vector of indices i that represent an angle-valued
coordinate in configuration space, and should be automatically bounded. If the
joint is not encompassed by the continuous revolute framework, return an empty
vector. */
std::vector<int> revolute_joint_indices(const multibody::Joint<double>& joint) {
  if (joint.type_name() == multibody::RevoluteJoint<double>::kTypeName) {
    DRAKE_ASSERT(joint.num_positions() == 1);
    // RevoluteJoints store their configuration as (θ)
    if (joint.position_lower_limits()[0] ==
            -std::numeric_limits<float>::infinity() &&
        joint.position_upper_limits()[0] ==
            std::numeric_limits<float>::infinity()) {
      return std::vector<int>{joint.position_start() + 0};
    }
  }
  if (joint.type_name() == multibody::PlanarJoint<double>::kTypeName) {
    DRAKE_ASSERT(joint.num_positions() == 3);
    // PlanarJoints store their configuration as (x, y, θ)
    if (joint.position_lower_limits()[2] ==
            -std::numeric_limits<float>::infinity() &&
        joint.position_upper_limits()[2] ==
            std::numeric_limits<float>::infinity()) {
      return std::vector<int>{joint.position_start() + 2};
    }
  }
  if (joint.type_name() == multibody::RpyFloatingJoint<double>::kTypeName) {
    DRAKE_ASSERT(joint.num_positions() == 6);
    // RpyFloatingJoints store their configuration as (qx, qy, qz, x, y, z),
    // i.e., the first three positions are the revolute components.
    std::vector<int> continuous_revolute_indices;
    for (int i = 0; i < 3; ++i) {
      if (joint.position_lower_limits()[i] ==
              -std::numeric_limits<float>::infinity() &&
          joint.position_upper_limits()[i] ==
              std::numeric_limits<float>::infinity()) {
        continuous_revolute_indices.push_back(joint.position_start() + i);
      }
    }
    return continuous_revolute_indices;
  }
  // TODO(cohnt): Add support for other joint types that may be compatible with
  // the continuous revolute framework.
  return std::vector<int>{};
}
}  // namespace

HPolyhedron IrisNP2(Eigen::VectorXd q, const CollisionChecker& checker,
                    const IrisNP2Options& options) {
  const auto& plant = checker.plant();
  const auto& context = checker.UpdatePositions(q);

  // Check the inputs.
  plant.ValidateContext(context);
  const int nq = plant.num_positions();
  const Eigen::VectorXd seed = q;

  Eigen::VectorXd lower_limits = plant.GetPositionLowerLimits();
  Eigen::VectorXd upper_limits = plant.GetPositionUpperLimits();

  DRAKE_THROW_UNLESS(options.convexity_radius_stepback < M_PI_2);
  for (multibody::JointIndex index : plant.GetJointIndices()) {
    const multibody::Joint<double>& joint = plant.get_joint(index);
    if (joint.type_name() == QuaternionFloatingJoint<double>::kTypeName) {
      throw std::runtime_error(
          "IrisInConfigurationSpace does not support QuaternionFloatingJoint. "
          "Consider using RpyFloatingJoint instead.");
    }
    const std::vector<int> continuous_revolute_indices =
        revolute_joint_indices(joint);
    for (const int i : continuous_revolute_indices) {
      lower_limits[i] = seed[i] - M_PI_2 + options.convexity_radius_stepback;
      upper_limits[i] = seed[i] + M_PI_2 - options.convexity_radius_stepback;
    }
  }

  // Make the polytope and ellipsoid.
  MatrixXd A_init = MatrixXd::Zero(2 * ssize(lower_limits), nq);
  VectorXd b_init = VectorXd::Zero(2 * ssize(lower_limits));
  int row_count = 0;
  for (int i = 0; i < ssize(upper_limits); ++i) {
    if (std::isfinite(upper_limits[i])) {
      A_init(row_count, i) = 1;
      b_init(row_count) = upper_limits[i];
      ++row_count;
    }
    if (std::isfinite(lower_limits[i])) {
      A_init(row_count, i) = -1;
      b_init(row_count) = -lower_limits[i];
      ++row_count;
    }
  }
  A_init.conservativeResize(row_count, nq);
  b_init.conservativeResize(row_count);
  HPolyhedron P(A_init, b_init);

  bool boundedness_error = false;
  if (options.bounding_region) {
    DRAKE_DEMAND(options.bounding_region->ambient_dimension() == nq);
    P = P.Intersection(*options.bounding_region);
    if (options.verify_domain_boundedness) {
      if (!P.IsBounded()) {
        boundedness_error = true;
      }
    }
  } else {
    if (lower_limits.array().isInf().any() ||
        upper_limits.array().isInf().any()) {
      boundedness_error = true;
    }
  }

  if (boundedness_error) {
    throw std::runtime_error(
        "IrisInConfigurationSpace requires that the initial domain be bounded. "
        "Make sure all joints have position limits (unless that joint is a "
        "RevoluteJoint or the revolute component of a PlanarJoint or "
        "RpyFloatingJoint), or ensure that the intersection of the joint "
        "limits and options.bounding_region is bounded.");
  }

  const double kEpsilonEllipsoid = 1e-2;
  Hyperellipsoid E = options.starting_ellipse.value_or(
      Hyperellipsoid::MakeHypersphere(kEpsilonEllipsoid, seed));

  // Make all of the convex sets and supporting quantities.
  auto query_object =
      plant.get_geometry_query_input_port().Eval<QueryObject<double>>(context);
  const SceneGraphInspector<double>& inspector = query_object.inspector();
  IrisConvexSetMaker maker(query_object, inspector.world_frame_id());
  std::unordered_map<GeometryId, copyable_unique_ptr<ConvexSet>> sets{};
  std::unordered_map<GeometryId, const multibody::Frame<double>*> frames{};
  const std::vector<GeometryId> geom_ids =
      inspector.GetAllGeometryIds(Role::kProximity);
  copyable_unique_ptr<ConvexSet> temp_set;
  for (GeometryId geom_id : geom_ids) {
    // Make all sets in the local geometry frame.
    FrameId frame_id = inspector.GetFrameId(geom_id);
    maker.set_reference_frame(frame_id);
    maker.set_geometry_id(geom_id);
    inspector.GetShape(geom_id).Reify(&maker, &temp_set);
    sets.emplace(geom_id, std::move(temp_set));
    frames.emplace(geom_id, &plant.GetBodyFromFrameId(frame_id)->body_frame());
  }

  auto pairs = inspector.GetCollisionCandidates();
  const int n = static_cast<int>(pairs.size());
  auto same_point_constraint =
      std::make_shared<SamePointConstraint>(&plant, context);
  std::map<std::pair<GeometryId, GeometryId>, std::vector<VectorXd>>
      counter_examples;

  // As a surrogate for the true objective, the pairs are sorted by the
  // distance between each collision pair from the seed point configuration.
  // This could improve computation times and produce regions with fewer
  // faces.
  std::vector<GeometryPairWithDistance> sorted_pairs;
  for (const auto& [geomA, geomB] : pairs) {
    const double distance =
        query_object.ComputeSignedDistancePairClosestPoints(geomA, geomB)
            .distance;
    if (distance < 0.0) {
      throw std::runtime_error(
          fmt::format("The seed point is in collision; geometry {} is in "
                      "collision with geometry {}",
                      inspector.GetName(geomA), inspector.GetName(geomB)));
    }
    sorted_pairs.emplace_back(geomA, geomB, distance);
  }
  std::sort(sorted_pairs.begin(), sorted_pairs.end());

  // On each iteration, we will build the collision-free polytope represented
  // as {x | A * x <= b}.  Here we pre-allocate matrices with a generous
  // maximum size.
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A(
      P.A().rows() + 2 * n, nq);
  VectorXd b(P.A().rows() + 2 * n);
  A.topRows(P.A().rows()) = P.A();
  b.head(P.A().rows()) = P.b();
  int num_initial_constraints = P.A().rows();

  DRAKE_THROW_UNLESS(P.PointInSet(seed, 1e-12));

  double best_volume = E.Volume();
  int iteration = 0;
  VectorXd closest(nq);
  RandomGenerator generator(options.random_seed);

  auto solver = solvers::MakeFirstAvailableSolver(
      {solvers::SnoptSolver::id(), solvers::IpoptSolver::id()});

  VectorXd guess = seed;

  // For debugging visualization.
  Vector3d point_to_draw = Vector3d::Zero();
  int num_points_drawn = 0;
  bool do_debugging_visualization = options.meshcat && nq <= 3;

  const std::string seed_point_error_msg =
      "IrisInConfigurationSpace: require_sample_point_is_contained is true but "
      "the seed point exited the initial region. Does the provided "
      "options.starting_ellipse not contain the seed point?";
  const std::string seed_point_msg =
      "IrisInConfigurationSpace: terminating iterations because the seed point "
      "is no longer in the region.";
  const std::string termination_error_msg =
      "IrisInConfigurationSpace: the termination function returned false on "
      "the computation of the initial region. Are the provided "
      "options.starting_ellipse and options.termination_func compatible?";
  const std::string termination_msg =
      "IrisInConfigurationSpace: terminating iterations because "
      "options.termination_func returned false.";

  while (true) {
    log()->info("IrisInConfigurationSpace iteration {}", iteration);
    int num_constraints = num_initial_constraints;
    HPolyhedron P_candidate = HPolyhedron(A.topRows(num_initial_constraints),
                                          b.head(num_initial_constraints));
    DRAKE_ASSERT(best_volume > 0);
    // Find separating hyperplanes

    // Use the fast nonlinear optimizer until it fails
    // num_collision_infeasible_samples consecutive times.
    for (const auto& pair_w_distance : sorted_pairs) {
      std::pair<GeometryId, GeometryId> geom_pair(pair_w_distance.geomA,
                                                  pair_w_distance.geomB);
      int consecutive_failures = 0;
      ClosestCollisionProgram prog(
          same_point_constraint, *frames.at(pair_w_distance.geomA),
          *frames.at(pair_w_distance.geomB), *sets.at(pair_w_distance.geomA),
          *sets.at(pair_w_distance.geomB), E, A.topRows(num_constraints),
          b.head(num_constraints));
      std::vector<VectorXd> prev_counter_examples =
          std::move(counter_examples[geom_pair]);
      // Sort by the current ellipsoid metric.
      std::sort(prev_counter_examples.begin(), prev_counter_examples.end(),
                [&E](const VectorXd& x, const VectorXd& y) {
                  return (E.A() * x - E.center()).squaredNorm() <
                         (E.A() * y - E.center()).squaredNorm();
                });
      std::vector<VectorXd> new_counter_examples;
      int counter_example_searches_for_this_pair = 0;
      bool warned_many_searches = false;
      while (consecutive_failures < 5) {
        // First use previous counter-examples for this pair as the seeds.
        if (counter_example_searches_for_this_pair <
            ssize(prev_counter_examples)) {
          guess = prev_counter_examples[counter_example_searches_for_this_pair];
        } else {
          MakeGuessFeasible(P_candidate, &guess);
          guess = P_candidate.UniformSample(&generator, guess,
                                            options.mixing_steps);
        }
        ++counter_example_searches_for_this_pair;
        if (do_debugging_visualization) {
          ++num_points_drawn;
          point_to_draw.head(nq) = guess;
          std::string path = fmt::format("iteration{:02}/{:03}/guess",
                                         iteration, num_points_drawn);
          options.meshcat->SetObject(path, Sphere(0.01),
                                     geometry::Rgba(0.1, 0.1, 0.1, 1.0));
          options.meshcat->SetTransform(path,
                                        RigidTransform<double>(point_to_draw));
        }
        if (prog.Solve(*solver, guess, options.solver_options, &closest)) {
          if (do_debugging_visualization) {
            point_to_draw.head(nq) = closest;
            std::string path = fmt::format("iteration{:02}/{:03}/found",
                                           iteration, num_points_drawn);
            options.meshcat->SetObject(path, Sphere(0.01),
                                       geometry::Rgba(0.8, 0.1, 0.8, 1.0));
            options.meshcat->SetTransform(
                path, RigidTransform<double>(point_to_draw));
          }
          consecutive_failures = 0;
          new_counter_examples.emplace_back(closest);
          AddTangentToPolytope(E, closest, options.configuration_space_margin,
                               &A, &b, &num_constraints);
          P_candidate =
              HPolyhedron(A.topRows(num_constraints), b.head(num_constraints));
          MakeGuessFeasible(P_candidate, &guess);
          if (options.require_sample_point_is_contained) {
            const bool seed_point_requirement =
                A.row(num_constraints - 1) * seed <= b(num_constraints - 1);
            if (!seed_point_requirement) {
              if (iteration == 0) {
                throw std::runtime_error(seed_point_error_msg);
              }
              log()->info(seed_point_msg);
              return P;
            }
          }
          prog.UpdatePolytope(A.topRows(num_constraints),
                              b.head(num_constraints));
        } else {
          if (do_debugging_visualization) {
            point_to_draw.head(nq) = closest;
            std::string path = fmt::format("iteration{:02}/{:03}/closest",
                                           iteration, num_points_drawn);
            options.meshcat->SetObject(path, Sphere(0.01),
                                       geometry::Rgba(0.1, 0.8, 0.8, 1.0));
            options.meshcat->SetTransform(
                path, RigidTransform<double>(point_to_draw));
          }
          if (counter_example_searches_for_this_pair >
              ssize(counter_examples[geom_pair])) {
            // Only count the failures once we start the random guesses.
            ++consecutive_failures;
          }
        }
        if (!warned_many_searches &&
            counter_example_searches_for_this_pair -
                    ssize(counter_examples[geom_pair]) >=
                10 * 5) {
          warned_many_searches = true;
          log()->info(
              " Checking {} against {} has already required {} counter-example "
              "searches; still searching...",
              inspector.GetName(pair_w_distance.geomA),
              inspector.GetName(pair_w_distance.geomB),
              counter_example_searches_for_this_pair);
        }
      }
      counter_examples[geom_pair] = std::move(new_counter_examples);
      if (warned_many_searches) {
        log()->info(
            " Finished checking {} against {} after {} counter-example "
            "searches.",
            inspector.GetName(pair_w_distance.geomA),
            inspector.GetName(pair_w_distance.geomB),
            counter_example_searches_for_this_pair);
      }
    }

    P = HPolyhedron(A.topRows(num_constraints), b.head(num_constraints));

    iteration++;
    if (iteration >= options.iteration_limit) {
      log()->info(
          "IrisInConfigurationSpace: Terminating because the iteration limit "
          "{} has been reached.",
          options.iteration_limit);
      break;
    }

    E = P.MaximumVolumeInscribedEllipsoid();
    const double volume = E.Volume();
    const double delta_volume = volume - best_volume;
    if (delta_volume <= options.termination_threshold) {
      log()->info(
          "IrisInConfigurationSpace: Terminating because the hyperellipsoid "
          "volume change {} is below the threshold {}.",
          delta_volume, options.termination_threshold);
      break;
    } else if (delta_volume / best_volume <=
               options.relative_termination_threshold) {
      log()->info(
          "IrisInConfigurationSpace: Terminating because the hyperellipsoid "
          "relative volume change {} is below the threshold {}.",
          delta_volume / best_volume, options.relative_termination_threshold);
      break;
    }
    best_volume = volume;
  }
  return P;
}

}  // namespace planning
}  // namespace drake
