#pragma once

#include <map>
#include <memory>
#include <optional>

#include <common_robotics_utilities/parallelism.hpp>

#include "drake/geometry/meshcat.h"
#include "drake/geometry/optimization/convex_set.h"
#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/planning/collision_checker.h"
#include "drake/solvers/solver_interface.h"

namespace drake {
namespace planning {

using geometry::optimization::HPolyhedron;
using geometry::optimization::Hyperellipsoid;

// List of things we may want to support in the future, but aren't including
// yet:
// TODO(cohnt): Implement archive serialization.
// TODO(cohnt): Implement additional constraints.
// TODO(cohnt): Support a termination function.
// TODO(cohnt): Support automatically handling the convexity radius.
// TODO(cohnt): Support max hyperplanes per inner iteration.
// TODO(cohnt): Support specifying a sampler object that enables different
// sampling strategies.
// TODO(cohnt): Implement sampler object for greedy sampler.
// TODO(rhjiang): Implement sampler object for ray sampler.
struct IrisNP2Options {
  /** The initial polytope is guaranteed to contain the point if that point is
  collision-free. However, the IRIS alternation objectives do not include (and
  can not easily include) a constraint that the original sample point is
  contained. Therefore, the IRIS paper recommends that if containment is a
  requirement, then the algorithm should simply terminate early if alternations
  would ever cause the set to not contain the point. */
  bool require_sample_point_is_contained{false};

  /** Maximum number of outer iterations (bilinear alternations). */
  int iteration_limit{100};

  /** IRIS will terminate if the change in the *volume* of the hyperellipsoid
  between iterations is less that this threshold. This termination condition can
  be disabled by setting to a negative value. */
  double termination_threshold{2e-2};  // from rdeits/iris-distro.

  /** IRIS will terminate if the change in the *volume* of the hyperellipsoid
  between iterations is less that this percent of the previous best volume.
  This termination condition can be disabled by setting to a negative value. */
  double relative_termination_threshold{1e-3};  // from rdeits/iris-distro.

  // TODO(russt): Improve the implementation so that we can clearly document the
  // units for this margin.
  /** For IRIS in configuration space, we retreat by this margin from each
  C-space obstacle in order to avoid the possibility of requiring an infinite
  number of faces to approximate a curved boundary.
  */
  double configuration_space_margin{1e-2};

  /** The initial hyperellipsoid that IRIS will use for calculating hyperplanes
  in the first iteration. If no hyperellipsoid is provided, a small hypershpere
  centered at the given sample will be used. */
  std::optional<Hyperellipsoid> starting_ellipse{};

  /** Optionally allows the caller to restrict the space within which IRIS
  regions are allowed to grow. By default, IRIS regions are bounded by the
  `domain` argument in the case of `Iris` or the joint limits of the input
  `plant` in the case of `IrisInConfigurationSpace`. If this option is
  specified, IRIS regions will be confined to the intersection between the
  domain and `bounding_region` */
  std::optional<HPolyhedron> bounding_region{};

  /** If the user knows the intersection of bounding_region and the domain (for
  IRIS) or plant joint limits (for IrisInConfigurationSpace) is bounded,
  setting this flag to `false` will skip the boundedness check that IRIS and
  IrisInConfigurationSpace perform (leading to a small speedup, as checking
  boundedness requires solving optimization problems). If the intersection turns
  out to be unbounded, this will lead to undefined behavior. */
  bool verify_domain_boundedness{true};

  // TODO(cohnt): Document how randomness is used.
  int random_seed{1234};

  // TODO(cohnt): Allow finer control of logging.
  bool verbose{true};

  /** Descision threshold for unadaptive test.*/
  double tau = 0.5;

  /** Upper bound on the error probability that the fraction-in-collision
   * `epsilon` is not met.*/
  double delta = 5e-2;

  /** Admissible fraction of the region volume allowed to be in collision.*/
  double admissible_proportion_in_collision = 1e-2;

  /** Maximum number of "inner iterations" per iteration. For each inner
  iteration, we draw samples, check the Bernoulli trial, and add hyperplanes if
  it fails. */
  int max_iterations_separating_planes{20};

  /** Maximum number of hyperplanes added per inner iteration. */
  int max_hyperplanes_per_iteration{INT_MAX};

  /** The `mixing_steps` parameters is passed to HPolyhedron::UniformSample to
  control the total number of hit-and-run steps taken for each new random
  sample. */
  int mixing_steps{10};

  /** Artificial joint limits are added to continuous revolute joints and planar
  joints with an unbounded revolute degree-of-freedom on a per-region basis. If
  the seed point value for that joint is θ, then the limits are
  θ - π/2 + convexity_radius_stepback and θ + π/2 - convexity_radius_stepback.
  Setting this to a negative number allows growing larger regions, but those
  regions must then be partitioned to be used with GcsTrajectoryOptimization.
  See @ref geometry_optimization_geodesic_convexity for more details.
  IrisInConfigurationSpace throws if this value is not smaller
  than π/2. */
  double convexity_radius_stepback{1e-3};

  /** Optionally specify the solver to use for the counterexample search
   * programs. */
  const solvers::SolverInterface* solver{nullptr};

  /** Optionally specify the SolverOptions used for the counterexample search
   * programs. */
  std::optional<solvers::SolverOptions> solver_options;

  /** Degree of parallelism to be used for collision checking when drawing
   * samples. */
  const Parallelism parallelism = Parallelism::Max();

  /** Passing a meshcat instance may enable debugging visualizations; this
  currently and when the
  configuration space is <= 3 dimensional.*/
  std::shared_ptr<geometry::Meshcat> meshcat{};
};

HPolyhedron IrisNP2(Eigen::VectorXd q, const CollisionChecker& checker,
                    const IrisNP2Options& options = IrisNP2Options());

}  // namespace planning
}  // namespace drake