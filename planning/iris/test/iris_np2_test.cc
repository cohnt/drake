#include "drake/planning/iris/iris_np2.h"

#include <gtest/gtest.h>

#include "drake/common/find_resource.h"
#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/common/test_utilities/maybe_pause_for_user.h"
#include "drake/geometry/meshcat.h"
#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/geometry/optimization/vpolytope.h"
#include "drake/geometry/test_utilities/meshcat_environment.h"
#include "drake/multibody/inverse_kinematics/inverse_kinematics.h"
#include "drake/planning/robot_diagram_builder.h"
#include "drake/planning/scene_graph_collision_checker.h"

namespace drake {
namespace planning {
namespace {

using common::MaybePauseForUser;
using Eigen::Vector2d;
using Eigen::VectorXd;
using geometry::Meshcat;
using geometry::Rgba;
using geometry::Sphere;
using geometry::optimization::HPolyhedron;
using geometry::optimization::Hyperellipsoid;
using geometry::optimization::VPolytope;

// Helper method for testing IRIS-NP2 from a urdf string.
HPolyhedron IrisNP2FromUrdf(const std::string urdf, const VectorXd& q,
                            const IrisNP2Options& options) {
  CollisionCheckerParams params;
  RobotDiagramBuilder<double> builder(0.0);

  builder.parser().package_map().AddPackageXml(FindResourceOrThrow(
      "drake/multibody/parsing/test/box_package/package.xml"));
  params.robot_model_instances =
      builder.parser().AddModelsFromString(urdf, "urdf");

  auto plant_ptr = &(builder.plant());
  plant_ptr->Finalize();

  params.model = builder.Build();
  params.edge_step_size = 0.01;

  planning::SceneGraphCollisionChecker checker(std::move(params));
  // plant.SetPositions(&plant.GetMyMutableContextFromRoot(context.get()),
  // sample);
  return IrisNP2(q, checker, options);
}

// One prismatic link with joint limits.  Iris should return the joint limits.
GTEST_TEST(IrisNP2Test, JointLimits) {
  const std::string limits_urdf = R"(
<robot name="limits">
  <link name="movable">
    <collision>
      <geometry><box size="1 1 1"/></geometry>
    </collision>
  </link>
  <joint name="movable" type="prismatic">
    <axis xyz="1 0 0"/>
    <limit lower="-2" upper="2"/>
    <parent link="world"/>
    <child link="movable"/>
  </joint>
</robot>
)";

  const Vector1d sample = Vector1d::Zero();
  IrisNP2Options options;
  options.verbose = true;
  HPolyhedron region = IrisNP2FromUrdf(limits_urdf, sample, options);

  EXPECT_EQ(region.ambient_dimension(), 1);

  const double kTol = 1e-5;
  const double qmin = -2.0, qmax = 2.0;
  EXPECT_TRUE(region.PointInSet(Vector1d{qmin + kTol}));
  EXPECT_TRUE(region.PointInSet(Vector1d{qmax - kTol}));
  EXPECT_FALSE(region.PointInSet(Vector1d{qmin - kTol}));
  EXPECT_FALSE(region.PointInSet(Vector1d{qmax + kTol}));
}

// A simple double pendulum with link lengths `l1` and `l2` with a sphere at the
// tip of radius `r` between two (fixed) walls at `w` from the origin.  The
// true configuration space is - w + r ≤ l₁s₁ + l₂s₁₊₂ ≤ w - r.  These regions
// are visualized at https://www.desmos.com/calculator/ff0hbnkqhm.
GTEST_TEST(IrisNP2Test, DoublePendulum) {
  const double l1 = 2.0;
  const double l2 = 1.0;
  const double r = .5;
  const double w = 1.83;
  const std::string double_pendulum_urdf = fmt::format(
      R"(
<robot name="double_pendulum">
  <link name="fixed">
    <collision name="right">
      <origin rpy="0 0 0" xyz="{w_plus_one_half} 0 0"/>
      <geometry><box size="1 1 10"/></geometry>
    </collision>
    <collision name="left">
      <origin rpy="0 0 0" xyz="-{w_plus_one_half} 0 0"/>
      <geometry><box size="1 1 10"/></geometry>
    </collision>
  </link>
  <joint name="fixed_link_weld" type="fixed">
    <parent link="world"/>
    <child link="fixed"/>
  </joint>
  <link name="link1"/>
  <joint name="joint1" type="revolute">
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57"/>
    <parent link="world"/>
    <child link="link1"/>
  </joint>
  <link name="link2">
    <collision name="ball">
      <origin rpy="0 0 0" xyz="0 0 -{l2}"/>
      <geometry><sphere radius="{r}"/></geometry>
    </collision>
  </link>
  <joint name="joint2" type="revolute">
    <origin rpy="0 0 0" xyz="0 0 -{l1}"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57"/>
    <parent link="link1"/>
    <child link="link2"/>
  </joint>
</robot>
)",
      fmt::arg("w_plus_one_half", w + .5), fmt::arg("l1", l1),
      fmt::arg("l2", l2), fmt::arg("r", r));

  const Vector2d sample = Vector2d::Zero();
  std::shared_ptr<Meshcat> meshcat = geometry::GetTestEnvironmentMeshcat();
  meshcat->Delete();
  IrisNP2Options options;
  options.verbose = true;
  options.meshcat = meshcat;
  HPolyhedron region = IrisNP2FromUrdf(double_pendulum_urdf, sample, options);

  EXPECT_EQ(region.ambient_dimension(), 2);
  // Confirm that we've found a substantial region.
  EXPECT_GE(region.MaximumVolumeInscribedEllipsoid().Volume(), 2.0);

  EXPECT_TRUE(region.PointInSet(Vector2d{.4, 0.0}));
  EXPECT_FALSE(region.PointInSet(Vector2d{.5, 0.0}));
  EXPECT_TRUE(region.PointInSet(Vector2d{.3, .3}));
  EXPECT_FALSE(region.PointInSet(Vector2d{.4, .3}));
  EXPECT_TRUE(region.PointInSet(Vector2d{-.4, 0.0}));
  EXPECT_FALSE(region.PointInSet(Vector2d{-.5, 0.0}));
  EXPECT_TRUE(region.PointInSet(Vector2d{-.3, -.3}));
  EXPECT_FALSE(region.PointInSet(Vector2d{-.4, -.3}));

  {
    meshcat->Set2dRenderMode(math::RigidTransformd(Eigen::Vector3d{0, 0, 1}),
                             -3.25, 3.25, -3.25, 3.25);
    meshcat->SetProperty("/Grid", "visible", true);
    Eigen::RowVectorXd theta2s =
        Eigen::RowVectorXd::LinSpaced(100, -1.57, 1.57);
    Eigen::Matrix3Xd points = Eigen::Matrix3Xd::Zero(3, 2 * theta2s.size() + 1);
    const double c = -w + r;
    for (int i = 0; i < theta2s.size(); ++i) {
      const double a = l1 + l2 * std::cos(theta2s[i]),
                   b = l2 * std::sin(theta2s[i]);
      // wolfram solve a*sin(q) + b*cos(q) = c for q
      points(0, i) =
          2 * std::atan((std::sqrt(a * a + b * b - c * c) + a) / (b + c)) +
          M_PI;
      points(1, i) = theta2s[i];
      points(0, points.cols() - i - 2) =
          2 * std::atan((std::sqrt(a * a + b * b - c * c) + a) / (b - c)) -
          M_PI;
      points(1, points.cols() - i - 2) = theta2s[i];
    }
    points.col(points.cols() - 1) = points.col(0);
    meshcat->SetLine("True C_free", points, 2.0, Rgba(0, 0, 1));
    VPolytope vregion = VPolytope(region).GetMinimalRepresentation();
    points.resize(3, vregion.vertices().cols() + 1);
    points.topLeftCorner(2, vregion.vertices().cols()) = vregion.vertices();
    points.topRightCorner(2, 1) = vregion.vertices().col(0);
    points.bottomRows<1>().setZero();
    meshcat->SetLine("IRIS Region", points, 2.0, Rgba(0, 1, 0));

    MaybePauseForUser();
  }
}

const char block_urdf[] = R"(
<robot name="block">
  <link name="fixed">
    <collision name="ground">
      <origin rpy="0 0 0" xyz="0 0 -1"/>
      <geometry><box size="10 10 2"/></geometry>
    </collision>
  </link>
  <joint name="fixed_link_weld" type="fixed">
    <parent link="world"/>
    <child link="fixed"/>
  </joint>
  <link name="link1"/>
  <joint name="joint1" type="prismatic">
    <axis xyz="0 0 1"/>
    <limit lower="0" upper="3.0"/>
    <parent link="world"/>
    <child link="link1"/>
  </joint>
  <link name="link2">
    <collision name="block">
      <geometry><box size="2 1 1"/></geometry>
    </collision>
  </link>
  <joint name="joint2" type="revolute">
    <axis xyz="0 1 0"/>
    <limit lower="-3.14159" upper="3.14159"/>
    <parent link="link1"/>
    <child link="link2"/>
  </joint>
</robot>
)";

// A block on a vertical track, free to rotate (in the plane) with width `w` of
// 2 and height `h` of 1, plus a ground plane at z=0.  The true configuration
// space is min(q₀ ± .5w sin(q₁) ± .5h cos(q₁)) ≥ 0, where the min is over the
// ±. This region is also visualized at
// https://www.desmos.com/calculator/ok5ckpa1kp.
GTEST_TEST(IrisNP2Test, BlockOnGround) {
  const Vector2d sample{1.0, 0.0};
  std::shared_ptr<Meshcat> meshcat = geometry::GetTestEnvironmentMeshcat();
  meshcat->Delete();
  IrisNP2Options options;
  options.verbose = true;
  options.meshcat = meshcat;
  HPolyhedron region = IrisNP2FromUrdf(block_urdf, sample, options);

  EXPECT_EQ(region.ambient_dimension(), 2);
  // Confirm that we've found a substantial region.
  EXPECT_GE(region.MaximumVolumeInscribedEllipsoid().Volume(), 2.0);

  {
    meshcat->Set2dRenderMode(math::RigidTransformd(Eigen::Vector3d{0, 0, 1}), 0,
                             3.25, -3.25, 3.25);
    meshcat->SetProperty("/Grid", "visible", true);
    Eigen::RowVectorXd thetas = Eigen::RowVectorXd::LinSpaced(100, -M_PI, M_PI);
    const double w = 2, h = 1;
    Eigen::Matrix3Xd points = Eigen::Matrix3Xd::Zero(3, 2 * thetas.size() + 1);
    for (int i = 0; i < thetas.size(); ++i) {
      const double a = 0.5 *
                       (-w * std::sin(thetas[i]) - h * std::cos(thetas[i])),
                   b = 0.5 *
                       (-w * std::sin(thetas[i]) + h * std::cos(thetas[i])),
                   c = 0.5 *
                       (+w * std::sin(thetas[i]) - h * std::cos(thetas[i])),
                   d = 0.5 *
                       (+w * std::sin(thetas[i]) + h * std::cos(thetas[i]));
      points(0, i) = std::max({a, b, c, d});
      points(1, i) = thetas[i];
      points(0, points.cols() - i - 2) = 3.0;
      points(1, points.cols() - i - 2) = thetas[i];
    }
    points.col(points.cols() - 1) = points.col(0);
    meshcat->SetLine("True C_free", points, 2.0, Rgba(0, 0, 1));
    VPolytope vregion = VPolytope(region).GetMinimalRepresentation();
    points.resize(3, vregion.vertices().cols() + 1);
    points.topLeftCorner(2, vregion.vertices().cols()) = vregion.vertices();
    points.topRightCorner(2, 1) = vregion.vertices().col(0);
    points.bottomRows<1>().setZero();
    meshcat->SetLine("IRIS Region", points, 2.0, Rgba(0, 1, 0));

    MaybePauseForUser();
  }
}

// A (somewhat contrived) example of a concave configuration-space obstacle
// (resulting in a convex configuration-space, which we approximate with
// polytopes):  A simple pendulum of length `l` with a sphere at the tip of
// radius `r` on a vertical track, plus a ground plane at z=0.  The
// configuration space is given by the joint limits and z + l*cos(theta) >= r.
// The region is also visualized at
// https://www.desmos.com/calculator/flshvay78b. In addition to testing the
// convex space, this was originally a test for which Ibex found
// counter-examples that Snopt missed; now Snopt succeeds due to having
// options.num_collision_infeasible_samples > 1.
GTEST_TEST(IrisNP2Test, ConvexConfigurationSpace) {
  const double l = 1.5;
  const double r = 0.1;

  std::shared_ptr<Meshcat> meshcat = geometry::GetTestEnvironmentMeshcat();
  meshcat->Delete();
  meshcat->Set2dRenderMode(math::RigidTransformd(Eigen::Vector3d{0, 0, 1}),
                           -3.25, 3.25, -3.25, 3.25);
  meshcat->SetProperty("/Grid", "visible", true);
  Eigen::RowVectorXd theta1s = Eigen::RowVectorXd::LinSpaced(100, -1.5, 1.5);
  Eigen::Matrix3Xd points = Eigen::Matrix3Xd::Zero(3, 2 * theta1s.size());
  for (int i = 0; i < theta1s.size(); ++i) {
    points(0, i) = r - l * cos(theta1s[i]);
    points(1, i) = theta1s[i];
    points(0, points.cols() - i - 1) = 0;
    points(1, points.cols() - i - 1) = theta1s[i];
  }
  meshcat->SetLine("True C_free", points, 2.0, Rgba(0, 0, 1));

  const std::string convex_urdf = fmt::format(
      R"(
<robot name="pendulum_on_vertical_track">
  <link name="fixed">
    <collision name="ground">
      <origin rpy="0 0 0" xyz="0 0 -1"/>
      <geometry><box size="10 10 2"/></geometry>
    </collision>
  </link>
  <joint name="fixed_link_weld" type="fixed">
    <parent link="world"/>
    <child link="fixed"/>
  </joint>
  <link name="cart">
  </link>
  <joint name="track" type="prismatic">
    <axis xyz="0 0 1"/>
    <limit lower="-{l}" upper="0"/>
    <parent link="world"/>
    <child link="cart"/>
  </joint>
  <link name="pendulum">
    <collision name="ball">
      <origin rpy="0 0 0" xyz="0 0 {l}"/>
      <geometry><sphere radius="{r}"/></geometry>
    </collision>
  </link>
  <joint name="pendulum" type="revolute">
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57"/>
    <parent link="cart"/>
    <child link="pendulum"/>
  </joint>
</robot>
)",
      fmt::arg("l", l), fmt::arg("r", r));

  const Vector2d sample{-0.5, 0.0};
  IrisNP2Options options;

  options.iteration_limit = 1;
  options.max_iterations_separating_planes = 5;

  // This point should be outside of the configuration space (in collision).
  // The particular value was found by visual inspection using meshcat.
  const double z_test = 0, theta_test = -1.55;
  // Confirm that the pendulum is colliding with the wall with true kinematics:
  EXPECT_LE(z_test + l * std::cos(theta_test), r);

  // Turn on meshcat for addition debugging visualizations.
  // This example is truly adversarial for IRIS. After one iteration, the
  // maximum-volume inscribed ellipse is approximately centered in C-free. So
  // finding a counter-example in the bottom corner (near the test point) is
  // not only difficult because we need to sample in a corner of the polytope,
  // but because the objective is actually pulling the counter-example search
  // away from that corner. Open the meshcat visualization to step through the
  // details!
  options.meshcat = meshcat;
  options.verbose = true;
  // std::this_thread::sleep_for(std::chrono::milliseconds(100));
  HPolyhedron region = IrisNP2FromUrdf(convex_urdf, sample, options);
  // TODO(russt): Expecting the test point to be outside the verified region is
  // too strong of a requirement right now. If we can improve the algorithm then
  // we should make this EXPECT_FALSE.
  if (!region.PointInSet(Vector2d{z_test, theta_test})) {
    log()->info("Our test point is not in the set");
  }

  EXPECT_EQ(region.ambient_dimension(), 2);
  // Confirm that we've found a substantial region.
  EXPECT_GE(region.MaximumVolumeInscribedEllipsoid().Volume(), 0.5);

  {
    VPolytope vregion = VPolytope(region).GetMinimalRepresentation();
    points.resize(3, vregion.vertices().cols() + 1);
    points.topLeftCorner(2, vregion.vertices().cols()) = vregion.vertices();
    points.topRightCorner(2, 1) = vregion.vertices().col(0);
    points.bottomRows<1>().setZero();
    meshcat->SetLine("IRIS Region", points, 2.0, Rgba(0, 1, 0));

    meshcat->SetObject("Test point", Sphere(0.03), Rgba(1, 0, 0));
    meshcat->SetTransform("Test point", math::RigidTransform(Eigen::Vector3d(
                                            z_test, theta_test, 0)));

    MaybePauseForUser();
  }
}

}  // namespace
}  // namespace planning
}  // namespace drake