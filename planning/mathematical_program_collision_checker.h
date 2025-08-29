#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "drake/planning/collision_checker.h"
#include "drake/planning/collision_checker_params.h"
#include "drake/solvers/mathematical_program.h"

namespace drake {
namespace planning {

class MathematicalProgramCollisionChecker final : public CollisionChecker {
 public:
  /** @name     Does not allow copy, move, or assignment. */
  /** @{ */
  // N.B. The copy constructor is private for use in implementing Clone().
  void operator=(const MathematicalProgramCollisionChecker&) = delete;
  /** @} */

  /** Creates a new checker with the given params. */
  explicit MathematicalProgramCollisionChecker(CollisionCheckerParams params, std::unique_ptr<solvers::MathematicalProgram> prog);

 private:
  // To support Clone(), allow copying (but not move nor assign).
  explicit MathematicalProgramCollisionChecker(
      const MathematicalProgramCollisionChecker&);

  std::unique_ptr<CollisionChecker> DoClone() const final;

  // No additional actions are required to update positions.
  void DoUpdateContextPositions(CollisionCheckerContext*) const final {}

  bool DoCheckContextConfigCollisionFree(
      const CollisionCheckerContext& model_context) const final;

  std::optional<geometry::GeometryId> DoAddCollisionShapeToBody(
      const std::string&, const multibody::RigidBody<double>&,
      const geometry::Shape&, const math::RigidTransform<double>&) final {
    throw std::runtime_error(
        "AddCollisionShapeToBody is not supported for "
        "MathematicalProgramCollisionChecker.");
  }

  void RemoveAddedGeometries(
      const std::vector<CollisionChecker::AddedShape>&) final {
    throw std::runtime_error(
        "RemoveAddedGeometries is not supported for "
        "MathematicalProgramCollisionChecker.");
  }

  void UpdateCollisionFilters() final {
    throw std::runtime_error(
        "UpdateCollisionFilters is not supported for "
        "MathematicalProgramCollisionChecker.");
  };

  RobotClearance DoCalcContextRobotClearance(const CollisionCheckerContext&,
                                             double) const final {
    throw std::runtime_error(
        "CalcContextRobotClearance is not supported for "
        "MathematicalProgramCollisionChecker.");
  }

  std::vector<RobotCollisionType> DoClassifyContextBodyCollisions(
      const CollisionCheckerContext&) const final {
    throw std::runtime_error(
        "ClassifyContextBodyCollisions is not supported for "
        "MathematicalProgramCollisionChecker.");
  }

  int DoMaxContextNumDistances(const CollisionCheckerContext&) const final {
    throw std::runtime_error(
        "MaxContextNumDistances is not supported for "
        "MathematicalProgramCollisionChecker.");
  }

  copyable_unique_ptr<solvers::MathematicalProgram> prog_;
};

}  // namespace planning
}  // namespace drake
