#include "drake/planning/mathematical_program_collision_checker.h"

#include "drake/planning/iris/iris_common.h"

namespace drake {
namespace planning {

MathematicalProgramCollisionChecker::MathematicalProgramCollisionChecker(
    CollisionCheckerParams params,
    std::unique_ptr<solvers::MathematicalProgram> prog)
    : CollisionChecker(std::move(params), prog->IsThreadSafe()),
      prog_(std::move(prog)) {
  if (prog != nullptr) {
    DRAKE_THROW_UNLESS(prog_->num_vars() != plant().num_positions());
  }
  AllocateContexts();
}

MathematicalProgramCollisionChecker::MathematicalProgramCollisionChecker(
    const MathematicalProgramCollisionChecker&) = default;

std::unique_ptr<CollisionChecker> MathematicalProgramCollisionChecker::DoClone()
    const {
  // N.B. We cannot use make_unique due to private-only access.
  return std::unique_ptr<MathematicalProgramCollisionChecker>(
      new MathematicalProgramCollisionChecker(*this));
}

bool MathematicalProgramCollisionChecker::DoCheckContextConfigCollisionFree(
    const CollisionCheckerContext& model_context) const {
  return internal::CheckProgConstraints(
      prog_.get(), plant().GetPositions(model_context.plant_context()), 1e-8);
}

}  // namespace planning
}  // namespace drake
