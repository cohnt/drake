#include "drake/bindings/pydrake/documentation_pybind.h"
#include "drake/bindings/pydrake/planning/planning_py.h"
#include "drake/bindings/pydrake/pydrake_pybind.h"
#include "drake/planning/iris/iris_common.h"
#include "drake/planning/visibility_graph.h"

namespace drake {
namespace pydrake {
namespace internal {

void DefinePlanningVisibilityGraph(py::module m) {
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  using namespace drake::planning;
  constexpr auto& doc = pydrake_doc.drake.planning;

  m.def("VisibilityGraph", &planning::VisibilityGraph,
      py::arg("ambient_checker"), py::arg("points"),
      py::arg("parallelize") = true,
      py::arg("parameterization") = IrisParameterizationFunction(),
      py::arg("parameterized_checker") = nullptr,
      py::call_guard<py::gil_scoped_release>(), doc.VisibilityGraph.doc);
}

}  // namespace internal
}  // namespace pydrake
}  // namespace drake
