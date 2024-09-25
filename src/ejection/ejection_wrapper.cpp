// src/ejection/ejection_wrapper.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ejection_ext.cpp"

namespace py = pybind11;

PYBIND11_MODULE(ejection, m) {
    py::class_<EjectionChecker>(m, "EjectionChecker")
        .def(py::init<>())
        .def("find_nearest_neighbour", &EjectionChecker::find_nearest_neighbour);
}