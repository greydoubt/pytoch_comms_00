PYBIND11_MODULE(_comms_my_backend, m) {
  py::class_<MyBackend, std::shared_ptr<MyBackend>>(m, "MyBackend");
}
