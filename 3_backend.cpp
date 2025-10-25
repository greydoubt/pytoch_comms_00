namespace {
class MyBackendRegistration {
 public:
  MyBackendRegistration() {
    TorchCommFactory::get().register_backend(
        "my_backend", []() { return std::make_shared<MyBackend>(); });
  }
};

static MyBackendRegistration registration{};
}
