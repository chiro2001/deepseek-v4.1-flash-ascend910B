"""Private startup facade only; official SDK and frozen bridge bytes unchanged."""
from parallel_tiny.dspark_api_counter_bridge_cpu import service_bootstrap
from parallel_tiny.dspark_api_native_diagnostic_v2.http_facade import NativeHttpFacade

if __name__ == '__main__':
    # This is our private extension's class binding, not an official SDK patch.
    service_bootstrap.PrivateOnlyASGI = NativeHttpFacade
    service_bootstrap.main()
