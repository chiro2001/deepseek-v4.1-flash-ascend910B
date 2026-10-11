"""Module entry adapter; the frozen API bridge and native math stay unchanged."""
import runpy

if __name__ == '__main__':
    runpy.run_module('parallel_tiny.dspark_api_counter_bridge_cpu.service_bootstrap', run_name='__main__')
