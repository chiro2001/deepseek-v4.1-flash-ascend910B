"""CPU source/guard tests with SDK stubs; never starts a model or device."""
import ast
import importlib.util
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
builder = load("physical_builder_cpu", "build_physical_worker.py")
loader = load("delivery_matrix.physical_worker", "physical_worker.py")
COMPONENT = HERE / "tp8_physical_0_7_20261013_v1"
PARENT = "parallel_tiny.dspark_formal_prep"
ALIAS = PARENT + "._delivery_physical_0_7"


class PhysicalWorkerTests(unittest.TestCase):
    def source(self): return (COMPONENT / "worker.py").read_text().replace(builder.NEW, builder.OLD)

    def test_only_one_literal_changes_all_model_sdk_async_graph_checks_stay(self):
        original = self.source()
        derived = builder.derive(original)
        before, after = ast.parse(original), ast.parse(derived)
        changes = [(a.value, b.value) for a, b in zip(ast.walk(before), ast.walk(after))
                   if isinstance(a, ast.Constant) and isinstance(b, ast.Constant) and a.value != b.value]
        self.assertEqual(changes, [("8,9,10,11,12,13,14,15", builder.CHIPS)])
        self.assertEqual(derived.replace(builder.NEW, builder.OLD), original)
        with self.assertRaises(ValueError): builder.derive(original + "\n# drift\n")

    def setup_runtime(self):
        task = TemporaryDirectory(dir=os.environ["TMPDIR"])
        self.addCleanup(task.cleanup)
        root = Path(task.name)
        original = root / builder.BASE_RELATIVE
        original.parent.mkdir(parents=True)
        original.write_text(self.source())
        native_calls = []
        class Native:
            def load_model(self, *a, **kw): native_calls.append(True); return "loaded"
        class Backbone: pass
        torch = ModuleType("torch")
        torch.no_grad = lambda: (lambda function: function)
        sdk = ModuleType(PARENT + ".sdk_gate")
        sdk.inspect_native_factory = lambda: {"compatible": True}
        native = ModuleType("vllm_ascend.worker.worker")
        native.NPUWorker = Native
        model = ModuleType("vllm_ascend.models.deepseek_v41.model")
        model.DeepseekV41Model = Backbone
        distributed = ModuleType("vllm.distributed")
        distributed.get_tensor_model_parallel_rank = lambda: 0
        counter = ModuleType("formal_dspark_counters")
        counter.install_runner = lambda *a, **kw: {"CPU_stub": True}
        parent = ModuleType(PARENT)
        parent.__path__ = []
        modules = {PARENT: parent, "torch": torch, sdk.__name__: sdk, native.__name__: native,
                   model.__name__: model, distributed.__name__: distributed, counter.__name__: counter}
        environment = {key: builder.CHIPS for key in ("STACK_PHYSICAL_CHIPS", "STACK_EXPECTED_CHIPS",
                                                       "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES")}
        environment.update(HCCL_DETERMINISTIC="strict", STACK_TP8_ARM="tp8base",
                           STACK_FP32_DECODE_REDUCTION="0", STACK_REAL_AUDIT="0",
                           STACK_DSPARK_FORMAL_OUTPUT=str(root), DSPARK_ASYNC_COUNTER="1",
                           DSPARK_ASYNC_SCHEDULING="1")
        self.addCleanup(lambda: sys.modules.pop(ALIAS, None))
        config = NS(model_config=NS(hf_text_config=NS(hidden_size=5120, num_hidden_layers=40,
                    n_routed_experts=384, num_experts_per_tok=6, engram_layer_ids=[1,14],
                    num_nextn_predict_layers=3, dspark_target_layer_ids=[37,38,39], dspark_block_size=4)),
                    speculative_config=NS(method="dspark", num_speculative_tokens=5),
                    load_config=NS(load_format="auto"), scheduler_config=NS(async_scheduling=True))
        runner = NS(model=NS(modules=lambda: [Backbone()]), drafter=NS(model=NS(model="draft-backbone")))
        return root, original, modules, environment, config, runner, native_calls

    def test_separate_class_uses_original_sdk_namespace_without_env_or_source_spoofing(self):
        root, original, modules, env, config, runner, calls = self.setup_runtime()
        before = builder.sha(original)
        with patch.dict(os.environ, env), patch.dict(sys.modules, modules):
            cls, receipt = loader.load_base(COMPONENT, root, builder.BASE_CLASS)
            instance = cls()
            instance.vllm_config, instance.model_runner = config, runner
            self.assertEqual(instance.load_model(), "loaded")
            self.assertEqual(calls, [True])
            self.assertEqual(os.environ["STACK_PHYSICAL_CHIPS"], builder.CHIPS)
            self.assertEqual(cls.__module__, ALIAS)
            self.assertFalse(receipt["actual_quality_passed"])
            self.assertEqual(receipt["derived_source_sha256"], builder.sha(COMPONENT / "worker.py"))
        self.assertEqual(builder.sha(original), before)

    def test_wrong_visibility_original_source_or_class_rejected(self):
        root, original, modules, env, *_ = self.setup_runtime()
        with patch.dict(os.environ, env), patch.dict(sys.modules, modules):
            with self.assertRaises(ValueError): loader.load_base(COMPONENT, root, "another.Worker")
            with patch.dict(os.environ, {"ASCEND_VISIBLE_DEVICES": "8,9,10,11,12,13,14,15"}):
                with self.assertRaises(ValueError): loader.load_base(COMPONENT, root, builder.BASE_CLASS)
            original.write_text(original.read_text() + "\n# drift\n")
            with self.assertRaises(ValueError): loader.load_base(COMPONENT, root, builder.BASE_CLASS)

    def test_actual_delivery_description_rechecks_geometry_manifest(self):
        root, _, modules, env, config, runner, _ = self.setup_runtime()
        env.update(DELIVERY_BASE_WORKER=builder.BASE_CLASS, STACK_SRC_ROOT=str(root),
                   DELIVERY_PHYSICAL_WORKER=str(COMPONENT), DELIVERY_FORM="a2-tp8-a3sim",
                   DELIVERY_PHYSICAL_WORKER_MANIFEST_SHA256=builder.sha(COMPONENT / "manifest.json"))
        config.parallel_config = NS(tensor_parallel_size=8, decode_context_parallel_size=1)
        runner.drafter.use_cuda_graph = True
        with patch.dict(os.environ, env), patch.dict(sys.modules, modules):
            worker_module = load("delivery_matrix.physical_delivery_cpu", "worker.py")
            worker = worker_module.DeliveryWorker()
            worker.vllm_config, worker.model_runner = config, runner
            description = worker.delivery_describe()
            self.assertEqual(description["physical_chips"], builder.CHIPS)
            self.assertEqual(description["physical_worker_derivation"]["derived_source_sha256"],
                             builder.sha(COMPONENT / "worker.py"))
            with patch.dict(os.environ, {"DELIVERY_PHYSICAL_WORKER_MANIFEST_SHA256": "f" * 64}):
                with self.assertRaisesRegex(ValueError, "Actually loaded physical worker"):
                    worker.delivery_describe()

    def test_dummy_wrong_full40_async_and_sdk_negative_controls_still_fail(self):
        root, _, modules, env, config, runner, calls = self.setup_runtime()
        with patch.dict(os.environ, env), patch.dict(sys.modules, modules):
            cls, _ = loader.load_base(COMPONENT, root, builder.BASE_CLASS)
            instance = cls()
            instance.vllm_config, instance.model_runner = config, runner
            config.load_config.load_format = "dummy"
            with self.assertRaises(AssertionError): instance.load_model()
            config.load_config.load_format = "auto"
            config.model_config.hf_text_config.num_hidden_layers = 39
            with self.assertRaises(AssertionError): instance.load_model()
            config.model_config.hf_text_config.num_hidden_layers = 40
            with patch.dict(os.environ, {"DSPARK_ASYNC_COUNTER": "0"}):
                with self.assertRaises(AssertionError): instance.load_model()
            modules[PARENT + ".sdk_gate"].inspect_native_factory = lambda: {"compatible": False}
            with self.assertRaisesRegex(ValueError, "bias_vl"): instance.load_model()
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
