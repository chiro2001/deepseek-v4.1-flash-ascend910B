# a3-21 实际 vLLM API middleware 时序（只读源码审查）

日期：2026-10-13。只核验实际 vLLM / middleware loader 源码和哈希；未导入 Python/Torch，未启动或修改服务，未触碰 P/D workload。哈希查询在现有容器中只读执行；middleware dependency 源码另以固定 base image 的 `--read-only --network none` 只读 inspect，并与目标容器 SHA 对照。

## 身份与源码绑定

| 对象 | 身份 / SHA-256 |
|---|---|
| a3-21 目标容器 | `dsv41-real-stack-tp8-20261010-a321`，ID `b1aa15d8800346aff8f755a049feaae148b9d6e315e92bd32dcdf1063b429eb0` |
| 固定 base image | `sha256:b69e58820f929ac7a77f93774a3d96da1961f507628239e9c7d7b24066b7a8ea` |
| vLLM HEAD | `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| 实际 `vllm/entrypoints/openai/api_server.py` | 目标容器路径 `/vllm-workspace/vllm/vllm/entrypoints/openai/api_server.py`；SHA-256 `cd4b83e85dc9d5aae808348d336e59b3064bee697012750d23c786de082a1c53`（目标容器文件直读；与该 HEAD 的 git blob 导出哈希一致） |
| 实际 `vllm/entrypoints/serve/sagemaker/api_router.py` | 目标容器路径 `/vllm-workspace/vllm/vllm/entrypoints/serve/sagemaker/api_router.py`；SHA-256 `f866fbdd60b39377bb493d1e15e48b85ccabad5d8bc3b2b159b3bfd238c5ba30`（目标容器文件直读；与该 HEAD 一致） |
| `model_hosting_container_standards` | version `0.1.16`；`sagemaker/__init__.py` SHA-256 `71b66f5d62a8ed37e43e2c1fb55644c61224228f241a006629e5cf81a720bb09`；`common/fastapi/middleware/core.py` SHA-256 `0a8c5df31f0acdfdea523eed7229b31ddd89cd241ac9fad23e9424833030c989` |
| Starlette | version `0.50.0`；`starlette/applications.py` SHA-256 `c4adbc2c83f6519e59e1a418205a77de495689bb6401d0ce9cb370840485a2b3` |

后三个 dependency 文件在目标容器与固定 base image 的只读文件 SHA 相同；因此下面的 middleware 时序不依赖服务运行状态，也不是只凭网上版本推测。

## `build_app()` 注册顺序

精确 vLLM HEAD 的 `vllm/entrypoints/openai/api_server.py`：

1. 在创建 FastAPI app 后，`build_app()` 先注册 routes/plugins。之后 `:279-285` **无条件**调用 `app.add_middleware(CORSMiddleware, ...)`；`:320-321` 再**无条件**调用 `app.add_middleware(ScalingMiddleware)`。API-key、request-id、realtime、debug middleware 是条件项。
2. `:339-349` 遍历 `args.middleware`。每个配置是 `module.path.symbol` dotted path：用 `middleware.rsplit('.', 1)` 分离模块与对象，再 `importlib.import_module()` / `getattr()` 解析。若对象是 class，则 `app.add_middleware(imported)`；若是 coroutine function，则 `app.middleware("http")(imported)`；其他对象会报 `Invalid middleware`。这正是异步函数或 middleware class 的受支持接入方式。
3. `:351-352` **最后**执行 `app = sagemaker_standards_bootstrap(app)` 然后返回。对应 `api_router.py:101-102` 将其转给 `model_hosting_container_standards.sagemaker.bootstrap(app)`。

## 为什么从 `build_app()` 返回后加 middleware 会失败

实际容器的 SageMaker package `sagemaker/__init__.py:204-231` 在 `bootstrap()` 内调用 `core_load_middlewares(app, ...)`。loader `common/fastapi/middleware/core.py:69-90` 先抓取并保留已有 `app.user_middleware`；由于 `build_app()` 在调用它前已无条件加入 CORS 与 Scaling，`existing_middlewares` 至少含这两项。它们会使 `added_count > 0`。随后 loader 的 `:97-100` 明确执行：

```python
app.middleware_stack = app.build_middleware_stack()
```

所以此处并不是“Uvicorn 已经接收了请求”才冻结 app；middleware stack 在 `sagemaker_standards.bootstrap()` 中、**`build_app()` 返回之前**已经物化。

Starlette `applications.py:72,103-107` 显示 `middleware_stack` 初始化为 `None`，通常会在 ASGI `__call__()` 首次执行时惰性构建；但本镜像的 SageMaker middleware loader 在更早阶段显式构建了它。Starlette `:225-242` 的 `app.middleware("http")` decorator 会调用 `self.add_middleware(...)`；而 `:118-126` 的 `add_middleware()` 在 `middleware_stack is not None` 时抛出 `RuntimeError("Cannot add middleware after an application has started")`。该错误文案表达的是 stack 已成为已构建状态，不等同于 HTTP listener 已打开；v2 的错误 receipt 也记录服务在开始监听前因该调用退出（见 [`ced_v2_startup_attempt_20261013.md`](ced_v2_startup_attempt_20261013.md)）。

## 对 D v2 修复的判断

将 `plan['http_middleware']` 赋给 `config.middleware` 后交给官方 `build_app(config, ...)`，会让其在 `api_server.py:339-349`、SageMaker bootstrap `:351` 之前完成安装；SageMaker loader 随后统一重建 stack，并保留这些 app middleware。该修复满足官方构造时序，不需要在 `build_app()` 返回后改写 `app.middleware_stack`，也不应通过篡改 stack 来规避生命周期检查。

D v2 使用的 async middleware dotted path 与这条 API 兼容；若后续改成同步函数对象，`inspect.iscoroutinefunction()` 分支不会接收它，应改用 middleware class/async callable 形态并在新 job 启动前做静态校验。本轮未重跑任何服务或测试。
