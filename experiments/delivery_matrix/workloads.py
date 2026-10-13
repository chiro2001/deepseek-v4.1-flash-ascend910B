"""Frozen client workloads and isolated code-answer scoring for delivery_matrix.

The Dihuo text is intentionally loaded from the user's local corpus file at run
time.  The repository does not distribute the copyrighted source text.
"""

from __future__ import annotations

import ast
import collections
import functools
import hashlib
import itertools
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
DIHUO_PATH = ROOT / "data" / "dihuo.txt"
DIHUO_SUFFIX_DIR = ROOT / "data" / "dihuo_local"
DIHUO_SHA256 = "00aefd13430d4be45424db3c3f832b6c012d9c60c199518f45feed5d0a57e52b"
DIHUO_SUFFIX_NAMES = (
    "continue.txt",
    "extract.txt",
    "qa.txt",
    "quote.txt",
    "summarize.txt",
)


@dataclass(frozen=True)
class CodeTask:
    task_id: str
    function_name: str
    signature: str
    description: str
    example: str


CODE_TASKS: tuple[CodeTask, ...] = (
    CodeTask(
        "lru_cache",
        "lru_cache",
        "def lru_cache(operations, capacity):",
        "实现一个 LRU 缓存操作器。operations 是按顺序排列的列表，每项是 "
        "[\"get\", key] 或 [\"put\", key, value]。容量为非负整数。get 命中时返回值并将该键设为最近使用；"
        "get 未命中时返回 None；put 更新已有键并刷新其使用顺序；超出容量时删除最久未使用项。"
        "函数返回所有 get 操作结果组成的列表，不修改 operations。",
        "例如 capacity=2 且操作为 put(a,1), put(b,2), get(a), put(c,3), get(b)，结果是 [1, None]。",
    ),
    CodeTask(
        "merge_intervals",
        "merge_intervals",
        "def merge_intervals(intervals):",
        "合并闭区间。输入是 [start, end] 数值区间列表，保证 start <= end。按 start 排序后，"
        "重叠或相接的区间都要合并；返回新的、按 start 升序排列的区间列表，不修改输入。空输入返回空列表。",
        "例如 [[1,3], [3,5], [8,9]] 返回 [[1,5], [8,9]]。",
    ),
    CodeTask(
        "topological_order",
        "topological_order",
        "def topological_order(nodes, edges):",
        "返回有向无环图的字典序最小拓扑序。nodes 是互不重复的字符串节点列表，edges 中的 [u, v]"
        "表示 u 必须排在 v 前面；重复边只算一次。每一步若有多个入度为零的节点，选择字典序最小者。"
        "若图有环，返回空列表。输入列表不得被修改。",
        "例如 nodes=[\"b\",\"a\",\"c\"], edges=[[\"a\",\"c\"]] 返回 [\"a\",\"b\",\"c\"]。",
    ),
    CodeTask(
        "longest_unique_substring",
        "longest_unique_substring",
        "def longest_unique_substring(text):",
        "返回字符串中不含重复字符的最长连续子串长度。按 Unicode 字符比较；空串返回 0。",
        "例如 \"abba\" 返回 2。",
    ),
    CodeTask(
        "kth_largest",
        "kth_largest",
        "def kth_largest(numbers, k):",
        "返回 numbers 中第 k 大的数，k 从 1 开始，重复值分别计数。保证 numbers 非空且 "
        "1 <= k <= len(numbers)。不得修改输入列表。",
        "例如 numbers=[3,1,3,2], k=2 返回 3。",
    ),
    CodeTask(
        "min_window",
        "min_window",
        "def min_window(text, pattern):",
        "返回 text 中覆盖 pattern 全部字符及其重复次数的最短连续子串。若有多个同长答案，"
        "返回最靠左者；pattern 为空或不存在满足条件的子串时返回空字符串。大小写敏感。",
        "例如 text=\"ADOBECODEBANC\", pattern=\"ABC\" 返回 \"BANC\"。",
    ),
    CodeTask(
        "group_anagrams",
        "group_anagrams",
        "def group_anagrams(words):",
        "把变位词分组。两个字符串当且仅当字符及其出现次数完全相同才属于同一组。"
        "组的顺序按该组第一次出现的位置排列，组内字符串保留原始相对顺序。空输入返回空列表；"
        "大小写敏感，不修改输入。",
        "例如 [\"eat\",\"tea\",\"tan\",\"ate\"] 返回 [[\"eat\",\"tea\",\"ate\"],[\"tan\"]]。",
    ),
    CodeTask(
        "flatten_dict",
        "flatten_dict",
        "def flatten_dict(data, sep=\".\"):",
        "递归展开只含字符串键和字典/JSON 标量叶子的嵌套字典。返回新字典，键由路径段用 sep 连接；"
        "根空字典返回空字典；非根空字典作为空字典叶子保留。不得修改输入。",
        "例如 {\"a\": {\"b\": 2}, \"c\": 3} 返回 {\"a.b\": 2, \"c\": 3}。",
    ),
)

CODE_TASK_BY_ID = {task.task_id: task for task in CODE_TASKS}
CODE_TASK_IDS = tuple(task.task_id for task in CODE_TASKS)

# These stable notes supply long-context pressure for code prompts.  They are
# deliberately generic and are explicitly marked as background, not hidden
# requirements.  The task statement is placed after the padding.
CODE_CONTEXT_NOTES = (
    "Repository note: public functions should validate through their documented behavior.\n",
    "Repository note: prefer deterministic results when multiple answers are valid.\n",
    "Repository note: preserve caller-owned input objects unless mutation is required.\n",
    "Repository note: use Python's standard library and avoid external services.\n",
    "Repository note: keep edge cases such as empty collections well-defined.\n",
    "Repository note: return values should be ordinary JSON-compatible Python values.\n",
    "Repository note: complexity should be reasonable for inputs with thousands of items.\n",
    "Repository note: names and behavior should follow the exact function contract.\n",
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_dihuo_material(
    corpus_path: str | Path = DIHUO_PATH,
    suffix_dir: str | Path = DIHUO_SUFFIX_DIR,
    *,
    verify_sha256: bool = True,
) -> dict[str, Any]:
    """Read the frozen source and five suffixes; never fetch the copyrighted text."""
    corpus_path = Path(corpus_path)
    if not corpus_path.is_file():
        raise FileNotFoundError(
            f"地火正文不存在：{corpus_path}。仓库不分发原文；请在本机准备 tools/fetch_corpus.sh "
            "产生的 data/dihuo.txt，或用 --corpus 指定同一份本地正文。"
        )
    raw = corpus_path.read_bytes()
    digest = sha256_bytes(raw)
    if verify_sha256 and digest != DIHUO_SHA256:
        raise ValueError(
            f"地火正文 SHA-256 不匹配：{digest}；预期 {DIHUO_SHA256}。"
            "拒绝用不同语料产生不可比较的测量。"
        )
    corpus = raw.decode("utf-8")
    if not corpus.strip():
        raise ValueError("地火正文为空。")

    suffix_dir = Path(suffix_dir)
    suffixes: list[dict[str, str]] = []
    suffix_hashes: dict[str, str] = {}
    for name in DIHUO_SUFFIX_NAMES:
        path = suffix_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"地火任务后缀缺失：{path}")
        suffix_raw = path.read_bytes()
        text = suffix_raw.decode("utf-8")
        if not text.strip():
            raise ValueError(f"地火任务后缀为空：{path}")
        suffixes.append({"name": name, "text": text})
        suffix_hashes[name] = sha256_bytes(suffix_raw)
    return {
        "corpus": corpus,
        "corpus_sha256": digest,
        "corpus_path": str(corpus_path.resolve()),
        "suffixes": suffixes,
        "suffix_dir": str(suffix_dir.resolve()),
        "suffix_sha256": suffix_hashes,
    }


def repeated_corpus_slice(corpus: str, offset: int, chars: int) -> str:
    """Return a deterministic cyclic slice, allowing contexts longer than the novel."""
    if not corpus:
        raise ValueError("corpus cannot be empty")
    if chars < 0:
        raise ValueError("chars must be non-negative")
    offset %= len(corpus)
    if chars == 0:
        return ""
    doubled = corpus[offset:] + corpus[:offset]
    repeats = (chars + len(corpus) - 1) // len(corpus)
    if repeats <= 1:
        return doubled[:chars]
    # Start each repetition at the same stable offset so the context is a true
    # repeat of one corpus slice rather than a character-count estimate.
    return (corpus * (repeats + 1))[offset : offset + chars]


def dihuo_prompt(sample_index: int, body: str, suffix: str) -> str:
    marker = f"[delivery-matrix-dihuo-{sample_index + 1:02d}]"
    return f"{marker}\n上文：\n{body}\n\n{suffix}"


def build_code_prompt(task_id: str, context_padding: str = "") -> str:
    task = CODE_TASK_BY_ID[task_id]
    background = ""
    if context_padding:
        background = (
            "背景资料：以下重复的通用仓库说明仅用于长上下文负载，不是额外任务要求；"
            "请以末尾明确的问题定义为准。\n"
            + context_padding
            + "\n"
        )
    return (
        f"[delivery-matrix-code-{task_id}]\n"
        "你正在为 Python 3.10+ 实现一个独立函数。只使用 Python 标准库。"
        "最终只输出 Python 代码，可以用一个 ```python 代码块；不要附加解释。\n"
        f"{background}"
        f"任务：{task.description}\n"
        f"函数签名：{task.signature}\n"
        f"示例：{task.example}\n"
        "请严格遵守函数契约，并覆盖空输入、边界和重复值等情况。"
    )


def code_padding_prefix(chars: int, sample_index: int) -> str:
    """Build stable varied filler; final prompt is still calibrated by a tokenizer."""
    if chars <= 0:
        return ""
    offset = sample_index % len(CODE_CONTEXT_NOTES)
    parts: list[str] = []
    total = 0
    cursor = offset
    while total < chars:
        piece = CODE_CONTEXT_NOTES[cursor]
        parts.append(piece)
        total += len(piece)
        cursor = (cursor + 1) % len(CODE_CONTEXT_NOTES)
    return "".join(parts)[:chars]


def _case(case_id: str, args: list[Any], expected: Any) -> dict[str, Any]:
    return {"id": case_id, "args": args, "expected": expected}


def _reference_lru(operations: list[list[Any]], capacity: int) -> list[Any]:
    cache: collections.OrderedDict[Any, Any] = collections.OrderedDict()
    outputs: list[Any] = []
    for operation in operations:
        if operation[0] == "get":
            key = operation[1]
            if key not in cache:
                outputs.append(None)
            else:
                cache.move_to_end(key)
                outputs.append(cache[key])
        elif capacity > 0:
            key, value = operation[1], operation[2]
            if key in cache:
                del cache[key]
            cache[key] = value
            if len(cache) > capacity:
                cache.popitem(last=False)
    return outputs


def _reference_merge(intervals: list[list[int]]) -> list[list[int]]:
    ordered = sorted((start, end) for start, end in intervals)
    result: list[list[int]] = []
    for start, end in ordered:
        if not result or start > result[-1][1]:
            result.append([start, end])
        else:
            result[-1][1] = max(result[-1][1], end)
    return result


def _reference_topological(nodes: list[str], edges: list[list[str]]) -> list[str]:
    ordered_nodes = sorted(nodes)
    prerequisites = {(u, v) for u, v in edges}
    for candidate in itertools.permutations(ordered_nodes):
        position = {node: i for i, node in enumerate(candidate)}
        if all(position[u] < position[v] for u, v in prerequisites):
            return list(candidate)
    return []


def _reference_longest_unique(text: str) -> int:
    best = 0
    for start in range(len(text)):
        seen: set[str] = set()
        for end in range(start, len(text)):
            if text[end] in seen:
                break
            seen.add(text[end])
            best = max(best, end - start + 1)
    return best


def _reference_kth_largest(numbers: list[int], k: int) -> int:
    return sorted(numbers, reverse=True)[k - 1]


def _reference_min_window(text: str, pattern: str) -> str:
    if not pattern:
        return ""
    need = collections.Counter(pattern)
    best = ""
    for left in range(len(text)):
        seen: collections.Counter[str] = collections.Counter()
        for right in range(left, len(text)):
            seen[text[right]] += 1
            if all(seen[ch] >= count for ch, count in need.items()):
                candidate = text[left : right + 1]
                if not best or len(candidate) < len(best):
                    best = candidate
                break
    return best


def _reference_group_anagrams(words: list[str]) -> list[list[str]]:
    groups: list[list[str]] = []
    keys: list[tuple[str, ...]] = []
    for word in words:
        key = tuple(sorted(word))
        try:
            i = keys.index(key)
        except ValueError:
            keys.append(key)
            groups.append([word])
        else:
            groups[i].append(word)
    return groups


def _reference_flatten(data: dict[str, Any], sep: str = ".") -> dict[str, Any]:
    result: dict[str, Any] = {}

    def visit(value: dict[str, Any], prefix: str) -> None:
        if not value:
            if prefix:
                result[prefix] = {}
            return
        for key, item in value.items():
            path = f"{prefix}{sep}{key}" if prefix else key
            if isinstance(item, dict):
                visit(item, path)
            else:
                result[path] = item

    visit(data, "")
    return result


REFERENCE_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "lru_cache": _reference_lru,
    "merge_intervals": _reference_merge,
    "topological_order": _reference_topological,
    "longest_unique_substring": _reference_longest_unique,
    "kth_largest": _reference_kth_largest,
    "min_window": _reference_min_window,
    "group_anagrams": _reference_group_anagrams,
    "flatten_dict": _reference_flatten,
}


def _random_cases(task_id: str) -> list[dict[str, Any]]:
    rng = random.Random(0xD5_4100 + CODE_TASK_IDS.index(task_id))
    cases: list[dict[str, Any]] = []
    for index in range(12):
        case_id = f"random-{index:02d}"
        if task_id == "lru_cache":
            capacity = rng.randrange(0, 5)
            operations: list[list[Any]] = []
            for _ in range(rng.randrange(1, 18)):
                key = f"k{rng.randrange(0, 7)}"
                if rng.random() < 0.58:
                    operations.append(["put", key, rng.randrange(-20, 21)])
                else:
                    operations.append(["get", key])
            args = [operations, capacity]
        elif task_id == "merge_intervals":
            intervals = []
            for _ in range(rng.randrange(0, 13)):
                start = rng.randrange(-12, 13)
                end = start + rng.randrange(0, 9)
                intervals.append([start, end])
            args = [intervals]
        elif task_id == "topological_order":
            nodes = [chr(ord("a") + i) for i in range(rng.randrange(2, 7))]
            edges = [
                [u, v]
                for u in nodes
                for v in nodes
                if u != v and rng.random() < 0.18
            ]
            if edges and rng.random() < 0.2:
                edges.append(list(edges[0]))
            args = [nodes, edges]
        elif task_id == "longest_unique_substring":
            alphabet = "abcz中🙂"
            args = ["".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 22)))]
        elif task_id == "kth_largest":
            values = [rng.randrange(-30, 31) for _ in range(rng.randrange(1, 20))]
            args = [values, rng.randrange(1, len(values) + 1)]
        elif task_id == "min_window":
            alphabet = "abc"
            text = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 12)))
            pattern = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 5)))
            args = [text, pattern]
        elif task_id == "group_anagrams":
            alphabet = "abc"
            words = [
                "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 6)))
                for _ in range(rng.randrange(0, 14))
            ]
            args = [words]
        elif task_id == "flatten_dict":
            def make_dict(depth: int) -> dict[str, Any]:
                result: dict[str, Any] = {}
                for j in range(rng.randrange(0, 4)):
                    key = f"k{j}"
                    if depth > 0 and rng.random() < 0.5:
                        result[key] = make_dict(depth - 1)
                    else:
                        result[key] = rng.choice([None, True, False, -2, 0, 7, "v"])
                return result

            args = [make_dict(3), rng.choice([".", "/", "::"])]
        else:  # pragma: no cover - guarded by fixed task list
            raise KeyError(task_id)
        expected = REFERENCE_FUNCTIONS[task_id](*args)
        cases.append(_case(case_id, args, expected))
    return cases


def fixed_unit_cases(task_id: str) -> list[dict[str, Any]]:
    """Return fixed boundary cases plus seeded, independent reference cases."""
    if task_id not in CODE_TASK_BY_ID:
        raise KeyError(f"unknown code task: {task_id}")
    if task_id == "lru_cache":
        boundaries = [
            _case("empty", [[], 2], []),
            _case("miss", [[["get", "x"]], 2], [None]),
            _case(
                "evicts-least-recent",
                [[
                    ["put", "a", 1],
                    ["put", "b", 2],
                    ["get", "a"],
                    ["put", "c", 3],
                    ["get", "b"],
                    ["get", "c"],
                ], 2],
                [1, None, 3],
            ),
            _case("zero-capacity", [[["put", "x", 1], ["get", "x"]], 0], [None]),
            _case("update-existing", [[["put", "x", 1], ["put", "x", 9], ["get", "x"]], 1], [9]),
        ]
    elif task_id == "merge_intervals":
        boundaries = [
            _case("empty", [[]], []),
            _case("single-point", [[[2, 2]]], [[2, 2]]),
            _case("touching", [[[1, 3], [3, 5], [8, 9]]], [[1, 5], [8, 9]]),
            _case("nested-and-unsorted", [[[8, 10], [1, 7], [3, 4], [12, 12]]], [[1, 7], [8, 10], [12, 12]]),
        ]
    elif task_id == "topological_order":
        boundaries = [
            _case("empty", [[], []], []),
            _case("tie-break", [["b", "a", "c"], [["a", "c"]]], ["a", "b", "c"]),
            _case("duplicate-edge", [["a", "b", "c"], [["a", "b"], ["a", "b"], ["b", "c"]]], ["a", "b", "c"]),
            _case("cycle", [["a", "b", "c"], [["a", "b"], ["b", "c"], ["c", "a"]]], []),
        ]
    elif task_id == "longest_unique_substring":
        boundaries = [
            _case("empty", [""], 0),
            _case("all-same", ["aaaa"], 1),
            _case("middle-repeat", ["abba"], 2),
            _case("unicode", ["你🙂好你"], 3),
        ]
    elif task_id == "kth_largest":
        boundaries = [
            _case("single", [[7], 1], 7),
            _case("duplicate-rank", [[3, 1, 3, 2], 2], 3),
            _case("negative", [[-5, -1, -9], 3], -9),
        ]
    elif task_id == "min_window":
        boundaries = [
            _case("empty-pattern", ["abc", ""], ""),
            _case("no-match", ["abc", "d"], ""),
            _case("multiplicity", ["aabdec", "aabc"], "aabdec"),
            _case("leftmost-tie", ["abxxab", "ab"], "ab"),
            _case("classic", ["ADOBECODEBANC", "ABC"], "BANC"),
        ]
    elif task_id == "group_anagrams":
        boundaries = [
            _case("empty", [[]], []),
            _case("empty-words", [["", "a", ""]], [["", ""], ["a"]]),
            _case("stable-order", [["eat", "tea", "tan", "ate"]], [["eat", "tea", "ate"], ["tan"]]),
            _case("case-sensitive", [["Ab", "ba", "ab"]], [["Ab"], ["ba", "ab"]]),
        ]
    else:  # flatten_dict
        boundaries = [
            _case("root-empty", [{}], {}),
            _case("nested-empty-leaf", [{"a": {"b": {}}, "c": 1}], {"a.b": {}, "c": 1}),
            _case("custom-separator", [{"a": {"b": 2}, "c": 3}, "/"], {"a/b": 2, "c": 3}),
        ]
    return boundaries + _random_cases(task_id)


def extract_python_code(output: str, function_name: str) -> str:
    """Extract a Python fence if present and require the requested function."""
    fences = re.findall(r"```(?:python|py)?\s*\n(.*?)```", output, flags=re.IGNORECASE | re.DOTALL)
    source = fences[0].strip() if fences else output.strip()
    if not source:
        raise ValueError("empty code output")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"Python syntax error: {exc.msg} at line {exc.lineno}") from exc
    top_level_allowed = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    for node in tree.body:
        if not isinstance(node, top_level_allowed):
            raise ValueError(f"unsupported top-level statement: {type(node).__name__}")
    if not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
        for node in ast.walk(tree)
    ):
        raise ValueError(f"required function not found: {function_name}")
    return source + "\n"


def _temp_root() -> Path:
    configured = os.environ.get("TMPDIR")
    root = Path(configured) if configured else Path.home() / "tmp" / "dsv41-workload-bench"
    root.mkdir(parents=True, exist_ok=True)
    return root


_CHILD_RUNNER = r'''import json, os, resource, socket, sys

payload = json.load(sys.stdin)
memory_bytes = int(payload["memory_mb"]) * 1024 * 1024
resource.setrlimit(resource.RLIMIT_CPU, (int(payload["cpu_seconds"]), int(payload["cpu_seconds"])))
resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
resource.setrlimit(resource.RLIMIT_FSIZE, (int(payload["file_size_mb"]) * 1024 * 1024, int(payload["file_size_mb"]) * 1024 * 1024))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
if hasattr(resource, "RLIMIT_NPROC"):
    resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))

def deny(event, args):
    if event.startswith("socket.") or event in {
        "subprocess.Popen", "os.system", "os.fork", "os.forkpty", "os.exec",
        "os.posix_spawn", "os.spawn", "pty.spawn",
    }:
        raise PermissionError("sandbox denied " + event)

sys.addaudithook(deny)
namespace = {"__name__": "__submission__"}
try:
    exec(compile(payload["code"], "<generated-code>", "exec"), namespace, namespace)
    function = namespace.get(payload["function_name"])
    if not callable(function):
        raise TypeError("required function is not callable")
    results = []
    for case in payload["cases"]:
        try:
            actual = function(*case["args"])
            actual_json = json.dumps(actual, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            expected_json = json.dumps(case["expected"], ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            passed = actual_json == expected_json
            results.append({"id": case["id"], "passed": passed, "actual": actual_json[:1000]})
        except BaseException as exc:
            results.append({"id": case["id"], "passed": False, "error": type(exc).__name__ + ": " + str(exc)[:300]})
    sys.__stdout__.write(json.dumps({"case_results": results}, ensure_ascii=False) + "\n")
except BaseException as exc:
    sys.__stdout__.write(json.dumps({"runner_error": type(exc).__name__ + ": " + str(exc)[:500]}, ensure_ascii=False) + "\n")
'''


@functools.lru_cache(maxsize=1)
def _sandbox_runtime_paths() -> tuple[str, tuple[str, ...]]:
    """Select a system Python and only its stdlib/runtime files for bwrap."""
    candidates = [Path("/usr/bin/python3")]
    current = Path(sys.executable).resolve()
    if current not in candidates:
        candidates.append(current)
    errors: list[str] = []
    for candidate in candidates:
        try:
            executable = candidate.resolve(strict=True)
            if not str(executable).startswith(("/usr/", "/opt/")):
                errors.append(f"refusing non-system Python path {executable}")
                continue
            probe = subprocess.run(
                [str(executable), "-I", "-S", "-c", (
                    "import json,sys,sysconfig; "
                    "print(json.dumps({'exe':sys.executable,'stdlib':sysconfig.get_path('stdlib'),"
                    "'dynload':sysconfig.get_config_var('DESTSHARED')}))"
                )],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
                env={"PATH": "/usr/bin:/bin"},
            )
            paths = json.loads(probe.stdout.strip())
            runtime_paths = {str(Path(paths["stdlib"]).resolve(strict=True))}
            dynload = Path(paths["dynload"]).resolve(strict=True)
            runtime_paths.add(str(dynload))
            runtime_paths.add(str(executable))

            readelf = shutil.which("readelf")
            if readelf:
                program = subprocess.run(
                    [readelf, "-l", str(executable)],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                    env={"PATH": "/usr/bin:/bin"},
                )
                interpreter = re.search(r"Requesting program interpreter: ([^]]+)", program.stdout)
                if interpreter:
                    loader = interpreter.group(1).strip()
                    if loader.startswith(("/usr/", "/lib/", "/lib64/")):
                        # Preserve the ELF-requested path; it can be a symlink
                        # whose resolved target would not satisfy the loader.
                        runtime_paths.add(loader)

            # Include the executable's loader and shared libraries, plus direct
            # dependencies of CPython's standard extension modules.  Paths are
            # restricted to system runtime directories; no home/workspace mount
            # can enter the sandbox through ldd output.
            library_roots = ("/usr/lib/", "/usr/lib64/", "/lib/", "/lib64/", "/usr/local/lib/")
            shared_objects = [executable]
            shared_objects.extend(dynload.glob("*.so"))
            ldd = shutil.which("ldd")
            if not ldd:
                raise RuntimeError("ldd is required to construct a minimal Python runtime root")
            for shared_object in shared_objects:
                result = subprocess.run(
                    [ldd, str(shared_object)],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    env={"PATH": "/usr/bin:/bin"},
                )
                for line in result.stdout.splitlines():
                    candidate_path = None
                    if "=>" in line:
                        candidate_path = line.split("=>", 1)[1].split("(", 1)[0].strip()
                    elif line.lstrip().startswith("/"):
                        candidate_path = line.strip().split(" ", 1)[0]
                    if candidate_path and candidate_path.startswith("/"):
                        resolved = (
                            candidate_path
                            if candidate_path.startswith(("/lib/", "/lib64/"))
                            and Path(candidate_path).is_symlink()
                            else str(Path(candidate_path).resolve())
                        )
                        if resolved.startswith(library_roots):
                            runtime_paths.add(resolved)
            return str(executable), tuple(sorted(runtime_paths))
        except Exception as exc:  # try the next installed system interpreter
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
    raise RuntimeError("no safe system Python runtime found for bwrap: " + "; ".join(errors))


def _sandbox_command(sandbox_temp: Path) -> tuple[list[str], str]:
    bwrap = shutil.which("bwrap")
    if bwrap:
        python, runtime_paths = _sandbox_runtime_paths()
        dirs: set[str] = {"/usr", "/tasktmp", "/dev", "/proc"}
        for entry in runtime_paths:
            parent = Path(entry).parent
            while str(parent) != "/":
                dirs.add(str(parent))
                parent = parent.parent
        command = [
            bwrap,
            "--die-with-parent",
            "--new-session",
            "--unshare-net",
            "--tmpfs",
            "/",
        ]
        for directory in sorted(dirs, key=lambda value: (value.count("/"), value)):
            command.extend(["--dir", directory])
        command.extend(["--dev", "/dev", "--proc", "/proc", "--bind", str(sandbox_temp), "/tasktmp"])
        for entry in runtime_paths:
            command.extend(["--ro-bind", entry, entry])
        command.extend(["--chdir", "/tasktmp", python, "-I", "-S", "-c", _CHILD_RUNNER])
        return command, "bubblewrap: minimal Python/stdlib/libs, private task tmp, read-only runtime, private network namespace"
    raise RuntimeError("执行不可信生成代码必须安装 bubblewrap (bwrap)；不提供弱隔离回退。")


def run_untrusted_cases(
    code: str,
    task_id: str,
    cases: list[dict[str, Any]] | None = None,
    *,
    timeout_seconds: float = 3.0,
    memory_mb: int = 512,
    cpu_seconds: int = 2,
) -> dict[str, Any]:
    """Execute generated code in a network-isolated, resource-limited subprocess."""
    task = CODE_TASK_BY_ID[task_id]
    selected_cases = fixed_unit_cases(task_id) if cases is None else cases
    payload = {
        "code": code,
        "function_name": task.function_name,
        "cases": selected_cases,
        "memory_mb": int(memory_mb),
        "cpu_seconds": max(1, int(cpu_seconds)),
        "file_size_mb": 1,
    }
    with tempfile.TemporaryDirectory(prefix="code-score-", dir=_temp_root()) as temp_dir:
        sandbox_temp = Path(temp_dir) / "task-tmp"
        sandbox_temp.mkdir()
        output_path = Path(temp_dir) / "child-output.jsonl"
        isolation: str | None = None
        try:
            command, isolation = _sandbox_command(sandbox_temp)
            with output_path.open("wb") as output_file:
                proc = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=output_file,
                    stderr=subprocess.STDOUT,
                    cwd=temp_dir,
                    env={
                        "PATH": "/usr/bin:/bin",
                        "HOME": "/nonexistent",
                        "TMPDIR": "/tasktmp",
                        "PYTHONIOENCODING": "utf-8",
                    },
                    start_new_session=True,
                )
                try:
                    proc.communicate(json.dumps(payload, ensure_ascii=False).encode("utf-8"), timeout=timeout_seconds)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.communicate()
                    return {
                        "status": "timeout",
                        "error": f"execution exceeded {timeout_seconds:.3f}s wall limit",
                        "passed": 0,
                        "total": len(selected_cases),
                        "case_results": [],
                        "isolation": isolation,
                    }
            raw_output = output_path.read_bytes()
        except Exception as exc:
            return {
                "status": "sandbox_error",
                "error": f"{type(exc).__name__}: {exc}",
                "passed": 0,
                "total": len(selected_cases),
                "case_results": [],
                "isolation": isolation,
            }

    lines = raw_output.decode("utf-8", errors="replace").splitlines()
    result = None
    for line in reversed(lines):
        try:
            result = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    if not isinstance(result, dict) or proc.returncode != 0:
        return {
            "status": "sandbox_error",
            "error": (lines[-1][:1000] if lines else f"sandbox exited {proc.returncode}"),
            "passed": 0,
            "total": len(selected_cases),
            "case_results": [],
            "isolation": isolation,
        }
    if result.get("runner_error"):
        return {
            "status": "execution_error",
            "error": result["runner_error"],
            "passed": 0,
            "total": len(selected_cases),
            "case_results": [],
            "isolation": isolation,
        }
    case_results = result.get("case_results", [])
    passed = sum(bool(row.get("passed")) for row in case_results)
    return {
        "status": "pass" if passed == len(selected_cases) else "fail",
        "passed": passed,
        "total": len(selected_cases),
        "case_results": case_results,
        "failed_case_ids": [row.get("id") for row in case_results if not row.get("passed")],
        "isolation": isolation,
    }


def score_code_output(
    task_id: str,
    output: str,
    *,
    timeout_seconds: float = 3.0,
    memory_mb: int = 512,
    cpu_seconds: int = 2,
) -> dict[str, Any]:
    task = CODE_TASK_BY_ID[task_id]
    try:
        source = extract_python_code(output, task.function_name)
    except ValueError as exc:
        return {
            "status": "parse_error",
            "error": str(exc),
            "extracted_code": None,
            "extracted_code_sha256": None,
            "passed": 0,
            "total": len(fixed_unit_cases(task_id)),
            "case_results": [],
            "failed_case_ids": [],
        }
    result = run_untrusted_cases(
        source,
        task_id,
        timeout_seconds=timeout_seconds,
        memory_mb=memory_mb,
        cpu_seconds=cpu_seconds,
    )
    return {
        **result,
        "extracted_code": source,
        "extracted_code_sha256": sha256_text(source),
    }


def output_cleanliness(output: str) -> dict[str, Any]:
    """Conservative text-integrity screen; semantic correctness is scored separately."""
    reasons: list[str] = []
    stripped = output.strip()
    if not stripped:
        reasons.append("empty")
    if "\ufffd" in output:
        reasons.append("replacement_character")
    # Flag long exact repeated blocks, which catches common degeneration without
    # rejecting ordinary repeated words, lists, or code formatting.
    if len(output) >= 240:
        compact = re.sub(r"\s+", " ", output).strip()
        for width in (80, 120, 160):
            block = compact[:width]
            if len(block) == width and compact.count(block) >= 3:
                reasons.append("repeated_block")
                break
    return {"clean": not reasons, "reasons": reasons}

