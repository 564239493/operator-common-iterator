"""Test case generator — public API.

This package is the single source of truth for the case generation logic.

The previous mock implementation (case_builder, dtype_picker, shape_sampler, …)
has been replaced by the formal generation pipeline ported from
``operator_case_generator``:

  ``json_constraints`` (raw dict)  →  ``single_operator_handle``  →  ``list[CaseConfig]``

Migration overview
------------------

* The script files from ``operator_case_generator/operator_case_generator/scripts``
  were moved under this package and re-rooted at ``agent.generators.*``.
* ``operator_handle_main.single_operator_handle`` now accepts either a constraint
  JSON file path or an in-memory constraint dict.
* ``facade.TestCaseGenerator`` is the single entry point for node / route / MCP
  callers; the public API stays stable.
* 用例生成主链路直接消费从 MCP / DB 取出的原始 ``json_constraints`` dict，
  不再做 ``GeneratorContext`` 中间层转换；返回值是 ``single_operator_handle``
  的原始输出 ``list[CaseConfig]``。

Public re-exports
-----------------
"""

from __future__ import annotations

from generators.common_model_definition import (
    InterConstraintsRuleType,
    InterParamConstraint,
    OperatorRule,
    ParamAttributes,
    ValueWithSrcText,
)
from generators.facade import (
    DEFAULT_COUNT,
    DEFAULT_SEED,
    TestCaseGenerator,
)
from generators.operator_handle_main import (
    batch_operator_handel,
    single_operator_handle,
)

__all__ = [
    # Public facade
    "DEFAULT_COUNT",
    "DEFAULT_SEED",
    "TestCaseGenerator",
    # Formal generation entry point
    "single_operator_handle",
    "batch_operator_handel",
    # Common constraint models
    "InterConstraintsRuleType",
    "InterParamConstraint",
    "OperatorRule",
    "ParamAttributes",
    "ValueWithSrcText",
]
