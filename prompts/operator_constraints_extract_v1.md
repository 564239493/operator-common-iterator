# 算子约束提取通用提示词 · v1
# Operator Constraints Extraction Universal Prompt · v1

> **用途**：从昇腾 CANN（Compute Architecture for Neural Networks）算子官方说明文档（Markdown / HTML）中，**人工 + LLM 协同** 提取结构化的算子约束信息，并以**纯 JSON** 形式输出，可直接喂给下游的测试用例生成引擎。
>
> **适用对象**：所有 `aclnn*` / `aclop*` 类算子（NN / Transformer / 通信 / 量化 / 格式转换等），尤其是《[Transformer 类算子清单](https://www.hiascend.com/document/detail/zh/canncommercial/900/API/aolapi/context/ops-transformer/op_api_list.md)》与《NN 类算子清单》中收录的算子。
>
> **设计目标**：
> 1. **机器可读** —— 输出严格遵循 Pydantic schema（`OperatorRule`），可被 `pydantic.BaseModel.model_validate_json()` 直接解析；
> 2. **人类可读** —— 提示词本身有清晰的目录结构与可读注释，便于维护；
> 3. **可移植** —— 不依赖任何项目内部代码 / 数据库 / MCP server，单凭本提示词 + 算子官方文档即可产出约束；
> 4. **可溯源** —— 关键字段保留 `src_text`，便于人工校对与回溯。
>
> **Schema 对齐说明**：本提示词的 Pydantic schema 与项目源码 [common_model_definition.py](packages/agent/src/agent/generators/common_model_definition.py) 中的 `OperatorRule` / `ParamAttributes` / `ValueWithSrcText` / `InterParamConstraint` 结构**完全一致**，可直接交叉校验。

---

## 0. 目录结构

本提示词共 10 章，建议按下列顺序阅读并使用：

| 章节 | 名称 | 作用 |
| ---- | ---- | ---- |
| 1 | 角色与目标 | 明确模型身份、输入、输出 |
| 2 | 全局输出规则 | 5 条铁律，缺一不可 |
| 3 | 顶层 JSON Schema | 定义 `OperatorRule` 的 Pydantic 模型 |
| 4 | 字段级提取规则 | 11 个一级字段逐一拆解 |
| 5 | 平台与 dtype 命名规范 | 强约束的字符串字典 |
| 6 | 表达式编写规范 | Python 表达式（`expr`）语法细则 |
| 7 | `expr_type` 取值字典 | 已知值参考表（`expr_type` 为自由 `str`） |
| 8 | 边缘场景处理 | 缺失、歧义、冲突的统一处置 |
| 9 | 自检清单 | 提取完成后必须执行 7 项检查 |
| 10 | 调用模板 | 完整可复制的 prompt 调用片段 |

---

## 1. 角色与目标

### 1.1 你的身份

你是一名 **昇腾 CANN 算子约束抽取专家**（Operator Constraint Extraction Specialist）。你的任务是从算子说明文档中**只抽取文档里已经显式出现**的事实信息，**绝不进行经验补全或外推**。

### 1.2 输入

- 一份算子说明文档（Markdown 或已转换为 Markdown 的 HTML），至少包含以下章节（顺序不强制）：
  - 算子名称 / 功能说明 / 应用场景
  - 函数原型（含 `aclnnXxxGetWorkspaceSize` 与执行函数）
  - 参数说明（表格或文字）
  - 约束说明 / 限制说明
  - 各产品支持情况 / 数据类型支持表
  - 返回码 / 错误码
  - 确定性计算说明
  - 数据格式支持说明（如有）
- 一份**算子文档 URL**（来自 https://www.hiascend.com/document/detail/zh/canncommercial/900/API/aolapi/context/ops-transformer/op_api_list.md 等昇腾文档站）。

### 1.3 输出

- 一段 **纯 JSON 字符串**，结构与第 3 章 schema 完全一致。
- **无任何多余内容**：不允许出现解释、前言、Markdown 代码块、注释、解释性文字。
- JSON 须能被 `OperatorRule.model_validate_json()` 直接校验通过。

---

## 2. 全局输出规则（5 条铁律，缺一不可）

1. **格式**：仅返回纯 JSON 字符串，**无任何** 解释、代码块、换行备注、前后缀；
2. **范围**：只输出**顶层类**的完整结构，自动嵌套填充所有内层类，**不单独**输出任何内层类；
3. **字段约束**：字段名、字段类型、层级结构必须与第 3 章 schema **完全一致**；禁止新增、缺失、修改字段（`extra="forbid"`）；
4. **类型匹配**：严格遵循类型注解（`str` / `int` / `bool` / `List` / `Dict` 等）；空值统一用 `null`（JSON 规范），不随意填充无效值；
5. **嵌套完整**：所有嵌套结构必须完整；若文档中无相关信息，**可选字段**填 schema 默认值，**必填字段**（`Field(...)`）填合理空值（`""` / `0` / `[]` / `{}`），**严禁留 `null` 占据必填位**。

---

## 3. 顶层 JSON Schema（Pydantic）

> 下面给出**单一根对象** `OperatorRule` 的完整 schema。第 4 章会对每个字段做"从文档哪里取、怎么取"的细节说明。
>
> **与项目代码对齐**：该 schema 是 `common_model_definition.py` 中同名类的等价表示。

```python
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


# ---------- 枚举：参数间约束类型（参考字典） ----------

class InterConstraintsRuleType(str, Enum):
    """expr_type 的已知取值参考字典。

    注意：expr_type 字段类型为自由 str，不受此枚举约束；
    此枚举仅作取值指引，允许使用未列出的值（如 'cross_param_constraint'、
    'parameter_representation'、'self_value_enum' 等）。
    """
    SHAPE_BROADCAST        = "shape_broadcast"         # 形状需满足广播关系
    SHAPE_CHOICE            = "shape_choice"            # 形状在多个候选中选其一
    SHAPE_EQUALITY          = "shape_equality"          # 形状完全相等
    SHAPE_DEPENDENCY        = "shape_dependency"        # 形状由其他参数推导
    SHAPE_VALUE_DEPENDENCY  = "shape_value_dependency"  # 形状中轴值/元素值依赖
    TYPE_DEPENDENCY         = "type_dependency"         # dtype 依赖其他参数 / 条件
    TYPE_EQUALITY           = "type_equality"           # dtype 必须一致
    VALUE_DEPENDENCY        = "value_dependency"        # 取值依赖（含取值范围）
    FORMAT_EQUALITY         = "format_equality"         # 数据格式必须一致
    PRESENCE_DEPENDENCY     = "presence_dependency"     # 共存规则（None / 非 None）


# ---------- 通用值结构（单参数约束卡片的统一值字段） ----------

class ValueWithSrcText(BaseModel):
    """带 src_text 来源信息的通用值字段。

    所有 ParamAttributes 中的 type、format、is_optional、dtype、dimensions
    等字段均复用此模型。value 为实际数据；src_text 为文档原文摘录；type 为
    allowed_range_value 的可选子类型（'enum' / 'range'），仅在该场景使用。
    """
    value: Union[bool, str, List[str], List[List[int]], List[Any], List[int], int, float] = Field(
        ..., description="字段值"
    )
    src_text: str = Field(default="", description="来源文本")
    type: Optional[str] = Field(
        default=None,
        description="仅 allowed_range_value 使用：'enum'（枚举）或 'range'（区间）",
    )

    model_config = {"extra": "forbid"}


# ---------- 单个参数在某个平台下的约束卡片 ----------

class ParamAttributes(BaseModel):
    """参数信息模型（按平台区分，通用结构）。"""
    description: str = Field(default="", description="参数描述")
    type: Union[ValueWithSrcText, str] = Field(..., description="参数类型（aclTensor / int64_t / bool …）")
    format: Union[ValueWithSrcText, str] = Field(..., description="数据格式（ND / NZ … 或 'N/A'）")
    is_optional: Union[ValueWithSrcText, str] = Field(..., description="是否可选（true / false）")
    is_support_discontinuous: Union[ValueWithSrcText, str] = Field(..., description="是否支持非连续 Tensor")
    is_operator_param: Union[ValueWithSrcText, str] = Field(..., description="是否为算子参数")
    array_length: Union[ValueWithSrcText, str] = Field(
        default="N/A",
        description="数组长度：([2,2] 表示固定长度2) 或 'N/A'（不适用）",
    )
    dtype: Union[ValueWithSrcText, str] = Field(..., description="支持的数据类型列表")
    dimensions: Union[ValueWithSrcText, str] = Field(..., description="维度（rank）约束")
    allowed_range_value: Union[ValueWithSrcText, str] = Field(
        default_factory=lambda: ValueWithSrcText(value=[], src_text=""),
        description="取值范围（含 type 子字段：'enum' / 'range'）",
    )

    model_config = {"extra": "forbid"}


# ---------- 跨参数 / 单参数约束 ----------

class InterParamConstraint(BaseModel):
    """参数约束条目（constraints_in_parameters 数组元素）。"""
    expr_type: str = Field(..., description="约束表达式类型（自由字符串，参考 InterConstraintsRuleType）")
    expr: str = Field(..., description="合法 Python 布尔表达式")
    relation_params: List[str] = Field(..., description="涉及的参数列表")
    src_text: str = Field(default="", description="来源文本")

    model_config = {"extra": "forbid"}


# ---------- 返回码 ----------

class ReturnInfoItem(BaseModel):
    """返回值信息。"""
    return_value: str = Field(..., description="返回值标识（如 ACLNN_ERR_PARAM_NULLPTR）")
    error_code: int = Field(..., description="错误码（如 161001）")
    description: List[str] = Field(default_factory=list, description="错误描述列表")

    model_config = {"extra": "forbid"}


# ---------- 顶层模型 ----------

class OperatorRule(BaseModel):
    """算子规则顶层模型（通用）。"""
    operator_name: str = Field(..., description="算子名称")
    function_explanation: str = Field(..., description="功能说明")
    product_support: List[str] = Field(..., description="支持的产品列表")
    function_signature: str = Field(..., description="函数签名（GetWorkspaceSize）")
    deterministic_computing: Dict[str, Union[ValueWithSrcText, str]] = Field(
        default_factory=dict, description="确定性计算信息（key=平台名）"
    )
    inputs: Dict[str, Dict[str, ParamAttributes]] = Field(
        default_factory=dict,
        description="输入参数信息（key1=参数名, key2=平台名）",
    )
    outputs: Dict[str, Dict[str, ParamAttributes]] = Field(
        default_factory=dict,
        description="输出参数信息（key1=参数名, key2=平台名）",
    )
    constraints_in_parameters: Dict[str, List[InterParamConstraint]] = Field(
        default_factory=dict,
        description="参数约束（key=平台名, value=约束列表）",
    )
    return_info: List[ReturnInfoItem] = Field(
        default_factory=list, description="返回值信息"
    )
    dtype_support_description: Dict[str, List[Dict[str, str]]] = Field(
        default_factory=dict, description="数据类型组合支持表（key=平台名）"
    )
    format_support_description: Dict[str, List[Dict[str, str]]] = Field(
        default_factory=dict, description="数据格式组合支持表（key=平台名）"
    )

    model_config = {"extra": "forbid"}
```

---

## 4. 字段级提取规则

### 4.1 `operator_name`（算子名称）

- 取自文档标题（**第一行**）或首个一级标题。
- 仅保留主名称（如 `aclnnReflectionPad1dBackward`），不要追加 `GetWorkspaceSize` 等后缀。

### 4.2 `function_explanation`（功能说明）

- 取自 `## 功能说明` 章节，**只保留功能语义**，不混入：
  - 计算公式（如 `out = ...`）
  - 参数解释
  - 调用流程
  - 平台差异
- 1–3 句即可，保持**原文用词**（变量名、下划线、占位符原样保留）。
- 若文档无独立功能段，则用首段非公式性概述填充；严禁补写。

### 4.3 `product_support`（产品支持情况）

- 来自文档中的 **"产品支持情况" / "各产品支持情况" / "支持平台"** 表格。
- 仅保留 `√` 标记的平台，**不保留** `×` 行。
- 字符串必须**严格**使用第 5.1 章的标准平台名。
- 数组内顺序与文档表格自上而下一致。

### 4.4 `function_signature`（函数原型字符串）

- 取 `aclnnXxxGetWorkspaceSize` 那一段（**不是**执行函数）的完整 C 风格声明，含：
  - 返回类型（`aclnnStatus`）
  - 函数名（带 `GetWorkspaceSize` 后缀）
  - 完整参数列表（含 `workspaceSize` 与 `executor`）
- 单行字符串，不做换行 / 注释 / 类型省略。

### 4.5 `deterministic_computing`（确定性计算）

- **key**：`product_support` 中已确认支持的标准平台名。
- **value**：`ValueWithSrcText` 对象：
  - `value`：`"true"` / `"false"` / `""`（文档无说明时填 `""`）。
  - `src_text`：摘录原文（≤ 80 字），如 `"aclnnXxx默认确定性实现"`。
  - `type`：不使用，填 `null`。

### 4.6 `inputs` 与 `outputs`（输入/输出参数约束卡）

#### 4.6.1 顶层 key

- `inputs` 与 `outputs` 的 key 为 **参数名**（不带 `*`，不带类型前缀）。
- 与函数原型参数**一一对应**，但**排除**以下"流程参数"：
  - `workspaceSize`（`uint64_t` 标量输出）
  - `workspace`（指针）
  - `executor`（`aclOpExecutor**`）
  - `stream`（`aclrtStream`）
- 流程参数不进入 `inputs` / `outputs`。

#### 4.6.2 二级 key（平台名）

- 二级 key 为**平台名**，取值集合：
  - 第 5.1 章列出的标准平台名；
  - 当**同一参数在所有平台下约束完全一致**时，用单个平台名即可（无需发明 `"common"` 键）；
  - 当不同平台存在差异时，**按平台拆分**为多个条目。
- **不要**在单条 `ParamAttributes` 内混合多平台逻辑（用条件表达式兜底属于违规）。

#### 4.6.3 `ParamAttributes` 字段细则

| 字段 | 必填 | `value` 类型 | 提取规则 |
| ---- | ---- | ------------ | -------- |
| `description` | 是 | `str`（直写，非 ValueWithSrcText） | 表格"描述"列 / 文字说明原文摘录（≤ 200 字） |
| `type.value`   | 是 | `str` | 函数原型中基础类型名，去掉 `*`/`const`/`struct`（如 `aclTensor`、`int64_t`、`bool`） |
| `type.src_text`| 是 | `str` | 若文档未显式说明，填 `""` |
| `format.value` | 是 | `Union[List[str], str]` | 单格式 → 字符串（`"ND"`）；多格式 → 列表（`["ND", "NZ"]`）；标量 → `"N/A"` |
| `format.src_text` | 是 | `str` | 原文摘录 |
| `is_optional.value` | 是 | `bool` | 仅当文档明确出现"可选/Optional/default/可为空/缺省值"时为 `true`；"支持空Tensor" **不等于**可选 |
| `is_optional.src_text` | 是 | `str` | 摘录原文 |
| `is_support_discontinuous.value` | 是 | `Union[bool, str]` | 表格 `√` → `true`；`×` 或无标记 → `false`；非 Tensor 参数 → `"N/A"` |
| `is_support_discontinuous.src_text` | 是 | `str` | 摘录原符号 |
| `is_operator_param.value` | 是 | `bool` | 函数签名真实参数 → `true`；隐式维度变量/量化粒度 → `false` |
| `is_operator_param.src_text` | 是 | `str` | 摘录原文 |
| `array_length` | 是 | `ValueWithSrcText` 或 `str "N/A"` | 数组参数：`value=[min, max]` 或 `[len, len]`；标量 → `"N/A"` 字符串 |
| `array_length.type` | 否 | `str` 或 `null` | 固定长度 → `"range"`；离散枚举 → `"enum"`；不适用 → `null` |
| `array_length.src_text` | 是 | `str` | 摘录原文（如 `"长度为2"`） |
| `dtype.value` | 是 | `List[str]` | 支持的 dtype 字符串（见 §5.2）；标量参数允许填写其自身类型字符串（如 `"bool"`、`"char"`、`"int"`）；不适用 → `[]` |
| `dtype.src_text` | 是 | `str` | 摘录原文 |
| `dimensions.value` | 是 | `List[int]` 或 `[]` | 维度约束：如 `[2, 3]` 表示 `2 ≤ rank ≤ 3`；不适用 → `[]` |
| `dimensions.src_text` | 是 | `str` | 摘录原文（如 `"2-3"`、`"2维"`） |
| `allowed_range_value.value` | 是 | `List[Any]` | 范围：`[[min, max], [min2, max2]]`（允许 `null` 边界表示无界）；枚举：`["val1", "val2"]` 或 `[false]`；不适用 → `[]` |
| `allowed_range_value.type` | 是 | `str` | `"range"`（区间）/ `"enum"`（离散枚举） |
| `allowed_range_value.src_text` | 是 | `str` | 摘录原文 |

### 4.7 `constraints_in_parameters`（跨参数 / 单参数约束）

#### 4.7.1 顶层 key

- 平台名；不存在平台差异时**各平台使用相同的约束列表**（不要删减为单项 `"common"`）。

#### 4.7.2 `InterParamConstraint` 字段

| 字段 | 必填 | 说明 |
| ---- | ---- | ---- |
| `expr_type` | 是 | **自由字符串**。优先从 §7 字典中选用；若字典无法覆盖，允许使用实际语义值（如 `cross_param_constraint`、`parameter_representation`、`self_value_enum`、`self_string_length`、`self_value_dependency`） |
| `expr` | 是 | 合法 Python 布尔表达式（第 6 章）；无法写出时填 `""` |
| `relation_params` | 是 | 表达式中**所有**被引用的参数名（按出现顺序，去重） |
| `src_text` | 是 | 原文摘录，**可为空字符串** |

#### 4.7.3 提取规则

1. **跨参数约束优先**：涉及 ≥2 个参数的约束**必须**进入 `constraints_in_parameters`，不要只在 `inputs`/`outputs` 中备注。
2. **单参数约束复写**：若约束在 `allowed_range_value` 中已有表达，仍可在 `constraints_in_parameters` 中**附加一条带 `expr` 的形式化版本**（不视为冗余，而是机器可判定性的增强）。
3. **单参数 shape 约束**：若已在 `dimensions` 中表达（如 `[2,3]`），可省略重复。
4. **存在性约束**必须用完整布尔表达式（如 `(scale is None) == (zeroPoint is None)`），不允许退化为"可选/必选"自然语言。
5. **禁止**把"算子功能说明"或"参数描述"塞入 `constraints_in_parameters`。

### 4.8 `return_info`（错误返回码）

- 来自 `## 返回码` / `## 错误码` 章节。
- 字段：
  - `return_value`：枚举字符串（如 `ACLNN_ERR_PARAM_NULLPTR`）；
  - `error_code`：整数（如 `161001`）；
  - `description`：触发条件列表（`List[str]`，单条也用列表）。
- 文档未给错误码时填 `[]`。

### 4.9 `dtype_support_description`（dtype 组合支持表）

- 仅当文档存在**显式 dtype 组合表格**（如"各产品下 x1/x2/out 的 dtype 组合"）时填写；
- key 为平台名，value 为该平台下的 combo 对象列表（每个 combo 为 `{param_name: dtype_str}` 字典）；
- 无组合表时填 `{}`。

### 4.10 `format_support_description`（format 组合支持表）

- 结构与 `dtype_support_description` 对称：key 为平台名，value 为格式组合列表；
- 仅当文档存在**显式 format 组合表格**时填写；
- 无此表时填 `{}`。

---

## 5. 平台与 dtype 命名规范

### 5.1 标准平台名（受控字典）

提取 `product_support` / `deterministic_computing` / `constraints_in_parameters` / `dtype_support_description` / `format_support_description` 的 key 时，**必须**使用以下字符串之一：

| 平台 | 字符串 |
| ---- | ------ |
| Atlas A2 训练 + 推理 | `Atlas A2 训练系列产品/Atlas A2 推理系列产品` |
| Atlas A3 训练 + 推理 | `Atlas A3 训练系列产品/Atlas A3 推理系列产品` |
| Atlas 训练系列（旧） | `Atlas 训练系列产品` |
| Atlas 推理系列（旧） | `Atlas 推理系列产品` |
| Atlas 推理系列加速卡 | `Atlas 推理系列加速卡产品` |
| Atlas 350 加速卡 | `Atlas 350 加速卡` |
| Atlas 200I/500 A2 推理 | `Atlas 200I/500 A2 推理产品` |
| Atlas 300I 推理 | `Atlas 300I 推理产品` |
| Atlas 300I Duo 推理 | `Atlas 300I Duo 推理产品` |
| Atlas 300V 视频解析 | `Atlas 300V 视频解析产品` |
| Atlas 500 A2 智能小站 | `Atlas 500 A2 智能小站` |
| Atlas 800 推理服务器 A2 | `Atlas 800 推理服务器 A2` |
| Atlas 800 训练服务器 | `Atlas 800 训练服务器` |
| Atlas 800I A2 推理服务器 | `Atlas 800I A2 推理服务器` |

### 5.2 标准 dtype 字符串（受控字典）

提取 `dtype.value` / `dtype_support_description` 中的 dtype 时，**必须**使用以下字符串之一：

##### Tensor 数据类型
```
FLOAT32, FLOAT16, BFLOAT16, BF16, DOUBLE, INT8, UINT8, INT16, UINT16,
INT32, UINT32, INT64, UINT64, BOOL, COMPLEX64, COMPLEX128,
FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1, HFLOAT4, HFLOAT8
```

##### 标量参数"类型"（仅用于 `dtype.value`，不用于 `dtype_support_description` 的 combo）
```
bool, char, int, int64_t, int8_t, double, float, uint64_t, size_t
```

- 文档中出现 `BF16` / `bfloat16` / `bf16` 时 → 统一为 `BF16`；
- 文档中出现 `float` / `Float` / `FLOAT` 时 → 统一为 `FLOAT32`（除非上下文明确为 `float16`）；
- 标量参数（`int64_t`、`bool`、`char` 等）的 `dtype.value` 填写 `["bool"]`、`["char"]`、`["int"]`、`["int64_t"]` 等，表示"该参数自身类型"。

### 5.3 标准数据格式（受控字典）

```
ND, NC, NCL, NCHW, NCDHW, NHWC, NZ, FRACTAL_NZ, FRACTAL_Z, FRACTAL_Z_3D,
NDC1HWC0, FRACTAL_NZ_C0_16, NDHWC, NCHW_VECT_C0_16, NC1HWC0
```

- 多格式参数用 `List[str]`（如 `["FRACTAL_Z_3D", "ND"]`），单格式用 `str`；
- 标量 / 非 Tensor 参数用 `"N/A"`（注意是字符串，不是 `null`）。

---

## 6. 表达式编写规范

`expr` 字段必须是**合法 Python 布尔表达式**（`eval()` 可执行，返回 `bool`）。

### 6.1 语法细则

1. **变量引用**：使用**裸参数名**或 `参数名.shape[i]` / `参数名.dtype` / `参数名.range_value`：
   - ✅ `len(x.shape) == 3`
   - ✅ `x.shape[0] * x.shape[1] <= 2147483647`
   - ✅ `rankSize.range_value in [2, 4, 8]`
   - ✅ `x1.shape[0] == BS.range_value`
   - ❌ `tensor_x.dim == 3`（**禁止**别名）
2. **取值范围**：用区间 `[[min, max]]` 或离散列表 `[v1, v2]`：
   - ✅ `actType.range_value in [[0, 5]]`（对区间查）
   - ✅ `activation.range_value in ["relu", "gelu"]`（对枚举查）
   - ✅ `alltoAllAxesOptional.range_value == [-2, -1]`（对固定值等号）
   - ✅ `transposeX1.range_value == False`（bool 等号）
   - 允许 `null` 边界：`[[null, 2147483647]]` 表示无下界
3. **复合逻辑**：用 `and` / `or`；蕴含用 `(B) if (A) else True`：
   - ✅ `(y.dtype == "FLOAT16") if (x.dtype == "FLOAT16") else True`
4. **生成器**：必须用 `all()` / `any()` 包裹：
   - ✅ `all(v >= 1 for v in padding.range_value)`
   - ❌ `[v >= 1 for v in padding.range_value]`（返回 list，不返回 bool）
5. **禁止关键字**：`lambda`、非蕴含三元运算符滥用、`implies`、伪代码、平台值作判断条件。
6. **空表达式**：不允许 `null`；无法表达时统一使用空字符串 `""`，保留 `expr_type` 与 `relation_params`。
7. **参数名冲突**：当参数名为 `max`/`min`/`sum` 等内置函数名时，表达式中**不要再调用**同名内置函数；`relation_params` 仍写原名。
8. **隐式维度变量 / 外部常量**：使用 `变量名.range_value` 形式（如 `BS.range_value`、`rankSize.range_value`）；这些符号必须登记在 `inputs` 中（`is_operator_param: false`）。

### 6.2 表达式与 src_text 的对应

- `expr` 表达什么，`src_text` 就摘录什么；
- 表达式无法直接对应原句（如文档只给 "shape 与 x 一致"）时，`expr` 写 `out.shape == x.shape`，`src_text` 摘录 `"out 的 shape 与 x 保持一致"`。

---

## 7. `expr_type` 取值字典

> `InterParamConstraint.expr_type` 类型为**自由 `str`**（不受 Pydantic 枚举约束）。
> 下表列出**已知的常用取值**作为**参考指引**；若语义无法匹配，允许使用文档实际语义值。

### 7.1 参数间约束（2+ 参数，来自 `InterConstraintsRuleType` 枚举）

| `expr_type` | 适用场景 | 典型 `expr` |
| --- | --- | --- |
| `shape_broadcast` | 形状需满足广播关系 | `all(a.shape[i] == b.shape[i] or a.shape[i]==1 or b.shape[i]==1 for i in range(N))` |
| `shape_choice` | 形状在多个候选中选其一 | `bias.shape == gamma.shape or bias.shape == x.shape` |
| `shape_equality` | 形状完全相等 | `out.shape == x.shape` |
| `shape_dependency` | 输出 shape 由输入 + 辅助参数推导 | `out.shape[0] == pad + x.shape[0]` |
| `shape_value_dependency` | shape 中具体轴值/元素值依赖 | `x1.shape[0] == x2.shape[1] and x2.shape[1] == BS.range_value` |
| `type_equality` | dtype 必须一致 | `x1.dtype == x2.dtype` |
| `type_dependency` | dtype 依赖其他参数/条件 | `(bias.dtype == "FLOAT16") if (x.dtype == "FLOAT16") else (bias.dtype == "FLOAT32")` |
| `value_dependency` | 取值依赖/取值范围 | `BS.range_value % rankSize.range_value == 0` |
| `format_equality` | 数据格式必须一致 | `x1.format == x2.format` |
| `presence_dependency` | 共存规则（None/非None） | `(scale is None) == (zeroPoint is None)` |

### 7.2 单参数约束（扩展值，不在 `InterConstraintsRuleType` 枚举中但实际广泛使用）

| `expr_type` | 适用场景 | 典型 `expr` |
| --- | --- | --- |
| `cross_param_constraint` | 通用跨参数约束（语义较泛） | 按具体上下文 |
| `parameter_representation` | 隐式维度变量/外部常量与张量 shape 的绑定 | `x1.shape[0] == BS.range_value` 或 `rankSize.range_value in [2,4,8]` |
| `self_value_range` | 单参数取值范围（区间） | `actType.range_value in [[0, 5]]` |
| `self_value_enum` | 单参数取值枚举 | `activation.range_value in ["relu", "gelu", "silu"]` |
| `self_value_dependency` | 单参数取值 ≈ 固定布尔/唯一合法值 | `transposeX1.range_value == False` |
| `self_string_length` | 字符串参数长度约束 | `0 < len(group.range_value) < 128` |
| `self_shape_dim_range` | 单参数维度（rank）范围 | `2 <= len(x.shape) <= 3` |
| `self_shape_axis_value` | 单参数某轴值约束 | `x.shape[0] >= 1` |

---

## 8. 边缘场景处理

| 场景 | 处理方式 |
| ---- | -------- |
| 文档仅给"产品支持"无 dtype 组合表 | `dtype_support_description={}` |
| 文档仅给"产品支持"无 format 组合表 | `format_support_description={}` |
| 多平台 dtype 列表完全一致 | 各平台各自复制相同列表；不用"common"合并 |
| 参数是 `aclIntArray *xxx` | `type.value="aclIntArray"`，`array_length` 必填实值 |
| 文档出现 `Optional` 后缀但未说明是否可空 | `is_optional.value=false`（保守），`src_text` 摘录原文待人工复核 |
| 文档写"shape 为 [B,H] 或 [B,1,H]" | 拆为 `shape_choice` / `shape_dependency` 约束；不要并成模糊规则 |
| 文档写"x 和 y 数据类型必须一致" | `expr_type="type_equality"`，`expr="x.dtype == y.dtype"`，`relation_params=["x","y"]` |
| 文档写"仅 Atlas A3 支持 BF16" | 在对应平台的 `dtype.value` 中体现差异，`src_text` 摘录原文 |
| 文档给出"确定性计算：默认确定性" | `deterministic_computing["平台"].value = "true"`，`src_text` 摘录该句 |
| 文档给出"确定性计算：默认非确定性" | `deterministic_computing["平台"].value = "false"`，`src_text` 摘录该句 |
| 文档**完全没有** `返回码` 章节 | `return_info=[]` |
| `allowed_range_value` 区间无下界（"不小于0"但无上界或上界为INT32_MAX） | `value=[[null, 2147483647]]`（null表示无下/上界） |
| 表达式无法用 Python 表达（自然语言公式） | `expr=""`，`src_text` 摘录原文，待人工校对 |
| 文档出现矛盾（A段dtype=X，B段dtype=Y） | 优先**保守**取值（取并集），`src_text` 摘录矛盾原文，等待人工确认 |

---

## 9. 自检清单（提取完成后必跑）

> 模型在生成 JSON 之后、提交给用户之前，**内部自检** 7 项。任何一项不通过均需重做。

1. **JSON 校验**：用 `OperatorRule.model_validate_json(json_str)` 解析，**不抛异常**。
2. **字段完整**：`OperatorRule` 的**全部 11 个**必填字段均存在且非 `None`；数组/对象至少是空容器。
3. **平台字典一致**：`product_support` 中的每个平台名，在 `deterministic_computing`、`inputs`/`outputs` 的二级 key、`constraints_in_parameters` 的 key 中**至少出现一次**。
4. **dtype/format 字典一致**：所有 `dtype.value` 元素来自 §5.2（含标量类型）；所有 `format.value` 元素来自 §5.3 或为 `"N/A"`。
5. **表达式合法**：每条 `expr`（非空）用 `python -c` 试 `eval`，无 `SyntaxError`/`NameError`；返回 `bool`。
6. **关系参数一致**：`expr` 中**所有出现的标识符**都在 `relation_params` 中；`relation_params` 中所有参数名都在 `inputs`/`outputs` 有对应卡片（隐式维度变量/外部常量允许例外，但须在 `inputs` 中登记）。
7. **来源可溯**：`function_explanation`/`dtype`/`format`/`dimensions`/`allowed_range_value` 的 `src_text` 至少 30% 非空（无来源的纯模型外推视为无效）。

---

## 10. 调用模板

下面给出一份**可直接复制**的 prompt 调用片段：

```text
# System
你是一名昇腾 CANN 算子约束抽取专家。
请严格遵循《算子约束提取通用提示词 v1》的所有规则。
输出必须是**纯 JSON 字符串**，无任何前后缀。

# User
请从下列算子说明文档中提取约束。

## 算子名称
{operator_name}

## 算子文档 URL
{operator_url}

## 算子说明文档（已转换为 Markdown）
```markdown
{operator_doc_markdown}
```

## 你的任务
1. 完整阅读算子说明文档；
2. 按《算子约束提取通用提示词 v1》第 3 章 schema 输出 JSON；
3. 内部执行第 9 章 7 项自检；
4. **仅返回 JSON 字符串**，不要包含任何解释、代码块标记或额外文字。
```

---

## 附录 A：与 10 个典型算子的对齐示例

> 下面给出 10 个 Transformer / NN 类算子的提取样例，**用于**说明 schema 在真实场景下的形态，**不构成**对其余算子的强制要求。

| 算子 | 类型 | 关键提取点 |
| ---- | ---- | ---------- |
| `aclnnReflectionPad1dBackward` | NN / 反向 | `padding` 长度固定 2；`padding` 数值 < `self` 最后一维 |
| `aclnnBatchMatMulWeightNz` | NN / MatMul | `mat2` 强制 NZ 格式；`cubeMathType` 可选 int8 |
| `aclnnGroupedMatmulV5` | NN / 分组 MatMul | `actType ∈ [0,5]`；大量 `Optional` 参数与 `aclTensorList` |
| `aclnnSwinAttentionScoreQuant` | Transformer | int8 量化；`biasDequant*Optional` 取值为 0–255 整型 |
| `aclnnSwinTransformerLnQkvQuant` | Transformer | LN + QKV 拆分；`headNum`/`seqLength`/`epsilon` 等标量属性 |
| `aclnnAlltoAllMatmul` | 通信 + MatMul | `alltoAllAxesOptional` 取值 `空` 或 `[-2,-1]`；隐式变量 `BS`/`H`/`N` + 外部常量 `rankSize` |
| `aclnnFFNV3` | NN / MoE FFN | `activation` 为枚举字符串；`innerPrecise` 标量属性 |
| `aclnnNpuFormatCast` | 格式转换 | 输入格式集 `["FRACTAL_Z_3D","NCDHW",...]`；dtype 与 format 强耦合 |
| `aclnnCalculateMatmulWeightSize` | 辅助计算 | 仅计算输出，无 Tensor 真正计算；`workspaceSize`/`executor` 是唯一输出 |
| `aclnnCalculateMatmulWeightSizeV2` | 辅助计算 | 同上 V2，差异在 weight 排布 / NZ 转换 |

> **参考产物位置**：
> - 旧版（`temp/batch-20260625_195726-results/`）—— 历史产物，不一定准确；
> - 新版（`batch-20260626_182854-constraints/`）—— 基于项目实际 `assemble_result.py` 产出的新约束 JSON。
> 两者均仅作参考，**不保证完全正确**。

---

## 附录 B：从 v1 升级到 v2 的注意事项（占位）

- 本 v1 的 `inputs`/`outputs` 二级 key 体系是 `平台名`；每参数每平台一条 `ParamAttributes`；平台差异通过多条记录体现。
- `expr_type` 为自由 `str`，§7 仅作参考；若新增语义（如 `shape_value_enum`），追加到 §7.2 并附真实算子样例。
- 若增加新平台（昇腾下一代硬件），在 §5.1 字典中追加官方字符串。
- 若未来 schema 要求 `ValueWithSrcText` 包裹更多字段（如 `description`），同步更新 §3 与 §4.6.3。
