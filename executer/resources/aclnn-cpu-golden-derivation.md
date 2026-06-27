---
name: aclnn-cpu-golden-derivation
description: Derive PyTorch CPU computation from ACLNN C++ operator signatures for ATK test scripts. Use when asked to replace dummy CPU output with real PyTorch computation in an ATK-generated test file, or when converting aclnn operator signatures to torch calls.
metadata:
  type: skill
---

# ACLNN Signature → PyTorch CPU Golden Derivation

## When to Use

Use this skill when:
- You have an ATK-generated test script (e.g. `test_aclnnAddmv.py`) with a `BaseApi` subclass that returns dummy `torch.ones()` outputs
- You need to replace the dummy computation with real PyTorch CPU calls
- The file contains `# TODO: CPU_GOLDEN` and `# END_CPU_GOLDEN` markers with the C++ signature in comments

## Step 0: Read the ACLNN Operator Documentation (MUST DO FIRST)

**Before deriving the CPU golden, ALWAYS read the operator's CANN documentation to extract parameter constraints.**
This is the single most important step — it prevents runtime errors that would otherwise only surface during testing.

**How to find the doc:**
- Directory: `D:\software\markitdown\CANN-aclnn-api-reference\context\`
- Sub-directory by domain: `ops-nn/`, `ops-math/`, `ops-cv/`, `ops-transformer/`
- File: `aclnn{OperatorName}.md` (e.g. `aclnnBinaryCrossEntropyWithLogits.md`)
- If the user specifies a different path, use that instead

**What to extract from the doc's "参数说明" table:**

For each parameter, read the **"使用说明"** (Usage Notes) column. This is the most critical column — it contains:

| What to look for | Example from doc | How to apply |
|---|---|---|
| **Integer attr valid values** | `reduction: 支持0(none)\|1(mean)\|2(sum)` | Code must clamp to `[0, 2]`; JSON `range_values` must be int 0/1/2 |
| **Broadcast constraints** | `weightOptional: shape需要能够broadcast到target` | JSON weight/posWeight shape must broadcast to self/target; or use `[1]` |
| **Shape relationships** | `target: 与self保持一致` | JSON self/target shapes must be identical |
| **Output shape rules** | `out: 如果reduction=0，shape与self一致，其他情况shape为[1]` | CPU golden must return correct shape; ATK output inference depends on it |
| **Data type constraints** | `epsilon: 取值仅支持1e-5` | JSON must use exactly that value; code should default to it |
| **Data type relationships** | `target: 与self保持一致` | JSON self/target dtype must match |
| **Supported data types** | `self: FLOAT16、FLOAT、BFLOAT16` | Verify JSON dtypes are in the supported set |

**Also read the "返回值" table (Error Codes):**
- `ACLNN_ERR_PARAM_INVALID (161002)` entries list exactly which conditions cause errors
- Example: `weightOptional、posWeightOptional不能扩展成self/target形状` — this tells you the broadcast direction

**Apply findings in TWO places:**

1. **JSON test data:** Ensure shapes, dtypes, and attr values satisfy all documented constraints. Fix BEFORE running the test.
2. **Python CPU golden code:** Add appropriate defaults, None checks, and clamping based on the doc's value ranges.

**If the doc is not found:** Proceed with signature-only derivation, but be aware that runtime errors are more likely and will need iterative fixes.

## Step-by-Step Procedure

### Step 1: Read the Signature

Find the C++ signature in the generated file. It appears in two places:
- `_SIG_STR` class attribute
- Comment line `# C++ signature: aclnnStatus aclnnXXXGetWorkspaceSize(...)`

Parse each parameter to determine its role:

| C++ Type | Role | Python Type in kwargs |
|---|---|---|
| `const aclTensor*` | Input tensor | `torch.Tensor` |
| `aclTensor*` (non-const) | Output tensor | Not in kwargs — this is what we compute |
| `const aclTensorList*` | Input tensor list | `List[torch.Tensor]` |
| `const aclScalarList*` | Input scalar list | `List[float]` or `List[int]` |
| `const aclScalar*` | Scalar value | `float` or `int` |
| `const aclIntArray*` | Integer array | `List[int]` |
| `const aclFloatArray*` | Float array | `List[float]` |
| `const aclBoolArray*` | Bool array | `List[bool]` |
| `const aclString*` / `const char*` | String | `str` |
| `int64_t`, `int32_t`, `int8_t` | Integer attr | `int` |
| `uint64_t`, `uint32_t`, `uint8_t` | Unsigned int attr | `int` |
| `double`, `float` | Float attr | `float` |
| `bool`, `attr_bool` | Boolean attr | `bool` |
| `aclDataType` | dtype attr | `torch.dtype` or `int` |
| `uint64_t* workspaceSize` | Framework | **Ignore** |
| `aclOpExecutor** executor` | Framework | **Ignore** |

### Step 2: Derive the PyTorch Function Name

From the ACLNN operator name, derive the PyTorch function:

**Rule A: Simple prefix removal (most common)**
- `aclnn` + `OpName` → `torch.opname`
- `aclnnAbs` → `torch.abs`
- `aclnnAdd` → `torch.add`
- `aclnnMul` → `torch.mul`
- `aclnnDiv` → `torch.div`
- `aclnnExp` → `torch.exp`
- `aclnnSqrt` → `torch.sqrt`
- `aclnnSin` → `torch.sin`
- `aclnnCos` → `torch.cos`
- `aclnnTanh` → `torch.tanh`
- `aclnnLog` → `torch.log`

**Rule B: CamelCase → snake_case (for multi-word ops)**
- `aclnnLayerNorm` → `torch.layer_norm`
- `aclnnGroupNorm` → `torch.group_norm`
- `aclnnBatchNorm` → `torch.batch_norm`
- `aclnnAddLayerNorm` → Manual computation (Add + LayerNorm fusion, see Rule F)
- `aclnnAddmv` → `torch.addmv`
- `aclnnBmm` → `torch.bmm`
- `aclnnMatmul` → `torch.matmul`
- `aclnnEinsum` → `torch.einsum`
- `aclnnTopk` → `torch.topk`
- `aclnnSort` → `torch.sort`
- `aclnnArgMax` → `torch.argmax`
- `aclnnArgMin` → `torch.argmin`
- `aclnnArgmax` → `torch.argmax`

**Rule F: Fused operators (cannot directly map to single torch call)**
- `aclnnAddLayerNorm` → Manual: `residual = x1 + x2 [+ biasOptional]`, then manual LayerNorm along last axis (`dim=-1`) using mean/var/rstd
- These operators combine multiple operations (add + normalize) that have no single PyTorch equivalent
- gamma/beta are broadcast scalars, NOT normalized_shape vectors
- See the aclnnAddLayerNorm example below for full derivation

**Rule C: Scalar variants (suffix `s`)**
- `aclnnAdds` → `torch.add(self, other=scalar, alpha=...)`
- `aclnnMuls` → `torch.mul(self, other=scalar)`
- `aclnnSubs` → `torch.sub(self, other=scalar)`
- `aclnnDivs` → `torch.div(self, other=scalar)`
- The `s` suffix means the second operand is a scalar, not a tensor

**Rule D: Inplace operators**
- `aclnnInplaceAbs` → `self.abs_()` or `torch.abs(self)` (return self)
- `aclnnInplaceAdd` → `self.add_(other, alpha=alpha)`
- Strip `Inplace` prefix, derive the base op name, use the `xxx_()` in-place method
- Signature has **no output tensor** parameter — only `selfRef` and inputs

**Rule E: Special operators (no simple naming pattern)**

| ACLNN Name | PyTorch Call |
|---|---|
| `aclnnUpsampleBilinear2d` | `F.interpolate(self, size=output_size, mode='bilinear', align_corners=align_corners)` |
| `aclnnUpsampleBicubic2d` | `F.interpolate(self, size=output_size, mode='bicubic', align_corners=align_corners)` |
| `aclnnUpsampleNearest1d/2d/3d` | `F.interpolate(self, size=output_size, mode='nearest')` |
| `aclnnUpsampleLinear1d` | `F.interpolate(self, size=output_size, mode='linear', align_corners=align_corners)` |
| `aclnnUpsampleTrilinear3d` | `F.interpolate(self, size=output_size, mode='trilinear', align_corners=align_corners)` |
| `aclnnReflectionPad1d/2d/3d` | `F.pad(self, padding, mode='reflect')` |
| `aclnnReplicationPad1d/2d/3d` | `F.pad(self, padding, mode='replicate')` |
| `aclnnConstantPadNd` | `F.pad(self, padding, mode='constant', value=value)` |
| `aclnnCircularPad2d/3d` | `F.pad(self, padding, mode='circular')` |
| `aclnnAffineGrid` | `F.affine_grid(theta, size, align_corners=align_corners)` |
| `aclnnAdaLayerNorm` | `torch.layer_norm(self, normalized_shape, weight=scale, bias=shift, eps=eps)` |
| `aclnnDropout` | `torch.dropout(self, p=p, train=train)` |
| `aclnnCat` | `torch.cat(tensors, dim=dim)` |
| `aclnnStack` | `torch.stack(tensors, dim=dim)` |
| `aclnnOneHot` | `torch.nn.functional.one_hot(self, num_classes)` |
| `aclnnEmbedding` | `torch.nn.functional.embedding(indices, weight)` |
| `aclnnNllLoss2DBackward` | (skip, complex) |
| `aclnnArange` | `torch.arange(start, end, step, dtype=dtype, device=device)` |
| `aclnnFull` | `torch.full(size, fill_value, dtype=dtype)` |
| `aclnnZeros` | `torch.zeros(size, dtype=dtype)` |
| `aclnnOnes` | `torch.ones(size, dtype=dtype)` |
| `aclnnEye` | `torch.eye(n, m, dtype=dtype)` |
| `aclnnRandperm` | `torch.randperm(n, generator=g)` |
| `aclnnBernoulli` | `torch.bernoulli(self, p=prob)` |
| `aclnnLinspace` | `torch.linspace(start, end, steps, dtype=dtype)` |
| `aclnnMeshgrid` | `torch.meshgrid(*tensors, indexing='ij')` |
| `aclnnRepeat` | `self.repeat(repeats)` |
| `aclnnExpand` | `self.expand(size)` |
| `aclnnPermute` | `self.permute(dims)` |
| `aclnnTranspose` | `self.transpose(dim0, dim1)` |
| `aclnnFlip` | `torch.flip(self, dims)` |
| `aclnnChunk` | `self.chunks(chunks, dim=dim)` |
| `aclnnGather` | `self.gather(dim, index)` |
| `aclnnScatter` | `self.scatter(dim, index, src)` |
| `aclnnIndexSelect` | `torch.select(self, dim, index)` |
| `aclnnWhere` | `torch.where(condition, self, other)` |
| `aclnnClamp` | `torch.clamp(self, min=clipValueMin, max=clipValueMax)` |
| `aclnnUnfold` | `self.unfold(dim, size, step)` |
| `aclnnIm2col` | `torch.nn.functional.unfold(self, kernel_size, dilation, padding, stride)` |

### Step 3: Map Parameters

Map ACLNN signature parameters to PyTorch function arguments:

**Common parameter name mappings:**

| ACLNN Param | PyTorch Arg |
|---|---|
| `self`, `input`, `x`, `selfRef` | First positional arg |
| `other`, `tensor1` | Second positional arg |
| `mat`, `matrix`, `weight` | Positional (context-dependent) |
| `vec`, `vector`, `bias` | Positional (context-dependent) |
| `alpha`, `beta` | `alpha=`, `beta=` keyword args |
| `dim`, `dimOptional` | `dim=` |
| `keepDim`, `keepdim`, `keepDimOptional` | `keepdim=` |
| `size`, `shape` | `size=` or positional |
| `start`, `end`, `step` | Positional or keyword |
| `eps`, `epsilon` | `eps=` |
| `seed`, `offset` | For random ops, use `torch.manual_seed(seed)` |
| `p`, `prob` | `p=` (dropout probability) |
| `padding` | `padding=` (from aclIntArray as list) |
| `mode`, `string` | Convert to Python string, pass as `mode=` |
| `out` | **Omit** — PyTorch returns the result |
| `workspaceSize`, `executor` | **Ignore** |
| `cubeMathType` | **Ignore** (NPU-specific) |
| `dtype` | Convert aclDataType to torch.dtype |

### Step 4: Handle Special Cases

**Multi-output operators:**
```python
# aclnnMaxDim → torch.max
values, indices = torch.max(self, dim=dim, keepdim=keepdim)
return [values, indices]

# aclnnSort → torch.sort
values, indices = torch.sort(self, dim=dim, descending=descending)
return [values, indices]

# aclnnTopk → torch.topk
values, indices = torch.topk(self, k, dim=dim, largest=largest, sorted=sorted)
return [values, indices]

# aclnnAminmax → torch.aminmax
min_out, max_out = torch.aminmax(self, dim=dim, keepdim=keepdim)
return [min_out, max_out]

# aclnnSvd → torch.svd
U, S, V = torch.svd(self)
return [U, S, V]
```

**Inplace operators (no output tensor in signature):**
```python
# aclnnInplaceAdd
self_ref.add_(other, alpha=alpha)
return self_ref

# aclnnInplaceAbs
self_ref.abs_()
return self_ref
```

**TensorList input:**
```python
# aclnnCat
tensors = _get_param("tensors")  # List[torch.Tensor]
result = torch.cat(tensors, dim=dim)
return result
```

**ScalarList input (e.g. aclnnForeachSubScalarList, aclnnForeachAddScalarList, etc.):**
When the signature has `const aclScalarList *scalars`, the CPU golden receives a `List[float]` or `List[int]`. Use `zip()` to iterate over tensors and scalars together:
```python
# aclnnForeachSubScalarList: y_i = x_i - scalars[i] for each tensor in the list
x = _get_param("x")          # List[torch.Tensor]
scalars = _get_param("scalars")  # List[float]
out = [t - s for t, s in zip(x, scalars)]
return out  # Return List[torch.Tensor] — not wrapped in another list
```

**TensorList / ScalarList output (e.g. foreach ops — aclnnForeachAbs, aclnnForeachAddList, aclnnForeachSubScalarList, etc.):**
When the signature has `const aclTensorList *out` as an output parameter, the CPU golden should return `List[torch.Tensor]` directly. ATK will handle the conversion. The generated `init_by_input_data` already contains logic to handle `aclTensorList*` / `aclScalarList*` outputs correctly (it collects `OutputData` entries and passes them as a list to `convert_output_data`, triggering `create_x_list`).

```python
# aclnnForeachAbs: y_i = |x_i| for each tensor in the list
x = _get_param("x")  # List[torch.Tensor]
out = [torch.abs(t) for t in x]
return out  # Return List[torch.Tensor] — not wrapped in another list
```

**Key: The return should be `List[torch.Tensor]` directly.** ATK's `get_output_data_infos` will see the list, recurse into it, and produce nested `[[info1, info2, ...]]`. However, `update_output_info_list` flattens this to `[OutputData(info1), OutputData(info2), ...]`. The `init_by_input_data` code (auto-generated by generator.py) detects `aclTensorList` in the output's `raw_type`, collects the correct number of `OutputData` entries (based on the input tensor list length), and re-packages them into a list for `convert_output_data`, which then calls `create_x_list` to produce `AclTensorList*`.

**If testing fails with** `位置 X 类型不匹配：传入类型 [LP_AclTensor]，预期类型 [LP_AclTensorList]`, it means either:
1. The `_SIG_ORDER` has `out`'s `kind` set to `'tensorList'` instead of `'output'` — check that the JSON has `"outputs": "out"` (or the correct output name)
2. The `init_by_input_data` doesn't have the `aclTensorList` handling code — regenerate with the updated generator.py

**String parameters (mode):**
```python
# The mode comes as a Python string from kwargs
mode = _get_param("mode", "bilinear")
result = F.interpolate(self, size=size, mode=mode, align_corners=align_corners)
```

**aclIntArray parameters:**
```python
# Comes as a Python list from kwargs
padding = _get_param("padding")  # e.g. [1, 2, 1, 2]
dims = _get_param("dims")  # e.g. [0, 2, 1]
```

### Step 5: Write the Code

Replace the code between `# TODO: CPU_GOLDEN` and `# END_CPU_GOLDEN` with:

```python
def __call__(self, input_data: InputDataset, with_output: bool = False):
    # Use _get_param(name) to extract inputs — handles both kwargs and args modes
    self_t = _get_param("self")  # example: first input tensor
    other = _get_param("other", None)  # example: optional param with default
    result = torch.abs(self_t)
    return result
```

**Use `_get_param(name)` for ALL parameter extraction.** This helper function is auto-generated in the class and handles ATK data-passing transparently:

**How ATK passes data (CRITICAL):** ATK's `base_dataset.py` appends every generated parameter value to `input_data.args` in JSON `inputs[]` order. **`input_data.kwargs` is always empty** for ACLNN operators. The `_get_param` helper builds a name→value mapping by zipping `input_data.args` with `self.task_result.case_config.inputs[].name`, so you can look up by parameter name regardless of JSON ordering.

**NEVER access `input_data.kwargs` or `input_data.args` directly** in CPU golden code — always use `_get_param(name)` or `_get_tensor(name)`.

**Do NOT remove:**
- `_OP_NAME` and `_SIG_STR` class attributes
- `_INPUT_PARAM_NAMES` class attribute
- The `with_output` parameter
- The method signature
- The `_get_param` helper function and `_arg_idx` counter

**Do NOT modify existing import lines** — leave every existing `import` / `from ... import ...` line exactly as it is. Do not rename module paths, do not reorder, do not remove, do not change aliases. Example: if the file has `from atk.tasks.api_execute.base_api import BaseApi`, keep it exactly that — do NOT "correct" it to `from atk.tasks.base_api import BaseApi`. You MAY add new imports after the existing ones if the CPU golden needs them (e.g. `import torch.nn.functional as F`).

**Do NOT modify:**
- **Existing import lines** — never remove or change existing `import` / `from ... import ...` lines. You MAY add new imports if needed.
- The `@register()` decorator arguments
- The `get_cpp_func_signature_type` method

**You may modify:**
- The `BaseApi` subclass's `__call__` method (this is the primary target — CPU golden logic)
- The `AclnnBaseApi` class's `init_by_input_data` and `after_call` methods (e.g., when `convert_output_data` doesn't handle `aclTensorList*` / `aclScalarList*` outputs correctly, or when `after_call` needs to handle list conversion)

**Do remove:**
- The `# [FALLBACK]` dummy computation block
- The `_dummy_output` inner function

### Step 6: Handle Broadcast Differences

Some ACLNN operators have different tensor shapes between parameters than their PyTorch equivalent. When a `torch.*` call's direct argument broadcast doesn't match the ACLNN semantics, use manual computation (mean/var/unsqueeze/expand) instead.

**Common broadcast patterns:**

| ACLNN Operator | Shape Mismatch | Solution |
|---|---|---|
| `aclnnAdaLayerNorm` | `x=[..., S, H]`, `scale/shift=[B, H]` or `[B, 1, H]` — scale may lack the S dimension | Dynamic: `if scale.dim() < x.dim(): scale = scale.unsqueeze(-2)` — only unsqueeze when dims don't match |
| `aclnnLayerNorm` | `scale/bias=[H]`, `x=[..., H]` | `torch.layer_norm` handles this directly via `normalized_shape` |
| `aclnnAddLayerNorm` | `gamma/beta=[1]` or same shape as x, not `normalized_shape` vectors | Manual mean/var along `dim=-1` only; gamma/beta are broadcast scalars |

**PITFALL: Do NOT use `gamma.shape` as `normalized_shape` for `torch.layer_norm` when gamma is a broadcast scalar ([1] or same shape as input).**

For `aclnnAddLayerNorm` and similar fused operators:
- gamma and beta are broadcast coefficients, NOT `normalized_shape` vectors
- Using `gamma.shape` as `normalized_shape` for `torch.layer_norm` will either error out (shape mismatch) or normalize over wrong dimensions
- Instead, manually compute: `mean = x.mean(dim=-1, keepdim=True)`, `var = x.var(dim=-1, keepdim=True, unbiased=False)`, `rstd = 1.0 / sqrt(var + eps)`

**When `torch.*` cannot handle the broadcast directly:**
```python
# Compute the norm manually, then apply scale/shift with dynamic unsqueeze
# IMPORTANT: scale/shift may already have the right number of dims (e.g. [B, 1, H] matches x=[..., S, H])
# Only unsqueeze when scale/shift have fewer dims than x — don't assume a fixed pattern
if scale.dim() < x.dim():
    scale = scale.unsqueeze(-2)
if shift.dim() < x.dim():
    shift = shift.unsqueeze(-2)
mean = x.mean(dim=(-2, -1), keepdim=True)
var = x.var(dim=(-2, -1), keepdim=True, unbiased=False)
x_norm = (x - mean) / (var.sqrt() + epsilon)
result = x_norm * scale + shift
```

### Step 7: Handle Output Dtype Alignment

**CRITICAL: ATK uses the CPU golden's return tensor dtype to infer the NPU output tensor dtype.**

The NPU's `GetWorkspaceSize` function validates that ALL tensor dtypes (inputs + outputs) match one of its pre-defined `SupportInfo` combinations. If the CPU golden returns a dtype that doesn't match the NPU's expectation, you'll get error 161002: `Io input dtype or format is not supported`.

**How to determine the correct output dtype:**

1. Look at the NPU error message's `SupportInfo` list — it shows every supported input+output dtype combination
2. Match the user's JSON input dtypes to find which `SupportInfo[N]` entry applies
3. The output dtypes in that entry are what your CPU golden MUST return

**Common dtype rules:**

| Output Type | Rule | Example |
|---|---|---|
| Main output (`yOut`, `out`, etc.) | Same as input tensor dtype (`x1.dtype`) | If x1=fp16, yOut must be fp16 |
| Statistics outputs (`meanOut`, `varOut`, `rstdOut`, `mean`) | **Always fp32**, regardless of input dtype | All SupportInfo entries show `meanOut(DT_FLOAT) rstdOut(DT_FLOAT)` |
| Intermediate normalized output (`xOut`) | Same as input tensor dtype | If x1=fp16, xOut must be fp16 |
| Index outputs (`indices` from topk/argmax/sort) | `int64` | `torch.topk` returns `torch.int64` indices |

**How to apply in code:**

```python
# ALWAYS promote to fp32 for computation, then cast outputs to match NPU expectations
residual_f32 = residual.float()
mean = residual_f32.mean(dim=reduce_axis, keepdim=True)  # fp32
var = residual_f32.var(dim=reduce_axis, keepdim=True, unbiased=False)  # fp32
rstd = (var + epsilon).reciprocal().sqrt()  # fp32
xOut = (residual_f32 - mean) * rstd  # fp32
ln_out = xOut * gamma.float() + beta.float()  # fp32

# Cast to match NPU SupportInfo expectations
out_dtype = x1.dtype        # The input dtype (fp16/bf16/fp32)
ln_out = ln_out.to(out_dtype)  # yOut → input dtype
xOut = xOut.to(out_dtype)      # xOut → input dtype
mean = mean.float()            # meanOut → always fp32
rstd = rstd.float()            # rstdOut → always fp32
```

**Why this is needed:**
- PyTorch's `mean()` / `var()` on fp16 tensors returns fp16 — but NPU expects fp32 for statistics
- PyTorch's type promotion: `fp32 * fp16` → fp32 — but NPU may expect fp16 for the main output
- Without explicit casting, the CPU golden's return dtype depends on PyTorch's promotion rules, which don't match NPU's `SupportInfo`

### Step 8: Verify

After modifying, verify:
1. `python -c "import ast; ast.parse(open('file.py', encoding='utf-8').read())"` — syntax is valid
2. All input parameters use `_get_param(name)` — NOT direct `input_data.kwargs` or `input_data.args` access
3. **The return tensor's shape must match what the ACLNN operator expects for `out`** — ATK uses the CPU golden's return to infer the NPU output tensor's shape. If the CPU golden returns a tensor of wrong shape (e.g. `torch.ones([1])` from a dummy fallback), the NPU `GetWorkspaceSize` will fail with `Expected tensor for outputTensor.out to have same size as [correct_shape], but got [1]`. **Always remove the `_dummy_output` fallback completely.**
4. **The return tensor's dtype must match the NPU `SupportInfo` for the given input dtype combination** — see Step 7 above. Statistics outputs (mean/var/rstd) are always fp32; main outputs follow input dtype.
5. The return type matches: single tensor → `return result`; multiple tensors → `return [a, b]`
6. No NPU-specific imports (`torch_npu`, `acl`, `ascendcl`) in the CPU class
7. NPU-specific params (`cubeMathType`, `workspaceSize`, `executor`) are not passed to torch

## Example: aclnnAddmv

```
Signature:
aclnnStatus aclnnAddmvGetWorkspaceSize(
    const aclTensor* self,
    const aclTensor* mat,
    const aclTensor* vec,
    const aclScalar* alpha,
    const aclScalar* beta,
    aclTensor* out,
    int8_t cubeMathType,
    uint64_t* workspaceSize,
    aclOpExecutor** executor
)

Derivation:
1. aclnnAddmv → torch.addmv
2. torch.addmv(input, mat, vec, *, alpha, beta)
3. Parameters: self→input, mat→mat, vec→vec, alpha→alpha, beta→beta
4. cubeMathType is NPU-specific → ignore
5. Single output: out

Result:
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        self_t = _get_param("self")
        mat = _get_param("mat")
        vec = _get_param("vec")
        alpha = _get_param("alpha", 1.0)
        beta = _get_param("beta", 1.0)
        result = torch.addmv(self_t, mat, vec, alpha=alpha, beta=beta)
        return result
```

## Example: aclnnLayerNorm

```
Signature:
aclnnStatus aclnnLayerNormGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* scale,
    const aclTensor* shift,
    const aclIntArray* normalizedShape,
    double eps,
    aclTensor* meanOut,
    aclTensor* varOut,
    aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor
)

Derivation:
1. aclnnLayerNorm → torch.layer_norm
2. torch.layer_norm(input, normalized_shape, weight, bias, eps)
3. Multiple outputs: meanOut, varOut, out → torch.layer_norm doesn't return mean/var
4. Use torch.nn.functional.layer_norm for just the output, or decompose

Result:
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x = _get_param("x")
        scale = _get_param("scale")
        shift = _get_param("shift")
        normalized_shape = list(_get_param("normalizedShape"))
        eps = _get_param("eps", 1e-5)
        out = torch.layer_norm(x, normalized_shape, weight=scale, bias=shift, eps=eps)
        # Compute mean/var for the extra outputs — always fp32 for NPU
        n_dims = len(x.shape) - len(normalized_shape)
        dim = tuple(range(-len(normalized_shape), n_dims))
        mean = x.float().mean(dim=dim, keepdim=True).expand_as(out).float()
        var = x.float().var(dim=dim, keepdim=True, unbiased=False).expand_as(out).float()
        return [mean, var, out]
```

## Example: aclnnAdaLayerNorm

```
Signature (from aclnn_extracted.txt):
aclnnStatus aclnnAdaLayerNormGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* scale,
    const aclTensor* shift,
    const aclTensor* weightOptional,
    const aclTensor* biasOptional,
    double epsilon,
    aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor
)

ACLNN shape constraints (from CANN docs):
- x: shape [B, S, H], B supports 0-6 dims (total 2-8 dims)
- scale: [B, H] or [B, 1, H], B/H match x
- shift: [B, H] or [B, 1, H], B/H match x
- weightOptional: [H], optional
- biasOptional: [H], optional
- out: same shape as x

Derivation:
1. aclnnAdaLayerNorm → torch.layer_norm (semantics: out = ((x-mean)/std)*scale + shift)
2. torch.layer_norm requires weight.shape == normalized_shape, but scale=[B,H] and x=[...,S,H]
3. scale/shift lack the S dimension → cannot pass directly to torch.layer_norm or broadcast with x
4. Manual norm + unsqueeze: compute mean/var over last two dims, then unsqueeze(-2) on scale/shift
5. weightOptional/biasOptional are optional → use .get() with None default
6. workspaceSize/executor → ignore

Result:
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x = _get_param("x")
        scale = _get_param("scale")
        shift = _get_param("shift")
        weightOptional = _get_param("weightOptional", None)
        biasOptional = _get_param("biasOptional", None)
        epsilon = _get_param("epsilon", 1e-5)
        # Semantics: out = ((x - mean)/std) * scale + shift
        # scale/shift=[B, H], x=[..., S, H] — need unsqueeze(-2) to broadcast over S
        scale_expanded = scale.unsqueeze(-2)  # [B, H] -> [B, 1, H]
        shift_expanded = shift.unsqueeze(-2)  # [B, H] -> [B, 1, H]
        mean = x.mean(dim=(-2, -1), keepdim=True)
        var = x.var(dim=(-2, -1), keepdim=True, unbiased=False)
        x_norm = (x - mean) / (var.sqrt() + epsilon)
        result = x_norm * scale_expanded + shift_expanded
        return result
```

**Key lessons from aclnnAdaLayerNorm:**
- **ATK uses CPU golden's return to infer NPU output tensor shape** — if the CPU golden returns a wrong shape (e.g. `torch.ones([1])` from dummy fallback), NPU `GetWorkspaceSize` will fail with `Expected tensor for outputTensor.out to have same size as [X], but got [1]`. Always remove `_dummy_output` completely.
- **ACLNN parameter broadcast differs from PyTorch** — when scale/shift don't directly broadcast with x, use `unsqueeze` to insert singleton dims rather than forcing `torch.layer_norm` with mismatched shapes.

## Example: aclnnAddLayerNorm

```
Signature:
aclnnStatus aclnnAddLayerNormGetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *gamma,
    const aclTensor *beta,
    const aclTensor *biasOptional,
    double epsilon,
    bool additionalOutput,
    const aclTensor *yOut,
    const aclTensor *meanOut,
    const aclTensor *rstdOut,
    const aclTensor *xOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor
)

ACLNN semantics (from torch_npu.npu_add_layer_norm):
- x1, x2: same shape, the two inputs to add (residual connection)
- gamma, beta: scale and shift, can be [1] or same shape as input (broadcast)
- biasOptional: optional extra bias added BEFORE normalization
- epsilon: numerical stability constant
- additionalOutput: if true, return [y, mean, rstd, xOut]; if false, return [y, mean, rstd]

CRITICAL DIFFERENCE from torch.layer_norm:
- torch.layer_norm(input, normalized_shape, weight, bias, eps) REQUIRES weight.shape == normalized_shape
- aclnnAddLayerNorm's gamma/beta are NOT "normalized_shape vectors" — they are broadcast scalars/tensors
- The normalization axis is the LAST axis only (reduce_axis=-1), NOT the last len(normalized_shape) axes
- Therefore CANNOT use torch.layer_norm directly; must compute manually:
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    rstd = 1.0 / sqrt(var + epsilon)

Derivation:
1. residual = x1 + x2 [+ biasOptional] (add BEFORE normalization)
2. Compute mean, var, rstd along last axis (dim=-1)
3. xOut = (residual - mean) * rstd
4. yOut = xOut * gamma + beta

Result:
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x1 = _get_param("x1")
        x2 = _get_param("x2")
        gamma = _get_param("gamma")
        beta = _get_param("beta")
        biasOptional = _get_param("biasOptional", None)
        epsilon = _get_param("epsilon", 1e-5)
        additionalOutput = _get_param("additionalOutput", True)
        reduce_axis = -1
        if biasOptional is not None:
            residual = torch.add(torch.add(x1, x2), biasOptional)
        else:
            residual = torch.add(x1, x2)
        # Promote to fp32 for numerical stability and correct dtype inference
        residual_f32 = residual.float()
        mean = residual_f32.mean(dim=reduce_axis, keepdim=True)
        var = residual_f32.var(dim=reduce_axis, keepdim=True, unbiased=False)
        rstd = (var + epsilon).reciprocal().sqrt()
        xOut = (residual_f32 - mean) * rstd
        ln_out = xOut * gamma.float() + beta.float()
        mean = mean.expand_as(ln_out)
        rstd = rstd.expand_as(ln_out)
        # Cast outputs to match NPU SupportInfo expectations:
        # yOut → input dtype, meanOut/rstdOut → always fp32, xOut → input dtype
        out_dtype = x1.dtype
        ln_out = ln_out.to(out_dtype)
        xOut = xOut.to(out_dtype)
        mean = mean.float()
        rstd = rstd.float()
        if additionalOutput:
            return [ln_out, mean, rstd, xOut]
        return [ln_out, mean, rstd]
```

**Key lessons from aclnnAddLayerNorm:**
- **Fused Add+LayerNorm operators**: When an operator combines addition (residual) with normalization, the addition happens BEFORE the normalization. The biasOptional parameter also participates in the pre-normalization sum.
- **gamma/beta as broadcast scalars, not normalized_shape vectors**: In some ACLNN normalization operators, `gamma` and `beta` are broadcast coefficients (can be `[1]` or same shape as input), NOT `normalized_shape` vectors for `torch.layer_norm`. Do NOT use `gamma.shape` as `normalized_shape` for `torch.layer_norm` — the normalization axis may differ.
- **Normalization axis is last axis only (dim=-1)**: For `aclnnAddLayerNorm` and similar operators, the mean/var computation is over the LAST axis only, not over the last N axes as `torch.layer_norm` would do with a multi-dimensional `normalized_shape`. Always use manual `mean(dim=-1)` / `var(dim=-1)` computation for these operators.
- **`additionalOutput` flag controls return count**: Some operators have a boolean flag controlling how many outputs to return. The return count must match what the NPU expects, otherwise ATK output comparison will fail.
- **Output dtype casting is mandatory**: Always compute in fp32 (`residual.float()`), then explicitly cast each output to its NPU-expected dtype. PyTorch's `mean()` on fp16 returns fp16, but NPU always expects fp32 for statistics. Similarly, `fp32 * fp16` promotes to fp32, but NPU may expect fp16 for the main output. Without explicit `.to(input_dtype)` and `.float()` casts, error 161002 will occur.
- **NPU dtype combinations are all-or-nothing**: The NPU `SupportInfo` requires ALL input AND output dtypes to match a pre-defined combination. You can't mix and match — if gamma is fp32, beta must also be fp32 (SupportInfo[4/5]). If all inputs are fp16, match SupportInfo[6/7]. The JSON input dtypes must be consistent with one SupportInfo entry.

## Example: aclnnLayerNormWithImplMode

```
Signature:
aclnnStatus aclnnLayerNormWithImplModeGetWorkspaceSize(
    const aclTensor *input,
    const aclIntArray *normalizedShape,
    const aclTensor *weightOptional,
    const aclTensor *biasOptional,
    double eps,
    aclTensor *out,
    aclTensor *meanOutOptional,
    aclTensor *rstdOutOptional,
    int32_t implMode,
    uint64_t *workspaceSize,
    aclOpExecutor **executor
)

Derivation:
1. aclnnLayerNormWithImplMode → torch.layer_norm (same as aclnnLayerNorm, implMode is NPU-specific)
2. implMode is an NPU implementation mode selector → not needed for CPU golden, extract but don't use
3. Multi-output: out, meanOutOptional, rstdOutOptional → torch.layer_norm only returns out
4. Compute mean and rstd manually; rstd = 1/sqrt(var + eps)
5. normalizedShape comes as Python list from kwargs

Result:
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        input_t = _get_param("input")
        weightOptional = _get_param("weightOptional", None)
        biasOptional = _get_param("biasOptional", None)
        normalized_shape = list(_get_param("normalizedShape"))
        eps = _get_param("eps", 1e-5)
        out = torch.layer_norm(input_t, normalized_shape, weight=weightOptional,
                               bias=biasOptional, eps=eps)
        n_dims = len(input_t.shape) - len(normalized_shape)
        dim = tuple(range(-len(normalized_shape), n_dims))
        # Statistics outputs always fp32 for NPU SupportInfo matching
        input_f32 = input_t.float()
        mean = input_f32.mean(dim=dim, keepdim=True).expand_as(out).float()
        var = input_f32.var(dim=dim, keepdim=True, unbiased=False).expand_as(out).float()
        rstd = (var + eps).reciprocal().sqrt().expand_as(out).float()
        return [out, mean, rstd]
```

## Important: NPU-Specific Parameters to Ignore in CPU Golden

Some ACLNN operators have NPU-specific parameters that have no PyTorch equivalent. These should be extracted from kwargs (so the NPU backend can use them) but **should NOT be passed to the PyTorch function** in the CPU golden:

| NPU-Specific Param | Description | How to handle |
|---|---|---|
| `implMode` (int32_t) | NPU implementation mode (e.g., 0=standard, 1=fused) | Extract but don't use in torch call |
| `cubeMathType` (int8_t) | NPU cube library math precision | Ignore in CPU golden |
| `workspaceSize` / `executor` | Framework params | Already filtered by generator |

When deriving the CPU golden, extract these params from kwargs only if needed for conditional logic; otherwise skip them entirely.

## Important: ATK Integer Attr Type Coercion

**Problem:** ATK's `convert_input_data` converts Python `int` to `ctypes.c_long` for integer attrs. But the C signature may require `c_int32`, `c_int64`, etc. This causes `TypeError: an integer is required (got type c_long)` at runtime.

**Solution (already built into generator.py):** The generated `init_by_input_data` method includes automatic type coercion:

```python
_INT_TYPES = {'int8_t', 'int32_t', 'int64_t', 'uint8_t', 'uint32_t', 'uint64_t', 'bool'}
if _raw_type in _INT_TYPES and isinstance(data, list) and len(data) == 1:
    data = [self._ATTR_TYPE_MAP[_raw_type](int(getattr(data[0], 'value', data[0])))]
```

**Key details:**
- Only coerce **integer** types, never `float`/`double` (ATK handles them correctly as `c_double`/`c_float`)
- Use `int(getattr(data[0], 'value', data[0]))` to extract the Python int from any ctypes type — `c_long(1)` has `.value=1`, but `ctypes.c_int32()` constructor doesn't accept `c_long` directly in Python 3.9
- This logic is auto-generated by `generator.py`; you don't need to write it manually

## Checklist Before Declaring Done

- [ ] **Read the ACLNN operator doc from `CANN-aclnn-api-reference` (Step 0)**
- [ ] Extracted all parameter constraints from the doc's "使用说明" column
- [ ] JSON shapes satisfy all documented broadcast/shape relationships
- [ ] JSON attr values are within documented valid ranges (integers are integers, not floats)
- [ ] JSON dtypes match documented SupportInfo combinations
- [ ] Read the `_SIG_STR` or `# C++ signature:` comment
- [ ] Identified all input params, output params, and NPU-specific params to ignore
- [ ] Derived the correct `torch.*` function name
- [ ] Mapped each ACLNN parameter to the correct PyTorch argument position
- [ ] Extracted all inputs using `_get_param(name)` by signature param name
- [ ] Handled multi-output correctly (list return)
- [ ] Handled inplace correctly (return modified tensor)
- [ ] Handled broadcast differences between ACLNN params (unsqueeze/expand where needed)
- [ ] Computed intermediate values in fp32 (`residual.float()` / `x.float()`)
- [ ] Each output tensor is explicitly cast to its NPU-expected dtype (main output → input dtype; statistics → fp32)
- [ ] CPU golden return shape matches the ACLNN `out` tensor shape (no dummy `_dummy_output` fallback remaining)
- [ ] JSON input dtypes are consistent with one NPU `SupportInfo` combination (e.g. gamma/beta same dtype)
- [ ] For `aclTensorList*` output params: JSON has `"outputs": "out"` (or correct name) so `_SIG_ORDER` has `kind: 'output'` not `'tensorList'`
- [ ] For `aclTensorList*` output params: CPU golden returns `List[torch.Tensor]` directly (not wrapped in another list)
- [ ] For `aclScalarList*` input params: JSON `inputs` has `type='scalars'` with `length` field — generator.py handles the expansion; CPU golden uses `zip(x, scalars)` pattern
- [ ] Verified Python syntax
- [ ] No NPU-specific imports or parameters in the CPU code
- [ ] Existing imports are preserved (no removal/modification); new imports are added only when necessary


---

# Distributed Operator CPU Golden Derivation

For ACLNN operators that require **inter-card communication** (all-to-all, all-reduce, all-gather, reduce-scatter, send/recv, etc.), the CPU golden cannot be a single-process computation. It requires `torch.distributed` with multi-process launch.

## Detection: Is This a Distributed Operator?

Check for these signals in the signature or JSON config:
- Parameters named `group`, `hcclGroup`, `comm` (type `const char*` or `aclUint64`)
- `alltoAll`, `all_to_all`, `reduce`, `gather`, `scatter`, `send`, `recv` in the operator name
- JSON config contains `dist_api_type` field
- Input params include `world_size`, `rank`, `hasAlltoall`, `hasReduce` etc.
- Input tensor shapes don't match for direct matmul/compute (e.g. x1=[1024,1280], x2=[2560,2560] — dimensions only align after all-to-all)

## Architecture Overview

```
ATK dist mode launch:
  atk node --backend pyaclnn --devices 0,1,2,3 --is_dist true --dist_backend hccl \
     node --backend cpu --is_dist true --dist_backend gloo \
     task -c case.json -p test.py --task accuracy

Each rank runs as a separate celery worker process:
  rank 0: pyaclnn on NPU:0  +  cpu on CPU (gloo)
  rank 1: pyaclnn on NPU:1  +  cpu on CPU (gloo)
  ...

ATK calls torch.distributed.init_process_group automatically in run_dist_opp_task():
  - For pyaclnn: backend=hccl
  - For cpu: backend=gloo
  - rank/world_size are set via task_result.dist_task_info
```

## Key ATK Interfaces for Dist

### `task_result.dist_task_info` (DiskTaskInfo)
Accessible from any `BaseApi` subclass via `self.task_result.dist_task_info`:
- `.rank` — current process rank (0, 1, 2, ...)
- `.world_size` — total number of ranks
- `.dist_backend` — "hccl", "gloo", or "nccl"
- `.master_ip` — master node IP for init_process_group
- `.master_port` — master port
- `.init_method` — "tcp://master_ip:master_port"
- `.is_bm` — whether running benchmark mode

### `self.name` and `self.device`
From `BaseApi.__init__()`:
- `self.name` — node name (set by ATK, e.g. "cpu" for CPU node)
- `self.device` — backend type string

### `self.task_result.is_dist`
Boolean flag: `True` when running in dist mode.

## Writing a Distributed CPU Golden

### Step 1: Class Structure

```python
@register("function_aclnn_alltoall_matmul")
class FunctionAclnnAlltoallMatmul(BaseApi):
    _OP_NAME = "aclnnAlltoAllMatmul"
    _SIG_STR = """..."""

    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)
        self.rank = self.task_result.dist_task_info.rank
        self.world_size = self.task_result.dist_task_info.world_size
```

### Step 2: `__call__` with Dist Communication

```python
def __call__(self, input_data: InputDataset, with_output: bool = False):
    import torch.distributed as dist
    from atk.configs.dataset_config import InputDataset

    rank = self.rank
    world_size = self.world_size

    # Extract inputs
    x = input_data.kwargs["x"]          # [M_local, K] — local shard
    weight = input_data.kwargs["weight"] # [M_global, N]
    bias = input_data.kwargs.get("bias")
    hasBias = input_data.kwargs.get("hasBias", False)
    hasAlltoall = input_data.kwargs.get("hasAlltoall", False)

    if self.name == "cpu":
        # ---- CPU golden path ----
        x = x.cpu().to(torch.float32)
        weight = weight.cpu().to(torch.float32)
        bias = bias.cpu().to(torch.float32) if bias is not None else None

        M_local = x.shape[0]
        K = x.shape[1]

        # ---- All-to-All communication ----
        # Split x into world_size chunks along dim=1, each chunk sent to one rank
        # Receive chunks from all ranks and concatenate along dim=1
        x_splits_to_ranks = torch.chunk(x, world_size, dim=1)  # each: [M_local, K/world_size]
        x_splits_from_ranks = [torch.empty_like(x_splits_to_ranks[0]) for _ in range(world_size)]

        if world_size > 1:
            self._manual_all_to_all(x_splits_from_ranks, x_splits_to_ranks, rank, world_size)

        # After all-to-all: A = [M_local, K * world_size] = [M_local, K_global]
        A = torch.cat(x_splits_from_ranks, dim=1)

        # ---- Matmul ----
        output = torch.matmul(A, weight)
        if hasBias and bias is not None:
            output = output + bias.to(A.dtype)

        if hasAlltoall:
            return [output, A]
        return output

    # ---- NPU / benchmark path (is_bm) ----
    if self.task_result.dist_task_info.is_bm:
        ...  # NPU reference path using dist.all_to_all_single
    else:
        pass
```

### Step 3: Manual All-to-All Implementation

For CPU (gloo backend), use `dist.send`/`dist.recv`:

```python
def _manual_all_to_all(self, x_splits_from_ranks, x_splits_to_ranks, rank, world_size):
    """
    Simulate all-to-all: each rank splits its tensor into world_size chunks,
    sends chunk[i] to rank i, and receives chunk[i] from rank i.

    x_splits_to_ranks:  list of tensors TO send (rank[i]'s chunk for destination rank i)
    x_splits_from_ranks: list of empty tensors TO receive (from rank i)
    """
    import torch.distributed as dist

    if rank == 0:
        # Rank 0: send to others, then receive from others
        for target_rank in range(1, world_size):
            dist.send(tensor=x_splits_to_ranks[target_rank], dst=target_rank,
                      tag=rank * 1000 + target_rank)
        for src_rank in range(1, world_size):
            dist.recv(tensor=x_splits_from_ranks[src_rank], src=src_rank,
                      tag=src_rank * 1000 + rank)
    else:
        # Other ranks: first receive from rank 0, then send to others
        dist.recv(tensor=x_splits_from_ranks[0], src=0,
                  tag=0 * 1000 + rank)
        for target_rank in range(1, world_size):
            if target_rank != rank:
                dist.send(tensor=x_splits_to_ranks[target_rank], dst=target_rank,
                          tag=rank * 1000 + target_rank)
            for src_rank in range(1, world_size):
                if src_rank != rank and src_rank != 0:
                    dist.recv(tensor=x_splits_from_ranks[src_rank], src=src_rank,
                              tag=src_rank * 1000 + rank)
```

> **IMPORTANT**: The send/recv order must avoid deadlock. Rank 0 sends first, others receive first. Then others send, and others receive among themselves.

### Step 4: NPU Benchmark Path (Optional)

For `is_bm=True`, use `dist.all_to_all_single` (NPU/HCLL path):

```python
if self.task_result.dist_task_info.is_bm:
    split_sizes = [M_local * K_per_rank] * world_size
    x_flat = x.flatten()
    x_recv = torch.empty(world_size * M_local * K_per_rank, dtype=x.dtype, device=x.device)
    dist.all_to_all_single(x_recv, x_flat)
    A = x_recv.reshape(world_size, M_local, K_per_rank).permute(1, 0, 2).reshape(M_local, -1).contiguous()
    output = torch.matmul(A, weight)
    if hasBias:
        output = output + bias.to(A.dtype)
```

## ATK Dist Launch Command

```bash
# Single-machine multi-card:
atk node --backend pyaclnn --devices 0,1 --is_dist true --dist_backend hccl \
   node --backend cpu --is_dist true --dist_backend gloo \
   task -c aclnnAllToAllMatmul.json -p test_aclnnAllToAllMatmul.py --task accuracy

# The --is_dist flag triggers:
# - run_dist_opp_task() instead of run_opp_task()
# - torch.distributed.init_process_group is called automatically
# - dist_task_info.rank / world_size are populated
# - Each device gets a separate worker with its own rank
```

## Common Distributed Patterns

| ACLNN Operator | Communication Pattern | CPU Golden Implementation |
|---|---|---|
| `aclnnAlltoAllMatmul` | all-to-all + matmul | `_manual_all_to_all()` + `torch.matmul()` |
| `aclnnAllReduce` | all-reduce | `dist.all_reduce()` |
| `aclnnAllGather` | all-gather | `dist.all_gather()` |
| `aclnnReduceScatter` | reduce-scatter | `dist.reduce_scatter()` |
| `aclnnSendRecv` | point-to-point | `dist.send()` / `dist.recv()` |
| `aclnnEmbeddingShard` | all-gather + embedding | `dist.all_gather()` + `F.embedding()` |

## Checklist for Distributed CPU Golden

- [ ] Override `__init__` to extract `rank` and `world_size` from `self.task_result.dist_task_info`
- [ ] Guard communication code with `if self.name == "cpu":` (CPU golden path)
- [ ] All-to-all uses `_manual_all_to_all()` with proper send/recv ordering (no deadlock)
- [ ] NPU-specific params (`group` as HCCL comm) are replaced with `dist.all_to_all_single` or manual send/recv
- [ ] Input tensor chunks are split along the correct axis for the communication pattern
- [ ] Output shape matches ACLNN expectation after communication + computation
- [ ] Output dtype matches NPU SupportInfo (`.to(input_dtype)` for main output, `.float()` for statistics)
- [ ] Multi-output operators return list `[output, intermediate]` when `alltoAllOutOptional` etc. exist
- [ ] Verified with `--is_dist true` launch (not single-process)
- [ ] No deadlock: rank 0 sends first, others receive first (or use non-blocking `dist.isend`/`dist.irecv`)

---

# CPU Golden Robustness: Attr Parameter Handling

These rules ensure the CPU golden `__call__` method never crashes when ATK passes unexpected values
(`None`, float-for-int, or mismatched types) in attr parameters.

## Rule 1: Integer Attr May Be `None` — Always Check Before Converting

**Problem:** ATK passes `null` in JSON for an attr parameter. When the `_param_map` maps this value,
`_get_param("reduction", 0)` returns `None` (because the value in args is literally `None`), and
`int(None)` raises `TypeError: int() argument must be a string, a bytes-like object or a number, not 'NoneType'`.

**Fix — pattern for integer attrs with enum semantics (reduction, mode, etc.):**
```python
reduction = _get_param("reduction", 0)
if reduction is None:
    reduction = 0  # default to "none" / identity behavior — VALUE FROM Step 0 doc
```

**Fix — pattern for integer attrs with scalar semantics (alpha, beta, dim, etc.):**
```python
alpha = _get_param("alpha", 1.0)
if alpha is None:
    alpha = 1.0
```

**General principle:** Always explicitly check `is None` after extracting an attr. Never directly pass
a potentially-`None` value to `int()`, `float()`, or use it as a list index. The default value should
come from the operator doc (Step 0) — e.g., the doc says `reduction: 支持0(none)|1(mean)|2(sum)` so
default to `0`.

## Rule 2: Integer Attr May Be Float — Clamp / Round to Valid Range

**Problem:** ATK's random data generator produces float values for integer attrs (e.g. `reduction: 0.99`,
`1.01`, `1.99`, `-0.01`). Using these directly as list indices or enum values causes `IndexError` or
incorrect behavior.

**Fix — for enum-like integer attrs (reduction, mode, etc. with discrete valid values):**

The valid range comes from the operator doc (Step 0). For example, the doc says
`reduction: 支持0(none)|1(mean)|2(sum)` — that gives you `[0, 2]` as the valid range:

```python
reduction = _get_param("reduction", 0)
if reduction is None:
    reduction = 0
# ATK may generate float (0.99, 1.01, 1.99) — round and clamp to valid range
# Valid range [0, 2] from doc: "支持0(none)|1(mean)|2(sum)"
reduction = max(0, min(2, round(reduction)))
reduction_str = ["none", "mean", "sum"][reduction]
```

**General principle:** Read the valid value range from the operator doc's "使用说明" column (Step 0).
Always `round()` then `clamp()` after the `None` check: `max(min_val, min(max_val, round(v)))`.

## Rule 3: Output Dtype Must Match NPU SupportInfo — Always Cast

**Problem:** PyTorch functions return results in promoted dtypes (e.g. `fp32`) while the NPU `SupportInfo`
expects the input dtype (`fp16`). ATK uses the CPU golden's return dtype to infer NPU output tensor dtype.
A dtype mismatch causes `Io input dtype or format is not supported` (error 161002).

**Fix — always cast the final result before returning:**
```python
result = F.binary_cross_entropy_with_logits(self_t, target, ...)
result = result.to(self_t.dtype)  # Cast to input dtype to match NPU SupportInfo
return result
```

**General principle:**
- Main output → always cast to `self_t.dtype` (the input dtype)
- Statistics output (mean/var/rstd) → always `.float()` (fp32)
- Index output (argmax/topk/sort indices) → `torch.int64`

## Rule 4: `_get_param` — Name-Based Lookup via `_param_map` (from args by case_config)

**How ATK passes data:** ATK's `base_dataset.py` appends every generated value to `input_data.args`
in JSON `inputs[]` order. **`input_data.kwargs` is always empty** for ACLNN operators.

**The auto-generated `_get_param` helper builds a `_param_map` dict by matching `input_data.args[i]`
with `case_config.inputs[i].name`:**
```python
# In __call__:
_param_map = {}
if input_data.args and hasattr(self, 'task_result') and self.task_result.case_config.inputs:
    flat_configs = self.task_result.case_config.flatten_list(self.task_result.case_config.inputs)
    for idx, conf in enumerate(flat_configs):
        if idx < len(input_data.args) and conf.name:
            _param_map[conf.name] = input_data.args[idx]
            _param_map[conf.name.lower()] = input_data.args[idx]

def _get_param(name, default=None):
    v = _param_map.get(name)
    if v is None:
        v = _param_map.get(name.lower())
    if v is not None:
        return v
    return default
```

**IMPORTANT:**
- This approach is **name-based**, not positional — it works regardless of JSON `inputs[]` order vs C++ signature order
- The lookup supports case-insensitive matching (e.g. `negativeslope` vs `negativeSlope`)
- If a JSON input entry has no `name` field, it will NOT be in `_param_map` — use `_get_param` with a sensible default
- **Do NOT write this mapping code yourself** — it is auto-generated by `generator.py`. You only call `_get_param(name)` and `_get_tensor(name)` in the CPU golden code
- **NEVER access `input_data.kwargs` directly** — it is empty for ACLNN function-type operators
- The `self.task_result.case_config.inputs` list may contain nested lists (for grouped params like backward pairs). Use `case_config.flatten_list()` to get a flat list matching `input_data.args`

## Rule 5: Optional Tensor Params — Pass `None` to PyTorch, Not Empty Tensors

**Problem:** `weightOptional` and `posWeightOptional` are optional ACLNN params. When ATK generates
test data, the JSON may define them with a shape that doesn't broadcast with `self`/`target`, causing
`RuntimeError: The size of tensor a (X) must match the size of tensor b (Y)`.

**Fix — in CPU golden code, optional tensor params should use `None` as default in `_get_param`:**
```python
weightOptional = _get_param("weightOptional", None)
posWeightOptional = _get_param("posWeightOptional", None)
```

This lets PyTorch handle the param as "not provided" when ATK passes `None`.

**However:** The real root cause is usually the **JSON test data** having incompatible shapes.
The fix for that is in the JSON (see next section).

## Rule 6: Tensor Param Extraction — Use `_get_tensor()`, Not Raw `_get_param()`

**Problem:** `_get_param("weight1")` may return a non-tensor value (int, float, dict) when ATK's
data generator produces incorrect types or the JSON `dtype` field is non-standard (e.g. `"int"` instead
of `"int8"`). When the CPU golden code calls `.float()` or `.to()` on the result, it crashes with
`AttributeError: 'int' object has no attribute 'float'`.

**Fix — always wrap tensor extraction with `_get_tensor()`:**
```python
def _get_tensor(name, default=None):
    v = _get_param(name, default)
    return v if isinstance(v, torch.Tensor) else default

# Usage:
weight1 = _get_tensor("weight1")
weight2 = _get_tensor("weight2")
x = _get_tensor("x")
scaleOptional = _get_tensor("scaleOptional", None)  # optional
```

**Why:** `_get_param` is a blind value extractor — it returns whatever ATK put in the dict.
`_get_tensor` adds a type gate that silently falls back to `None` for corrupt entries,
preventing the `'int' object has no attribute 'float'` class of errors.

**Application:**
- Use `_get_tensor()` for ALL params expected to be `atk.Tensor`
- Use `_get_param()` only for scalars (float, int), attrs (enum ints), and strings
- The auto-generated test file skeleton from `generator.py` already includes `_get_tensor`
  and uses it for tensor params — when writing CPU golden code, follow this pattern

**Note:** If `_get_tensor` returns `None` for a required tensor, the underlying issue is
corrupt JSON test data — fix the JSON `dtype` and `shape` fields, then re-run.

---

# ATK JSON Test Data Validation & Repair

When ATK test cases fail with shape/dtype errors, the root cause is often the JSON test data itself
(ATK's random data generator doesn't enforce operator-specific constraints). Use these rules to
diagnose and fix.

## Problem Patterns & Fixes

### Pattern 1: Optional Weight/posWeight Shape Cannot Broadcast

```
RuntimeError: The size of tensor a (16) must match the size of tensor b (7) at non-singleton dimension 0
```

**Cause:** `weightOptional`/`posWeightOptional` shape doesn't broadcast with `self`/`target`.

**Fix:** Set the weight/posWeight shape to either:
- `[1]` (scalar, broadcasts to any shape)
- Same shape as `self`/`target` broadcast result

### Pattern 2: Output Shape Mismatch from Broadcast

```
RuntimeError: output with shape [1] doesn't match the broadcast shape [65534]
```

**Cause:** ATK computes the expected output shape from the broadcast of ALL inputs including
weight/posWeight, but `self`/`target` are too small, so PyTorch's output shape doesn't match.

**Fix:** Set `self` and `target` shape to match the broadcast result of all input tensors.
When `posWeightOptional` is large (e.g. `[65534]`), `self`/`target` should also be `[65534]`.

### Pattern 3: Integer Attr Has Float Values

**Cause:** ATK random generator produces `0.99`, `1.01`, `-0.01` for integer attrs.

**Fix in JSON:** Change `range_values` to an integer within valid range.
**Fix in Python:** Add `round()` + `clamp()` (see Rule 2 above).

### Pattern 4: Integer Attr Is `null` / `None`

**Cause:** JSON has `"range_values": null` for an integer attr, or ATK generates `None`.

**Fix in Python:** Always check `is None` before using (see Rule 1 above).

## General JSON Shape Validation Rules for Loss / Reduction Operators

For operators like `BCELoss`, `NLLLoss`, `CrossEntropyLoss`, `MSELoss`, `L1Loss`, etc.:

1. `self` and `target` shapes must be identical or broadcastable
2. `weightOptional` (if present) must broadcast to `self`/`target`'s broadcast shape, or be `[1]`
3. `posWeightOptional` (if present) must broadcast to `self`/`target`'s broadcast shape, or be `[1]`
4. `reduction` must be `0` (none), `1` (mean), or `2` (sum) — no float values
5. When `reduction=0` (none), output shape = broadcast of `self`+`target`
6. When `reduction=1/2` (mean/sum), output is a scalar — ATK should expect shape `[]` or `[1]`

## General JSON Shape Validation Rules for MoE/FFN Operators

For operators like `aclnnFFNV3`, `aclnnMOE`, or any MoE (Mixture of Experts) operator:

### Expert Dimension Rules

1. `expertTokensOptional` shape must be `[E]` where `E` = number of experts (matches first dimension of all per-expert weights)
2. `sum(expertTokensOptional)` must equal `M` — the batch/sequence dimension of input `x`
3. `x.shape` should be `[M, K]` where `K` = input feature dim (must match `weight1[e].shape[0]`)
4. Per-expert weight shapes: `weight1` is `[E, K1, N1]`, `weight2` is `[E, K2, N2]`
5. NPU constraint: `N1 == K2` (output of first linear = input of second linear)
6. NPU constraint: `N2 == K1` typically (residual connection requires matching input/output dims)

### Quantization Constraint Rules

7. **Quant and antiquant params MUST NOT coexist** — if the case has `antiquantScale*Optional` / `antiquantOffset*Optional`,
   remove `scaleOptional`/`deqScaleOptional`/`scale`/`offset` params (and vice versa)
8. Pseudo-quantized input `x` must be `fp16` or `fp32` — NOT integer dtypes (the de-quant + compute happens internally)
9. Antiquant scale/offset shapes are per-expert: `[E, N1]` and `[E, N2]` — NOT flat `[N1]`/`[N2]`

### Activation Attr Rules

10. `activation.range_values` must be a valid enum string (`"gelu"`, `"relu"`, `"silu"`, `"geglu"`, `"swiglu"`, `"reglu"`) — NOT `null`
11. `innerPrecise.range_values` must be `0` or `1` — NOT a float like `-0.01`

### CPU Golden Derivation for MoE FFN

When deriving CPU golden for MoE FFN operators with pseudo-quantization:
- Antiquant computation MUST be done **per-expert inside the dispatch loop**, NOT at the top level
- Per-expert weight `we1 = W1[e]` has shape `[K1, N1]`, and `antiquantOffset1Optional[e]` has shape `[N1]` — dimensions align only after per-expert slicing
- Top-level attempt: `(W1.float() + antiquantOffset1Optional.float())` fails because `W1` is `[E, K1, N1]` and offset is `[E, N1]` — the K1 dimension doesn't broadcast

---

# WeightNZ Operators (NZ Sparse Format)

Operators whose name contains `WeightNz` or `WeightNZ` use the **NZ (FRACTAL_NZ)** sparse format for the weight tensor (`mat2`). This is an Ascend NPU-internal memory layout that requires **both NPU-side format conversion and CPU-side NZ→dense decompression**.

**Affected operators:** `aclnnBatchMatMulWeightNz`, `aclnnMatmulWeightNz`, `aclnnAddmmWeightNz`, `aclnnGroupedMatmulWeightNz`, and any other operator with `WeightNz` in the name.

## Step 0 Addition: Detect NZ Format from Doc

When reading the operator doc (Step 0), check the mat2 parameter's **"数据格式"** column:
- If it says **`NZ`** or **`FRACTAL_NZ`**, this is a WeightNZ operator
- The doc specifies the NZ shape dimensions: e.g. for `aclnnBatchMatMulWeightNz`, mat2 is 5D `[b, n1, k1, k0=16, n0=16]`
- The doc specifies `k0 = 16, n0 = 16` (fixed constants)
- The doc specifies the shape relationship: `ceil(k / k0) = k1`, `ceil(n / n0) = n1`

**This determines TWO things:**
1. **JSON**: mat2 must be defined with the **NZ shape** (5D for batch, 4D for non-batch), `format: "NZ"`
2. **Python**: Both `AclnnFunction` (NPU side) and `Function` (CPU side) need NZ-specific handling

## Step 0 Addition: NZ Format JSON Constraints

When writing the JSON test data for a WeightNZ operator, enforce ALL of the following from the doc:

| Constraint | Rule | Example |
|---|---|---|
| **mat2 dimensions** | Must match doc's NZ dimension count (5D for batch, 4D for non-batch) | `[b, n1, k1, k0, n0]` |
| **k0, n0** | Must be exactly `16` | `nz.shape[3] == 16`, `nz.shape[4] == 16` |
| **k1 = ceil(k/16)** | Derived from self's k dimension | `k=32 → k1=2`; `k=48 → k1=3` |
| **n1 = ceil(n/16)** | Derived from desired output n | `n=16 → n1=1`; `n=64 → n1=4` |
| **self shape** | 3D `[b, m, k]` for batch; 2D `[m, k]` for non-batch | Must satisfy `ceil(k/16) = k1` |
| **dtype consistency** | self and mat2 must have the **same dtype** | Both `fp16` or both `bf16`; **NOT mixed** |
| **cubeMathType** | Must be integer `0` (KEEP_DTYPE) or `2` (USE_FP16, not with BFLOAT16) | `range_values: 0` |
| **k ≠ 1, n ≠ 1** | Doc says "不支持k=1或n=1" | k ≥ 2, n ≥ 2 |
| **b broadcast** | self.b and mat2.b must satisfy broadcast relationship | Usually identical, or one is 1 |

**Computing correct NZ shape from desired dense shape:**

```python
import math
# Given: self shape [b, m, k], desired output [b, m, n]
k0, n0 = 16, 16
k1 = math.ceil(k / k0)  # e.g. k=32 → k1=2
n1 = math.ceil(n / n0)  # e.g. n=16 → n1=1
# NZ shape: [b, n1, k1, k0, n0]
mat2_nz_shape = [b, n1, k1, k0, n0]
```

## Python Side: Two-Part Implementation

A WeightNZ operator requires changes in **both** the `AclnnFunction` (NPU) and `Function` (CPU) classes.

### Part A: AclnnFunction — NPU-Side ND→NZ Format Conversion

ATK creates the mat2 tensor as **ND format** from the JSON data, but the NPU interface requires **FRACTAL_NZ format**. You must add three methods to `AclnnFunction`:

**1. Required imports** (at file top):
```python
import math
import torch.nn.functional as F
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat, nnopbase
```

**2. `get_format` method** — tells ATK the actual format of each tensor:
```python
def get_format(self, input_data: InputDataset, index=None, name=None):
    if name == "mat2":
        return AclFormat.ACL_FORMAT_FRACTAL_NZ
    return AclFormat.ACL_FORMAT_ND
```

**3. `get_storage_shape` method** — returns the NZ storage shape for aclTensor creation:
```python
def get_storage_shape(self, input_data: InputDataset, index=None, name=None):
    if name == "mat2" and input_data.kwargs and "mat2" in input_data.kwargs:
        mat2 = input_data.kwargs["mat2"]
        if mat2.dim() == 5:
            # Already 5D NZ shape [b, n1, k1, k0, n0], use directly
            return mat2.shape
        # 3D dense [b, k, n] → derive 5D NZ shape
        b = mat2.shape[0]
        k = mat2.shape[-2]
        n = mat2.shape[-1]
        k0, n0 = 16, 16
        k1 = math.ceil(k / k0)
        n1 = math.ceil(n / n0)
        return torch.Size([b, n1, k1, k0, n0])
    elif name is not None and input_data.kwargs and name in input_data.kwargs:
        return input_data.kwargs[name].shape
    return None
```

**4. `init_by_input_data` — ND→NZ conversion at the end:**

After the auto-generated loop builds `input_args`, add NZ conversion for mat2:

```python
# mat2: convert ND format to FRACTAL_NZ format
mat2_nd = self.acl_tensor_to_torch(input_args[<mat2_index>])
import torch_npu
mat2_nz = mat2_nd.contiguous().npu()
input_args[<mat2_index>] = nnopbase.create_acl_tensor(
    mat2_nz, AclFormat.ACL_FORMAT_FRACTAL_NZ, mat2_nd.shape
)
```

**Where `<mat2_index>` is the position of mat2 in `input_args`**, determined by `_SIG_ORDER` — count non-framework, non-output params before mat2, plus output positions. For `aclnnBatchMatMulWeightNz` with `_SIG_ORDER = [self, mat2, out, cubeMathType, ...]`: `self=0, mat2=1, out=2, cubeMathType=3`.

### Part B: Function — CPU-Side NZ→Dense Decompression

On CPU, PyTorch cannot compute with NZ format. You must decompress NZ→dense before the matmul.

**How NZ format works (from reference implementation):**

The forward conversion (dense → NZ) done by `aclnnTransMatmulWeight`:
1. Dense `[b, k, n]` → pad to `[b, k_padded, n_padded]` (multiples of 16)
2. Reshape to `[b, k1, k0, n1, n0]` where `k1=k_padded/16`, `n1=n_padded/16`
3. Permute `(0, 3, 1, 2, 4)` → `[b, n1, k1, k0, n0]` = NZ layout

The **inverse** (NZ → dense) for CPU golden:
1. NZ `[b, n1, k1, k0, n0]` → permute `(0, 2, 3, 1, 4)` → `[b, k1, k0, n1, n0]`
2. Reshape to `[b, k_padded, n_padded]`
3. Slice to actual `k`: `[..., :k_actual, :]`

**CPU golden code pattern:**

```python
k0, n0 = 16, 16

if mat2_t.dim() == 5:
    # NZ format: determine layout from self's k dimension
    nz = mat2_t.float()
    b = nz.shape[0]
    k_actual = self_t.shape[-1]
    k1_expected = (k_actual + k0 - 1) // k0  # ceil(k/16)
    
    if nz.shape[2] == k1_expected and nz.shape[3] == k0 and nz.shape[4] == n0:
        # Non-transposed: [b, n1, k1, k0, n0]
        n1 = nz.shape[1]
        k1 = nz.shape[2]
        dense_padded = nz.permute(0, 2, 3, 1, 4).reshape(b, k1 * k0, n1 * n0)
    elif nz.shape[1] == k1_expected and nz.shape[3] == n0 and nz.shape[4] == k0:
        # Transposed: [b, k1, n1, n0, k0]
        k1 = nz.shape[1]
        n1 = nz.shape[2]
        dense_padded = nz.permute(0, 1, 4, 2, 3).reshape(b, k1 * k0, n1 * n0)
    else:
        # Fallback: assume non-transposed [b, n1, k1, k0, n0]
        n1 = nz.shape[1]
        k1 = nz.shape[2]
        dense_padded = nz.permute(0, 2, 3, 1, 4).reshape(b, k1 * k0, n1 * n0)
    dense_t = dense_padded[..., :k_actual, :]
else:
    # 3D fallback: treat as dense directly [b, k, n]
    dense_t = mat2_t.float()

result = torch.bmm(self_t.float(), dense_t)
result = result.to(self_t.dtype)
return result
```

**For non-batch WeightNZ operators** (e.g. `aclnnMatmulWeightNz`):
- NZ shape is 4D `[n1, k1, k0, n0]` instead of 5D
- Use `torch.matmul` instead of `torch.bmm`
- Permute adjusts: `permute(2, 3, 0, 1)` for non-transposed `[n1, k1, k0, n0]` → `[k1, k0, n1, n0]`
- The `k1_expected` check on `nz.shape[2]` (k1) and `nz.shape[1]` (n1)

## Checklist for WeightNZ Operators

- [ ] **Read doc** — confirmed mat2 format is NZ, extracted k0=16, n0=16
- [ ] **JSON mat2 shape** — 5D `[b, n1, k1, 16, 16]` with correct k1=ceil(k/16), n1=ceil(n/16)
- [ ] **JSON mat2 format** — `"format": "NZ"`
- [ ] **JSON mat2 dtype** — **same** as self dtype (no fp16/bf16 mixing)
- [ ] **JSON self shape** — 3D `[b, m, k]` where `ceil(k/16) = k1` matches mat2
- [ ] **JSON cubeMathType** — integer `0` or `2`, NOT float
- [ ] **JSON k ≥ 2, n ≥ 2** — no k=1 or n=1
- [ ] **Python AclnnFunction** — added `get_format` returning `ACL_FORMAT_FRACTAL_NZ` for mat2
- [ ] **Python AclnnFunction** — added `get_storage_shape` for mat2
- [ ] **Python AclnnFunction** — `init_by_input_data` converts ND→FRACTAL_NZ via `nnopbase.create_acl_tensor`
- [ ] **Python Function** — CPU golden decompresses NZ→dense via permute+reshape+slice
- [ ] **Python Function** — handles both 5D NZ and 3D dense fallback
- [ ] **Python Function** — uses `self_t.shape[-1]` to determine k1 layout (transposed vs non-transposed)
- [ ] **Python imports** — added `math`, `torch.nn.functional as F`, `AclFormat`, `nnopbase`
- [ ] **NPU error 161002** — if you see "Expected X dimension input, but got otherTensor with sizes [...]", the mat2 format is wrong (ND instead of NZ) → check `get_format` and `init_by_input_data`
- [ ] **CPU error "batch2 must be a 3D tensor"** — mat2 is 5D NZ on CPU → need NZ→dense decompression
