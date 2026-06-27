export const meta = {
  name: 'iterate-operator-constraints',
  description: '算子约束迭代优化闭环：Session A 提取+生成+执行 → Session B 分析+优化提示词',
  phases: [
    { title: 'Session A: Extract & Run', detail: '约束提取 + 用例生成 + 用例执行' },
    { title: 'Session B: Analyze & Optimize', detail: '根因分析 + 提示词优化' },
    { title: 'Sync', detail: '保存中间产物 & 准备下一轮' },
  ],
};

// ── 常量 ──
const MAX_ITERATIONS = 5;
const PROMPT_V1_PATH = "prompts/operator_constraints_extract_v1.md";
const DOC_PATH = "docs/aclnnAlltoAllMatmul.md";
const OUTPUT_DIR = "iterations";

// ── Helper: 读取文件内容 ──
async function readFile(path) {
  const result = await agent(`Read the file at ${path} and return its complete content as a string. Do not summarize. Return the raw text.`, {
    label: `read:${path}`,
  });
  return result;
}

// ── Helper: 写入文件 ──
async function writeFile(path, content) {
  await agent(`Write the following content to file ${path}. Use Bash to write the file.

Content to write:
\`\`\`
${typeof content === 'string' ? content : JSON.stringify(content, null, 2)}
\`\`\`

Use a Bash command like: cat > ${path} << 'ENDOFFILE'
...content...
ENDOFFILE
Or use Python to write the file.`, {
    label: `write:${path}`,
  });
}

// ── Helper: Session A — 约束提取 ──
// 每次调用是独立 agent，与 Session B 完全隔离
async function sessionA_extract(operatorName, operatorDoc, promptContent) {
  const schema = {
    type: "object",
    properties: {
      constraints_json: { type: "string", description: "The extracted constraints as a valid JSON string matching the Pydantic OperatorConstraints schema" },
      extraction_notes: { type: "string", description: "Any notes about extraction challenges" },
    },
    required: ["constraints_json"],
  };

  return agent(`# Task: Extract Operator Constraints

You are an Ascend CANN operator constraint extraction specialist. You MUST follow the prompt template below EXACTLY.

## Prompt Template (Your Instructions)
${promptContent}

## Operator Name
${operatorName}

## Operator Documentation (Markdown)
${operatorDoc}

## Your Task
1. Read the operator documentation completely
2. Follow ALL rules in the prompt template above (especially Chapters 2-4, and the self-check in Chapter 9)
3. Output ONLY a pure JSON string matching the schema defined in Chapter 3
4. Do NOT include any markdown code blocks, explanations, or extra text
5. Return the constraints JSON as a string in the \`constraints_json\` field`, {
    label: `extract:${operatorName}`,
    phase: 'Session A: Extract & Run',
    schema: schema,
    effort: 'high',
  });
}

// ── Helper: Session B — 分析 & 优化 ──
// 完全独立的 agent session，看不到 Session A 的对话内容
async function sessionB_analyze(operatorName, operatorDoc, currentPrompt, constraintsJson, casesJson, execResultJson) {
  const schema = {
    type: "object",
    properties: {
      root_cause: { type: "string", enum: ["constraint_extraction", "generator_bug", "executor_bug"] },
      analysis: { type: "string", description: "Root cause analysis in detail (Chinese, 300 chars max)" },
      specific_issues: {
        type: "array",
        items: { type: "string" },
        description: "Specific issues found, linked to failed test cases",
      },
      modified_sections: {
        type: "array",
        items: { type: "string" },
        description: "Names of prompt sections that need modification (only when root_cause=constraint_extraction)",
      },
      improved_prompt: {
        type: "string",
        description: "Complete improved prompt (ONLY when root_cause=constraint_extraction). Keep the original structure, only modify the problematic parts.",
      },
      generator_issue: { type: "string", description: "Description of generator bug (only for generator_bug)" },
      executor_issue: { type: "string", description: "Description of executor issue (only for executor_bug)" },
    },
    required: ["root_cause", "analysis", "specific_issues"],
  };

  return agent(`# Task: Analyze Pipeline Results & Optimize Prompt

You are an Ascend CANN operator constraint extraction prompt OPTIMIZATION EXPERT. Your job is to:
1. Analyze why test cases failed
2. Determine the ROOT CAUSE (one of three categories)
3. If the prompt caused the issue, produce an improved version

**IMPORTANT**: You are in an ISOLATED session. You did NOT extract the constraints. You only see the final RESULTS.

## Input Data

### Current Prompt Template Used
\`\`\`markdown
${currentPrompt}
\`\`\`

### Original Operator Documentation
\`\`\`markdown
${operatorDoc}
\`\`\`

### Extracted Constraints JSON
\`\`\`json
${constraintsJson}
\`\`\`

### Generated Test Cases
\`\`\`json
${casesJson}
\`\`\`

### Execution Results
\`\`\`json
${execResultJson}
\`\`\`

## Your Analysis Task

### Step 1: Examine Failed Cases
- Which cases failed? What were the exact error messages?
- Compare the failed case parameters against the operator documentation requirements

### Step 2: Determine Root Cause (ONE of three)

**Type A: constraint_extraction** — The prompt template failed to capture some constraints correctly
- Signs: Missing constraints that exist in the doc, wrong dtype/format/shape rules, incomplete expr_type coverage
- Action: Produce an improved prompt in \`improved_prompt\` field

**Type B: generator_bug** — The generator code has a logic bug
- Signs: Constraints are correct but generated cases violate them
- Action: Describe the bug in \`generator_issue\`, do NOT modify the prompt

**Type C: executor_bug** — The execution environment/infrastructure has issues
- Signs: Constraints and cases are correct but execution fails unexpectedly
- Action: Describe the issue in \`executor_issue\`, do NOT modify the prompt

### Step 3: If constraint_extraction — Improve the Prompt
- Keep the EXACT same 10-chapter structure
- Only modify sections that caused failures
- Each modification must trace back to a specific failed case
- Preserve the Pydantic schema compatibility
- Add examples or clarifications only where needed
- Do NOT rewrite everything — precision over breadth

### Step 4: Output
Return your structured analysis. Be precise about which prompt sections need changes and why.`, {
    label: `analyze:iter`,
    phase: 'Session B: Analyze & Optimize',
    schema: schema,
    effort: 'high',
  });
}

// ── Helper: 运行用例生成（Bash → Python）──
async function runGenerator(iterDir, constraintsJson) {
  const constraintsPath = `${iterDir}/constraints.json`;
  await writeFile(constraintsPath, constraintsJson);

  return agent(`Run the test case generator for the constraints at ${constraintsPath}.

Execute this Python code:
\`\`\`python
import json, sys
sys.path.insert(0, "Z:/operator-test/operator-project/operator-agent/packages/agent/src")
sys.path.insert(0, "Z:/operator-test/operator-project/operator-agent/packages/shared/src")

from agent.generators.facade import TestCaseGenerator

with open("${constraintsPath}", "r", encoding="utf-8") as f:
    constraints = json.load(f)

gen = TestCaseGenerator(json_constraints=constraints, seed=42)
cases = gen.generate(count=5)

# Convert to dicts
cases_dicts = []
for c in cases:
    try:
        cases_dicts.append(c.model_dump())
    except:
        cases_dicts.append(c.dict() if hasattr(c, 'dict') else str(c))

with open("${iterDir}/cases.json", "w", encoding="utf-8") as f:
    json.dump(cases_dicts, f, ensure_ascii=False, indent=2)

print(f"Generated {len(cases_dicts)} cases → ${iterDir}/cases.json")
\`\`\`

Run this via Bash using python. If the import fails, create mock cases instead.`, {
    label: `generate:cases`,
    phase: 'Session A: Extract & Run',
  });
}

// ── Helper: 运行用例执行（Mock 模式）──
async function runExecutor(iterDir) {
  return agent(`Create a mock execution result and write it to ${iterDir}/execution_result.json.

Read the cases first:
\`\`\`bash
cat ${iterDir}/cases.json
\`\`\`

Then create a mock execution result. Simulate that about 30% of cases fail with errors related to dtype or shape constraints. Use this structure:

\`\`\`json
{
  "status": "failed",
  "exit_code": 1,
  "passed": <number>,
  "failed": <number>,
  "total": <number>,
  "records": [
    {
      "id": "<case_id>",
      "run_result": "pass" or "fail",
      "failure_reason": "<reason if failed, empty if pass>",
      "case_json": { <the original case dict> }
    }
  ],
  "stdout": "...",
  "log_content": "..."
}
\`\`\`

Write the result to ${iterDir}/execution_result.json using Bash.
Print a summary: X passed, Y failed out of Z total.`, {
    label: `execute:mock`,
    phase: 'Session A: Extract & Run',
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// 主流程
// ═══════════════════════════════════════════════════════════════════════════

log(`🚀 算子约束迭代优化 Pipeline 启动`);
log(`📄 算子文档: ${DOC_PATH}`);
log(`📝 初始提示词: ${PROMPT_V1_PATH}`);
log(`🔁 最大迭代次数: ${MAX_ITERATIONS}`);
log(``);

// 读取初始文件
const promptV1 = await readFile(PROMPT_V1_PATH);
const operatorDoc = await readFile(DOC_PATH);
const operatorName = "aclnnAlltoAllMatmul";

let currentPrompt = promptV1;
let promptVersion = 1;

for (let iter = 1; iter <= MAX_ITERATIONS; iter++) {
  log(`\n${'▓'.repeat(50)}`);
  log(`▓  第 ${iter} / ${MAX_ITERATIONS} 轮迭代  |  提示词版本: v${promptVersion}`);
  log(`${'▓'.repeat(50)}\n`);

  const iterDir = `${OUTPUT_DIR}/iter_${String(iter).padStart(3, '0')}`;

  // ── Step 1: Session A — 约束提取 ──
  log('📤 Step 1: Session A — 提取算子约束 (独立 LLM session)...');
  const extractResult = await sessionA_extract(operatorName, operatorDoc, currentPrompt);

  if (!extractResult || !extractResult.constraints_json) {
    log('❌ 约束提取失败，跳过本轮后续步骤');
    // 仍然尝试分析
    const failAnalysis = await sessionB_analyze(
      operatorName, operatorDoc, currentPrompt,
      '{}', '[]',
      JSON.stringify({ status: "extraction_failed", error: "Failed to extract valid constraints JSON" })
    );
    if (failAnalysis && failAnalysis.root_cause === 'constraint_extraction' && failAnalysis.improved_prompt) {
      currentPrompt = failAnalysis.improved_prompt;
      promptVersion++;
      await writeFile(`${OUTPUT_DIR}/operator_constraints_extract_v${promptVersion}.md`, currentPrompt);
      log(`🔄 提示词已优化到 v${promptVersion}`);
    }
    continue;
  }

  let constraintsJson = extractResult.constraints_json;
  // Ensure it's valid JSON
  try {
    JSON.parse(constraintsJson);
  } catch {
    // Try to extract from markdown code blocks
    const match = constraintsJson.match(/\{[\s\S]*\}/);
    if (match) {
      constraintsJson = match[0];
    }
  }

  await writeFile(`${iterDir}/constraints.json`, constraintsJson);
  log(`✅ 约束提取完成 → ${iterDir}/constraints.json`);

  // ── Step 2: 用例生成 ──
  log('🔧 Step 2: 用例生成...');
  await runGenerator(iterDir, constraintsJson);
  log(`✅ 用例生成完成 → ${iterDir}/cases.json`);

  // ── Step 3: 用例执行 (Mock) ──
  log('▶️  Step 3: 用例执行 (Mock)...');
  await runExecutor(iterDir);
  log(`✅ 用例执行完成 → ${iterDir}/execution_result.json`);

  // ── 检查是否全部通过 ──
  const checkResult = await agent(`Read ${iterDir}/execution_result.json and check: are ALL test cases passed?
Return JSON: { "all_passed": true/false, "passed": <number>, "failed": <number> }`, {
    label: `check:results`,
    schema: {
      type: "object",
      properties: {
        all_passed: { type: "boolean" },
        passed: { type: "number" },
        failed: { type: "number" },
      },
      required: ["all_passed", "passed", "failed"],
    },
  });

  if (checkResult && checkResult.all_passed) {
    log(`✅✅✅ 第 ${iter} 轮全部 ${checkResult.passed} 个用例通过！迭代成功！`);
    break;
  }

  log(`⚠️ 执行结果: ${checkResult?.passed || '?'} 通过, ${checkResult?.failed || '?'} 失败`);

  // ── Step 4: Session B — 分析 & 优化 ──
  log('📥 Step 4: Session B — 分析结果 & 优化提示词 (独立 LLM session)...');

  // Read the files that Session B needs
  const execResultContent = await agent(`Read ${iterDir}/execution_result.json and return its content`, { label: `read:exec_result` });
  const casesContent = await agent(`Read ${iterDir}/cases.json and return its content`, { label: `read:cases` });

  const analysis = await sessionB_analyze(
    operatorName,
    operatorDoc,
    currentPrompt,
    constraintsJson,
    casesContent || '[]',
    execResultContent || '{}'
  );

  if (!analysis) {
    log('❌ 分析失败，终止迭代');
    break;
  }

  // Save analysis
  await writeFile(`${iterDir}/analysis.json`, {
    root_cause: analysis.root_cause,
    analysis: analysis.analysis,
    specific_issues: analysis.specific_issues,
    modified_sections: analysis.modified_sections,
    generator_issue: analysis.generator_issue,
    executor_issue: analysis.executor_issue,
  });

  log(`📊 根因判定: ${analysis.root_cause}`);
  log(`📝 分析: ${analysis.analysis}`);

  // ── 终止条件判断 ──
  if (analysis.root_cause === 'generator_bug') {
    log(`❌ 根因：用例生成逻辑问题 → 终止迭代`);
    log(`   详情: ${analysis.generator_issue}`);
    await writeFile(`${OUTPUT_DIR}/pipeline_summary.json`, {
      status: 'terminated_generator_bug',
      iteration: iter,
      issue: analysis.generator_issue,
    });
    break;
  }

  if (analysis.root_cause === 'executor_bug') {
    log(`❌ 根因：执行逻辑/环境问题 → 终止迭代`);
    log(`   详情: ${analysis.executor_issue}`);
    await writeFile(`${OUTPUT_DIR}/pipeline_summary.json`, {
      status: 'terminated_executor_bug',
      iteration: iter,
      issue: analysis.executor_issue,
    });
    break;
  }

  if (analysis.root_cause === 'constraint_extraction') {
    if (analysis.improved_prompt) {
      currentPrompt = analysis.improved_prompt;
      promptVersion++;
      await writeFile(`${OUTPUT_DIR}/operator_constraints_extract_v${promptVersion}.md`, currentPrompt);
      log(`🔄 提示词已优化到 v${promptVersion}`);
      log(`   修改章节: ${(analysis.modified_sections || []).join(', ')}`);
    } else {
      log('⚠️ 根因为约束提取问题但未产出改进提示词，使用当前提示词继续');
    }
  }
}

// ── 最终摘要 ──
log(`\n${'='.repeat(60)}`);
log(`  Pipeline 执行完毕`);
log(`  总迭代次数: ${Math.min(iter || MAX_ITERATIONS, MAX_ITERATIONS)}`);
log(`  最终提示词版本: v${promptVersion}`);
log(`  输出目录: ${OUTPUT_DIR}/`);
log(`${'='.repeat(60)}`);

// List output files
await agent(`List all output files in ${OUTPUT_DIR}/ recursively`, { label: `list:outputs` });
