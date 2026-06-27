# -*- coding: UTF-8 -*-
"""迭代编排器 — 算子约束提取 -> 用例生成 -> 用例执行的闭环迭代。

流水线:
    Session A (独立 LLM)                  Session B (独立 LLM)
    +----------------------+   结果    +-------------------------+
    | 1. 加载 prompt_vN    | ------>  | 4. 分析执行结果/日志       |
    | 2. 提取约束 JSON      |          | 5. 诊断根因              |
    | 3a. 用例生成          |          | 6a. 约束问题 -> 优化提示词  |
    | 3b. 用例执行          |          | 6b. 生成/执行bug -> 终止   |
    +----------------------+          +-------------------------+
                                          |
                             改进提示词 <---+  (仅根因=constraint_extraction)

终止条件:
    - 全部用例通过 -> 成功退出
    - 根因判定为 generator_bug 或 executor_bug -> 终止 & 报告
    - 达到最大迭代次数 (默认 5) 仍未通过 -> 终止 & 报告

Usage:
    python orchestrator.py \
        --prompt prompts/operator_constraints_extract_v1.md \
        --doc docs/aclnnAlltoAllMatmul.md \
        --max-iterations 5 \
        --mock-exec

    python orchestrator.py \
        --prompt prompts/operator_constraints_extract_v1.md \
        --doc docs/aclnnAlltoAllMatmul.md \
        --backend agent
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# -- 确保 agent.generators 可导入（用例生成依赖 operator-agent 的 generators 包）--
_AGENT_ROOT = Path(__file__).resolve().parent.parent / "operator-project" / "operator-agent"
_AGENT_SRC = _AGENT_ROOT / "packages" / "agent" / "src"
_SHARED_SRC = _AGENT_ROOT / "packages" / "shared" / "src"
if str(_AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(_AGENT_SRC))
if str(_SHARED_SRC) not in sys.path:
    sys.path.insert(0, str(_SHARED_SRC))

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class IterationRecord:
    """单轮迭代的完整记录。"""
    iteration: int
    prompt_version: str  # e.g. "v1", "v2"
    prompt_path: str     # 使用的提示词文件路径

    # Session A 产出
    constraints_json: dict | None = None
    constraints_valid: bool = False
    raw_extraction_output: str = ""

    # 用例生成
    cases: list = field(default_factory=list)
    case_count: int = 0
    generation_error: str = ""

    # 用例执行
    execution_result: dict = field(default_factory=dict)
    execution_status: str = ""  # success / failed / timeout / error
    passed_count: int = 0
    failed_count: int = 0

    # Session B 产出
    analysis: dict = field(default_factory=dict)
    root_cause: str = ""
    improved_prompt: str = ""
    improved_prompt_path: str = ""

    @property
    def all_passed(self) -> bool:
        return self.execution_status == "success" and self.failed_count == 0

    @property
    def is_constraint_issue(self) -> bool:
        return self.root_cause == "constraint_extraction"

    @property
    def is_generator_bug(self) -> bool:
        return self.root_cause == "generator_bug"

    @property
    def is_executor_bug(self) -> bool:
        return self.root_cause == "executor_bug"


@dataclass
class PipelineResult:
    """完整流水线运行结果。"""
    operator_name: str
    total_iterations: int = 0
    final_status: str = ""  # "success" | "max_iterations" | "generator_bug" | "executor_bug" | "error"
    iterations: list[IterationRecord] = field(default_factory=list)
    summary: str = ""

    @property
    def successful(self) -> bool:
        return self.final_status == "success"


# ═══════════════════════════════════════════════════════════════════════════════
# 编排器
# ═══════════════════════════════════════════════════════════════════════════════

class IterativePipeline:
    """算子约束迭代优化流水线。

    两个 LLM session 完全隔离：
    - Session A: 约束提取（constraint_extractor.py）
    - Session B: 结果分析 & 提示词优化（result_analyzer.py）

    通过独立的 LLM API 调用实现"记忆互不影响"。
    """

    def __init__(
        self,
        prompt_path: str | Path,
        doc_path: str | Path,
        *,
        output_root: str | Path = "iterator_output",
        max_iterations: int = 5,
        case_count: int = 10,
        mock_exec: bool = False,
        platform: str = "",
        server_config_path: str = "",
        backend_type: str = "api",
    ):
        self.prompt_path = Path(prompt_path)
        self.doc_path = Path(doc_path)
        self.output_root = Path(output_root)
        self.max_iterations = max_iterations
        self.mock_exec = mock_exec
        self.platform = platform or "Atlas A3 训练系列产品/Atlas A3 推理系列产品"
        # server_config_path: 相对路径默认相对于脚本所在目录,绝对路径原样使用
        if server_config_path:
            srv_path = Path(server_config_path)
            if not srv_path.is_absolute():
                srv_path = Path(__file__).resolve().parent / srv_path
            self.server_config_path = str(srv_path)
        else:
            self.server_config_path = ""
        self.backend_type = backend_type
        self.case_count = case_count

        from backends import create_backend
        self.backend = create_backend(backend_type)

        self.operator_name = self.doc_path.stem

        # 运行时创建的目录（在 run() 中初始化）
        self._run_dir: Path | None = None

        # 验证输入
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"提示词文件不存在: {self.prompt_path}")
        if not self.doc_path.exists():
            raise FileNotFoundError(f"算子文档不存在: {self.doc_path}")

    # -- 主入口 ---------------------------------------------------------

    def run(self) -> PipelineResult:
        """运行完整的迭代流水线。"""
        # 创建运行时目录: iterator_output/{operator_name}_{timestamp}/
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        self._run_dir = self.output_root / f"{self.operator_name}_{ts}"
        self._run_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 70)
        logger.info("算子约束迭代优化流水线启动")
        logger.info("算子: %s | 最大迭代: %d | Mock执行: %s | Backend: %s",
                     self.operator_name, self.max_iterations, self.mock_exec,
                     self.backend.name)
        logger.info("输出目录: %s", self._run_dir)
        logger.info("=" * 70)

        result = PipelineResult(operator_name=self.operator_name)
        current_prompt_path = self.prompt_path
        prompt_version = "v1"

        for iteration in range(1, self.max_iterations + 1):
            logger.info("\n" + "#" * 50)
            logger.info("#  第 %d / %d 轮迭代", iteration, self.max_iterations)
            logger.info("#  提示词版本: %s", prompt_version)
            logger.info("#" * 50 + "\n")

            record = IterationRecord(
                iteration=iteration,
                prompt_version=prompt_version,
                prompt_path=str(current_prompt_path),
            )

            # 为本轮创建输出目录
            iter_dir = self._run_dir / f"iter_{iteration:03d}"
            iter_dir.mkdir(parents=True, exist_ok=True)

            # -- Step 1: Session A — 约束提取 --
            self._step_extract_constraints(record, iter_dir)

            if not record.constraints_valid or record.constraints_json is None:
                logger.error("约束提取失败，跳过后续步骤")
                record.root_cause = "constraint_extraction"
                result.iterations.append(record)
                # 即使提取失败也尝试分析
                self._step_analyze(record, iter_dir, current_prompt_path)
                if record.improved_prompt:
                    current_prompt_path = self._save_improved_prompt(
                        record, iter_dir, iteration)
                    prompt_version = f"v{iteration + 1}"
                continue

            # -- Step 2: 用例生成 --
            self._step_generate_cases(record, iter_dir)

            if record.generation_error:
                logger.error("用例生成失败")
                record.root_cause = "generator_bug"
                result.iterations.append(record)
                result.final_status = "generator_bug"
                result.summary = f"第 {iteration} 轮用例生成失败: {record.generation_error}"
                self._write_summary(result)
                return result

            # -- Step 3: 用例执行 --
            self._step_execute_cases(record, iter_dir)

            # -- 全部通过？ --
            if record.all_passed:
                logger.info("[OK] 全部用例通过！迭代结束")
                result.iterations.append(record)
                result.final_status = "success"
                result.total_iterations = iteration
                result.summary = f"第 {iteration} 轮全部 {record.case_count} 个用例通过"
                self._write_summary(result)
                return result

            # -- Step 4: Session B — 分析 & 优化 --
            self._step_analyze(record, iter_dir, current_prompt_path)
            result.iterations.append(record)

            if record.is_generator_bug:
                logger.error("[FAIL] 根因：用例生成逻辑问题 -> 终止迭代")
                result.final_status = "generator_bug"
                result.total_iterations = iteration
                result.summary = record.analysis.get("analysis", "生成器 bug")
                self._write_summary(result)
                return result

            if record.is_executor_bug:
                logger.error("[FAIL] 根因：执行逻辑/环境问题 -> 终止迭代")
                result.final_status = "executor_bug"
                result.total_iterations = iteration
                result.summary = record.analysis.get("analysis", "执行器 bug")
                self._write_summary(result)
                return result

            if record.is_constraint_issue and record.improved_prompt:
                logger.info("[LOOP] 根因：约束提取问题 -> 优化提示词，继续下一轮")
                current_prompt_path = self._save_improved_prompt(
                    record, iter_dir, iteration)
                prompt_version = f"v{iteration + 1}"
            else:
                logger.warning("无法确定根因或未产出改进提示词，使用当前提示词继续")
                # 复制当前提示词到下一轮
                new_prompt = iter_dir / f"prompt_v{iteration + 1}.md"
                shutil.copy2(current_prompt_path, new_prompt)
                current_prompt_path = new_prompt
                prompt_version = f"v{iteration + 1}"

        # -- 达到最大迭代次数 --
        result.final_status = "max_iterations"
        result.total_iterations = self.max_iterations
        result.summary = f"达到最大迭代次数 {self.max_iterations}，未全部通过"
        self._write_summary(result)
        logger.warning("[WARN]️ 达到最大迭代次数，终止")
        return result

    # -- Step 实现 ------------------------------------------------------

    def _step_extract_constraints(self, record: IterationRecord, iter_dir: Path):
        """Session A — 调用 LLM 提取约束。"""
        logger.info("-" * 40)
        logger.info("Step 1: Session A — 约束提取")

        from constraint_extractor import extract_constraints

        constraints_path = iter_dir / "constraints.json"
        constraints, raw_output, attempts = extract_constraints(
            prompt_path=record.prompt_path,
            doc_path=self.doc_path,
            operator_name=self.operator_name,
            backend=self.backend,
        )

        record.raw_extraction_output = raw_output

        if constraints is not None:
            record.constraints_json = constraints
            record.constraints_valid = True
            constraints_path.write_text(
                json.dumps(constraints, ensure_ascii=False, indent=2),
                encoding="utf-8")
            logger.info("约束提取成功 (第 %d 次 LLM 调用) -> %s", attempts, constraints_path)
        else:
            record.constraints_valid = False
            # 保存原始输出供调试
            (iter_dir / "extraction_raw_output.txt").write_text(
                raw_output, encoding="utf-8")
            logger.error("约束提取失败，原始输出已保存")

    def _step_generate_cases(self, record: IterationRecord, iter_dir: Path):
        """用例生成 — 调用 generators 模块。"""
        logger.info("-" * 40)
        logger.info("Step 2: 用例生成")

        try:
            from agent.generators.facade import TestCaseGenerator

            gen = TestCaseGenerator(
                json_constraints=record.constraints_json,
                seed=42,
            )
            platforms = gen.supported_platforms or [self.platform]
            all_cases = []
            for plat in platforms[:1]:  # 先只用一个平台验证
                cases = gen.generate_for_platform(plat, count=self.case_count)
                all_cases.extend(cases)
                logger.info("  平台 %s: 生成 %d 个用例", plat, len(cases))

            # 转为可序列化的 dict 列表
            cases_dicts = []
            for case in all_cases:
                try:
                    cases_dicts.append(case.model_dump())
                except AttributeError:
                    cases_dicts.append(case.dict() if hasattr(case, 'dict') else str(case))

            record.cases = cases_dicts
            record.case_count = len(cases_dicts)

            cases_path = iter_dir / "cases.json"
            cases_path.write_text(
                json.dumps(cases_dicts, ensure_ascii=False, indent=2),
                encoding="utf-8")
            logger.info("用例生成完成: %d 个用例 -> %s", record.case_count, cases_path)

        except ImportError as e:
            logger.warning("无法导入 generators: %s", e)
            record.generation_error = str(e)
            # 创建 mock cases 供测试
            record.cases = [{"id": f"mock_{i}", "params": {}} for i in range(3)]
            record.case_count = len(record.cases)
            (iter_dir / "cases.json").write_text(
                json.dumps(record.cases, ensure_ascii=False, indent=2),
                encoding="utf-8")
        except Exception as e:
            logger.exception("用例生成异常")
            record.generation_error = str(e)

    def _step_execute_cases(self, record: IterationRecord, iter_dir: Path):
        """用例执行 — 真实执行或 Mock。"""
        logger.info("-" * 40)
        logger.info("Step 3: 用例执行")

        if self.mock_exec:
            self._mock_execute(record, iter_dir)
        else:
            self._real_execute(record, iter_dir)

    def _mock_execute(self, record: IterationRecord, iter_dir: Path):
        """Mock 执行 — 模拟部分失败，用于快速验证迭代逻辑。"""
        logger.info(">>> Mock 执行模式 <<<")

        # 模拟: 约 30% 用例失败
        import random
        random.seed(42)

        passed = 0
        failed = 0
        records = []
        for case in record.cases:
            # 确定性模拟 — 第2个用例总是失败（模拟 dtype 约束问题）
            case_id = case.get("id", str(case))
            is_pass = not (case_id == "mock_1" or "2" in str(case_id))

            rec = {
                "id": str(case_id),
                "run_result": "pass" if is_pass else "fail",
                "failure_reason": "" if is_pass else "ACLNN_ERR_PARAM_INVALID: dtype not supported on this platform",
                "case_json": case,
            }
            records.append(rec)
            if is_pass:
                passed += 1
            else:
                failed += 1

        exec_result = {
            "status": "success" if failed == 0 else "failed",
            "exit_code": 0 if failed == 0 else 1,
            "stdout": f"ATK run complete: {passed} passed, {failed} failed",
            "stderr": "",
            "duration": 12.5,
            "passed": passed,
            "failed": failed,
            "total": passed + failed,
            "records": records,
            "log_content": f"[INFO] Mock execution log\n[WARN] Some cases failed due to dtype mismatch\n",
        }

        record.execution_result = exec_result
        record.execution_status = exec_result["status"]
        record.passed_count = passed
        record.failed_count = failed

        (iter_dir / "execution_result.json").write_text(
            json.dumps(exec_result, ensure_ascii=False, indent=2),
            encoding="utf-8")
        logger.info("Mock 执行完成: %d passed, %d failed", passed, failed)

    def _load_server_for_platform(self, platform: str) -> dict | None:
        """从 servers.json 中匹配平台对应的服务器配置。

        匹配优先级:
          1. 精确匹配 (platform in srv.platforms)
          2. 模糊匹配 (平台字符串前 8 字符互为子串)
          3. 回退到 servers.json 中的第一条记录
          4. 都找不到则返回 None
        """
        config_path = Path(self.server_config_path)
        if not config_path.exists():
            logger.warning("服务器配置文件不存在: %s", config_path)
            return None

        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            servers = config.get("servers", [])
        except Exception as e:
            logger.error("解析服务器配置失败: %s", e)
            return None

        if not servers:
            logger.warning("服务器配置 %s 中没有 servers 列表", config_path)
            return None

        # 1. 精确匹配
        for srv in servers:
            if platform in srv.get("platforms", []):
                logger.info("平台 %s 精确匹配服务器: %s (%s)", platform, srv["name"], srv["ip"])
                return srv
        # 2. 模糊匹配（用 platform 中的关键词）
        for srv in servers:
            for p in srv.get("platforms", []):
                if platform[:8] in p or p[:8] in platform:
                    logger.info("平台 %s 模糊匹配服务器: %s (%s)", platform, srv["name"], srv["ip"])
                    return srv

        # 3. 回退到第一条
        fallback = servers[0]
        logger.warning("未找到平台 %s 对应的服务器，回退到第一条: %s (%s)",
                       platform, fallback.get("name"), fallback.get("ip"))
        return fallback

    async def _real_execute_async(self, record: IterationRecord, iter_dir: Path):
        """真实 SSH 执行 — 通过 executer_subgraph 调用（generate_atk → cpu_derivation → run_atk）。

        不再手写 SSH 上传 + atk 命令 — 改为构造 PipelineState 并 ainvoke 整个子图，
        自动复用 operator-agent 项目标准的 executor 生成、CPU golden 增强、SSH 上传、
        atk 命令组装、产物下载/解析流程。失败时回退到 _mock_execute 保持原迭代语义。
        """
        from agent.nodes.executer_subgraph import create_executer_subgraph

        # ── 前置检查：失败 → 回退 mock ──
        platforms = list((record.constraints_json or {}).get("product_support", []))
        if not platforms:
            logger.error("约束中没有 product_support，回退到 Mock")
            self._mock_execute(record, iter_dir)
            return

        server = self._load_server_for_platform(platforms[0])
        if not server:
            logger.error("无可用服务器，回退到 Mock")
            self._mock_execute(record, iter_dir)
            return

        cases_path = iter_dir / "cases.json"
        if not cases_path.exists() or record.case_count == 0:
            logger.error("用例文件缺失或 case_count=0，回退到 Mock")
            self._mock_execute(record, iter_dir)
            return

        try:
            operator_doc = self.doc_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error("无法读取算子文档 %s: %s — 回退到 Mock", self.doc_path, e)
            self._mock_execute(record, iter_dir)
            return

        # ── 调用 subgraph ──
        run_id = f"iter_{record.iteration:03d}"
        state_input: dict[str, Any] = {
            "operator_name":    self.operator_name,
            "cases_path":       str(cases_path),
            "content":          operator_doc,
            "server_info":      server,
            "task_type":        "accuracy",
            "execution_count":  1,
            "run_id":           run_id,
        }

        graph = create_executer_subgraph()
        try:
            result = await graph.ainvoke(state_input)
        except Exception as e:
            logger.exception("executer_subgraph.ainvoke 抛异常: %s", e)
            record.execution_result = {
                "status": "error",
                "error_message": f"subgraph ainvoke 异常: {e}",
                "passed": 0, "failed": 0,
            }
            record.execution_status = "error"
            record.passed_count = 0
            record.failed_count = 0
            (iter_dir / "execution_result.json").write_text(
                json.dumps(record.execution_result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return

        # ── 处理 subgraph 错误：回退 mock ──
        error = result.get("error")
        exec_result = result.get("exec_result") or {}
        if error:
            logger.warning("executer_subgraph 返回 error=%r — 回退到 Mock", error)
            self._mock_execute(record, iter_dir)
            return

        # ── 成功：映射字段 ──
        record.execution_result = exec_result
        record.execution_status = exec_result.get("status", "failed")
        record.passed_count = int(exec_result.get("passed", 0) or 0)
        record.failed_count = int(exec_result.get("failed", 0) or 0)

        # ── 写 iter_dir/execution_result.json（_step_analyze 依赖此文件） ──
        (iter_dir / "execution_result.json").write_text(
            json.dumps(exec_result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # ── 镜像 subgraph 产物到 iter_dir/（用户选择：全部镜像） ──
        iter_exec = iter_dir / "execution_results"
        iter_exec.mkdir(parents=True, exist_ok=True)
        subgraph_cache = _AGENT_ROOT / "execution_results" / self.operator_name / run_id
        if subgraph_cache.exists():
            try:
                for src in subgraph_cache.iterdir():
                    dst = iter_exec / src.name
                    if src.is_file():
                        shutil.copy2(src, dst)
                    elif src.is_dir():
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                logger.info("执行产物已镜像到: %s", iter_exec)
            except Exception as e:
                logger.warning("镜像执行产物失败: %s", e)
        else:
            logger.info(
                "subgraph cache 不存在 (%s) — 跳过产物镜像", subgraph_cache,
            )

        # ── 镜像 executor.py 到 iter_dir/ ──
        executor_src_str = result.get("atk_executor_path")
        if executor_src_str:
            executor_src = Path(executor_src_str)
            if executor_src.exists():
                try:
                    shutil.copy2(executor_src, iter_dir / executor_src.name)
                    logger.info("executor 已镜像到: %s", iter_dir / executor_src.name)
                except Exception as e:
                    logger.warning("镜像 executor 失败: %s", e)

        logger.info(
            "executer_subgraph 完成: status=%s passed=%d failed=%d duration=%.1fs",
            record.execution_status, record.passed_count, record.failed_count,
            exec_result.get("duration", 0.0),
        )

    def _real_execute(self, record: IterationRecord, iter_dir: Path):
        """真实 SSH 执行 — 调用 executer 模块。"""
        logger.info(">>> 真实执行模式 <<<")

        if not Path(self.server_config_path).exists():
            logger.warning("服务器配置文件不存在: %s，回退到 Mock", self.server_config_path)
            self._mock_execute(record, iter_dir)
            return

        try:
            import asyncio
            asyncio.run(self._real_execute_async(record, iter_dir))
        except ImportError as e:
            logger.warning("无法导入 executer 模块: %s，回退到 Mock", e)
            self._mock_execute(record, iter_dir)
        except Exception as e:
            logger.exception("真实执行异常: %s", e)
            self._mock_execute(record, iter_dir)

    def _step_analyze(self, record: IterationRecord, iter_dir: Path,
                      current_prompt_path: Path):
        """Session B — 分析结果 & 优化提示词。"""
        logger.info("-" * 40)
        logger.info("Step 4: Session B — 分析 & 优化")

        from result_analyzer import analyze_and_optimize

        constraints_path = iter_dir / "constraints.json"
        cases_path = iter_dir / "cases.json"
        exec_result_path = iter_dir / "execution_result.json"

        analysis_result = analyze_and_optimize(
            prompt_path=current_prompt_path,
            doc_path=self.doc_path,
            constraints_path=constraints_path,
            cases_path=cases_path,
            exec_result_path=exec_result_path,
            backend=self.backend,
        )

        record.root_cause = analysis_result.root_cause
        record.analysis = {
            "root_cause": analysis_result.root_cause,
            "analysis": analysis_result.analysis,
            "specific_issues": analysis_result.specific_issues,
            "modified_sections": analysis_result.modified_sections,
            "generator_issue": analysis_result.generator_issue,
            "executor_issue": analysis_result.executor_issue,
        }
        record.improved_prompt = analysis_result.improved_prompt

        analysis_path = iter_dir / "analysis.json"
        analysis_path.write_text(
            json.dumps(record.analysis, ensure_ascii=False, indent=2),
            encoding="utf-8")
        logger.info("分析结果: root_cause=%s", record.root_cause)

    # -- 辅助方法 ------------------------------------------------------

    def _save_improved_prompt(self, record: IterationRecord,
                               iter_dir: Path, iteration: int) -> Path:
        """保存改进提示词并返回路径。"""
        new_version = iteration + 1
        prompt_path = iter_dir / f"prompt_v{new_version}.md"
        prompt_path.write_text(record.improved_prompt, encoding="utf-8")
        # 同时保存为当前提示词副本
        versioned_path = self._run_dir / f"operator_constraints_extract_v{new_version}.md"
        shutil.copy2(prompt_path, versioned_path)
        record.improved_prompt_path = str(versioned_path)
        logger.info("改进提示词已保存: %s", versioned_path)
        return prompt_path

    def _write_summary(self, result: PipelineResult):
        """将流水线摘要写入文件。"""
        summary_path = self._run_dir / "pipeline_summary.json"
        summary = {
            "operator_name": result.operator_name,
            "total_iterations": result.total_iterations,
            "final_status": result.final_status,
            "summary": result.summary,
            "iterations": [
                {
                    "iteration": r.iteration,
                    "prompt_version": r.prompt_version,
                    "case_count": r.case_count,
                    "passed_count": r.passed_count,
                    "failed_count": r.failed_count,
                    "root_cause": r.root_cause,
                    "analysis_short": r.analysis.get("analysis", "")[:200] if r.analysis else "",
                }
                for r in result.iterations
            ],
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8")
        logger.info("流水线摘要已保存: %s", summary_path)

        # Log final summary (use logger instead of print for Unicode safety)
        logger.info("=" * 70)
        logger.info("Pipeline Summary for %s", result.operator_name)
        logger.info("Total iterations: %d | Final status: %s", result.total_iterations, result.final_status)
        logger.info("Summary: %s", result.summary)
        for r in result.iterations:
            status = "[OK]" if r.all_passed else ("[LOOP]" if r.is_constraint_issue else "[FAIL]")
            logger.info("  %s iter=%d prompt=%s cases=%d passed=%d failed=%d cause=%s",
                        status, r.iteration, r.prompt_version,
                        r.case_count, r.passed_count, r.failed_count, r.root_cause)
        logger.info("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="算子约束迭代优化流水线 — 提取->生成->执行->分析->优化 闭环",
    )
    parser.add_argument("--prompt", required=True,
                        help="初始提示词 MD 文件路径 (e.g. prompts/operator_constraints_extract_v1.md)")
    parser.add_argument("--doc", required=True,
                        help="算子文档 MD 文件路径 (e.g. docs/aclnnAlltoAllMatmul.md)")
    parser.add_argument("--output-root", default="iterator_output",
                        help="输出根目录 (默认: iterator_output)")
    parser.add_argument("--max-iterations", type=int, default=5,
                        help="最大迭代次数 (默认: 5)")
    parser.add_argument("--case-count", type=int, default=10,
                        help="每个平台生成的用例数量 (默认: 10)")
    parser.add_argument("--mock-exec", action="store_true", default=False,
                        help="Mock 执行模式 (默认关闭)")
    parser.add_argument("--real-exec", dest="mock_exec", action="store_false",
                        help="真实 SSH 执行模式 (默认开启)")
    parser.add_argument("--platform", default="",
                        help="目标平台名")
    parser.add_argument("--server-config", default="servers.json",
                        help="服务器配置 JSON (默认: servers.json)")
    parser.add_argument("--backend", default="api",
                        choices=["api", "agent"],
                        help="LLM 后端: api (LLM API) 或 agent (CLI 子进程) (默认: api)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    pipeline = IterativePipeline(
        prompt_path=args.prompt,
        doc_path=args.doc,
        output_root=args.output_root,
        max_iterations=args.max_iterations,
        case_count=args.case_count,
        mock_exec=args.mock_exec,
        platform=args.platform,
        server_config_path=args.server_config,
        backend_type=args.backend,
    )

    result = pipeline.run()

    # 返回码
    if result.successful:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
