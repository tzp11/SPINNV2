"""Export thesis-facing summaries and paper asset manifests.

This script intentionally does not run experiments. It collects the current
Windows-stage evidence into stable Markdown/JSON files that can be cited by the
thesis draft and extended as new experiments finish.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _copy_if_exists(src: Path, dst: Path, manifest: list[dict[str, Any]]) -> None:
    if not src.exists():
        manifest.append(
            {
                "asset": str(dst.as_posix()),
                "source": str(src.as_posix()),
                "status": "missing_source",
            }
        )
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    manifest.append(
        {
            "asset": str(dst.as_posix()),
            "source": str(src.as_posix()),
            "status": "copied",
        }
    )


def _related_work_markdown() -> str:
    return """# 相关工作与本文差异化对比

本文不能把“中间结果注入”“敏感层选择”“DMR/TMR/重复执行”单独作为创新点。当前创新边界应收紧到任务后果感知风险评估、资源预算保护配置优化、以及 ProtectionPlan 到 SPK/Runtime 的可执行落地闭环。

| 工作 | 故障对象 | 任务指标 | 保护机制 | 优化变量 | 资源约束 | 部署形态 | 与本文关系 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 师兄论文目录对应工作 | 推理框架、转译、FPGA/Zynq、容错相关章节 | 以目录和已有资料为准 | 软硬件结合容错 | 框架/硬件/容错设计 | 硬件资源 | SMTIF/FPGA/Zynq | 强相关，需要避免重复“框架+硬件容错”路线 |
| Research on Spaceborne Neural Network Accelerator and Its Fault Tolerance Design, Remote Sensing 2025 | 中间层输出、敏感层 | 分类/识别任务影响 | 时间冗余、投票类保护 | 敏感层选择 | 硬件/加速器资源 | 星载神经网络加速器 | 与“中间结果注入、敏感层保护”重叠，本文不能直接照写 |
| TensorFI 类故障注入工具 | 神经网络张量、层输出 | SDC、准确率或输出扰动 | 故障注入框架 | 注入位置和故障模式 | 通常不以平台预算为核心 | Python/框架级工具 | 注入工具不是本文创新，只是实验支撑 |
| ARM/嵌入式 DNN 软件容错研究 | 算子、层输出、执行数据 | 准确率、SDC、可靠性指标 | 软件冗余、选择性保护 | 保护位置/策略 | 时延和能耗 | 嵌入式 CPU | 证明“选择性软件保护”本身不是新概念 |
| 混合精度可靠性研究 | 权重、激活、量化位宽 | 准确率与软错误敏感性 | 混合精度、冗余或量化保护 | 位宽/保护组合 | 面积、能耗、时延 | 嵌入式/硬件平台 | 当前不走该路线，避免与近邻工作冲突 |
| 本文方法 | 运行时对象、算子输出、检测控制相关节点 | 分类错误、检测漏检/误检/定位退化、执行错误 | range guard、DMR、可选任务输出 guard | 保护对象与保护模式 | 时延预算、峰值内存预算 | SPINNV2 SPK/Runtime/Generated C | 重点是任务后果感知风险、预算优化、自动落地闭环 |

## 论文写作约束

- 不声称首次提出中间激活故障注入。
- 不声称首次提出敏感层保护。
- 不声称 DMR/TMR 或重复执行本身是新方法。
- 将 SPINNV2 的转译、静态内存和 Runtime 作为方法落地平台，不作为本文主创新。
"""


def _contribution_claims_markdown() -> str:
    return """# 最终创新点表述

## 创新点一：任务后果感知的运行时对象脆弱性评估方法

将运行时节点故障与遥感分类错误、目标漏检、误检、定位劣化和控制路径执行错误关联起来，形成可排序、可统计置信区间的任务级风险度量。

支撑证据：

- ResNet50 + EuroSAT 的运行时故障注入、风险排序与置信区间。
- YOLOv10n + DIOR 的检测任务故障模式统计，包括 task output failure 和 controlled execution error。

## 创新点二：风险损失受限的低内存保护配置优化方法

基于节点风险、保护收益、额外时延和额外内存开销，先求给定预算下最大风险降低方案，再允许可解释的风险降低损失上限，在该约束内最小化峰值额外保护内存。该方法避免为了极小的风险收益差异选择大缓冲 DMR，面向星载受限平台更强调低内存可部署性。

支撑证据：

- ResNet50 已有 multi-mode ILP 结果，可证明多机制预算优化优于简单单机制策略。
- ResNet50 bounded-loss memory ILP 结果显示，在 5% 风险降低损失容忍度下，部分预算点可以降低 50% 到 100% 峰值额外保护内存。
- YOLOv10n 当前主要证明 DMR 和 task output guard 保护有效，算法优势主要由 ResNet50 多机制和低内存优化实验支撑。

## 创新点三：面向遥感检测任务的任务效用一致性保护机制与可执行落地

针对 YOLO 检测输出定义目标数量、置信度总量、类别分布和框面积分布等任务效用一致性检查，并将保护配置编码为 ProtectionPlan，集成到 SPK 与 C Runtime，使保护方案可以被自动加载、执行、测量和复现实验。

支撑证据：

- YOLOv10n 10000 个 runtime fault event 中，Task-Utility Consistency Guard 将关键失效降低率从普通 task output guard 的 50.52% 提升到 97.21%，clean false positive 为 0。
- SPK ProtectionPlan section。
- C Runtime fault event、range guard、DMR 执行路径。
- 运行时同事件 protected/unprotected 对比结果。

## 不能作为创新单独表述的内容

- 中间结果故障注入。
- 敏感层或高风险层筛选。
- DMR/TMR/重复执行。
- 通用推理框架、ONNX 转译、静态内存规划。
"""


def _chapter_mapping_markdown() -> str:
    return """# 实验结果到论文章节映射

| 章节 | 主要内容 | 支撑产物 |
| --- | --- | --- |
| 第 1 章 绪论 | 星载遥感推理约束、软错误问题、研究意义、本文贡献 | related_work_comparison.md、contribution_claims.md |
| 第 2 章 相关工作与系统边界 | 师兄工作、Remote Sensing 2025、故障注入、软件容错、混合精度可靠性 | related_work_comparison.md |
| 第 3 章 任务级脆弱性评估方法 | 故障模型、任务失效定义、风险评分、排序稳定性 | 风险排序、置信区间、稳定性结果 |
| 第 4 章 资源预算保护优化方法 | range guard、DMR、成本模型、ILP、ProtectionPlan | 保护计划、优化对比、消融 |
| 第 5 章 实验与分析 | ResNet50、YOLOv10n、开销、可视化、边界实验 | JSON/CSV/图片资产 |
| 第 6 章 总结与展望 | 已完成结论、D2000/RISC-V 未完成限制、后续工作 | gap report、边界声明 |
"""


def _status_markdown(root: Path) -> str:
    formal = _read_json(root / "artifacts/reports/paper_assets/formal_windows_completion_summary.json")
    resnet = _read_json(
        root
        / "artifacts/reports/resnet50_windows_avx2_release/feedback2_calibrated/resnet50_final_runtime_summary.json"
    )
    yolo_activation = _read_json(
        root
        / "artifacts/reports/yolov10n_dior_full_e30_b16/ilp_dmr_activation_prior_comparison/yolov10n_control_dmr_comparison.json"
    )
    yolo_stratified = _read_json(
        root
        / "artifacts/reports/yolov10n_dior_full_e30_b16/ilp_dmr_stratified_comparison/yolov10n_control_dmr_comparison.json"
    )

    lines = ["# Windows 阶段当前结果摘要", ""]
    if formal:
        lines += ["## 正式完成实验", ""]
        for row in formal["rows"]:
            lines.append(
                "- `{model}` `{experiment}`: `{unprotected_critical_failures} -> {protected_critical_failures}`, "
                "reduction `{ratio:.2%}`, events `{events}`.".format(
                    ratio=row["critical_failure_reduction_ratio"], **row
                )
            )
        lines.append("")
    if resnet:
        clean = resnet["clean_evaluation"]
        fault = resnet["fault_evaluation"]
        budget = resnet["budget_optimization"]
        lines += [
            "## ResNet50 + EuroSAT",
            "",
            f"- clean samples: `{clean['samples']}`",
            f"- baseline/protected accuracy: `{clean['baseline_accuracy']:.6f}` / `{clean['protected_accuracy']:.6f}`",
            f"- prediction agreement: `{clean['prediction_agreement']:.6f}`",
            f"- false alarm rate: `{clean['false_alarm_rate']:.6f}`",
            f"- latency overhead: `{clean['latency_overhead_ms']:.3f} ms` (`{clean['latency_overhead_ratio']:.3%}`)",
            f"- fault events: `{fault['events']}`",
            f"- critical failures: `{fault['unprotected_critical_failures']} -> {fault['protected_critical_failures']}`",
            f"- observed reduction ratio: `{fault['observed_reduction_ratio']:.3%}`",
            f"- selected protected nodes: `{budget['selected_count']}`",
            f"- used peak extra memory: `{budget['used_peak_extra_memory_bytes']}` bytes",
            "",
        ]
    else:
        lines += ["## ResNet50 + EuroSAT", "", "- 当前摘要文件缺失。", ""]

    if yolo_activation:
        baseline = yolo_activation["baseline"]
        protected = yolo_activation["budget_ilp_dmr"]
        lines += [
            "## YOLOv10n + DIOR activation-prior",
            "",
            f"- workload: {yolo_activation['workload_scope']}",
            f"- fault sampling: {yolo_activation['fault_sampling']}",
            f"- critical failures: `{baseline['critical_failures']} -> {protected['critical_failures']}`",
            f"- reduction ratio: `{yolo_activation['critical_failure_reduction_ratio']:.3%}`",
            f"- latency overhead: `{yolo_activation['latency_overhead_ms']:.3f} ms` (`{yolo_activation['latency_overhead_ratio']:.3%}`)",
            f"- extra memory: `{protected['extra_memory_bytes']}` bytes",
            "",
        ]

    if yolo_stratified:
        baseline = yolo_stratified["baseline"]
        protected = yolo_stratified["budget_ilp_dmr"]
        lines += [
            "## YOLOv10n + DIOR stratified",
            "",
            f"- fault sampling: {yolo_stratified['fault_sampling']}",
            f"- critical failures: `{baseline['critical_failures']} -> {protected['critical_failures']}`",
            f"- reduction ratio: `{yolo_stratified['critical_failure_reduction_ratio']:.3%}`",
            f"- latency overhead: `{yolo_stratified['latency_overhead_ms']:.3f} ms` (`{yolo_stratified['latency_overhead_ratio']:.3%}`)",
            "",
        ]

    lines += [
        "## 当前不足",
        "",
        "- ResNet50 仍需 10000+ fault events 和多 seed 排序稳定性。",
        "- YOLOv10n 当前主要证明 DMR 保护有效，ILP 相对简单策略的优势还需要补强。",
        "- SPK/权重边界实验、工程 P95/P99 指标和 Generated C smoke 仍需补齐。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="SPINNV2 repository root.")
    parser.add_argument(
        "--out-dir",
        default="artifacts/reports/paper_assets",
        help="Output directory relative to repo root unless absolute.",
    )
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    generated_files = {
        "ch2_related_work_comparison.md": _related_work_markdown(),
        "contribution_claims.md": _contribution_claims_markdown(),
        "current_windows_status_summary.md": _status_markdown(root),
        "thesis_chapter_mapping.md": _chapter_mapping_markdown(),
    }
    for name, content in generated_files.items():
        path = out_dir / name
        _write(path, content)
        manifest.append({"asset": str(path.as_posix()), "status": "generated"})

    _copy_if_exists(
        root
        / "artifacts/reports/resnet50_windows_avx2_release/feedback2_calibrated/resnet50_budget_pareto.png",
        out_dir / "ch4_budget_risk_curve_resnet50.png",
        manifest,
    )
    _copy_if_exists(
        root
        / "artifacts/reports/resnet50_windows_avx2_release/feedback2_calibrated/resnet50_runtime_fault_mitigation.png",
        out_dir / "ch5_resnet50_runtime_fault_mitigation.png",
        manifest,
    )
    _copy_if_exists(
        root
        / "artifacts/reports/resnet50_windows_avx2_release/feedback2_calibrated/resnet50_runtime_latency_overhead.png",
        out_dir / "ch5_resnet50_latency_overhead.png",
        manifest,
    )
    _copy_if_exists(
        root
        / "artifacts/reports/yolov10n_dior_full_e30_b16/figures/yolov10n_fault_recovery_case.png",
        out_dir / "ch5_yolov10n_fault_recovery_case.png",
        manifest,
    )
    for name in [
        "spk_integrity_boundary.json",
        "weight_fault_boundary_resnet50.json",
        "weight_fault_boundary_yolov10n.json",
    ]:
        _copy_if_exists(
            root / "artifacts/reports/boundary_experiments" / name,
            out_dir / name,
            manifest,
        )
    for name in ["engineering_metrics.json", "engineering_metrics.csv", "generated_c_gap_report.md"]:
        source = out_dir / name
        if source.exists():
            manifest.append({"asset": str(source.as_posix()), "status": "generated_or_existing"})
    source = out_dir / "generated_c_protection_smoke.json"
    if source.exists():
        manifest.append({"asset": str(source.as_posix()), "status": "generated_or_existing"})
    for name in [
        "baseline_comparison.csv",
        "baseline_comparison.json",
        "risk_score_ablation.csv",
        "risk_score_ablation.json",
        "resnet50_bounded_loss_memory_comparison.csv",
        "resnet50_bounded_loss_memory_comparison.json",
        "resnet50_bounded_loss_memory_saving.png",
        "risk_stability_resnet50.csv",
        "risk_stability_resnet50.json",
        "risk_stability_yolov10n.csv",
        "risk_stability_yolov10n.json",
        "risk_stability_resnet50_formal.csv",
        "risk_stability_resnet50_formal.json",
        "risk_stability_yolov10n_formal.csv",
        "risk_stability_yolov10n_formal.json",
        "yolov10n_grouped_protection_comparison.json",
        "formal_windows_completion_summary.csv",
        "formal_windows_completion_summary.json",
        "formal_windows_completion_summary.md",
        "ch3_risk_rank_resnet50.png",
        "ch3_risk_rank_yolov10n.png",
        "ch3_risk_stability_resnet50.png",
        "ch3_risk_stability_yolov10n.png",
        "ch5_resnet50_method_comparison.png",
        "ch5_yolov10n_method_comparison.png",
        "ch5_yolov10n_grouped_protection.png",
        "ch5_formal_windows_completion_summary.png",
    ]:
        source = out_dir / name
        if source.exists():
            manifest.append({"asset": str(source.as_posix()), "status": "generated_or_existing"})
    screening_dir = out_dir / "mechanism_improvement_screening"
    if screening_dir.exists():
        for source in sorted(screening_dir.glob("*")):
            if source.is_file():
                manifest.append({"asset": str(source.as_posix()), "status": "generated_or_existing"})

    manifest_doc = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "research/reliability/export_paper_assets.py",
        "repo_root": str(root),
        "assets": manifest,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest_doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"out_dir": str(out_dir), "asset_count": len(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
