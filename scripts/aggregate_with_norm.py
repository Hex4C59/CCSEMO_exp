#!/usr/bin/env python3
"""
统计with_norm实验的K折交叉验证结果的均值和标准差
从各fold的best_results.txt中提取指标并计算统计量
"""
import re
from pathlib import Path
import numpy as np
from datetime import datetime


def parse_best_results(file_path):
    """解析best_results.txt文件，提取各项指标"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    results = {}
    
    # 提取CCC指标
    ccc_section = re.search(r'Best Test Results \(CCC.*?\):(.*?)(?=Best Test Results|\Z)', 
                           content, re.DOTALL)
    if ccc_section:
        v_match = re.search(r'V \(Valence\):\s+([\d.]+)', ccc_section.group(1))
        a_match = re.search(r'A \(Arousal\):\s+([\d.]+)', ccc_section.group(1))
        avg_match = re.search(r'Average:\s+([\d.]+)', ccc_section.group(1))
        results['CCC_V'] = float(v_match.group(1)) if v_match else None
        results['CCC_A'] = float(a_match.group(1)) if a_match else None
        results['CCC_Avg'] = float(avg_match.group(1)) if avg_match else None
    
    # 提取MSE指标
    mse_section = re.search(r'Best Test Results \(MSE.*?\):(.*?)(?=Best Test Results|\Z)', 
                           content, re.DOTALL)
    if mse_section:
        v_match = re.search(r'V \(Valence\):\s+([\d.]+)', mse_section.group(1))
        a_match = re.search(r'A \(Arousal\):\s+([\d.]+)', mse_section.group(1))
        avg_match = re.search(r'Average:\s+([\d.]+)', mse_section.group(1))
        results['MSE_V'] = float(v_match.group(1)) if v_match else None
        results['MSE_A'] = float(a_match.group(1)) if a_match else None
        results['MSE_Avg'] = float(avg_match.group(1)) if avg_match else None
    
    # 提取R2指标
    r2_section = re.search(r'Best Test Results \(R2.*?\):(.*?)(?=Best Test Results|\Z)', 
                          content, re.DOTALL)
    if r2_section:
        v_match = re.search(r'V \(Valence\):\s+([-\d.]+)', r2_section.group(1))
        a_match = re.search(r'A \(Arousal\):\s+([-\d.]+)', r2_section.group(1))
        avg_match = re.search(r'Average:\s+([-\d.]+)', r2_section.group(1))
        results['R2_V'] = float(v_match.group(1)) if v_match else None
        results['R2_A'] = float(a_match.group(1)) if a_match else None
        results['R2_Avg'] = float(avg_match.group(1)) if avg_match else None
    
    return results


def aggregate_folds(base_dir, pattern='*with_norm*fold*'):
    """聚合所有fold的结果并计算统计量"""
    base_path = Path(base_dir)
    
    # 查找所有匹配pattern的fold的best_results.txt
    fold_results = []
    for fold_dir in sorted(base_path.glob(pattern)):
        best_results_file = fold_dir / 'exp_1' / 'best_results.txt'
        if best_results_file.exists():
            fold_num = re.search(r'fold(\d+)', fold_dir.name)
            fold_num = int(fold_num.group(1)) if fold_num else None
            results = parse_best_results(best_results_file)
            fold_results.append((fold_num, results))
    
    # 按fold编号排序
    fold_results.sort(key=lambda x: x[0] if x[0] else 0)
    
    # 计算每个指标的均值和标准差
    metrics = ['CCC_V', 'CCC_A', 'CCC_Avg', 'MSE_V', 'MSE_A', 'MSE_Avg', 
               'R2_V', 'R2_A', 'R2_Avg']
    
    aggregated = {}
    for metric in metrics:
        values = [r[metric] for _, r in fold_results if r.get(metric) is not None]
        if values:
            aggregated[metric] = {
                'mean': np.mean(values),
                'std': np.std(values, ddof=1),  # 使用样本标准差
                'values': values
            }
    
    return aggregated, fold_results


def format_results(model_name, experiment_type, aggregated, fold_results):
    """格式化输出结果"""
    lines = []
    lines.append("=" * 80)
    lines.append(f"Model: {model_name}")
    lines.append(f"Experiment Type: {experiment_type}")
    lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Number of Folds: {len(fold_results)}")
    lines.append("=" * 80)
    lines.append("")
    
    # 输出各fold的详细结果
    lines.append("Individual Fold Results:")
    lines.append("-" * 80)
    for fold_num, results in fold_results:
        lines.append(f"Fold {fold_num}:")
        lines.append(f"  CCC - V: {results['CCC_V']:.4f}, A: {results['CCC_A']:.4f}, "
                    f"Avg: {results['CCC_Avg']:.4f}")
        lines.append(f"  MSE - V: {results['MSE_V']:.4f}, A: {results['MSE_A']:.4f}, "
                    f"Avg: {results['MSE_Avg']:.4f}")
        lines.append(f"  R2  - V: {results['R2_V']:.4f}, A: {results['R2_A']:.4f}, "
                    f"Avg: {results['R2_Avg']:.4f}")
    lines.append("")
    
    # 输出统计结果
    lines.append("Aggregated Results (Mean ± Std):")
    lines.append("-" * 80)
    
    # CCC结果
    lines.append("CCC (Concordance Correlation Coefficient):")
    if 'CCC_V' in aggregated:
        lines.append(f"  Valence: {aggregated['CCC_V']['mean']:.4f} ± "
                    f"{aggregated['CCC_V']['std']:.4f}")
    if 'CCC_A' in aggregated:
        lines.append(f"  Arousal: {aggregated['CCC_A']['mean']:.4f} ± "
                    f"{aggregated['CCC_A']['std']:.4f}")
    if 'CCC_Avg' in aggregated:
        lines.append(f"  Average: {aggregated['CCC_Avg']['mean']:.4f} ± "
                    f"{aggregated['CCC_Avg']['std']:.4f}")
    lines.append("")
    
    # MSE结果
    lines.append("MSE (Mean Squared Error):")
    if 'MSE_V' in aggregated:
        lines.append(f"  Valence: {aggregated['MSE_V']['mean']:.4f} ± "
                    f"{aggregated['MSE_V']['std']:.4f}")
    if 'MSE_A' in aggregated:
        lines.append(f"  Arousal: {aggregated['MSE_A']['mean']:.4f} ± "
                    f"{aggregated['MSE_A']['std']:.4f}")
    if 'MSE_Avg' in aggregated:
        lines.append(f"  Average: {aggregated['MSE_Avg']['mean']:.4f} ± "
                    f"{aggregated['MSE_Avg']['std']:.4f}")
    lines.append("")
    
    # R2结果
    lines.append("R2 (R-squared):")
    if 'R2_V' in aggregated:
        lines.append(f"  Valence: {aggregated['R2_V']['mean']:.4f} ± "
                    f"{aggregated['R2_V']['std']:.4f}")
    if 'R2_A' in aggregated:
        lines.append(f"  Arousal: {aggregated['R2_A']['mean']:.4f} ± "
                    f"{aggregated['R2_A']['std']:.4f}")
    if 'R2_Avg' in aggregated:
        lines.append(f"  Average: {aggregated['R2_Avg']['mean']:.4f} ± "
                    f"{aggregated['R2_Avg']['std']:.4f}")
    lines.append("")
    lines.append("")
    
    return '\n'.join(lines)


def main():
    # 配置路径
    base_dir = Path("runs/trash/5fold/CCSEMO/hubert_large")
    output_file = Path("runs/trash/5fold/CCSEMO/aggregated_results.txt")
    
    # 从目录名推断模型名
    model_name = base_dir.name
    experiment_type = "linear_with_norm"
    
    print(f"Processing results from: {base_dir}")
    print(f"Experiment type: {experiment_type}")
    print(f"Output will be appended to: {output_file}")
    
    # 聚合结果
    aggregated, fold_results = aggregate_folds(base_dir, pattern='*with_norm*fold*')
    
    if not fold_results:
        print("No matching fold results found!")
        return
    
    # 格式化输出
    formatted = format_results(model_name, experiment_type, aggregated, fold_results)
    
    # 追加到文件
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(formatted)
    
    print(f"\nResults successfully appended to {output_file}")
    print("\n" + formatted)


if __name__ == "__main__":
    main()
