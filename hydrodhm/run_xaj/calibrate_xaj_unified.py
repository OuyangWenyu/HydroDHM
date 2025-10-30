"""
Author: zhuanglaihong
Date: 2025-10-29
LastEditTime: 2025-10-29
LastEditors: zhuanglaihong
Description: XAJ model calibration script using the latest unified architecture
FilePath: \HydroDHM\hydrodhm\run_xaj\calibrate_xaj_unified.py
Copyright (c) 2023-2025 Wenyu Ouyang. All rights reserved.
"""

import argparse
import sys
import os
from pathlib import Path

# Add hydromodel to path
try:
    from hydromodel.trainers.unified_calibrate import calibrate
    from hydromodel.configs.config_manager import (
        setup_configuration_from_args,
        validate_and_show_config,
        save_config_to_file,
    )
except ImportError:
    print("Error: hydromodel package not found. Please install it first.")
    print("You can install it with: uv pip install hydromodel")
    sys.exit(1)


def load_simplified_config(
    config_path: str = None, simple_config: dict = None
) -> dict:
    """加载简化的配置文件并转换为统一格式

    Args:
        config_path: YAML配置文件路径
        simple_config: 直接提供的配置字典

    Returns:
        统一格式的配置字典
    """
    import yaml

    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            simple_config = yaml.safe_load(f)
    elif simple_config is None:
        raise ValueError("必须提供config_path或simple_config参数")

    # 验证简化配置的完整性
    required_sections = ["data", "model", "training", "evaluation"]
    for section in required_sections:
        if section not in simple_config:
            raise ValueError(f"配置缺少必需部分: {section}")

    data_cfg = simple_config["data"]
    model_cfg = simple_config["model"]
    training_cfg = simple_config["training"]
    eval_cfg = simple_config["evaluation"]

    # 转换为统一配置格式
    unified_config = {
        "data_cfgs": {
            "data_source_type": data_cfg.get("dataset", "selfmadehydrodataset"),
            "data_source_path": data_cfg["path"],
            "dataset_name": data_cfg.get("dataset", "selfmadehydrodataset"),
            "basin_ids": data_cfg["basin_ids"],
            "variables": data_cfg.get(
                "variables", ["prcp", "PET", "streamflow"]
            ),
            "train_period": data_cfg["train_period"],
            "test_period": data_cfg["test_period"],
            "warmup_length": data_cfg.get("warmup_length", 365),
        },
        "model_cfgs": {
            "model_name": model_cfg["name"],
            **model_cfg.get("params", {}),
        },
        "training_cfgs": {
            "algorithm": training_cfg["algorithm"],
            "loss_func": training_cfg["loss"],
            "output_dir": data_cfg.get("output_dir", "results"),
            "experiment_name": data_cfg.get(
                "experiment_name",
                f"{model_cfg['name']}_{training_cfg['algorithm']}"
            ),
            # 根据算法添加对应参数
            **training_cfg.get(training_cfg["algorithm"], {}),
        },
        "evaluation_cfgs": {
            "metrics": eval_cfg.get("metrics", ["NSE", "KGE", "RMSE"]),
        },
    }

    # 添加验证期（如果有）
    if "valid_period" in data_cfg:
        unified_config["data_cfgs"]["valid_period"] = data_cfg["valid_period"]

    # 添加交叉验证配置（如果有）
    if "cv_fold" in data_cfg and data_cfg["cv_fold"] > 1:
        unified_config["data_cfgs"]["cv_fold"] = data_cfg["cv_fold"]

    return unified_config


def parse_arguments():
    """解析命令行参数 - 简化版，支持配置文件"""
    parser = argparse.ArgumentParser(
        description="XAJ模型率定脚本 - 使用统一架构",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
配置文件格式（YAML）:
  data:     # 数据配置
    dataset: "selfmadehydrodataset"  # 数据集类型
    path: "C:\\\\Users\\\\wenyu\\\\OneDrive\\\\data\\\\FD_sources"  # 数据路径
    basin_ids: ["changdian_61561"]   # 流域ID列表
    warmup_length: 365               # 预热期（天）
    train_period: ["2014-10-01", "2018-09-30"]  # 训练期
    test_period: ["2017-10-01", "2021-09-30"]   # 测试期
    output_dir: "results"            # 结果目录
    experiment_name: "exp_xaj"       # 实验名称
    cv_fold: 1                       # 交叉验证折数（可选）

  model:    # 模型配置
    name: "xaj_mz"                   # 模型类型
    params:                          # 模型参数
      source_type: "sources"
      source_book: "HF"
      kernel_size: 15
      time_interval_hours: 24

  training: # 训练配置
    algorithm: "SCE_UA"              # 算法类型（SCE_UA/GA/scipy）
    loss: "RMSE"                     # 损失函数
    SCE_UA:                          # SCE_UA算法参数
      random_seed: 1234
      rep: 100000                    # 最大迭代次数
      ngs: 100                       # 复合体数量
      kstop: 50                      # 停止准则
      peps: 0.1                      # 收敛阈值
      pcento: 0.1                    # 收敛百分比
    # GA:                            # GA算法参数（示例）
    #   random_seed: 1234
    #   run_counts: 2
    #   pop_num: 50
    #   cross_prob: 0.5
    #   mut_prob: 0.5

  evaluation: # 评估配置
    metrics: ["NSE", "KGE", "RMSE"]  # 评估指标

使用示例:
  # 使用配置文件（推荐）
  python calibrate_xaj_unified.py --config config.yaml

  # 验证配置文件
  python calibrate_xaj_unified.py --config config.yaml --dry-run

  # 覆盖输出目录
  python calibrate_xaj_unified.py --config config.yaml --output-dir new_results
        """,
    )

    # 核心参数
    parser.add_argument(
        "--config",
        type=str,
        help="简化配置文件路径（YAML格式）",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只验证配置，不执行率定",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="覆盖配置中的输出目录",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        help="覆盖配置中的实验名称",
    )

    parser.add_argument(
        "--save-config",
        action="store_true",
        default=True,
        help="运行后保存配置文件（默认启用）",
    )

    return parser.parse_args()


def main():
    """主执行函数"""
    args = parse_arguments()

    try:
        # 从配置文件加载
        if args.config:
            if not os.path.exists(args.config):
                print(f"❌ 配置文件不存在: {args.config}")
                return 1

            print(f"📄 加载配置文件: {args.config}")
            config = load_simplified_config(args.config)
        else:
            print("❌ 请提供配置文件路径，使用 --config 参数")
            print("💡 示例: python calibrate_xaj_unified.py --config config.yaml")
            return 1

        if config is None:
            print("❌ 配置创建失败")
            return 1

        # 应用命令行覆盖
        if args.output_dir:
            config["training_cfgs"]["output_dir"] = args.output_dir
            print(f"✓ 输出目录覆盖为: {args.output_dir}")

        if args.experiment_name:
            config["training_cfgs"]["experiment_name"] = args.experiment_name
            print(f"✓ 实验名称覆盖为: {args.experiment_name}")

        # 验证配置
        print("\n🔍 验证配置...")
        if not validate_and_show_config(config, verbose=True):
            print("❌ 配置验证失败")
            return 1

        if args.dry_run:
            print("\n✅ 配置验证完成（dry-run 模式）")
            return 0

        # 执行率定
        print("\n🚀 开始率定...")
        results = calibrate(config)

        # 保存配置文件
        if args.save_config:
            training_cfgs = config.get("training_cfgs", {})
            output_dir = os.path.join(
                training_cfgs.get("output_dir", "results"),
                training_cfgs.get("experiment_name", "experiment"),
            )
            config_output_path = os.path.join(
                output_dir, "calibration_config.yaml"
            )
            os.makedirs(os.path.dirname(config_output_path), exist_ok=True)
            save_config_to_file(config, config_output_path)
            print(f"\n💾 配置已保存至: {config_output_path}")

        print("\n✅ XAJ率定完成！")
        return 0

    except KeyboardInterrupt:
        print("\n⚠️  率定被用户中断")
        return 1
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
