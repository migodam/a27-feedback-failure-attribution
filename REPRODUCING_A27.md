# 复现 A27：只使用已暴露数据和冻结网络

本项目从 A26 `2603f3bd16c48e48587fedc4363975779def17f5` 分支产生。物理是已知背景、24×24 二维 TM，八源、32接收器；不是三维 Vector-Maxwell。A27 不生成真值标签、不训练网络、不读取 locked ID/OOD。完整 Maxwell 调用发生在预测材料或已知背景，属于付费推理/诊断，不是新标签生成。

## 公开输入

从 [A26 冻结 Release](https://github.com/migodam/a26-om-feedback-prior-completion/releases/tag/a26-v0.1.0) 获取并解压：

- `A26_REPRO_INPUTS.zip`：已有 TRAIN/已暴露 VALIDATION、已知背景及几何。主要试验只使用 VALIDATION。
- `A26_CHECKPOINTS.zip`：N1/N2/N3 的 best/middle/final checkpoints。A27 主比较只用原 best。
- `A26_RAW_EVIDENCE.zip`：冻结特征、观测、预测和历史账本。A27 会核对 case 顺序并重现 N1/N3 预测。

A27 Release 提供 `A27_RAW_EVIDENCE.zip`，包含修正材料数组、原数值失败、全部原始诊断、局部谱、原生计时、逐算子账本与原环境记录。Git 内可以直接查看所有标量 CSV、图、配置和 terminal receipts。不需要私人 SSH 配置即可在自己的已有环境复现；XINAN 远端工具只是现有实验环境的连接适配器。

训练时中间量审计另外依赖已有 A25 TRAIN 电流。A26 简化输入没有电流，缺少它时该标签审计应记 `DATA_NOT_AVAILABLE`，不能用新仿真补齐。这不阻塞 post-decoder 成像比较。

## 环境和入口

原 GPU 环境：Python 3.10.9、PyTorch 2.6.0+cu124、NumPy 2.2.6、SciPy 1.15.3、RTX4060 Laptop 8GB。安装依赖可沿用 `requirements-a26.txt`；不要为复现擅自安装新的 CUDA/调度系统。CPU 测试使用已有 A25 Python 环境，其中 Clarabel 0.11.1 的独立参考测试可用。原 Windows 测试明确跳过缺少 Clarabel 的三项。

设置 `PYTHONPATH=src`、`OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`。从仓库根目录执行，输入在本仓库时显式传 `--data-root data --a26-root .`：

```sh
python -B tools/a27_driver.py --stage tests --job replica-tests-001 --device cpu --data-root data --a26-root .
python -B tools/a27_driver.py --stage audit --job replica-audit-001 --device cuda --data-root data --a26-root .
python -B tools/a27_driver.py --stage proxhealth --job replica-health-001 --device cuda --data-root data --a26-root .
python -B tools/a27_driver.py --stage postfgp --job replica-post-001 --device cuda --data-root data --a26-root .
python -B tools/a27_driver.py --stage diagnostics --job replica-diagnostics-001 --device cuda --data-root data --a26-root .
python -B tools/a27_driver.py --stage observability --job replica-spectrum-001 --device cuda --data-root data --a26-root .
python -B tools/a27_driver.py --stage latency --job replica-latency-001 --device cuda --data-root data --a26-root .
python -B tools/a27_driver.py --stage report --job replica-report-001 --device cpu --data-root data --a26-root .
```

这些命令仅作为另一个明确命名、独立有界 replication 的说明，不会在本项目自动执行。原 `results/a27/` 和账本不可覆盖，也不能删除后把本 campaign 的预算重置为零。要复算，应保留原目录作为冻结证据，在独立 replication 工作副本建立独立账本。共享 GPU 锁路径在 `src/a27/budget.py`；XINAN 必须使用既有唯一锁，其他主机应明确登记自己的安全锁。

`post` 是原 accelerated-PDHG 对照，387 个数值失败保留在原目录。最终共同方法 `postfgp` 只有十二项固定、无 truth 的数值健康检查通过后才执行。它使用相同目标、box、TV、步长/接受规则、KKT/gap/feasibility 门槛和 20,000 次/2秒上限。方法选择不是按材料误差挑选。详见 [数值修订](research/A27_NUMERICAL_AMENDMENT_01.md)。

## 结果阅读与计时边界

主分析目录为 `results/a27/artifacts_post_fgp/`，由明确的 `POST_SOURCE.json` 选择。先平均每 parent 的八个条件，再做 2,000 次 parent paired bootstrap。weak-only 表是后验描述切片，不替代完整 cohort。局部谱只覆盖四个固定 parent 的32个预测状态，不是16-parent全 cohort。

`NATIVE_INFERENCE` 是真实测量→encoder→CNN→box→主机图像的 resident warm 计时；`POST_RAW_TO_IMAGE` 还包含修正和所有 full-F 在线接受审核。`STAGE_PROFILES` 是额外带事件/dispatch观察器的执行，存在嵌套区间，不可把其时长相加冒充原生延迟。没有新增 disk-cold、吞吐或首次进程提速证据。

原始 JSON/CSV 保留原平台换行。历史 SHA 字段只作来源记录；本项目不执行 SHA256 检查。物理身份、数值安全、图像质量、部署时间和网页发布分别判断。
