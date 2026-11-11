# 跨业态推荐的反事实离线评估与约束重排

本项目保存百货多业态经营研究的合成资料及输入读取工具。当前程序只解析文件并显示输入类型与记录数量，不进行统计分析、模型训练、优化求解或业务推荐。所有记录均为虚构，不能用于陈述真实商场经营结果。

## 输入约定

每请求日志记录单个选中项目及候选集合，logging_propensity 是选中动作的已知记录概率。items 为带known_at及effective_at的属性版本。reward_redeem 是0或1，目标策略结果只涉及单项目奖励。

fixtures 为资料目录，src/impression_reader.py 为读取器，tests 为格式读取测试。输入在运行中保持不变；研究计算结果可写入独立的 .workspace-output 目录。

## 本地操作

Python 3.11 及以上版本，现有读取工具只依赖标准库，不需要下载第三方包。进入项目根目录：

    python -m src.impression_reader fixtures/impressions.jsonl

编译检查：

    python -m compileall -q src tests

格式读取测试：

    python -m unittest discover -s tests -v

## 固定运行环境

    docker build -t cross-format-ranking:local .
    docker run --rm cross-format-ranking:local

容器默认运行输入测试。镜像使用完整内容摘要固定，研究输入随项目一起复制。
