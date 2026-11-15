"""流水线的显式失败类型。任何一条都用异常终止，不做静默跳过。"""


class PipelineError(Exception):
    """所有可预期输入/配置问题的基类。"""


class InputContractError(PipelineError):
    """日志记录违反输入约定（字段、概率、奖励、时间）。"""


class DuplicateConflictError(InputContractError):
    """同一 request_id 的多条记录不完全一致。"""


class UnknownItemError(PipelineError):
    """日志引用了 items 里不存在的项目。"""


class NoVersionError(PipelineError):
    """请求时刻没有任何已生效且已登记的项目属性版本。"""


class SupportError(PipelineError):
    """目标策略在日志中有正概率质量落在零记录支持动作上。"""
