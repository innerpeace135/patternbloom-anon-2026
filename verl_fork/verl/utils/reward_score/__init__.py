"""Reward score dispatch for the PatternBloom training stages."""


def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    from . import qa_em_and_format
    res = qa_em_and_format.compute_score_format_answer(solution_str, ground_truth)
    return float(res) if isinstance(res, (int, float, bool)) else float(res[0])


def _default_compute_score_format(data_source, solution_str, extra_info=None):
    from . import qa_em_and_format
    res = qa_em_and_format.compute_score_format(solution_str)
    return float(res) if isinstance(res, (int, float, bool)) else float(res[0])


def _default_compute_score_answer_em(data_source, solution_str, ground_truth, extra_info=None):
    from . import qa_em_and_format
    res = qa_em_and_format.compute_score_em(solution_str, ground_truth)
    return float(res) if isinstance(res, (int, float, bool)) else float(res[0])


def _default_compute_score_answer_f1(data_source, solution_str, ground_truth, extra_info=None):
    from . import qa_em_and_format
    res = qa_em_and_format.compute_score_f1(solution_str, ground_truth)
    return float(res) if isinstance(res, (int, float, bool)) else float(res[0])


def _default_compute_score_format_answer(data_source, solution_str, ground_truth, extra_info=None):
    from . import qa_em_and_format
    res = qa_em_and_format.compute_score_format_answer(solution_str, ground_truth)
    return float(res) if isinstance(res, (int, float, bool)) else float(res[0])


def _compute_stage1_reward(data_source, solution_str, ground_truth, extra_info=None):
    from .path_reward import compute_stage1_reward
    return compute_stage1_reward(solution_str, ground_truth)


def _compute_stage2_reward(data_source, solution_str, ground_truth, extra_info=None):
    from .path_reward import compute_stage2_reward
    return compute_stage2_reward(solution_str, ground_truth)


def _compute_coverage_score(data_source, solution_str, ground_truth, extra_info=None):
    from .path_reward import compute_coverage_score
    return compute_coverage_score(solution_str, ground_truth)


def _compute_num_graph_blocks(data_source, solution_str, extra_info=None):
    from .path_reward import compute_num_graph_blocks
    return compute_num_graph_blocks(solution_str)
