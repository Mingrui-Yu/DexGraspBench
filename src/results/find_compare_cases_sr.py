import numpy as np
import yaml


def combine_mean_std(means, stds, counts):
    """
    means: np.array of shape (...), mean of each group
    stds:  np.array of same shape, std of each group
    counts: np.array of same shape, sample count of each group

    Returns: (global_mean, global_std)
    """
    means = np.asarray(means)
    stds = np.asarray(stds)
    counts = np.asarray(counts)

    total_n = np.sum(counts)
    # weighted mean
    global_mean = np.sum(means * counts) / total_n

    # pooled variance
    sq_diff = (means - global_mean) ** 2
    pooled_var = (np.sum((counts - 1) * stds**2) + np.sum(counts * sq_diff)) / (total_n - 1)

    global_std = np.sqrt(pooled_var)
    return global_mean, global_std


def main():
    exp_name = "learn"
    setting_name = "dist_2"
    hand_lst = [
        "shadow",
    ]
    method_lst = ["ours_ab2", "bs2"]
    # method_lst = ["ours_ab2", "op", "bs1", "bs2", "bs3"]

    failure_cases = {}

    for i_h, hand in enumerate(hand_lst):
        for i_m, method in enumerate(method_lst):
            file_path = f"output/{exp_name}_dummy_arm_{hand}/control_stat_res/{setting_name}_{method}.yaml"
            with open(file_path, "r") as f:
                results = yaml.safe_load(f)

                failure_cases[method] = results["failure_cases"]

    for key in failure_cases.keys():
        print(f"failure_cases {key}: ", failure_cases[key])

    ours_failures = set(failure_cases["ours_ab2"])
    other_methods = [m for m in failure_cases.keys() if m != "ours_ab2"]

    # 统计每个 case 在其他方法中的失败次数
    case_fail_counts = {}

    for method in other_methods:
        for case in failure_cases[method]:
            if case not in ours_failures:  # ours 成功
                case_fail_counts[case] = case_fail_counts.get(case, 0) + 1

    # 按失败次数排序（降序）
    sorted_cases = sorted(case_fail_counts.items(), key=lambda x: x[1], reverse=True)

    if setting_name == "dist_2":
        case_id = [n // 8 for n, _ in sorted_cases]
        pos_id = [n % 8 for n, _ in sorted_cases]

    print("Cases where ours_ab2 succeeded but others failed (sorted):")
    for i, (case, count) in enumerate(sorted_cases):
        print(f"case {case_id[i]} pos {pos_id[i]}: failed in {count}/{len(other_methods)} other methods")


if __name__ == "__main__":
    main()
