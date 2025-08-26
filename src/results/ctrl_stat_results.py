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
    setting_lst = ["dist_2"]
    hand_lst = ["shadow", "allegro", "leap_tac3d"]
    method_lst = ["ours", "op", "bs1", "bs2", "bs3", "bs4"]
    # method_lst = ["ours", "bs4"]

    n_valid = np.zeros((len(setting_lst), len(hand_lst), len(method_lst)))
    success_rate = np.zeros((len(setting_lst), len(hand_lst), len(method_lst)))
    obj_pos_err_mean = np.zeros((len(setting_lst), len(hand_lst), len(method_lst)))
    obj_pos_err_std = np.zeros((len(setting_lst), len(hand_lst), len(method_lst)))
    obj_rot_err_mean = np.zeros((len(setting_lst), len(hand_lst), len(method_lst)))
    obj_rot_err_std = np.zeros((len(setting_lst), len(hand_lst), len(method_lst)))
    norm_wrench_mean = np.zeros((len(setting_lst), len(hand_lst), len(method_lst)))
    norm_wrench_std = np.zeros((len(setting_lst), len(hand_lst), len(method_lst)))

    for i_s, setting_name in enumerate(setting_lst):
        for i_h, hand in enumerate(hand_lst):
            for i_m, method in enumerate(method_lst):
                file_path = f"output/learn_dummy_arm_{hand}/control_stat_res/{setting_name}_{method}.yaml"
                with open(file_path, "r") as f:
                    results = yaml.safe_load(f)

                    if "num_valid_cases" in results:
                        n_valid[i_s, i_h, i_m] = results["num_valid_cases"]
                    else:
                        n_valid[i_s, i_h, i_m] = (
                            100 - results["num_invalid_cases"]
                            if setting_name == "dist_0"
                            else 800 - results["num_invalid_cases"]
                        )

                    success_rate[i_s, i_h, i_m] = results["success_rate"]

                    obj_pos_err_mean[i_s, i_h, i_m] = results["ave_obj_pos_err"]["mean"]
                    obj_pos_err_std[i_s, i_h, i_m] = results["ave_obj_pos_err"]["std"]

                    obj_rot_err_mean[i_s, i_h, i_m] = results["ave_obj_angle_err"]["mean"]
                    obj_rot_err_std[i_s, i_h, i_m] = results["ave_obj_angle_err"]["std"]

                    norm_wrench_mean[i_s, i_h, i_m] = results["ave_normalized_wrench_all"]["mean"]
                    norm_wrench_std[i_s, i_h, i_m] = results["ave_normalized_wrench_all"]["std"]

    # print("-----------------------------------------------------")
    # print("Ave results among all settings and hands: ")
    # print("-----------------------------------------------------")
    # for i_m, method in enumerate(method_lst):
    #     n_success = success_rate[:, :, i_m] * n_valid[:, :, i_m]
    #     total_success_rate = np.sum(n_success) / np.sum(n_valid[:, :, i_m])
    #     total_pos_mean, total_pos_std = combine_mean_std(
    #         obj_pos_err_mean[:, :, i_m], obj_pos_err_std[:, :, i_m], n_valid[:, :, i_m]
    #     )
    #     total_rot_mean, total_rot_std = combine_mean_std(
    #         obj_rot_err_mean[:, :, i_m], obj_rot_err_std[:, :, i_m], n_valid[:, :, i_m]
    #     )

    #     print(f"{method} total success rate: {total_success_rate}")
    #     print(f"{method} total obj pos err: {total_pos_mean} +- {total_pos_std}")
    #     print(f"{method} total obj rot err: {total_rot_mean} +- {total_rot_std}")
    #     print("------")

    print("-----------------------------------------------------")
    print("Ave results among each settings and all hands: ")
    print("-----------------------------------------------------")
    for i_s, setting_name in enumerate(setting_lst):
        for i_m, method in enumerate(method_lst):
            n_success = success_rate[i_s, :, i_m] * n_valid[i_s, :, i_m]
            total_success_rate = np.sum(n_success) / np.sum(n_valid[i_s, :, i_m])
            total_pos_mean, total_pos_std = combine_mean_std(
                obj_pos_err_mean[i_s, :, i_m], obj_pos_err_std[i_s, :, i_m], n_valid[i_s, :, i_m]
            )
            total_rot_mean, total_rot_std = combine_mean_std(
                obj_rot_err_mean[i_s, :, i_m], obj_rot_err_std[i_s, :, i_m], n_valid[i_s, :, i_m]
            )

            print(f"{setting_name} {method} total success rate: {total_success_rate}")
            print(f"{setting_name} {method} total obj pos err: {total_pos_mean} +- {total_pos_std}")
            print(f"{setting_name} {method} total obj rot err: {total_rot_mean} +- {total_rot_std}")
            print("------")

    print("-----------------------------------------------------")
    print("Ave results among each settings and each hands: ")
    print("-----------------------------------------------------")
    for i_s, setting_name in enumerate(setting_lst):
        for i_h, hand in enumerate(hand_lst):
            for i_m, method in enumerate(method_lst):
                n_success = success_rate[i_s, i_h, i_m] * n_valid[i_s, i_h, i_m]
                total_success_rate = np.sum(n_success) / np.sum(n_valid[i_s, i_h, i_m])
                total_pos_mean, total_pos_std = combine_mean_std(
                    obj_pos_err_mean[i_s, i_h, i_m], obj_pos_err_std[i_s, i_h, i_m], n_valid[i_s, i_h, i_m]
                )
                total_rot_mean, total_rot_std = combine_mean_std(
                    obj_rot_err_mean[i_s, i_h, i_m], obj_rot_err_std[i_s, i_h, i_m], n_valid[i_s, i_h, i_m]
                )

                print(f"{setting_name} {hand} {method} total success rate: {total_success_rate}")
                print(f"{setting_name} {hand} {method} total obj pos err: {total_pos_mean} +- {total_pos_std}")
                print(f"{setting_name} {hand} {method} total obj rot err: {total_rot_mean} +- {total_rot_std}")
                print("------")


if __name__ == "__main__":
    main()
