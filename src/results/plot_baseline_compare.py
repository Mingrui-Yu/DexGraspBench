import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# from mr_utils.utils_plot import plotHistogram
from matplotlib import rcParams

X_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def lighten_color(color, amount=0.5):
    """
    将颜色变浅，amount 越接近 1 越接近白色
    color: 可以是 hex string 或者 matplotlib 内置颜色名
    amount: 0~1, 0 表示原色, 1 表示白色
    """
    try:
        c = mcolors.to_rgb(color)
    except ValueError:
        raise ValueError(f"Invalid color: {color}")
    return tuple((1 - amount) * x + amount * 1 for x in c)


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


def plotHistogram(  # noqa: PLR0913
    data,
    x_labels,
    bar_labels,
    bar_colors,
    data_bottom=None,
    x_width=0.8,
    bar_interval_ratio=0.0,
    border_width=0.0,
    edgecolor="k",
    linewidth=1,
    use_default_setting=True,
    scatter_points=None,
    scatter_colors="k",
    scatter_marker="D",
    scatter_zorder=2,
):
    """
    Input:
        data: shape (n_x_axis, n_bars_per_x)
        scatter_points: shape (n_x_axis, n_bars_per_x, n_points_per_bar)
    """
    data = np.array(data)
    if data.shape[0] != len(x_labels) or data.shape[1] != len(bar_labels):
        print("The dimension of input data is wrong.")
        return False

    if data_bottom is None:
        data_bottom = np.zeros(data.shape)
    elif data_bottom.shape != data.shape:
        print("The dimension of input data_bottom is wrong.")
        return False

    x = np.arange(len(x_labels))
    m = len(x_labels)
    n = len(bar_labels)
    if n == 1:
        bar_interval = 0.0
    else:
        bar_interval = (x_width * bar_interval_ratio) / float(n - 1)
    bar_width = (x_width - (n - 1) * bar_interval) / n

    plt.bar(
        x + (n - 1) / 2.0 * (bar_width + bar_interval), np.zeros((data.shape[0],)), width=bar_width, tick_label=x_labels
    )

    for i in range(n):
        plt.bar(
            x + i * (bar_width + bar_interval),
            data[:, i],
            bottom=data_bottom[:, i],
            width=bar_width,
            label=bar_labels[i],
            # color=bar_colors[i],
            color=[lighten_color(c, 0.3) for c in X_colors] if i == 0 else [lighten_color(c, 0.7) for c in X_colors],
            edgecolor=edgecolor,
            linewidth=linewidth,
            hatch="//" if i == 1 else None,
        )

        # scatter data points on the bar
        if scatter_points is not None:
            assert scatter_points.shape[0] == m
            assert scatter_points.shape[1] == n
            n_points_per_bar = scatter_points.shape[-1]
            for j in range(n_points_per_bar):
                plt.scatter(
                    x + i * (bar_width + bar_interval),
                    scatter_points[:, i, j],
                    color=scatter_colors[i],
                    zorder=scatter_zorder,
                    marker=scatter_marker,
                    edgecolors="k",
                )

    if use_default_setting:
        plt.xlim(
            [
                0 - bar_width / 2.0 - border_width,
                (m - 1) + (n - 1) * (bar_width + bar_interval) + bar_width / 2.0 + border_width,
            ]
        )


params = {
    "font.family": "Times New Roman",
    #                     # 'font.style':'italic',
    #                     'font.weight':'normal', #or 'bold'
    "mathtext.fontset": "stix",
    "font.size": 20,  # or large,small
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
rcParams.update(params)


def main():
    setting_lst = ["dist_0", "dist_2"]
    hand_lst = ["shadow", "allegro", "leap_tac3d"]

    exp_name = "learn_5k"
    method_lst = ["op", "bs1", "bs2", "bs3", "ours_ab2"]

    # exp_name = "learn_5k"
    # method_lst = ["op", "bs1", "bs2", "bs3", "ours_ab2"]

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
                file_path = f"output/{exp_name}_dummy_arm_{hand}/control_stat_res/{setting_name}_{method}.yaml"
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

    all_h_success_rate = np.zeros((len(setting_lst), len(method_lst)))
    all_h_obj_pos_err_mean = np.zeros((len(setting_lst), len(method_lst)))
    all_h_obj_rot_err_mean = np.zeros((len(setting_lst), len(method_lst)))
    all_h_norm_wrench_mean = np.zeros((len(setting_lst), len(method_lst)))

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
            total_wrench_mean, total_wrench_std = combine_mean_std(
                norm_wrench_mean[i_s, :, i_m], norm_wrench_std[i_s, :, i_m], n_valid[i_s, :, i_m]
            )

            all_h_success_rate[i_s, i_m] = total_success_rate
            all_h_obj_pos_err_mean[i_s, i_m] = total_pos_mean
            all_h_obj_rot_err_mean[i_s, i_m] = total_rot_mean
            all_h_norm_wrench_mean[i_s, i_m] = total_wrench_mean

            # print(
            #     f"{setting_name} {method} total success rate: {total_success_rate}, n: {np.sum(n_valid[i_s, :, i_m])}"
            # )
            # print(f"{setting_name} {method} total obj pos err: {total_pos_mean} +- {total_pos_std}")
            # print(f"{setting_name} {method} total obj rot err: {total_rot_mean} +- {total_rot_std}")
            # print(f"{setting_name} {method} total wrench err: {total_wrench_mean} +- {total_wrench_std}")
            # print("------")

    fig = plt.figure(figsize=(20, 3))

    # x_labels = ["Op", "bs1", "BS1", "BS2", "Ours"]
    x_labels = ["", "", "", "", ""]
    bar_labels = ["0cm", "2cm"]
    bar_colors = ["#1f77b4", "#ff7f0e"]

    plt.subplot(1, 4, 1)
    plt.title(r"Success rate (%) $\uparrow$")
    plotHistogram(100 * all_h_success_rate.T, x_labels=x_labels, bar_labels=bar_labels, bar_colors=bar_colors)
    plt.grid()
    plt.ylim([70, 100])
    legend = plt.legend(loc="upper left")
    for handle in legend.legend_handles:
        handle.set_facecolor("white")  # 条形图用set_facecolor

    plt.subplot(1, 4, 2)
    plt.title(r"Ave. obj. pos. err. (mm) $\downarrow$")
    plotHistogram(1000 * all_h_obj_pos_err_mean.T, x_labels=x_labels, bar_labels=bar_labels, bar_colors=bar_colors)
    plt.grid()

    plt.subplot(1, 4, 3)
    plt.title(r"Ave. obj. rot. err. ($^\circ$) $\downarrow$")
    plotHistogram(all_h_obj_rot_err_mean.T, x_labels=x_labels, bar_labels=bar_labels, bar_colors=bar_colors)
    plt.grid()

    plt.subplot(1, 4, 4)
    plt.title(r"Ave. normalized wrench $\downarrow$")
    plotHistogram(all_h_norm_wrench_mean.T, x_labels=x_labels, bar_labels=bar_labels, bar_colors=bar_colors)
    plt.grid()

    # 4. 【核心】创建共享图例 - 添加到图形上方
    legend_elements = []
    categories = ["Open-loop", "Feedback control", "No arm motion", "Independ. forces", "Ours"]
    for color, label in zip([lighten_color(c) for c in X_colors], categories):
        # 创建代表颜色的矩形补丁（不使用斜纹）
        patch = plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black", label=label)
        legend_elements.append(patch)

    # 将图例放置在图形上方中央
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),  # 精确控制位置
        ncol=5,  # 水平排列5个图例项
        frameon=True,
        fancybox=True,
        shadow=False,
    )

    # plt.tight_layout()

    plt.subplots_adjust(left=0.03, bottom=0.04, right=0.995, top=0.70, wspace=0.15, hspace=0.0)

    plt.show()


if __name__ == "__main__":
    main()
