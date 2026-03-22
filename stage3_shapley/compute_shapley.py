"""
阶段三：运行 Stratified SVARM 计算 Shapley Value。

克隆了 https://github.com/ZenDylan/Approximating-the-Shapley-Value-without-Marginal-Contributions
参考其 StratifiedSVARM 实现，自行适配 PreferenceDPOGame 接口。

支持：
- 断点续传（读取已有 JSONL 缓存）
- 可配置的 budget（默认 50 快速测试，建议后续扩展到 200）
- 精确边界计算（空集/单玩家/全集），warmup + 分层采样估算其余层
- 详细的 tqdm 进度条
"""

import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stage3_shapley.preference_game import PreferenceDPOGame
from utils.io import save_json


class StratifiedSVARM:
    """
    Stratified SVARM 算法（修正版）。

    与原始 SVARM 的区别：对每层（stratum）使用不同的采样分布，
    使得大子集和小子集的采样概率更均衡。

    关键修正：
    - _exact_calculation 只精确计算 O(n) 个边界点（空集、单玩家、全集），
      不再做 O(n²) 的遍历（200 玩家时 40000 次评估完全不可行）
    - 每层累积均值代替数组累加，避免精度问题
    """

    def __init__(self, game: PreferenceDPOGame, budget: int,
                 normalize: bool = False, warm_up: bool = True,
                 dist_type: str = "paper"):
        self.game = game
        self.budget = budget
        self.normalize = normalize
        self.warm_up = warm_up
        self.dist_type = dist_type
        self.n = game.get_player_number()

        self.v_empty = 0.0
        self.v_full = None
        self.single_values = np.zeros(self.n)
        self.single_counts = np.zeros(self.n)

        if self.normalize:
            self.v_full = self.game(tuple(range(self.n)))

        self.layer_plus = np.zeros(self.n)
        self.cnt_plus = np.zeros(self.n)
        self.layer_minus = np.zeros(self.n)
        self.cnt_minus = np.zeros(self.n)

        self.distribution = self._generate_distribution(dist_type)
        self.probs = [self.distribution[s] for s in range(self.n + 1)]

        self.eval_count = 0

    def _generate_distribution(self, dist_type: str) -> dict:
        distribution = {}
        if dist_type == "paper":
            H_n = sum(1.0 / s for s in range(1, self.n + 1))
            for s in range(self.n + 1):
                if s == 0 or s == self.n:
                    distribution[s] = 0.0
                else:
                    distribution[s] = (1.0 / s) / H_n
        else:
            for s in range(self.n + 1):
                distribution[s] = 1.0 / (self.n + 1)
        return distribution

    def _exact_calculation(self):
        print(f"\n  [精确边界计算] 空集 + {self.n} 个单玩家 + 全集 = {self.n + 1} 次评估")
        pbar_exact = tqdm(
            total=self.n + 1,
            desc="[SVARM] 精确边界",
            unit="次",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )

        full_coalition = tuple(range(self.n))
        self.v_full = self.game(full_coalition)
        self.eval_count += 1
        pbar_exact.set_postfix(val=f"v(empty)={self.v_empty:.4f}, v([n])={self.v_full:.4f}")
        pbar_exact.update(1)

        for i in range(self.n):
            v_single = self.game([i])
            n_c = self.single_counts[i]
            self.single_values[i] = (self.single_values[i] * n_c + v_single) / (n_c + 1)
            self.single_counts[i] = n_c + 1

            self._accumulate_plus(i, 0, v_single)
            self._accumulate_minus(i, 1, v_single)

            self.eval_count += 1
            pbar_exact.set_postfix(done=f"{self.eval_count}/{self.n + 1}", v_single=f"{v_single:.4f}")
            pbar_exact.update(1)

        for i in range(self.n):
            self._accumulate_minus(i, self.n - 1, self.v_full)

        pbar_exact.close()
        print(f"    精确边界计算完成，共 {self.eval_count} 次评估。")
        print(f"    v(empty) = 0.0, v([n]) = {self.v_full:.4f}")

    def _accumulate_plus(self, player: int, layer: int, value: float):
        if layer < 0 or layer >= self.n:
            return
        n_c = self.cnt_plus[layer]
        self.layer_plus[layer] = (self.layer_plus[layer] * n_c + value) / (n_c + 1)
        self.cnt_plus[layer] = n_c + 1

    def _accumulate_minus(self, player: int, layer: int, value: float):
        if layer < 0 or layer >= self.n:
            return
        n_c = self.cnt_minus[layer]
        self.layer_minus[layer] = (self.layer_minus[layer] * n_c + value) / (n_c + 1)
        self.cnt_minus[layer] = n_c + 1

    def _warmup(self):
        print("\n  [Warm-up] 分层均匀采样...")
        warmup_budget = min(self.budget // 4, 50)
        print(f"    Warm-up 预算: {warmup_budget} 次")

        pbar_warmup = tqdm(
            total=warmup_budget,
            desc="[SVARM] Warm-up",
            unit="次",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
        )

        eval_done = 0
        layer_samplers = list(range(2, self.n - 1))
        if len(layer_samplers) == 0:
            layer_samplers = [self.n // 2]

        samples_per_layer = max(1, warmup_budget // len(layer_samplers))

        for layer in layer_samplers:
            for _ in range(samples_per_layer):
                if eval_done >= warmup_budget:
                    break
                size = layer
                perm = np.random.permutation(self.n)
                coalition = perm[:size].tolist()

                v_coal = self.game(coalition)
                self._update_phi(coalition, size, v_coal)
                self.eval_count += 1
                eval_done += 1
                pbar_warmup.set_postfix(layer=layer, done=f"{eval_done}/{warmup_budget}")
                pbar_warmup.update(1)

        pbar_warmup.close()
        print(f"    Warm-up 完成，实际采样 {eval_done} 次。")

    def _update_phi(self, coalition: list, size: int, v_A: float):
        coalition_set = set(coalition)
        for i in coalition:
            self._accumulate_plus(i, size - 1, v_A)

        for i in range(self.n):
            if i not in coalition_set:
                self._accumulate_minus(i, size, v_A)

    def approximate_shapley_values(self) -> np.ndarray:
        print(f"\n[StratifiedSVARM] n={self.n}, budget={self.budget}, "
              f"warm_up={self.warm_up}, dist={self.dist_type}")

        self._exact_calculation()

        if self.warm_up:
            self._warmup()

        remaining_budget = self.budget - self.eval_count
        print(f"\n  [主循环] 剩余预算: {remaining_budget} 次评估")

        pbar = tqdm(
            total=self.budget,
            desc="[SVARM] 分层采样",
            unit="次",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        pbar.update(self.eval_count)

        while self.eval_count < self.budget:
            size = int(np.random.choice(range(0, self.n + 1), 1, p=self.probs)[0])
            if size == 0:
                coalition = []
            else:
                coalition = np.random.choice(self.n, size, replace=False).tolist()

            v_A = self.game(coalition)
            self._update_phi(coalition, size, v_A)
            self.eval_count += 1

            sv_estimates = self.get_estimates()
            elapsed = time.time() - self.game.start_time
            avg_time = elapsed / max(self.eval_count, 1)
            import psutil
            ram_gb = psutil.Process().memory_info().rss / 1024**3

            pbar.set_postfix(
                done=f"{self.eval_count}/{self.budget}",
                avg=f"{avg_time:.0f}s/次",
                RAM=f"{ram_gb:.1f}GB",
                SV_mean=f"{np.mean(sv_estimates):.4f}",
            )
            pbar.update(1)

        pbar.close()
        return self.get_estimates()

    def get_estimates(self) -> np.ndarray:
        shapley_values = np.zeros(self.n)

        for i in range(self.n):
            if self.cnt_plus[i] > 0:
                phi_plus = self.layer_plus[i]
            else:
                phi_plus = self.single_values[i] if self.single_counts[i] > 0 else 0.0

            if self.cnt_minus[i] > 0:
                phi_minus = self.layer_minus[i]
            else:
                phi_minus = self.v_full if self.v_full is not None else 0.0

            shapley_values[i] = phi_plus - phi_minus

        if self.normalize and self.v_full is not None:
            total = np.sum(shapley_values)
            if abs(total) > 1e-9:
                shapley_values *= (self.v_full / total)

        return shapley_values

    def get_name(self) -> str:
        name = f"StratSVARM_n{self.n}_b{self.budget}_{self.dist_type}"
        if self.warm_up:
            name += "_warmup"
        if self.normalize:
            name += "_nor"
        return name


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stratified SVARM for Preference DPO Game")
    parser.add_argument("--budget", type=int, default=50,
                        help="最大 value function 评估次数（默认 50，建议快速测试后扩展到 200+）")
    parser.add_argument("--warmup", action="store_true", default=True,
                        help="是否使用 warm-up（默认 True）")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false",
                        help="禁用 warm-up")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径（默认 outputs/shapley_values.json）")
    parser.add_argument("--dist", type=str, default="paper",
                        choices=["paper", "uniform"],
                        help="采样分布类型（默认 paper，即 1/k 分布）")
    args = parser.parse_args()

    budget = args.budget
    output_path = args.output or os.path.join(
        os.path.dirname(__file__), "..", "outputs", "shapley_values.json"
    )

    print("\n" + "#" * 60)
    print("# 阶段三：Stratified SVARM 计算 Shapley Value")
    print("#" * 60)
    print(f"  玩家数量: 200")
    print(f"  预算 (budget): {budget}")
    print(f"  Warm-up: {args.warmup}")
    print(f"  采样分布: {args.dist}")
    print(f"  输出文件: {output_path}")
    print(f"\n  预估耗时（RTX 4060, budget={budget}）:")
    print(f"    精确边界: 201 次（约数分钟/次）")
    print(f"    建议先用 --budget 10 做 smoke test，确认无误后再增加预算。")

    game = PreferenceDPOGame(budget=budget, n_workers=1)

    algo = StratifiedSVARM(
        game=game,
        budget=budget,
        normalize=False,
        warm_up=args.warmup,
        dist_type=args.dist,
    )

    start_time = time.time()
    shapley_values = algo.approximate_shapley_values()
    total_time = time.time() - start_time

    print(f"\n  计算完成! 总耗时: {total_time / 3600:.1f} 小时 ({total_time / 60:.1f} 分钟)")
    print(f"  总评估次数: {game.eval_count}")

    result = {
        "shapley_values": shapley_values.tolist(),
        "player_indices": list(range(len(shapley_values))),
        "budget": budget,
        "actual_evaluations": game.eval_count,
        "total_time_seconds": float(total_time),
        "algorithm": algo.get_name(),
    }
    save_json(result, output_path)

    print(f"\n  Shapley Value 统计:")
    print(f"    最小值: {shapley_values.min():.6f}")
    print(f"    最大值: {shapley_values.max():.6f}")
    print(f"    均值:   {shapley_values.mean():.6f}")
    print(f"    标准差: {shapley_values.std():.6f}")

    game.close()

    print("\n" + "#" * 60)
    print(f"# Shapley Value 计算完成! 结果: {output_path}")
    print("#" * 60)


if __name__ == "__main__":
    main()
