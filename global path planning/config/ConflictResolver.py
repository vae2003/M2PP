import math


class ConflictResolver:
    def __init__(self, safe_distance=1.0):
        self.safe_distance = safe_distance

    @staticmethod
    def _resample(path, target_len):
        if not path:
            return []
        if len(path) >= target_len:
            return path[:target_len]
        result = []
        for i in range(target_len):
            idx = int(round(i * (len(path) - 1) / max(1, target_len - 1)))
            result.append(path[idx])
        return result

    @staticmethod
    def _distance(p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def detect_conflicts(self, paths):
        conflicts = []
        if len(paths) < 2:
            return conflicts

        non_empty_lengths = [len(path) for path in paths if path]
        if not non_empty_lengths:
            return conflicts
        max_len = max(non_empty_lengths)
        norm_paths = [self._resample(path, max_len) for path in paths]

        for t in range(max_len):
            for i in range(len(norm_paths)):
                for j in range(i + 1, len(norm_paths)):
                    pi = norm_paths[i][t]
                    pj = norm_paths[j][t]
                    if self._distance(pi, pj) < self.safe_distance:
                        conflicts.append((i, j, t))
        return conflicts

    def apply_priority_wait(self, paths):
        """
        按 AUV 编号优先级处理冲突：编号大者在冲突时刻插入等待点。
        """
        if len(paths) < 2:
            return paths

        resolved = [list(path) for path in paths]
        conflicts = self.detect_conflicts(resolved)

        for i, j, t in conflicts:
            low = max(i, j)
            if not resolved[low]:
                continue
            wait_idx = min(t, len(resolved[low]) - 1)
            wait_point = resolved[low][wait_idx]
            resolved[low].insert(wait_idx, wait_point)

        return resolved
