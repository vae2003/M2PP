import copy
import numpy as np
import matplotlib.pyplot as plt
import math
from mapData.map_two import mapTwo


# 定义机器人状态
class RobotState:
    def __init__(self, x, y, yaw, v, omega):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.omega = omega


# 定义DWA参数
class DWAParams:
    def __init__(self):
        self.max_speed = 0.4
        self.min_speed = 0
        self.max_yaw_rate = np.pi / 3.0
        self.max_accel = 0.2
        self.max_delta_yaw_rate = np.pi / 8.0
        self.v_resolution = 0.01
        self.yaw_rate_resolution = np.pi / 180.0
        self.dt = 0.1
        self.predict_time = 2.0
        self.robot_radius = 0.15


# 障碍物类
class Obstacle:
    def __init__(self, x, y, vx, vy, r):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.r = r

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt


alpha = 1
beta = 1
gamma = 1
delta = 0.9
theta = 0.2
EPSILON = 1e-9


# 机器人运动模型
def motion(state, v, omega, dt):
    state.x += v * np.cos(state.yaw) * dt
    state.y += v * np.sin(state.yaw) * dt
    state.yaw += omega * dt
    state.v = v
    state.omega = omega
    return state


# 计算动态窗口
def calc_dynamic_window(state, params):
    Vs = [params.min_speed, params.max_speed, -params.max_yaw_rate, params.max_yaw_rate]
    Vd = [
        state.v - params.max_accel * params.dt,
        state.v + params.max_accel * params.dt,
        state.omega - params.max_delta_yaw_rate * params.dt,
        state.omega + params.max_delta_yaw_rate * params.dt
    ]
    return [
        max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
        max(Vs[2], Vd[2]), min(Vs[3], Vd[3])
    ]


# 目标朝向代价函数
def __heading(trajectory, goal):
    dx = goal[0] - trajectory[-1][0]
    dy = goal[1] - trajectory[-1][1]
    error_angle = math.atan2(dy, dx)
    cost_angle = error_angle - trajectory[-1][2]
    return math.pi - abs(cost_angle)


# 障碍物代价函数
def __dist(trajectory, obstacles, robot_radius):
    obstacle_cost = 10
    for obs in obstacles:
        for pos in trajectory:
            dist = np.linalg.norm(np.array([pos[0], pos[1]]) - np.array([obs.x, obs.y]))
            if dist <= robot_radius + obs.r:
                return float('inf')
            obstacle_cost = min(obstacle_cost, dist)
    return obstacle_cost


# 速度代价函数
def __vel(vel):
    return abs(vel)


# 转角代价函数
def __corn(state, trajectory, goal):
    m1 = math.atan2(state.x - goal[0], state.y - goal[1])
    m2 = math.atan2(trajectory[-1][0] - goal[0], trajectory[-1][1] - goal[1])
    cos_value = math.cos(abs(m1 - m2))
    return 1 + cos_value


# 动态障碍物代价函数
def __move(state, trajectory, obstacles, robot_radius):
    min_dist = float('inf')
    min_cost = 0
    for obs in obstacles:
        if obs.vx != 0 or obs.vy != 0:
            cost = cosine_of_angle([obs.x, obs.y], [obs.x + obs.vx, obs.y + obs.vy], [state.x, state.y],
                                   [trajectory[-1][0], trajectory[-1][1]])
            line_distance = np.linalg.norm(np.array([state.x, state.y]) - np.array([obs.x, obs.y]))
            if cost is not None and line_distance < robot_radius * 3 + obs.r and line_distance < min_dist:
                min_dist = line_distance
                min_cost = cost
    return min_cost


# 安全距离内是否存在障碍物
def isSafe(trajectory, obstacles, robot_radius):
    for pos in trajectory:
        for obs in obstacles:
            test_dist = np.linalg.norm(np.array([pos[0], pos[1]]) - np.array([obs.x, obs.y]))
            if test_dist < robot_radius * 2 + obs.r:
                return 1
    return 0


# Evaluate trajectory
def evaluate_trajectory(state, v, omega, params, goal, obstacles):
    predict_state = RobotState(state.x, state.y, state.yaw, v, omega)
    trajectory = [np.array([predict_state.x, predict_state.y, predict_state.yaw, predict_state.v, predict_state.omega])]
    time = 0.0

    while time <= params.predict_time:
        predict_state = motion(predict_state, v, omega, params.dt)
        trajectory.append(np.array([predict_state.x, predict_state.y, predict_state.yaw, predict_state.v, predict_state.omega]))
        time += params.dt

    return np.array(trajectory)


# 计算夹角余弦值
def cosine_of_angle(A, B, C, D):
    AB = np.array(B) - np.array(A)
    CD = np.array(D) - np.array(C)
    dot_product = np.dot(AB, CD)
    norm_AB = np.linalg.norm(AB)
    norm_CD = np.linalg.norm(CD)

    if norm_AB == 0 or norm_CD == 0:
        return 0

    return dot_product / (norm_AB * norm_CD)


# 判断动态窗口是否检测到动态障碍物
def isWindow(state, obstacles, trajectory, robot_radius):
    for obs in obstacles:
        if obs.vx != 0 or obs.vy != 0:
            for pos in trajectory:
                line_distance = np.linalg.norm(np.array([pos[0], pos[1]]) - np.array([obs.x, obs.y]))
                if line_distance < robot_radius * 2 + obs.r:
                    return True
    return False


# DWA主控制函数
def dwa_control(state, params, goal, obstacles):
    dw = calc_dynamic_window(state, params)
    min_cost = -float('inf')
    best_trajectory = None
    best_v = 0.0
    best_omega = 0.0

    sum_heading = 0
    sum_dist = 0
    sum_vel = 0
    sum_move = 0
    sum_corn = 0

    dynamic_state = False
    safe_state = False

    for v in np.arange(dw[0], dw[1], params.v_resolution):
        for omega in np.arange(dw[2], dw[3], params.yaw_rate_resolution):
            trajectory = evaluate_trajectory(state, v, omega, params, goal, obstacles)
            heading_eval = __heading(trajectory, goal)
            dist_eval = __dist(trajectory, obstacles, params.robot_radius)
            vel_eval = __vel(trajectory[-1][3])

            sum_vel += vel_eval
            sum_dist += dist_eval
            sum_heading += heading_eval
            sum_corn += __corn(state, trajectory, goal)

            if isSafe(trajectory, obstacles, params.robot_radius):
                safe_state = True
            move_eval = __move(state, trajectory, obstacles, params.robot_radius)
            if move_eval != 0:
                sum_move += 1 - move_eval
            if isWindow(state, obstacles, trajectory, params.robot_radius):
                dynamic_state = True

    # Prevent normalization denominator from being zero.
    sum_heading = max(sum_heading, EPSILON)
    sum_dist = max(sum_dist, EPSILON)
    sum_vel = max(sum_vel, EPSILON)
    sum_corn = max(sum_corn, EPSILON)

    for v in np.arange(dw[0], dw[1], params.v_resolution):
        for omega in np.arange(dw[2], dw[3], params.yaw_rate_resolution):
            trajectory = evaluate_trajectory(state, v, omega, params, goal, obstacles)

            heading_eval = alpha * __heading(trajectory, goal) / sum_heading
            dist_eval = beta * __dist(trajectory, obstacles, params.robot_radius) / sum_dist
            vel_eval = gamma * __vel(trajectory[-1][3]) / sum_vel
            move_eval = __move(state, trajectory, obstacles, params.robot_radius)
            corn_eval = theta * __corn(state, trajectory, goal) / sum_corn

            if dynamic_state and sum_move > 0 and (abs(move_eval) < 0.4 or move_eval < -0.7):
                cost = heading_eval + dist_eval + vel_eval + delta * (1 - move_eval) / sum_move
            elif safe_state:
                cost = heading_eval + dist_eval + vel_eval + corn_eval
            else:
                cost = heading_eval + dist_eval + vel_eval

            if cost > min_cost:
                min_cost = cost
                best_trajectory = trajectory
                best_v = v
                best_omega = omega

    return best_v, best_omega, best_trajectory


def _as_obstacles(obstacles):
    return [Obstacle(obs.x, obs.y, obs.vx, obs.vy, obs.r) for obs in obstacles]


def _other_auv_as_obstacles(states, current_idx, robot_radius):
    dynamic_obs = []
    for i, st in enumerate(states):
        if i == current_idx:
            continue
        vx = st.v * math.cos(st.yaw)
        vy = st.v * math.sin(st.yaw)
        dynamic_obs.append(Obstacle(st.x, st.y, vx, vy, robot_radius))
    return dynamic_obs


def _build_waypoints(starts, goals, global_paths=None):
    if global_paths is not None:
        return [[list(p) for p in path] for path in global_paths]
    return [[list(starts[i]), list(goals[i])] for i in range(len(starts))]


def _current_goal(waypoints, waypoint_idx, final_goal, state, reach_th=0.2):
    idx = waypoint_idx
    while idx < len(waypoints) - 1:
        wp = np.array(waypoints[idx])
        if np.linalg.norm(np.array([state.x, state.y]) - wp) <= reach_th:
            idx += 1
        else:
            break
    if idx >= len(waypoints):
        idx = len(waypoints) - 1
    return np.array(waypoints[idx]), idx


# 可视化函数
def visualize(states, trajectories, starts, goals, paths, obstacles, waypoint_targets=None):
    plt.clf()
    colors = ['r', 'b', 'g', 'm', 'c', 'y']

    for i, st in enumerate(states):
        color = colors[i % len(colors)]
        plt.plot(starts[i][0], starts[i][1], marker='s', color=color, markersize=7)
        plt.plot(goals[i][0], goals[i][1], marker='*', color=color, markersize=10)
        plt.plot(st.x, st.y, marker='x', color=color)
        if len(paths[i]) > 0:
            p = np.array(paths[i])
            plt.plot(p[:, 0], p[:, 1], color=color, linewidth=1.5, label=f"AUV-{i}")
        if trajectories[i] is not None:
            plt.plot(trajectories[i][:, 0], trajectories[i][:, 1], '--', color=color, linewidth=1)
        if waypoint_targets is not None:
            plt.plot(waypoint_targets[i][0], waypoint_targets[i][1], marker='o', color=color, markersize=4)

    for obs in obstacles:
        plt.plot(obs.x, obs.y, "ok", markersize=max(2, obs.r * 100))

    plt.grid(True)
    plt.axis("equal")
    plt.legend(loc='best')
    plt.pause(0.001)


# 主程序
# global_paths: 可选, 每个AUV一条waypoint列表, 例如 [[[x1,y1],[x2,y2],...], ...]
def main(global_paths=None):
    mapData = mapTwo()
    static_obstacles = mapData.getObstacles()

    starts = getattr(mapData, 'starts', [mapData.start])
    goals = getattr(mapData, 'goals', [mapData.goal])
    n = min(len(starts), len(goals))
    starts = starts[:n]
    goals = goals[:n]

    params = DWAParams()
    states = [RobotState(x=starts[i][0], y=starts[i][1], yaw=0.0, v=0.0, omega=0.0) for i in range(n)]
    paths = [[list(starts[i])] for i in range(n)]
    reached = [False for _ in range(n)]
    waypoints = _build_waypoints(starts, goals, global_paths)
    waypoint_idx = [1 if len(waypoints[i]) > 1 else 0 for i in range(n)]

    total_path_lengths = [0.0 for _ in range(n)]
    step = 0
    max_steps = 4000

    plt.figure()

    while not all(reached) and step < max_steps:
        trajectories = [None for _ in range(n)]
        controls = [(0.0, 0.0) for _ in range(n)]
        waypoint_targets = [np.array(goals[i]) for i in range(n)]

        for i in range(n):
            if reached[i]:
                continue

            target, idx = _current_goal(waypoints[i], waypoint_idx[i], goals[i], states[i])
            waypoint_idx[i] = idx
            waypoint_targets[i] = target

            dist_to_goal = np.linalg.norm(np.array([states[i].x, states[i].y]) - np.array(goals[i]))
            if dist_to_goal <= 0.25:
                reached[i] = True
                continue

            obs_for_i = _as_obstacles(static_obstacles) + _other_auv_as_obstacles(states, i, params.robot_radius)
            v, omega, trajectory = dwa_control(states[i], params, target, obs_for_i)
            controls[i] = (v, omega)
            trajectories[i] = trajectory

        for i in range(n):
            if reached[i]:
                continue
            prev_x, prev_y = states[i].x, states[i].y
            v, omega = controls[i]
            states[i] = motion(states[i], v, omega, params.dt)
            paths[i].append([states[i].x, states[i].y])
            step_length = np.linalg.norm(np.array([states[i].x, states[i].y]) - np.array([prev_x, prev_y]))
            total_path_lengths[i] += step_length

        for obs in static_obstacles:
            obs.update(params.dt)

        visualize(states, trajectories, starts, goals, paths, static_obstacles, waypoint_targets)
        step += 1

    print(f'iterations: {step}')
    for i in range(n):
        print(f'AUV-{i} reached={reached[i]}, path_length={total_path_lengths[i]:.2f}')
    plt.show()


if __name__ == '__main__':
    main()
