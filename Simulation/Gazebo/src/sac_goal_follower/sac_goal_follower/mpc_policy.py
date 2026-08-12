"""
Model Predictive Control (MPC) baseline for goal-following with obstacle avoidance.
Uses CasADi / IPOPT for nonlinear optimization.

Fixes over original version:
  1. Observation denormalization  — raw obs values are normalized; must scale back.
  2. Consistent robot frame       — goal AND obstacle positions/velocities all in
                                    robot-local frame, same frame as the optimizer.
  3. Proper obstacle avoidance    — scalar slack-variable soft constraints instead
                                    of the weak inverse-distance barrier.
  4. Correct validity check       — uses depth channel (col 2 < 0.95).
  5. Fixed warm-start             — guard on prev_u (not unused prev_v/prev_w).
  6. Solver-fail attribute        — evaluation script tracks fallback rate.
"""

import numpy as np
import casadi as ca
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Observation normalisation constants (must match GoalFollowerLSTMEnv)
# ---------------------------------------------------------------------------
GOAL_DIST_SCALE  = 15.0    # fixed[0]    = goal_dist   / 15.0
GOAL_BEAR_SCALE  = np.pi   # fixed[1]    = goal_bearing / pi
GOAL_VEL_SCALE   = 2.0     # fixed[2,3]  = goal_rel_v{x,y} / 2.0
OBS_DIST_SCALE   = 15.0    # obs[:,0]    = obs_dist  / 15.0
OBS_ANGLE_SCALE  = np.pi   # obs[:,1]    = obs_angle / pi
OBS_DEPTH_THRESH = 0.95    # obs[:,2] < 0.95 -> obstacle is active
OBS_VEL_SCALE    = 2.0     # obs[:,3,4]  = obs_v{x,y} / 2.0  (robot frame)


class MPCController:
    """
    Receding-horizon MPC for a differential-drive robot.

    Observation interface (GoalFollowerLSTMEnv) -- all values normalised:
      obs['fixed'] shape (14,):
        [0]  goal_dist    / 15.0
        [1]  goal_bearing / pi     (signed, robot frame)
        [2]  goal_rel_vx  / 2.0   (robot frame)
        [3]  goal_rel_vy  / 2.0
        [4]  v_cmd / v_max
        [5]  w_cmd / w_max
      obs['obstacles'] shape (MAX_OBS, 5):
        [:,0]  relative distance / 15.0
        [:,1]  relative angle    / pi    (robot frame)
        [:,2]  depth (< 0.95 -> active)
        [:,3]  rel_vx / 2.0             (robot frame, RELATIVE to robot -- see note)
        [:,4]  rel_vy / 2.0

      NOTE on obs[:,3:5] and obs['fixed'][2:4]: these are velocities of the
      obstacle/goal RELATIVE TO THE ROBOT (v_target - v_robot), not absolute
      velocities -- confirmed by goal_env_lstm's quintuplet builder, which
      computes them as `s["vx"] - robot_vx` (same convention that caused the
      earlier TTC "approach_speed" double-counting bug). See predict() for
      how this is un-done before use in the horizon prediction.

      obs['fixed'] shape is 15D as of the rear-ultrasonic addition (index 10
      is ultra_back); only indices [0:6] are read here, so the extra
      dimensions don't affect this controller.

    Action: [v_norm, w_norm] each in [-1, 1]
    """

    def __init__(
        self,
        horizon:             int   = 15,
        dt:                  float = 0.1,
        v_max:               float = 1.0,
        w_max:               float = 1.0,
        goal_weight:         float = 10.0,
        control_weight:      float = 0.1,
        terminal_weight:     float = 50.0,
        obstacle_weight:     float = 1000.0,
        safety_margin:       float = 0.6,
        max_obstacles:       int   = 10,
        obs_vel_is_relative: bool  = True,
        verbose:             bool  = False,
    ):
        self.horizon         = horizon
        self.dt              = dt
        self.v_max           = v_max
        self.w_max           = w_max
        self.goal_weight     = goal_weight
        self.control_weight  = control_weight
        self.terminal_weight = terminal_weight
        self.obstacle_weight = obstacle_weight
        self.safety_margin   = safety_margin
        self.max_obstacles   = max_obstacles
        self.verbose         = verbose

        # See predict() -- set False only if it turns out obs[:,3:5] is
        # already an absolute (world-frame) obstacle velocity, not relative
        # to the robot. Kept as a single toggle so this is a one-line revert.
        self.obs_vel_is_relative = obs_vel_is_relative

        self.prev_u: Optional[np.ndarray] = None
        self.last_solver_failed: bool = False

        self._build_optimizer()

    # ------------------------------------------------------------------
    def _build_optimizer(self):
        """Build CasADi optimisation problem (called once at construction)."""
        opti = ca.Opti()

        # Decision variables
        X = opti.variable(3, self.horizon + 1)           # [x, y, theta]
        U = opti.variable(2, self.horizon)                # [v, w]
        # Slack matrix: one scalar per (obstacle, timestep).
        # IMPORTANT: never apply a constraint to the whole matrix S at once --
        # CasADi rejects non-square matrix inequalities.  All constraints on S
        # are applied element-wise (S[i,t]) inside the loops below.
        S = opti.variable(self.max_obstacles, self.horizon)

        # Parameters
        x0       = opti.parameter(3)
        goal     = opti.parameter(2)
        obs_par  = opti.parameter(5, self.max_obstacles)  # [x,y,vx,vy,active]

        # Initial condition
        opti.subject_to(X[:, 0] == x0)

        cost = 0
        margin_sq = self.safety_margin ** 2

        for t in range(self.horizon):
            x, y, theta = X[0, t], X[1, t], X[2, t]
            v, w        = U[0, t], U[1, t]

            # Differential drive dynamics
            opti.subject_to(X[0, t+1] == x + v * ca.cos(theta) * self.dt)
            opti.subject_to(X[1, t+1] == y + v * ca.sin(theta) * self.dt)
            opti.subject_to(X[2, t+1] == theta + w * self.dt)

            # Control bounds
            opti.subject_to(opti.bounded(-self.v_max, v, self.v_max))
            opti.subject_to(opti.bounded(-self.w_max, w, self.w_max))

            # Goal tracking cost
            cost += self.goal_weight * ca.sqrt(
                (x - goal[0])**2 + (y - goal[1])**2 + 1e-6
            )

            # Control effort cost
            cost += self.control_weight * (v**2 + w**2)

            # Obstacle avoidance -- all constraints are scalar (element-wise)
            # to avoid CasADi's "matrix inequalities must be square" error.
            #
            # For each active obstacle i at timestep t we add:
            #   (a)  s_it >= 0                         [slack non-negative]
            #   (b)  dist_sq >= margin^2 - s_it - BIG_M*(1-active)
            #                                          [soft safety zone]
            #
            # BIG_M=1e6 makes constraint (b) trivially satisfied when active=0
            # so inactive obstacle slots have zero effect on the solution.
            BIG_M = 1e6
            for i in range(self.max_obstacles):
                s_it   = S[i, t]        # scalar CasADi variable -- safe for >=
                active = obs_par[4, i]  # scalar parameter (0.0 or 1.0)

                # (a) Slack non-negativity: scalar inequality, always accepted
                opti.subject_to(s_it >= 0)

                # Predicted obstacle position (constant-velocity, robot frame)
                obs_x_t = obs_par[0, i] + obs_par[2, i] * t * self.dt
                obs_y_t = obs_par[1, i] + obs_par[3, i] * t * self.dt

                dist_sq = (x - obs_x_t)**2 + (y - obs_y_t)**2

                # (b) Soft safety constraint: scalar inequality
                opti.subject_to(
                    dist_sq >= margin_sq - s_it - (1.0 - active) * BIG_M
                )

                # Cost: penalise slack heavily so avoidance is strongly preferred
                cost += self.obstacle_weight * active * s_it

        # Terminal cost
        cost += self.terminal_weight * ca.sqrt(
            (X[0, -1] - goal[0])**2 + (X[1, -1] - goal[1])**2 + 1e-6
        )

        opti.minimize(cost)

        opts = {
            'ipopt.print_level':           5 if self.verbose else 0,
            'print_time':                  self.verbose,
            'ipopt.max_iter':              150,
            'ipopt.tol':                   1e-4,
            'ipopt.acceptable_tol':        1e-3,
            'ipopt.warm_start_init_point': 'yes',
        }
        opti.solver('ipopt', opts)

        self.opti     = opti
        self.X        = X
        self.U        = U
        self.S        = S
        self.x0_par   = x0
        self.goal_par = goal
        self.obs_par  = obs_par

    # ------------------------------------------------------------------
    def predict(
        self,
        observation:   Dict[str, np.ndarray],
        state:         Optional[Tuple] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[Tuple]]:
        """
        Returns
        -------
        action : np.ndarray shape (2,) -- [v_norm, w_norm] in [-1, 1]
        state  : None  (MPC is memoryless)
        """
        fixed     = observation['fixed']
        obstacles = observation['obstacles']

        # 1. Denormalise goal -> robot frame, real units
        goal_dist    = float(fixed[0]) * GOAL_DIST_SCALE    # metres
        goal_bearing = float(fixed[1]) * GOAL_BEAR_SCALE    # radians
        goal_x = goal_dist * np.cos(goal_bearing)
        goal_y = goal_dist * np.sin(goal_bearing)

        # 2. Build obstacle parameter matrix (robot frame, real units)
        #
        # Current robot forward speed, used below to undo the "relative to
        # robot" subtraction in obs[:,3:5]. Under the unicycle/diff-drive
        # assumption the body-frame lateral velocity is zero, so only the
        # forward (x) component needs adding back.
        robot_vx = float(fixed[4]) * self.v_max

        obs_data = np.zeros((5, self.max_obstacles))
        n_obs    = min(len(obstacles), self.max_obstacles)

        for i in range(n_obs):
            obs = obstacles[i]
            if float(obs[2]) >= OBS_DEPTH_THRESH:
                continue   # inactive -- column stays zero (active=0)

            obs_dist  = float(obs[0]) * OBS_DIST_SCALE
            obs_angle = float(obs[1]) * OBS_ANGLE_SCALE
            obs_x     = obs_dist * np.cos(obs_angle)
            obs_y     = obs_dist * np.sin(obs_angle)

            # obs[3], obs[4] are (obstacle_v - robot_v) in the robot frame.
            # The horizon below is a frame FIXED at the robot's t=0 pose --
            # the robot's own future motion is already modelled by the
            # unicycle dynamics (X, U), so feeding a robot-relative velocity
            # into the obstacle's constant-velocity extrapolation would
            # double-count the robot's ego-motion (a static obstacle would
            # incorrectly appear to drift). Add the robot's current forward
            # speed back in to recover an approximate absolute velocity in
            # the fixed t=0 frame. This does not correct for any apparent
            # motion induced by the robot's angular velocity (a smaller,
            # second-order effect) -- flag if that turns out to matter.
            obs_vx = float(obs[3]) * OBS_VEL_SCALE
            obs_vy = float(obs[4]) * OBS_VEL_SCALE
            if self.obs_vel_is_relative:
                obs_vx += robot_vx

            obs_data[0, i] = obs_x
            obs_data[1, i] = obs_y
            obs_data[2, i] = obs_vx
            obs_data[3, i] = obs_vy
            obs_data[4, i] = 1.0

        # 3. Set optimisation parameters
        self.opti.set_value(self.x0_par,   [0.0, 0.0, 0.0])
        self.opti.set_value(self.goal_par, [goal_x, goal_y])
        self.opti.set_value(self.obs_par,  obs_data)

        # 4. Warm-start from previous solution
        if self.prev_u is not None:
            try:
                u_init         = np.zeros((2, self.horizon))
                u_init[:, :-1] = self.prev_u[:, 1:]
                u_init[:, -1]  = self.prev_u[:, -1]
                self.opti.set_initial(self.U, u_init)

                x_init = np.zeros((3, self.horizon + 1))
                for t in range(self.horizon):
                    th = x_init[2, t]
                    x_init[0, t+1] = x_init[0, t] + u_init[0, t] * np.cos(th) * self.dt
                    x_init[1, t+1] = x_init[1, t] + u_init[0, t] * np.sin(th) * self.dt
                    x_init[2, t+1] = th + u_init[1, t] * self.dt
                self.opti.set_initial(self.X, x_init)
                self.opti.set_initial(self.S, np.zeros((self.max_obstacles, self.horizon)))
            except Exception:
                pass

        # 5. Solve
        self.last_solver_failed = False
        try:
            sol   = self.opti.solve()
            u_opt = sol.value(self.U)
            v_opt = float(u_opt[0, 0])
            w_opt = float(u_opt[1, 0])
            self.prev_u = u_opt

        except Exception as exc:
            self.last_solver_failed = True
            self.prev_u = None
            if self.verbose:
                print(f"[MPC] Solver failed: {exc}")
            # Fallback proportional controller
            v_opt = float(np.clip(0.5 * goal_dist,    -self.v_max, self.v_max))
            w_opt = float(np.clip(2.0 * goal_bearing, -self.w_max, self.w_max))

        # 6. Normalise and clip
        action = np.clip(
            np.array([v_opt / self.v_max, w_opt / self.w_max], dtype=np.float32),
            -1.0, 1.0,
        )
        return action, None

    # ------------------------------------------------------------------
    def reset(self):
        """Call at the start of each episode to clear the warm-start cache."""
        self.prev_u             = None
        self.last_solver_failed = False