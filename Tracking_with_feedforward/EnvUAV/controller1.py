import torch
import torch.nn as nn
import numpy as np
import yaml
import pybullet as p


class Controller:
    def __init__(self, path, time_step=0.01):
        self.path = path
        with open(self.path + '/File/uva.yaml', 'r', encoding='utf-8') as F:
            param_dict = yaml.load(F, Loader=yaml.FullLoader)
        self.M = param_dict['M']
        self.G = param_dict['G']
        self.L = param_dict['L']
        self.J = param_dict['J']
        self.J_xx = self.J[0][0]
        self.J_yy = self.J[1][1]
        self.J_zz = self.J[2][2]
        self.acc_bound = 5
        self.ang_acc_bound = 20
        self.time_step = time_step

        self.actor_xy = DeterministicActor(s_dim=6, a_dim=1, hidden=32)
        self.actor_xy.load_state_dict(torch.load(path + '/XYActor.pth'))
        self.actor_z = DeterministicActor(s_dim=2, a_dim=1, hidden=32)
        self.actor_z.load_state_dict(torch.load(path + '/ZActor.pth'))
        self.actor_ang = DeterministicActor(s_dim=2, a_dim=1, hidden=32)
        self.actor_ang.load_state_dict(torch.load(path + '/AttitudeActor.pth'))

        self.last_p_d = self.current_p_d = None
        self.last_v_d = self.current_v_d = None
        self.last_acc_d = self.current_acc_d = None
        self.last_Rz_d = self.current_Rz_d = None
        self.last_R_tilde = self.current_R_tilde = None

    def reset(self):
        self.last_p_d = self.current_p_d = np.array([0., 0., 0.])
        self.last_v_d = self.current_v_d = np.array([0., 0., 0.])
        self.last_acc_d = self.current_acc_d = np.array([0., 0., 0.])
        self.last_Rz_d = self.current_Rz_d = np.array([0., 0., 1.])

    def step(self, time_index, target, pos, vel, last_R, current_R, ang_v):
        # update desired state
        self.last_p_d = self.current_p_d
        self.current_p_d = target[0:3]
        self.last_v_d = self.current_v_d
        self.current_v_d = (self.current_p_d - self.last_p_d) / self.time_step
        self.last_acc_d = self.current_acc_d
        self.current_acc_d = (self.current_v_d - self.last_v_d) / self.time_step
        self.last_Rz_d = self.current_Rz_d
        self.current_Rz_d = self.current_acc_d + np.array([0., 0., self.G])
        self.current_Rz_d = self.current_Rz_d / np.linalg.norm(self.current_Rz_d)
        self.last_R_tilde = self.current_R_tilde

        feedforward = time_index > 4
        # feedforward = False
        sx, sy, sz = self._get_pos_s(feedforward, pos, vel, last_R, current_R)
        # print(time_index, sx)
        with torch.no_grad():
            ax = self.actor_xy(torch.tensor(sx, dtype=torch.float)).item()
            print(time_index, sx, ax)
            ay = self.actor_xy(torch.tensor(sy, dtype=torch.float)).item()
            az = self.actor_z(torch.tensor(sz, dtype=torch.float)).item()

        acc = np.clip(self.acc_bound * np.array([ax, ay, az]) + (self.current_acc_d if feedforward else np.zeros(3)),
                      -self.acc_bound, self.acc_bound)
        fx, fy, fz = self.M * (acc + np.array([0, 0, self.G]))

        yaw = target[3]
        roll = np.arcsin((np.sin(yaw) * fx - np.cos(yaw) * fy) / np.linalg.norm([fx, fy, fz]))
        pitch = np.arctan((np.cos(yaw) * fx + np.sin(yaw) * fy) / fz)
        f = fz / current_R[2, 2]

        sroll, spitch, syaw = self._get_attitude_s(current_R, ang_v, np.array([roll, pitch, yaw]))

        fx_tilde = self.M * self.current_acc_d[0]
        fy_tilde = self.M * self.current_acc_d[1]
        fz_tilde = self.M * self.current_acc_d[2]
        roll_tilde = np.arcsin((np.sin(yaw) * fx_tilde - np.cos(yaw) * fy_tilde) / np.linalg.norm([fx_tilde, fy_tilde, fz_tilde]))
        pitch_tilde = np.arctan((np.cos(yaw) * fx_tilde + np.sin(yaw) * fy_tilde / fz_tilde))
        combined_quaternion = p.getQuaternionFromEuler([self.roll_tilde, pitch_tilde, yaw])
        self.current_R_tilde = p.getMatrixFromQuaternion(combined_quaternion)
        R_tilde_dot = (self.current_R_tilde - self.last_R_tilde) / self.time_step
        ang_vel_d_hat = np.matmul(self.current_R_tilde.T, R_tilde_dot)
        self.current_ang_v_d_tilde = np.array([ang_vel_d_hat[2, 1] - ang_vel_d_hat[1, 2],
                                         ang_vel_d_hat[0, 2] - ang_vel_d_hat[2, 0],
                                         ang_vel_d_hat[1, 0] - ang_vel_d_hat[1, 0]]) / 2
        ang_acc_d_tilde = (self.current_ang_v_d - self.last_ang_v_d) / self.time_step
        ang_acc_d_tilde = np.matmul(self.current_R, np.matmul(self.current_R_tilde.T, ang_acc_d_tilde))

        with torch.no_grad():
            tau_roll = (self.actor_ang(torch.tensor(sroll, dtype=torch.float)).item() + ang_acc_d_tilde[0]) * self.ang_acc_bound * self.J_xx
            tau_pitch = (self.actor_ang(torch.tensor(spitch, dtype=torch.float)).item() + ang_acc_d_tilde[1]) * self.ang_acc_bound * self.J_yy
            tau_yaw = (self.actor_ang(torch.tensor(syaw, dtype=torch.float)).item() + ang_acc_d_tilde[2]) * self.ang_acc_bound * self.J_zz

        return np.array([f, tau_roll, tau_pitch, tau_yaw])

    def _get_attitude_s(self, R, ang_v, ang_d):
        R_d = np.reshape(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(ang_d)), [3, 3])
        e_R = (np.matmul(R_d.T, R) - np.matmul(R.T, R_d)) / 2
        e = [e_R[1, 2], e_R[2, 0], e_R[0, 1]]
        s1, s2, s3 = [e[0], ang_v[0]], [e[1], ang_v[1]], [e[2], ang_v[2]]
        return s1, s2, s3

    def _get_pos_s(self, feedforward, pos, vel, last_R, current_R):
        ex, ey, ez = pos - self.current_p_d
        evx, evy, evz = vel - (self.current_v_d if feedforward else np.zeros(3))
        roll_ = np.arctan(current_R[1, 2] / current_R[2, 2])
        pitch_ = np.arctan(current_R[0, 2] / current_R[2, 2])
        roll_last = np.arctan(last_R[1, 2] / last_R[2, 2])
        pitch_last = np.arctan(last_R[0, 2] / last_R[2, 2])
        roll_v_ = (roll_ - roll_last) / self.time_step
        pitch_v_ = (pitch_ - pitch_last) / self.time_step

        if feedforward:
            roll_d = np.arctan(self.current_Rz_d[1] / self.current_Rz_d[2])
            pitch_d = np.arctan(self.current_Rz_d[0] / self.current_Rz_d[2])
            roll_last_d = np.arctan(self.last_Rz_d[1] / self.last_Rz_d[2])
            pitch_last_d = np.arctan(self.last_Rz_d[0] / self.last_Rz_d[2])
            roll_v_d = (roll_d - roll_last_d) / self.time_step
            pitch_v_d = (pitch_d - pitch_last_d) / self.time_step

            roll_ -= roll_d
            pitch_ -= pitch_d
            roll_v_ -= roll_v_d
            pitch_v_ -= pitch_v_d

        sx = np.array([ex, evx, np.sign(ex) * ez, np.sign(ex) * evz, pitch_, pitch_v_]) / 3
        sy = np.array([ey, evy, np.sign(ey) * ez, np.sign(ey) * evz, roll_, roll_v_]) / 3
        sz = np.array([ez, evz])
        return sx, sy, sz


class DeterministicActor(nn.Module):
    def __init__(self, s_dim,  a_dim, hidden):
        super(DeterministicActor, self).__init__()
        self.actor = nn.Sequential(nn.Linear(s_dim, hidden, bias=False),
                                   nn.Tanh(),
                                   nn.Linear(hidden, hidden, bias=False),
                                   nn.Tanh(),
                                   nn.Linear(hidden, a_dim, bias=False),
                                   nn.Tanh())

    def forward(self, s):
        return self.actor(s)
    