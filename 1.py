import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def system_equations(t, state, m, q, B0, omega):
    """
    微分方程组
    state = [x, y, vx, vy]
    """
    x, y, vx, vy = state
    
    cos_omega_t = np.cos(omega * t)
    sin_omega_t = np.sin(omega * t)
    
    dvx_dt = (-y * q * B0 * omega * cos_omega_t / 2 - q * B0 * vy * sin_omega_t) / m
    dvy_dt = (x * q * B0 * omega * cos_omega_t / 2 + q * B0 * vx * sin_omega_t) / m
    
    return [vx, vy, dvx_dt, dvy_dt]

def solve_system(m, q, B0, omega, r0, t_span=(0, 10), num_points=5000):
    """求解微分方程组"""
    initial_state = [r0, 0.0, 0.0, 0.0]  # [x0, y0, vx0, vy0]
    t_eval = np.linspace(t_span[0], t_span[1], num_points)
    
    solution = solve_ivp(
        system_equations,
        t_span,
        initial_state,
        args=(m, q, B0, omega),
        method='RK45',
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10
    )
    
    return solution

# 参数设置
m = 1.0      # 质量
q = 1.0      # 电荷
B0 = 1000    # 磁场强度
omega = 2.0  # 角频率
r0 = 0.01     # 初始位置

# 求解
solution = solve_system(m, q, B0, omega, r0)

# 提取结果
t = solution.t
x = solution.y[0]
y = solution.y[1]
vx = solution.y[2]
vy = solution.y[3]
speed = np.sqrt(vx**2 + vy**2)

# 绘制双图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 左图：速度-时间图
ax1.plot(t, speed, 'b-', linewidth=2)
ax1.set_xlabel('Time t', fontsize=12)
ax1.set_ylabel(r'$\sqrt{(dx/dt)^2 + (dy/dt)^2}$', fontsize=14)
ax1.set_title('Speed vs Time', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 10)

# 右图：粒子轨迹图
ax2.plot(x, y, 'r-', linewidth=1.5)
ax2.plot(x[0], y[0], 'go', markersize=8, label='Start')
ax2.plot(x[-1], y[-1], 'bo', markersize=8, label='End')
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('y', fontsize=12)
ax2.set_title('Particle Trajectory', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.axis('equal')
ax2.legend()

plt.tight_layout()

# 保存为PNG图片
plt.savefig('particle_motion.png', dpi=300, bbox_inches='tight')
print("图片已保存为 particle_motion.png")

# 关闭图形（释放内存）
plt.close()
