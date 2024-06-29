import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

G = 1 # gravity constant
m1 = 1
m2 = 1
m3 = 1

def rk4(f, y_prev, h):
    k1 = h * f(y_prev)
    k2 = h * f(y_prev + 0.5 * k1)
    k3 = h * f(y_prev + 0.5 * k2)
    k4 = h * f(y_prev + k3)
    return y_prev + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    
def rk4_3_8(f, y_prev, h):
    k1 = f(y_prev)
    k2 = f(y_prev + h/3 * k1)
    k3 = f(y_prev - h/3 * k1 + h*k2)
    k4 = f(y_prev + h * k1 - h * k2 + h * k3)

    return y_prev + h/8 * (k1 + 3*k2 + 3*k3 + k4)

def equations_of_motion(y):
    r1, r2, r3 = y[0:2], y[2:4], y[4:6]
    v1, v2, v3 = y[6:8], y[8:10], y[10:12]

    r12 = np.linalg.norm(r2 - r1)
    r13 = np.linalg.norm(r3 - r1)
    r23 = np.linalg.norm(r3 - r2)

    r1_pp = G * (r2 - r1) / r12**3 + G * (r3 - r1) / r13**3
    r2_pp = G * (r1 - r2) / r12**3 + G * (r3 - r2) / r23**3
    r3_pp = G * (r1 - r3) / r13**3 + G * (r2 - r3) / r23**3

    return np.concatenate((v1, v2, v3, r1_pp, r2_pp, r3_pp))
    
def equations_of_motion_softened(y):
    r1, r2, r3 = y[0:2], y[2:4], y[4:6]
    v1, v2, v3 = y[6:8], y[8:10], y[10:12]

    softening = 0.1

    r12 = r2 - r1
    r13 = r3 - r1
    r23 = r3 - r2

    # Accelerations
    r1_pp = G * (m2 * r12 / (np.linalg.norm(r12)**2 + softening**2)**1.5 + 
                 m3 * r13 / (np.linalg.norm(r13)**2 + softening**2)**1.5)
    r2_pp = G * (m3 * (-r12) / (np.linalg.norm(r12)**2 + softening**2)**1.5 + 
                 m1 * r23 / (np.linalg.norm(r23)**2 + softening**2)**1.5)
    r3_pp = G * (m1 * (-r13) / (np.linalg.norm(r13)**2 + softening**2)**1.5 - 
                 m2 * r23 / (np.linalg.norm(r23)**2 + softening**2)**1.5)

    return np.concatenate((v1, v2, v3, r1_pp, r2_pp, r3_pp))
    
def calculate_energy(y):
    r1, r2, r3 = y[0:2], y[2:4], y[4:6]
    v1, v2, v3 = y[6:8], y[8:10], y[10:12]

    # Kinetic energy
    KE = 0.5 * (m1 * np.dot(v1, v1) + m2 * np.dot(v2, v2) + m3 * np.dot(v3, v3))

    # Potential energy
    r12 = np.linalg.norm(r2 - r1)
    r13 = np.linalg.norm(r3 - r1)
    r23 = np.linalg.norm(r3 - r2)
    PE = -G * (m1 * m2 / r12 + m1 * m3 / r13 + m2 * m3 / r23)

    return KE + PE

def get_initial_conditions():
        
    print('0: custom input')
    print('1-7: examples')
    
    choice = input()

    if choice == '0':
        print("Enter initial positions and velocities for each body:")
        r1_0 = np.array([float(input("Body 1 x position: ")), float(input("Body 1 y position: "))])
        r2_0 = np.array([float(input("Body 2 x position: ")), float(input("Body 2 y position: "))])
        r3_0 = np.array([float(input("Body 3 x position: ")), float(input("Body 3 y position: "))])
        v1_0 = np.array([float(input("Body 1 x velocity: ")), float(input("Body 1 y velocity: "))])
        v2_0 = np.array([float(input("Body 2 x velocity: ")), float(input("Body 2 y velocity: "))])
        v3_0 = np.array([float(input("Body 3 x velocity: ")), float(input("Body 3 y velocity: "))])
    
    elif choice == '1':
        r1_0 = np.array([0.97000436, -0.24308753])
        r2_0 = np.array([-0.97000436, 0.24308753])
        r3_0 = np.array([0, 0])
        v1_0 = np.array([0.466203685, 0.43236573])
        v2_0 = np.array([0.466203685, 0.43236573])
        v3_0 = np.array([-0.93240737, -0.86473146])
        
    elif choice == '2':
        r1_0 = np.array([1, 0])
        r2_0 = np.array([-0.5, 0.866])
        r3_0 = np.array([-0.5, -0.866])
        v1_0 = np.array([0, 0.3])
        v2_0 = np.array([-0.259808, -0.15])
        v3_0 = np.array([0.259808, -0.15])
                
    elif choice == '3':
        r1_0 = np.array([1, 0])
        r2_0 = np.array([-0.5, 0.866])
        r3_0 = np.array([-0.5, -0.866])
        v1_0 = np.array([0, 0.4])
        v2_0 = np.array([-0.346, -0.2])
        v3_0 = np.array([0.346, -0.2])
        
    elif choice == '4':
        r1_0 = np.array([0, 1])
        r2_0 = np.array([-0.866, -0.5])
        r3_0 = np.array([0.866, -0.5])
        v1_0 = np.array([-0.5, 0])
        v2_0 = np.array([0.25, 0.433])
        v3_0 = np.array([0.25, -0.433])
        
    elif choice == '5':
        r1_0 = np.array([1, 0])
        r2_0 = np.array([-0.5, 0.866])
        r3_0 = np.array([-0.5, -0.866])
        v1_0 = np.array([0, 0.1])
        v2_0 = np.array([-0.0866, -0.05])
        v3_0 = np.array([0.0866, -0.05])
        
    elif choice == '6':
        r1_0 = np.array([-0.5, 0])
        r2_0 = np.array([0.5, 0])
        r3_0 = np.array([0, 1.5])
        v1_0 = np.array([0, -0.35])
        v2_0 = np.array([0, 0.35])
        v3_0 = np.array([-0.6, 0])
        
    elif choice == '7':
        r1_0 = np.array([-1, 0])
        r2_0 = np.array([1, 0])
        r3_0 = np.array([0, 0])
        v1_0 = np.array([0.3471128135, 0.532726851])
        v2_0 = np.array([0.3471128135, 0.532726851])
        v3_0 = np.array([-0.694225627, -1.065453702])
        
    else:
        print('Try again')
        return get_initial_conditions()

    return np.concatenate((r1_0, r2_0, r3_0, 
                               v1_0, v2_0, v3_0))

y0 = get_initial_conditions()

# Time steps
t0 = 0.0
tmax = 200.0
dt = 0.02
t = np.arange(t0, tmax + dt, dt)

# Calculate all positions, velocities, and energies
y = np.zeros((len(t), 12))
energy = np.zeros(len(t))
y[0] = y0
energy[0] = calculate_energy(y0)

for i in range(1, len(t)):
    y[i] = rk4_3_8(equations_of_motion_softened, y[i-1], dt)
    energy[i] = calculate_energy(y[i])

# Extract positions
r1 = y[:, 0:2]
r2 = y[:, 2:4]
r3 = y[:, 4:6]

fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
plt.tight_layout(pad=0)
ax.set_position([0, 0, 1, 1])

lines = []
for i in range(3):
    line, = ax.plot([], [], '-', linewidth=1)
    lines.append(line)

points = []
for i in range(3):
    point, = ax.plot([], [], 'o', markersize=8)
    points.append(point)
    
ax.set_title('Three-Body Problem')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xticks([])
ax.set_yticks([])
ax.grid(True)

values_label = ax.text(0.02, 0.98, '', transform=ax.transAxes, animated=True, color='white', verticalalignment='top')

def update(frame):
    for i, line in enumerate(lines):
        if i == 0:
            line.set_data(r1[:frame, 0], r1[:frame, 1])
            line.set_color('r')
        elif i == 1:
            line.set_data(r2[:frame, 0], r2[:frame, 1])
            line.set_color('g')
        else:
            line.set_data(r3[:frame, 0], r3[:frame, 1])
            line.set_color('b')
    
    # Update points for each body
    points[0].set_data(r1[frame, 0], r1[frame, 1])
    points[1].set_data(r2[frame, 0], r2[frame, 1])
    points[2].set_data(r3[frame, 0], r3[frame, 1])
    
    max_pos = max(np.max(np.abs(r1[:frame+1])), np.max(np.abs(r2[:frame+1])), np.max(np.abs(r3[:frame+1])))
    ax.set_xlim(-max_pos*1.1, max_pos*1.1)
    ax.set_ylim(-max_pos*1.1, max_pos*1.1)

    text = f'Total Energy: {energy[frame]:.12f}\n\n'
    
    for i, (r, v) in enumerate([(r1[frame], y[frame, 6:8]), 
                                (r2[frame], y[frame, 8:10]), 
                                (r3[frame], y[frame, 10:12])]):
        pos_str = f'{r[0]:.2f}, {r[1]:.2f}'
        vel_str = f'{v[0]:.2f}, {v[1]:.2f}'
        text += f'Body {i+1} - pos: ({pos_str}) vel: ({vel_str})\n'
    
    
    text = text.rstrip()    
    values_label.set_text(text)
    
    z_order = [r1[frame, 1], r2[frame, 1], r3[frame, 1]]
    indices = np.argsort(z_order)
    
    for i, idx in enumerate(indices):
        lines[idx].set_zorder(i)
        points[idx].set_zorder(i + 3)
    
    return lines + points + [values_label]

ani = animation.FuncAnimation(fig, update, frames=len(t), interval=50, blit=True)
ax.add_artist(values_label)
plt.show()
