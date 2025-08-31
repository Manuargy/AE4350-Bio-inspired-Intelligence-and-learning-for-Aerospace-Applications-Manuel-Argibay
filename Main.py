import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import pandas as pd
from scipy.spatial import cKDTree
from matplotlib.patches import Circle

# =======================
# Agent: Firefly
# =======================
class Firefly:
    def __init__(self, position, phase=0.0, velocity=None):
        self.position = position  # 2D position
        self.phase = phase        # Phase [0, 1)
        self.flashed = False      # Has flashed this time step?

        if velocity is None:
            # Random initial velocity
            angle = np.random.uniform(0, 2*np.pi)
            speed = 0.01  # adjust as needed
            self.velocity = np.array([np.cos(angle), np.sin(angle)]) * speed
        else:
            self.velocity = velocity

    def move(self, bounds=(0, 1)):
        # Small random turn
        angle_change = np.random.uniform(-0.1, 0.1)
        c, s = np.cos(angle_change), np.sin(angle_change)
        rot = np.array([[c, -s], [s, c]])
        self.velocity = rot @ self.velocity
        self.position += self.velocity

        # Bounce at walls
        for dim in range(2):
            if self.position[dim] < bounds[0]:
                self.position[dim] = bounds[0]
                self.velocity[dim] *= -1
            elif self.position[dim] > bounds[1]:
                self.position[dim] = bounds[1]
                self.velocity[dim] *= -1

    def update_phase(self, dt, threshold):
        self.phase += dt
        if self.phase >= threshold:
            self.phase -= threshold  # Reset after flashing
            self.flashed = True
        else:
            self.flashed = False

    def receive_flash(self, phase_increment, threshold):
        self.phase += phase_increment
        if self.phase >= threshold:
            self.phase -= threshold
            self.flashed = True
        else:
            self.flashed = False

    def detect_object(self, objects, detection_radius=0.05):
        for obj_pos in objects:
            if np.linalg.norm(self.position - obj_pos) < detection_radius:
                return True
        return False



# =======================
# Swarm of Fireflies
# =======================
class FireflySwarm:
    def __init__(self, num_agents, interaction_radius, dt=0.05, threshold=2, phase_increment=0.02):
        """
        Initialize the FireflySwarm.

        Parameters:
        num_agents (int): Number of fireflies in the swarm.
        interaction_radius (float): Radius within which fireflies interact.
        dt (float): Time step for phase updates.
        threshold (float): Phase threshold for flashing.
        phase_increment (float): Increment added to phase upon receiving a flash.
        """
        self.N = num_agents
        self.radius = interaction_radius
        self.dt = dt
        self.threshold = threshold
        self.phase_increment = phase_increment
        # Ensure random initial phases
        self.agents = [Firefly(position=np.random.rand(2), phase=np.random.uniform(0, self.threshold)) for _ in range(num_agents)]

    def step(self, objects=None):
        # Move all fireflies
        for firefly in self.agents:
            firefly.move()
            if objects is not None:
                if firefly.detect_object(objects):
                    firefly.found_object = True
                    firefly.phase = self.threshold  # Force flash

        # Update phases for all agents and record who flashed
        for firefly in self.agents:
            firefly.update_phase(self.dt, self.threshold)

        # Build k-d tree for current positions
        positions = np.array([f.position for f in self.agents])
        tree = cKDTree(positions)

        # 2. Iteratively apply local interactions until no new flashes
        flashed_this_round = True
        already_flashed = [f.flashed for f in self.agents]
        while flashed_this_round:
            flashed_this_round = False
            for i, f_i in enumerate(self.agents):
                if already_flashed[i]:
                    # Query all neighbors within radius (excluding self)
                    neighbors = tree.query_ball_point(f_i.position, self.radius)
                    for j in neighbors:
                        if i == j or already_flashed[j]:
                            continue
                        f_j = self.agents[j]
                        prev_phase = f_j.phase
                        f_j.receive_flash(self.phase_increment, self.threshold)
                        if f_j.flashed and not already_flashed[j]:
                            already_flashed[j] = True
                            flashed_this_round = True
        # Ensure all .flashed attributes are up to date
        for idx, f in enumerate(self.agents):
            f.flashed = already_flashed[idx]

    def handle_interactions(self):
        """
        Each firefly that flashed this timestep influences neighbors within the interaction radius.
        Neighbors adjust their phase if they did not flash already.
        """
        for i, f_i in enumerate(self.agents):
            if f_i.flashed:
                for j, f_j in enumerate(self.agents):
                    if i == j:
                        continue
                    distance = np.linalg.norm(f_i.position - f_j.position)
                    if distance <= self.radius and not f_j.flashed:
                        f_j.receive_flash(self.phase_increment, self.threshold)

    def get_flash_states(self):
        return [f.flashed for f in self.agents]

    def get_positions(self):
        return [f.position for f in self.agents]



# =======================
# Simulation Parameters
# =======================
NUM_AGENTS = 25
TIME_STEPS = 300
INTERACTION_RADIUS = 0.2


# Create swarm
swarm = FireflySwarm(num_agents=NUM_AGENTS, interaction_radius=INTERACTION_RADIUS)

# Metric tracking
phase_std_history = []
flash_count_history = []


# =======================
# Visualization with Matplotlib
# =======================
fig, ax = plt.subplots(figsize=(10, 10))  # Increased from (6, 6) to (10, 10)
positions = np.array(swarm.get_positions())
sc = ax.scatter(positions[:, 0], positions[:, 1], s=100, c='black')
title = ax.set_title("Firefly Swarm\nTime Step: 0")
title.set_animated(True)

# After creating the scatter plot
circle_patches = []
for pos in positions:
    circle = Circle(
        pos, INTERACTION_RADIUS,
        edgecolor='blue', facecolor='none', lw=1.2, ls='--', alpha=0.7
    )
    ax.add_patch(circle)
    circle_patches.append(circle)

ax.set_xlim(-INTERACTION_RADIUS, 1 + INTERACTION_RADIUS)
ax.set_ylim(-INTERACTION_RADIUS, 1 + INTERACTION_RADIUS)

def animate(frame):
    phases = [f.phase for f in swarm.agents]
    phase_std_history.append(np.std(phases))

    swarm.step()

    flashes = swarm.get_flash_states()
    flash_count_history.append(sum(flashes))

    positions = np.array(swarm.get_positions())
    colors = ['yellow' if flash else 'black' for flash in flashes]
    sc.set_offsets(positions)
    sc.set_color(colors)
    title.set_text(f"Firefly Swarm\nTime Step: {frame}")

    # Update circle positions
    for circle, pos in zip(circle_patches, positions):
        circle.center = pos

    return [sc, title] + circle_patches

ani = animation.FuncAnimation(fig, animate, frames=TIME_STEPS, interval=80, blit=True)
plt.show()

# =======================
# Plotting Synchronization Metrics
# =======================
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(phase_std_history, label='Phase Std. Dev.')
plt.xlabel('Time step')
plt.ylabel('Standard Deviation')
plt.title('Phase Synchronization Over Time')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(flash_count_history, label='Number of Flashes', color='orange')
plt.xlabel('Time step')
plt.ylabel('Number of Flashes')
plt.title('Flashes Per Time Step')
plt.grid(True)

plt.tight_layout()

# Create Figures directory if it doesn't exist
figures_dir = "Figures"
os.makedirs(figures_dir, exist_ok=True)
plt.savefig(os.path.join(figures_dir, "synchronization_metrics.png"))

plt.show()

# =======================
# Display Simulation Parameters Table
# =======================
initial_speed = np.linalg.norm(swarm.agents[0].velocity)

params = {
    "Parameter": [
        "Number of Agents",
        "Time Steps",
        "Interaction Radius",
        "Phase Threshold",
        "Phase Increment",
        "Phase dt",
        "Initial Speed",
        "Boundary Type"
    ],
    "Value": [
        NUM_AGENTS,
        TIME_STEPS,
        INTERACTION_RADIUS,
        swarm.threshold,
        swarm.phase_increment,
        swarm.dt,
        initial_speed,
        "Bouncing"
    ]
}

param_table = pd.DataFrame(params)
print("\nSimulation Parameters:\n")
print(param_table.to_string(index=False))
param_table.to_csv(os.path.join(figures_dir, "simulation_parameters.csv"), index=False)

OBJECTS = [np.array([0.7, 0.3]), np.array([0.2, 0.8])]  # Example object positions



