import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import pandas as pd
from scipy.spatial import cKDTree

# =======================
# Agent: Firefly
# =======================
class Firefly:
    def __init__(self, position, phase=0.0, velocity=None):
        self.position = position  # 2D position
        self.phase = phase        # Phase [0, 1)
        self.flashed = False      # Has flashed this time step?
        self.found_object = False # Has found (or been informed about) an object?
        self.target_object_idx = None # Index of the object being targeted/carried
        self.waiting_to_carry = False # Waiting to pick up an object?
        self.carrying = False  # Is this firefly carrying the object?

        if velocity is None:
            angle = np.random.uniform(0, 2*np.pi)
            speed = 0.01
            self.velocity = np.array([np.cos(angle), np.sin(angle)]) * speed
        else:
            self.velocity = velocity

    def move(self, bounds=(0, 1), objects=None, base_pos=None, object_states=None):
        # If carrying, move toward base
        if self.carrying and self.target_object_idx is not None and base_pos is not None:
            direction = base_pos - self.position
            distance = np.linalg.norm(direction)
            if distance > 0:
                direction = direction / distance
                speed = 0.008
                self.velocity = direction * speed
        elif self.waiting_to_carry and self.target_object_idx is not None:
            # Circle around the object
            obj_pos = objects[self.target_object_idx]["pos"]
            rel = self.position - obj_pos
            angle = np.arctan2(rel[1], rel[0]) + 0.15
            radius = 0.04 + 0.01 * np.random.randn()
            self.position = obj_pos + radius * np.array([np.cos(angle), np.sin(angle)])
            self.velocity = np.zeros(2)
        else:
            # If firefly has found an object, bias movement toward it
            if self.found_object and self.target_object_idx is not None:
                obj_pos = objects[self.target_object_idx]["pos"]
                direction = obj_pos - self.position
                distance = np.linalg.norm(direction)
                if distance > 0:
                    direction = direction / distance
                    angle_noise = np.random.uniform(-np.pi/8, np.pi/8)
                    c, s = np.cos(angle_noise), np.sin(angle_noise)
                    rot = np.array([[c, -s], [s, c]])
                    direction = rot @ direction
                    speed = 0.01 * (0.5 + 0.5 * np.clip(distance / 0.1, 0, 1))
                    self.velocity = direction * speed
            else:
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
            self.phase -= threshold
            self.flashed = True
        else:
            self.flashed = False

    def receive_flash(self, phase_increment, threshold, neighbor_found_object=False, neighbor_object_idx=None):
        self.phase += phase_increment
        if neighbor_found_object and neighbor_object_idx is not None:
            self.found_object = True
            self.target_object_idx = neighbor_object_idx
        if self.phase >= threshold:
            self.phase -= threshold
            self.flashed = True
        else:
            self.flashed = False

    def detect_object(self, objects, detection_radius=0.05):
        for idx, obj in enumerate(objects):
            if obj["state"] == "search" and np.linalg.norm(self.position - obj["pos"]) < detection_radius:
                self.target_object_idx = idx
                return idx
        return None

# =======================
# Swarm of Fireflies
# =======================
class FireflySwarm:
    def __init__(self, num_agents, interaction_radius, dt=0.05, threshold=2, phase_increment=0.02):
        self.N = num_agents
        self.radius = interaction_radius
        self.dt = dt
        self.threshold = threshold
        self.phase_increment = phase_increment
        self.agents = [Firefly(position=np.random.rand(2), phase=np.random.uniform(0, self.threshold)) for _ in range(num_agents)]

    def step(self, objects=None, base_pos=None, min_carriers=10):
        # Move all fireflies and handle detection/recruitment
        for firefly in self.agents:
            firefly.move(bounds=(ACTION_AREA_MIN, ACTION_AREA_MAX), objects=objects, base_pos=base_pos)
            if objects is not None:
                idx = firefly.detect_object(objects)
                MAX_CARRIERS_PER_OBJECT = 11  # (or any number you want)

                if idx is not None and len(objects[idx]["carriers"]) < MAX_CARRIERS_PER_OBJECT:
                    firefly.found_object = True
                    firefly.waiting_to_carry = True
                    objects[idx]["carriers"].add(firefly)
                    firefly.phase = self.threshold  # Force flash

        # For each object, check if enough carriers are present
        for idx, obj in enumerate(objects):
            if obj["state"] == "search" and len(obj["carriers"]) >= min_carriers:
                obj["state"] = "carrying"
                for f in obj["carriers"]:
                    f.carrying = True
                    f.waiting_to_carry = False

            # If carrying, move object with carriers
            if obj["state"] == "carrying":
                carriers = [f for f in obj["carriers"] if f.carrying]
                if carriers:
                    positions = np.array([f.position for f in carriers])
                    obj["pos"] = positions.mean(axis=0)
                    # If at base, drop object and reset
                    if np.linalg.norm(obj["pos"] - base_pos) < 0.05:
                        obj["state"] = "delivered"
                        for f in self.agents:
                            if f.target_object_idx == idx:
                                f.carrying = False
                                f.found_object = False
                                f.waiting_to_carry = False
                                f.target_object_idx = None
                        obj["carriers"].clear()

        # Update phases for all agents
        for firefly in self.agents:
            firefly.update_phase(self.dt, self.threshold)

        # Build k-d tree for current positions
        positions = np.array([f.position for f in self.agents])
        tree = cKDTree(positions)

        # Flash propagation (propagate object info)
        flashed_this_round = True
        already_flashed = [f.flashed for f in self.agents]
        MAX_CARRIERS_PER_OBJECT = 11  # (or any number you want)
        while flashed_this_round:
            flashed_this_round = False
            for i, f_i in enumerate(self.agents):
                if already_flashed[i]:
                    neighbors = tree.query_ball_point(f_i.position, self.radius)
                    for j in neighbors:
                        if i == j or already_flashed[j]:
                            continue
                        f_j = self.agents[j]
                        # Only propagate if the carrier set is NOT full
                        propagate_found = True
                        if (
                            f_i.found_object and
                            f_i.target_object_idx is not None and
                            (
                                len(objects[f_i.target_object_idx]["carriers"]) >= MAX_CARRIERS_PER_OBJECT or
                                objects[f_i.target_object_idx]["state"] == "delivered"
                            )
                        ):
                            propagate_found = False

                        f_j.receive_flash(
                            self.phase_increment,
                            self.threshold,
                            neighbor_found_object=f_i.found_object if propagate_found else False,
                            neighbor_object_idx=f_i.target_object_idx if propagate_found else None
                        )
                        if propagate_found and f_i.found_object and f_i.target_object_idx is not None:
                            f_j.found_object = True
                            f_j.target_object_idx = f_i.target_object_idx
                        if f_j.flashed and not already_flashed[j]:
                            already_flashed[j] = True
                            flashed_this_round = True
        for idx, f in enumerate(self.agents):
            f.flashed = already_flashed[idx]

    def get_flash_states(self):
        return [f.flashed for f in self.agents]

    def get_positions(self):
        return [f.position for f in self.agents]

    def get_found_states(self):
        return [f.found_object for f in self.agents]

# =======================
# Simulation Parameters
# =======================
NUM_AGENTS = 100
TIME_STEPS = 300
INTERACTION_RADIUS = 0.2

ACTION_AREA_MIN = 0.0
ACTION_AREA_MAX = 2.0

OBJECTS = [
    {"pos": np.array([0.7, 0.3]), "state": "search", "carriers": set()},
    {"pos": np.array([1.8, 0.2]), "state": "search", "carriers": set()},
    {"pos": np.array([0.3, 1.7]), "state": "search", "carriers": set()},
    {"pos": np.array([1.5, 1.5]), "state": "search", "carriers": set()},
    {"pos": np.array([0.5, 1.2]), "state": "search", "carriers": set()}
]
BASE_POSITION = np.array([0.2, 1.8])

swarm = FireflySwarm(num_agents=NUM_AGENTS, interaction_radius=INTERACTION_RADIUS)

phase_std_history = []
flash_count_history = []
found_count_history = []

# =======================
# Visualization with Matplotlib
# =======================
fig, ax = plt.subplots(figsize=(10, 10))
positions = np.array(swarm.get_positions())
sc = ax.scatter(positions[:, 0], positions[:, 1], s=100, c='black')
title = ax.set_title("Firefly Swarm\nTime Step: 0")
title.set_animated(True)

object_scatters = []
for obj in OBJECTS:
    obj_scatter = ax.scatter(obj["pos"][0], obj["pos"][1], s=200, c='red', marker='*', label='Object')
    object_scatters.append(obj_scatter)
ax.scatter([BASE_POSITION[0]], [BASE_POSITION[1]], s=200, c='blue', marker='s', label='Base')

ax.set_xlim(ACTION_AREA_MIN - INTERACTION_RADIUS, ACTION_AREA_MAX + INTERACTION_RADIUS)
ax.set_ylim(ACTION_AREA_MIN - INTERACTION_RADIUS, ACTION_AREA_MAX + INTERACTION_RADIUS)
ax.legend(loc='upper right')

def animate(frame):
    phases = [f.phase for f in swarm.agents]
    phase_std_history.append(np.std(phases))

    swarm.step(objects=OBJECTS, base_pos=BASE_POSITION)

    flashes = swarm.get_flash_states()
    founds = swarm.get_found_states()
    flash_count_history.append(sum(flashes))
    found_count_history.append(sum(founds))

    positions = np.array(swarm.get_positions())
    colors = []
    for firefly, flash in zip(swarm.agents, flashes):
        if firefly.found_object:
            colors.append('green')
        elif flash:
            colors.append('yellow')
        else:
            colors.append('black')
    sc.set_offsets(positions)
    sc.set_color(colors)
    title.set_text(f"Firefly Swarm\nTime Step: {frame}")

    # Update all object markers
    for idx, obj in enumerate(OBJECTS):
        if obj["state"] == "delivered":
            object_scatters[idx].set_offsets([BASE_POSITION])
        else:
            object_scatters[idx].set_offsets([obj["pos"]])

    return [sc, title] + object_scatters

ani = animation.FuncAnimation(fig, animate, frames=TIME_STEPS, interval=80, blit=True)
plt.show()

# =======================
# Plotting Synchronization & Object Finding Metrics
# =======================
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(phase_std_history, label='Phase Std. Dev.')
plt.xlabel('Time step')
plt.ylabel('Standard Deviation')
plt.title('Phase Synchronization Over Time')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(flash_count_history, label='Number of Flashes', color='orange')
plt.xlabel('Time step')
plt.ylabel('Number of Flashes')
plt.title('Flashes Per Time Step')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(found_count_history, label='Found Object', color='green')
plt.xlabel('Time step')
plt.ylabel('Fireflies Informed')
plt.title('Object Localization Spread')
plt.grid(True)

plt.tight_layout()

# Create Figures directory if it doesn't exist
figures_dir = "Figures"
os.makedirs(figures_dir, exist_ok=True)
plt.savefig(os.path.join(figures_dir, "object_localization_metrics.png"))

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



