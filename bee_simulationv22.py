import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from stable_baselines3 import PPO
from ctransformers import AutoModelForCausalLM
import concurrent.futures
import math


from OpenGL.GL import *
from OpenGL.GLU import *


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class BeeEnv(gym.Env):
    def __init__(self, grid_size=10, num_flowers=5):
        super(BeeEnv, self).__init__()
        self.grid_size = grid_size
        self.num_flowers = num_flowers

        # Aksiyon uzayı: 0: Up, 1: Down, 2: Left, 3: Right, 4: Dance
        self.action_space = spaces.Discrete(5)
        # Gözlem uzayı: [bee_x, bee_y, nearest_flower_x, nearest_flower_y] (normalized)
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

        self.bee_pos = None
        self.flowers = []
        self.energy = 100

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.bee_pos = np.array([self.grid_size // 2, self.grid_size // 2])
        self.flowers = [np.random.randint(0, self.grid_size, 2) for _ in range(self.num_flowers)]
        self.energy = 100
        return self._get_obs(), {}

    def _get_obs(self):
        closest_flower = min(self.flowers, key=lambda f: np.linalg.norm(f - self.bee_pos))
        return np.concatenate([
            self.bee_pos / self.grid_size,
            closest_flower / self.grid_size
        ]).astype(np.float32)

    def step(self, action):
        reward = 0
        done = False
        truncated = False

        # For actions 0-3: movement (up, down, left, right)
        if action < 4:
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            move = np.array(moves[action])
            self.bee_pos = np.clip(self.bee_pos + move, 0, self.grid_size - 1)
            self.energy -= 1

        # "Dance" action triggers extra reward for LLM call.
        if action == 4:
            reward += 0.5

        # If bee reaches a flower, reward and update the flower.
        for i, flower in enumerate(self.flowers):
            if np.array_equal(self.bee_pos, flower):
                reward += 10
                self.flowers.pop(i)
                self.flowers.append(np.random.randint(0, self.grid_size, 2))
                break

        if self.energy <= 0:
            done = True
            reward -= 5

        return self._get_obs(), reward, done, truncated, {}

# 2. LLM Entegrasyonu: Asynchronous message
class BeeCommunicator:
    def __init__(self):
        # Güncel model yolu
        model_path = r"C:\Users\.....\Llama-2-7B-Chat-GGUF\llama-2-7b-chat.Q4_0.gguf"  #change
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type='llama',
            max_new_tokens=290
        )
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def generate_message(self, state, stats):
        bee_x, bee_y, flower_x, flower_y = state.tolist()
        distance = math.sqrt((flower_x - bee_x) ** 2 + (flower_y - bee_y) ** 2)
        
        prompt = (
            "Simulation Statistics:\n"
            f"- Step: {stats['steps']}\n"
            f"- Total Reward: {stats['total_reward']:.2f}\n"
            f"- Energy: {stats['energy']}\n"
            f"- Efficiency: {stats['efficiency']:.2f}%\n"
            f"- FPS: {stats['fps']:.2f}\n\n"
            f"Bee position (normalized): [{bee_x:.2f}, {bee_y:.2f}]\n"
            f"Nearest flower position (normalized): [{flower_x:.2f}, {flower_y:.2f}]\n"
            f"Euclidean distance (normalized) between bee and flower: {distance:.2f}\n\n"
            "Please analyze the bee's movements mathematically based on the above statistics. Your analysis should include:\n"
            "1. Calculation of the average reward per step.\n"
            "2. The rate of energy consumption.\n"
            "3. Discussion on the relationship between the bee's distance from the flower and its movement dynamics.\n"
            "4. Overall efficiency and performance evaluation.\n\n"
            "Forecast any possible movement trends and explain your reasoning mathematically.\n\n"
            "Response:"
        )
        
        output = self.model(prompt, temperature=0.7, repetition_penalty=1.1)
        return output

    def generate_message_async(self, state, stats):
        return self.executor.submit(self.generate_message, state, stats)

# 3. Görselleştirme
class BeeVisualizer:
    def __init__(self, grid_size=10, cell_size=40, stats_width=300):
        pygame.init()
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.simulation_width = grid_size * cell_size
        self.stats_width = stats_width
        self.total_width = self.simulation_width + self.stats_width
        self.height = grid_size * cell_size

        self.screen = pygame.display.set_mode(
            (self.total_width, self.height),
            pygame.DOUBLEBUF | pygame.OPENGL
        )
        pygame.display.set_caption("Bee Simulation")

        self.colors = {
            'bg': (1.0, 1.0, 1.0),
            'bee': (1.0, 0.87, 0.0),
            'flower': (1.0, 0.0, 0.0),
            'grid': (0.8, 0.8, 0.8),
            'stats_bg': (0.9, 0.9, 0.9)
        }
        self.font = pygame.font.SysFont("Arial", 18)
        self.init_opengl()

    def init_opengl(self):
        glClearColor(*self.colors['bg'], 1.0)
        glViewport(0, 0, self.total_width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        gluOrtho2D(0, self.total_width, self.height, 0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def draw_circle(self, x, y, radius, color):
        glColor3f(*color)
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(x, y)
        segments = 36
        for i in range(segments + 1):
            angle = 2 * math.pi * i / segments
            glVertex2f(x + math.cos(angle) * radius, y + math.sin(angle) * radius)
        glEnd()

    def draw_text(self, x, y, text, font, color=(0, 0, 0, 255)):
        text_surface = font.render(text, True, color[:3])
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        glPushAttrib(GL_ENABLE_BIT)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glRasterPos2i(int(x), int(self.height - y - text_surface.get_height()))
        glDrawPixels(text_surface.get_width(), text_surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        glPopAttrib()

    def draw_grid(self):
        glColor3f(*self.colors['grid'])
        glLineWidth(1.0)
        glBegin(GL_LINES)
        for i in range(self.grid_size + 1):
            x = i * self.cell_size
            glVertex2f(x, 0)
            glVertex2f(x, self.height)
        for j in range(self.grid_size + 1):
            y = j * self.cell_size
            glVertex2f(0, y)
            glVertex2f(self.simulation_width, y)
        glEnd()

        for i in range(self.grid_size + 1):
            x = i * self.cell_size
            self.draw_text(x + 2, 2, f"{i}", self.font, (0, 0, 0, 255))
        for j in range(self.grid_size + 1):
            y = j * self.cell_size
            self.draw_text(2, y + 2, f"{j}", self.font, (0, 0, 0, 255))

    def draw_stats(self, stats):
        glColor3f(*self.colors['stats_bg'])
        glBegin(GL_QUADS)
        glVertex2f(self.simulation_width, 0)
        glVertex2f(self.total_width, 0)
        glVertex2f(self.total_width, self.height)
        glVertex2f(self.simulation_width, self.height)
        glEnd()

        glColor3f(0, 0, 0)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glVertex2f(self.simulation_width, 0)
        glVertex2f(self.simulation_width, self.height)
        glEnd()

        x_text = self.simulation_width + 10
        header_spacing = 25
        current_y = 10

        self.draw_text(x_text, current_y, "STATISTICS", self.font, (0, 0, 0, 255))
        current_y += header_spacing + 10

        self.draw_text(x_text, current_y, f"Step: {stats['steps']}", self.font, (0, 0, 0, 255))
        current_y += header_spacing
        self.draw_text(x_text, current_y, f"Total Reward: {stats['total_reward']:.2f}", self.font, (0, 0, 0, 255))
        current_y += header_spacing
        self.draw_text(x_text, current_y, f"Energy: {stats['energy']}", self.font, (0, 0, 0, 255))
        current_y += header_spacing

        efficiency = stats['efficiency']
        self.draw_text(x_text, current_y, f"Efficiency: {efficiency:.2f}%", self.font, (0, 0, 0, 255))
        bar_x = x_text
        bar_y = current_y + 20
        bar_width = 200
        bar_height = 15
        glColor3f(0.8, 0.8, 0.8)
        glBegin(GL_QUADS)
        glVertex2f(bar_x, bar_y)
        glVertex2f(bar_x + bar_width, bar_y)
        glVertex2f(bar_x + bar_width, bar_y + bar_height)
        glVertex2f(bar_x, bar_y + bar_height)
        glEnd()
        filled_width = bar_width * (efficiency / 100)
        glColor3f(0.0, 1.0, 0.0)
        glBegin(GL_QUADS)
        glVertex2f(bar_x, bar_y)
        glVertex2f(bar_x + filled_width, bar_y)
        glVertex2f(bar_x + filled_width, bar_y + bar_height)
        glVertex2f(bar_x, bar_y + bar_height)
        glEnd()

        current_y += header_spacing + 30
        self.draw_text(x_text, current_y, f"FPS: {stats['fps']:.2f}", self.font, (0, 0, 0, 255))

    def render(self, bee_pos, flowers, stats):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit("Window closed by user.")

        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()

        for f in flowers:
            cx = (f[0] + 0.5) * self.cell_size
            cy = (f[1] + 0.5) * self.cell_size
            self.draw_circle(cx, cy, self.cell_size / 4, self.colors['flower'])
        bx = (bee_pos[0] + 0.5) * self.cell_size
        by = (bee_pos[1] + 0.5) * self.cell_size
        self.draw_circle(bx, by, self.cell_size / 5, self.colors['bee'])

        self.draw_grid()
        self.draw_stats(stats)
        pygame.display.flip()

if __name__ == "__main__":
    env = BeeEnv(grid_size=10)
    model = PPO("MlpPolicy", env, verbose=1, device="cuda")
    visualizer = BeeVisualizer(grid_size=10, cell_size=40, stats_width=300)
    communicator = BeeCommunicator()

    print("Training starts...")
    model.learn(total_timesteps=25000)
    model.save("bee_model")

    print("Test phase:")
    obs, _ = env.reset()
    clock = pygame.time.Clock()
    steps = 0
    total_reward = 0.0
    llm_future = None

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit("Simulation terminated by user.")

        action, _ = model.predict(obs)
        
        if steps % 50 == 0:
            action = 4

        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward

        if steps > 0:
            avg_reward = total_reward / steps
            efficiency = (avg_reward / 10) * 100  # Ideal reward = 10 per step
            if efficiency > 100:
                efficiency = 100
        else:
            efficiency = 0

        fps = clock.get_fps()

        stats = {
            "steps": steps,
            "total_reward": total_reward,
            "energy": env.energy,
            "efficiency": efficiency,
            "fps": fps
        }

        visualizer.render(env.bee_pos, env.flowers, stats)
        clock.tick(30)

        
        if action == 4 and llm_future is None:
            llm_future = communicator.generate_message_async(obs, stats)
        if llm_future is not None and llm_future.done():
            message = llm_future.result()
            print(f"Bee Message (Mathematical Analysis):\n{message}\n")
            llm_future = None

        steps += 1
        if done or truncated:
            obs, _ = env.reset()
            steps = 0
            total_reward = 0.0