
import numpy as np 
import turtle
import argparse
import time

from maze import Maze, Particle, Robot, WeightedDistribution, weight_gaussian_kernel

def main(window_width, window_height, num_particles, sensor_limit_ratio, grid_height, grid_width, num_rows, num_cols, wall_prob, random_seed, robot_speed, kernel_sigma, particle_show_frequency):

	# sensor偵測範圍的上限
    sensor_limit = sensor_limit_ratio * max(grid_height * num_rows, grid_width * num_cols)

	# 設定畫布視窗大小
    window = turtle.Screen()#用turtle模組建立出初始畫布
    window.setup (width = window_width, height = window_height)# 設置畫布視窗的大小。

	# 建立迷宮(參數 --> 網格的寬/高度、行列數、牆出現的機率...)
    world = Maze(grid_height = grid_height, grid_width = grid_width, num_rows = num_rows, 
	             num_cols = num_cols, wall_prob = wall_prob, random_seed = random_seed)

	# 設定Robot初始位置
    x = np.random.uniform(0, world.width) # 從0~world.width(2500)中隨機取數
    y = np.random.uniform(0, world.height) # 從0~world.height(2500)中隨機取數
	
	# 建構Robot
    bob = Robot(x = x, y = y, maze = world, speed = robot_speed, sensor_limit = sensor_limit)

	# 用list儲存particle(3000個)
    particles = list()
	# 撒下particles(x, y 座標位置隨機)
    for i in range(num_particles): # num_particles --> 3000
        x = np.random.uniform(0, world.width) # 從0~world.width(2500)中隨機取數
        y = np.random.uniform(0, world.height) # 從0~world.height(2500)中隨機取數
		# random出 x, y 座標來建構particle，加入到list
		particles.append(Particle(x = x, y = y, maze = world, sensor_limit = sensor_limit))

    time.sleep(1)
	# 畫出剛剛建構的迷宮
    world.show_maze()
	# --> 畫完迷宮
	
    while True:
		# 觀察到的資訊 = robot sensor測量到與周圍的距離
		readings_robot = bob.read_sensor(maze = world)

		# 加總每個particle的weight
        particle_weight_total = 0
        for particle in particles:
			# 觀察到的資訊 = particle sensor測量到與周圍的距離
            readings_particle = particle.read_sensor(maze = world)
			# 會根據particle與robot的距離遠近，來賦予particle對應weight值 --> 離robot越近，particle.weight越高
            particle.weight = weight_gaussian_kernel(x1 = readings_robot, x2 = readings_particle, std = kernel_sigma)
			# 加總全部的particle.weight --> 等等要 Normalize particle weights
            particle_weight_total += particle.weight

        world.show_particles(particles = particles, show_frequency = particle_show_frequency)
        world.show_robot(robot = bob)
        world.show_estimated_location(particles = particles)# 用particles取平均值，算出robot可能的位置
        world.clear_objects()

        # Make sure normalization is not divided by zero
        if particle_weight_total == 0:
            particle_weight_total = 1e-8

        # Normalize particle weights --> 將每個particle.weight做標準化
        for particle in particles:
            particle.weight /= particle_weight_total

		
        distribution = WeightedDistribution(particles = particles)  # 算weight的機率分布 --> Resampling會用到
		
		### Resampling ###
		particles_new = list() # 存 Resampling particles
        for i in range(num_particles):
            particle = distribution.random_select()
			# 用andom_select()來隨機取樣一個particle
            if particle is None:# 如果選出來的particle是none的話，就自己再產生一個新的particle
                x = np.random.uniform(0, world.width)
                y = np.random.uniform(0, world.height)
				# random 出 x, y 座標來建構新的particle，加入到particles_new list
                particles_new.append(Particle(x = x, y = y, maze = world, sensor_limit = sensor_limit))
            else:# 如果選出來的particle不是none的話 --> 加入到particles_new list
                particles_new.append(Particle(x = particle.x, y = particle.y, maze = world, heading = particle.heading, sensor_limit = sensor_limit, noisy = True))

        particles = particles_new# 讓新的particles取代舊的(覆蓋)
 
        heading_old = bob.heading # robot移動前的heading方向
        bob.move(maze = world)
        heading_new = bob.heading # robot移動後 --> heading方向
        dh = heading_new - heading_old# robot移動前後的heading方位差

		# particle往robot所在處移動
        for particle in particles:
            particle.heading = (particle.heading + dh) % 360 # 兩者相加 --> 算出particle 新heading
            particle.try_move(maze = world, speed = bob.speed)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Particle filter in maze.')

    window_width_default = 800
    window_height_default = 800
    num_particles_default = 3000
    sensor_limit_ratio_default = 0.3
    grid_height_default = 100
    grid_width_default = 100
    num_rows_default = 25
    num_cols_default = 25
    wall_prob_default = 0.25
    random_seed_default = 100
    robot_speed_default = 10 # robot在地圖中的移動速度
    kernel_sigma_default = 500# 高斯距離內核的Sigma角度。
    particle_show_frequency_default = 10 # 調小時，顯示的 particle 變多

    parser.add_argument('--window_width', type = int, help = 'Window width.', default = window_width_default)
    parser.add_argument('--window_height', type = int, help = 'Window height.', default = window_height_default)
    parser.add_argument('--num_particles', type = int, help = 'Number of particles used in particle filter.', default = num_particles_default)
    parser.add_argument('--sensor_limit_ratio', type = float, help = 'Distance limit of sensors (real value: 0 - 1). 0: Useless sensor; 1: Perfect sensor.', default = sensor_limit_ratio_default)
    parser.add_argument('--grid_height', type = int, help = 'Height for each grid of maze.', default = grid_height_default)
    parser.add_argument('--grid_width', type = int, help = 'Width for each grid of maze.', default = grid_width_default)
    parser.add_argument('--num_rows', type = int, help = 'Number of rows in maze', default = num_rows_default)
    parser.add_argument('--num_cols', type = int, help = 'Number of columns in maze', default = num_cols_default)
    parser.add_argument('--wall_prob', type = float, help = 'Wall probability of a random maze.', default = wall_prob_default)
    parser.add_argument('--random_seed', type = int, help = 'Random seed for random maze and particle filter.', default = random_seed_default)
    parser.add_argument('--robot_speed', type = int, help = 'Robot movement speed in maze.', default = robot_speed_default)
    parser.add_argument('--kernel_sigma', type = int, help = 'Standard deviation for Gaussian distance kernel.', default = kernel_sigma_default)
    parser.add_argument('--particle_show_frequency', type = int, help = 'Frequency of showing particles on maze.', default = particle_show_frequency_default)

    argv = parser.parse_args()

    window_width = argv.window_width
    window_height = argv.window_height
    num_particles = argv.num_particles
    sensor_limit_ratio = argv.sensor_limit_ratio
    grid_height = argv.grid_height
    grid_width = argv.grid_width
    num_rows = argv.num_rows
    num_cols = argv.num_cols
    wall_prob = argv.wall_prob
    random_seed = argv.random_seed
    robot_speed = argv.robot_speed
    kernel_sigma = argv.kernel_sigma
    particle_show_frequency = argv.particle_show_frequency

    main(window_width = window_width, window_height = window_height, num_particles = num_particles, sensor_limit_ratio = sensor_limit_ratio, grid_height = grid_height, grid_width = grid_width, num_rows = num_rows, num_cols = num_cols, wall_prob = wall_prob, random_seed = random_seed, robot_speed = robot_speed, kernel_sigma = kernel_sigma, particle_show_frequency = particle_show_frequency)
