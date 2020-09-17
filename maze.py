'''
Particle Filter in Maze
'''

import numpy as np 
import turtle # 繪圖
import bisect
import argparse

class Maze(object):
    def __init__(self, grid_height, grid_width, maze = None, num_rows = None, num_cols = None, wall_prob = None, random_seed = None):
		#self.name 代表該物件自身的 name --> 讓傳入的參數與物件自身擁有的不至於混淆。
		'''
        maze: 2D numpy array -> passages are coded as a 4-bit number -> 上下左右用4-bit number來編碼,
        The 1s register corresponds with a square's top edge, (1)
            2s register                       the right edge, (10)
            4s register                      the bottom edge, (100)
            8s register                        the left edge. (1000)
		bit -> 0 if there is a wall and 
		    -> 1 if there is no wall. 
        '''
        self.grid_height = grid_height 
        self.grid_width = grid_width 

		# 如果maze已存在
        if maze is not None:
            self.maze = maze # 直接從外部傳入maze以及相關參數
            self.num_rows = maze.shape[0] # maze.shape[0] = 一維的長度
            self.num_cols = maze.shape[1] # maze.shape[1] = 二維的長度
            self.fix_maze_boundary() # 確定迷宮有被邊界包圍
            self.fix_wall_inconsistency() # 修正wall錯誤 --> 當相鄰cell定義的牆不一致時 --> 把牆補上
        else: # 如果maze不存在 --> random_maze隨機產生迷宮
            assert num_rows is not None and num_cols is not None and wall_prob is not None, 'Parameters for random maze should not be None.' 
            # 確保傳入的 maze 設定參數不為 None
			self.random_maze(num_rows = num_rows, num_cols = num_cols, wall_prob = wall_prob, random_seed = random_seed)
            # 把參數傳入 random_maze function 來建構 maze
			
        self.height = self.num_rows * self.grid_height # 算maze height = 25 * 100
        self.width = self.num_cols * self.grid_width # 算maze width = 25 * 100
        self.turtle_registration() # particle的圖形註冊

	# particle的圖形註冊
    def turtle_registration(self):
		# 註冊圖形'tri' --> 如此一來才能對其操作
		# shape 'tri' 為由(-3, -2), (0, 3), (3, -2), (0, 0)座標構成的多邊形: 箭頭形狀 
        turtle.register_shape('tri', ((-3, -2), (0, 3), (3, -2), (0, 0)))

    # 確認相鄰cell定義牆是否有不一致的地方
	def check_wall_inconsistency(self):

        wall_errors = list() # 用來存 wall_errors的list()

        # 檢查垂直的牆(左右)
        for i in range(self.num_rows):
            for j in range(self.num_cols-1): #j num_cols-1的原因是最右的cols的右邊就是最邊邊的牆壁了，不需要判斷要不要給牆
                if (self.maze[i,j] & 2 != 0) != (self.maze[i,j+1] & 8 != 0): 
					# maze[i,j]的右牆 = maze[i,j+1]的左牆 --> 所以要檢查兩cell定義是否一致
					# 如果maze[i,j]有右牆，但是maze[i,j+1]沒左牆 --> 代表出現wall_errors
                    wall_errors.append(((i,j), 'v'))
        
		# 檢查水平的牆(上下)
        for i in range(self.num_rows-1): # i num_rows-1的原因是最下的rows的右邊就是最邊邊的牆壁了，不需要判斷要不要給牆
            for j in range(self.num_cols):
                if (self.maze[i,j] & 4 != 0) != (self.maze[i+1,j] & 1 != 0):
					# maze[i,j]的底端牆 = maze[i,j+1]的頂端牆 --> 所以要檢查兩cell定義是否一致
					# 如果maze[i,j]有底端牆，但是maze[i,j+1]沒頂端牆 --> 代表出現wall_errors
                    wall_errors.append(((i,j), 'h'))

        return wall_errors

	# 修正wall錯誤 --> 當相鄰cell定義的牆不一致時 --> 把牆補上
    def fix_wall_inconsistency(self, verbose = True):
        wall_errors = self.check_wall_inconsistency() # 確認相鄰cell定義牆是否有不一致的地方

        if wall_errors and verbose:
            print('Warning: maze contains wall inconsistency.')

	    #修正wall list 中紀錄出錯誤的地方(i,j) --> 把缺少的牆補上
        for (i,j), error in wall_errors:
            if error == 'v': #垂直牆出錯(左右) --> 補牆以獲得一致性
                self.maze[i,j] |= 2
                self.maze[i,j+1] |= 8
            elif error == 'h': #水平牆出錯(上下) --> 補牆以獲得一致性
                self.maze[i,j] |= 4
                self.maze[i+1,j] |= 1
            else:
                raise Exception('Unknown type of wall inconsistency.')
        return

	# 讓迷宮被邊界包圍
    def fix_maze_boundary(self):
	
		# 把每個row開頭的cell加上左邊的牆(8)，把每個row結尾的cell加上右邊的牆(2)
        for i in range(self.num_rows):
            self.maze[i,0] |= 8 
            self.maze[i,-1] |= 2 
			
		# 把每個col開頭的cell加上頂端的牆(1)，把每個col結尾的cell加上底部的牆(4)
        for j in range(self.num_cols):
            self.maze[0,j] |= 1 
            self.maze[-1,j] |= 4 

    def random_maze(self, num_rows, num_cols, wall_prob, random_seed = None):
        if random_seed is not None:
            np.random.seed(random_seed) 
		# 用於隨機迷宮和粒子過濾器的隨機種子(None的話每次執行迷宮的樣子就都不一樣)。
		
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.maze = np.zeros((num_rows, num_cols), dtype = np.int8)
		# 把 maze 2D array 內值初始化為00000000(int8 --> 長度八位元)
		# np.zeros --> 返回来一个给定形状(num_rows, num_cols)和类型(dtype = np.int8)的用0填充的数组
		
		# 隨機產生迷宮
        for i in range(self.num_rows):
            for j in range(self.num_cols-1):
                if np.random.rand() < wall_prob: # 如果骰出的數字 < wall_prob(牆壁出現的機率) --> 產生牆壁(右)
                    self.maze[i,j] |= 2 
					# | --> 只要對應位置的兩個bit其中有一個為一，结果位就為1。
					# EX: (00000000 | 00000010 = 00000010) --> 把maze[i,j]設成 2 --> 產生牆壁(右)
        for i in range(self.num_rows-1):
            for j in range(self.num_cols):
                if np.random.rand() < wall_prob:
                    self.maze[i,j] |= 4 
					# EX: (00000000 | 00000100 = 00000100) --> 把maze[i,j]設成 4 --> 產生牆壁(下)
					# 若maze[i,j]值 = 6(00000110) --> 代表此cell有(右&下)的牆
        
		self.fix_maze_boundary()# 讓迷宮被邊界包圍
        self.fix_wall_inconsistency(verbose = False) # 修正wall錯誤 --> 當相鄰cell定義的牆不一致時 --> 把牆補上

	# 檢查給定cell內，哪些方向具有牆壁(上, 右, 下, 左)
    def permissibilities(self, cell):
        cell_value = self.maze[cell[0], cell[1]]
		# ex: 9 --> (False上, True右, True下, False左)
        return (cell_value & 1 == 0, cell_value & 2 == 0, cell_value & 4 == 0, cell_value & 8 == 0)

	# 測量與周遭最近牆壁的距離
    def distance_to_walls(self, coordinates):    
        x, y = coordinates # 傳入座標

		# 算與離自己最近的up wall的距離
		# 整數除法 -> //
        i = int(y // self.grid_height) # 算出座標在第幾個row
        j = int(x // self.grid_width) # 算出座標在第幾個col
        # 算出座標在cell內的確切高度位置(座標->cell頂端)
		d1 = y - y // self.grid_height * self.grid_height 
        while self.permissibilities(cell = (i,j))[0]: # 當 permissibilities[0] = true --> 代表沒頂端的牆
            i -= 1 
			# 因為當前cell的頂端沒有牆，所以退回上一格，上一格再繼續檢查有沒有頂端的牆
			# 上一格有頂端的牆 --> 與頂端最近牆壁的距離為，剛剛算出在cell內的確切高度位置(座標->cell頂端) + grid_height
			# 上一格沒頂端的牆 --> d1 + grid_height + grid_height --> 到上上cell繼續找
            d1 += self.grid_height
		
		# 算與離自己最近的right wall的距離
        i = int(y // self.grid_height)
        j = int(x // self.grid_width)
		# 算出在cell內的確切寬度位置(座標->cell右邊)
        d2 = self.grid_width - (x - x // self.grid_width * self.grid_width) 
		while self.permissibilities(cell = (i,j))[1]: # 當 permissibilities[1] = true --> 代表沒右邊的牆
            j += 1
            d2 += self.grid_width
		
		# 算與離自己最近的down wall的距離
        i = int(y // self.grid_height)
        j = int(x // self.grid_width)
		# 算出在cell內的確切寬度位置(座標->cell底端)
        d3 = self.grid_height - (y - y // self.grid_height * self.grid_height) 
        while self.permissibilities(cell = (i,j))[2]:
            i += 1
            d3 += self.grid_height

		# 算與離自己最近的left wall的距離
        i = int(y // self.grid_height)
        j = int(x // self.grid_width)
		# 算出在cell內的確切寬度位置(座標->cell左邊)
        d4 = x - x // self.grid_width * self.grid_width 
        while self.permissibilities(cell = (i,j))[3]:
            j -= 1
            d4 += self.grid_width

        return [d1, d2, d3, d4] # 回傳與(up, right, down, left)牆壁的距離

    def show_maze(self):

		# 設定畫布座標
        turtle.setworldcoordinates(0, 0, self.width * 1.005, self.height * 1.005)
        wally = turtle.Turtle()
        wally.speed(0) # 設定畫筆移動的速度(1~10)。數字愈大，畫筆移動愈快。
        wally.width(1.5) # 設定線條寬度
        wally.hideturtle() # 隱藏畫筆的位置，不然會在畫布上顯示出畫筆位置
        turtle.tracer(0, 0) 

        for i in range(self.num_rows):
            for j in range(self.num_cols):
				# permissibilities 傳回哪裡有牆 --> ex: (False上, True右, True下, False左)
				# True -> 沒牆
				# False -> 有牆
                permissibilities = self.permissibilities(cell = (i,j)) 
				turtle.up()
                wally.setposition((j * self.grid_width, i * self.grid_height)) # 設定起始位置 -> cell 左上角
                
				# Set turtle heading orientation
                # 0 - east, 90 - north, 180 - west, 270 - south
				
				wally.setheading(0) # 設定方向朝東
				# 畫top wall
                if not permissibilities[0]: # if (not False) = True
                    wally.down() # 畫筆放下(開始畫)
                else:
                    wally.up() # if (not True) = False --> 畫筆抬起
                wally.forward(self.grid_width) # 前進grid_width的長度
               
			    wally.setheading(90) # 設定方向朝北
                wally.up() # --> 畫筆抬起
				# 畫right wall
                if not permissibilities[1]:
                    wally.down()
                else:
                    wally.up()
                wally.forward(self.grid_height) # 前進grid_height的長度
                
				wally.setheading(180) # 設定方向朝西
                wally.up()
				# 畫bottom wall
                if not permissibilities[2]:
                    wally.down()
                else:
                    wally.up()
                wally.forward(self.grid_width) # 前進grid_width的長度
                
				wally.setheading(270) # 設定方向朝南
                wally.up()
				# 畫left wall
                if not permissibilities[3]:
                    wally.down()
                else:
                    wally.up()
                wally.forward(self.grid_height) # 前進grid_height的長度 --> 返回cell左上角起點
                wally.up()

        turtle.update()


	# 產生與 weight 對應的 color
    def weight_to_color(self, weight):
        return '#%02x00%02x' % (int(weight * 255), int((1 - weight) * 255))

    # 顯示灑下的particles
    def show_particles(self, particles, show_frequency = 10):
        turtle.shape('tri') # 前面定義過的箭頭圖形

		# show_frequency = 每10個才顯示一個(可調整)
        for i, particle in enumerate(particles):
            if i % show_frequency == 0:
                turtle.setposition((particle.x, particle.y))
                turtle.setheading(90 - particle.heading) # 將畫筆的方向設置為90 - particle.heading。
                turtle.color(self.weight_to_color(particle.weight)) # 根據particle weight來決定顏色 
                turtle.stamp() # 將烏龜形狀的副本標記在畫布上。
        
        turtle.update() # 更新畫布

	# 顯示出用particle預測出的位置(平均加權位置)(橘色)
    def show_estimated_location(self, particles):
        #初始化各參數
		x_accum = 0
        y_accum = 0
        heading_accum = 0
        weight_accum = 0
        num_particles = len(particles)

		# 把每個particle的x, y, heading乘上particle.weight後加總
        for particle in particles:
            weight_accum += particle.weight # 累積 weight
            x_accum += particle.x * particle.weight # 累積 x
            y_accum += particle.y * particle.weight # 累積 y
            heading_accum += particle.heading * particle.weight # 累積 heading
        if weight_accum == 0:
            return False

		# 算 x, y, heading 平均值--> 預測出的位置(平均加權位置)
        x_estimate = x_accum / weight_accum
        y_estimate = y_accum / weight_accum
        heading_estimate = heading_accum / weight_accum

		#顯示出用particle算出的平均加總位置
        turtle.color('orange')
        turtle.setposition(x_estimate, y_estimate) # 設定位置
        turtle.setheading(90 - heading_estimate)
        turtle.shape('turtle') # 以烏龜圖案顯示
        turtle.stamp() # 將烏龜形狀的副本標記在畫布上。
        turtle.update()#更新畫布

	# 顯示robot所在的位置(綠色烏龜)
    def show_robot(self, robot):
        turtle.color('green')
        turtle.shape('turtle') # 以烏龜圖案顯示
        turtle.shapesize(0.7, 0.7) # 設定烏龜圖案大小
        turtle.setposition((robot.x, robot.y)) # 設定烏龜位置
        turtle.setheading(90 - robot.heading)
        turtle.stamp() # 將烏龜形狀的副本標記在畫布上。
        turtle.update() # 更新畫布

    def clear_objects(self):
        turtle.clearstamps() # 刪掉maze以外戳記

class Particle(object):
    def __init__(self, x, y, maze, heading = None, weight = 1.0, sensor_limit = None, noisy = False):
        # 隨機產生一個角度
		if heading is None:
            heading = np.random.uniform(0,360)

		# 傳入參數
        self.x = x
        self.y = y
        self.heading = heading
        self.weight = weight 
        self.maze = maze
        self.sensor_limit = sensor_limit

		# 加上noisy雜訊誤差
        if noisy:
            std = max(self.maze.grid_height, self.maze.grid_width) * 0.2
            self.x = self.add_noise(x = self.x, std = std)
            self.y = self.add_noise(x = self.y, std = std)
            self.heading = self.add_noise(x = self.heading, std = 360 * 0.05)

		# 處理超出範圍的particles
        self.fix_invalid_particles()

	# 使 particles 不會超出maze範圍
    def fix_invalid_particles(self):	
        if self.x < 0:
            self.x = 0
        if self.x > self.maze.width:
            self.x = self.maze.width * 0.9999
        if self.y < 0:
            self.y = 0
        if self.y > self.maze.height:
            self.y = self.maze.height * 0.9999
			
        self.heading = self.heading % 360 # (除以360後的餘數)

    @property 
	# state --> 目前座標(x, y)與面向角度(heading)
    def state(self):
        return (self.x, self.y, self.heading)

	#回傳加上noise誤差的位置
    def add_noise(self, x, std):
		# np.random.normal(0, std) --> 從（均值0 ~ 標準差std）中獲得樣本
        return x + np.random.normal(0, std)
	
    def read_sensor(self, maze):
        readings = maze.distance_to_walls(coordinates = (self.x, self.y))
		# [readings[0], readings[1], readings[2], readings[3]] --> 與(up, right, down, left)牆壁的距離
        heading = self.heading % 360
		# 因為heading的關係，所以要把原本的方位，轉換成以heading指向北為準的方位
		
		# heading指向上 --> 方位不變(up, right, down, left)
		if heading >= 45 and heading < 135:
			readings = readings	
			
		# heading指向左 --> 以左為top重定方位
        elif heading >= 135 and heading < 225:  
            readings = readings[-1:] + readings[:-1] #readings = [readings[3], readings[0], readings[1], readings[2]]
			# [-1:] --> readings[3] 
			# [:-1] --> readings[0], readings[1], readings[2]
			
		# heading指向下 --> 以下為top重定方位
        elif heading >= 225 and heading < 315: 
            readings = readings[-2:] + readings[:-2] #readings = [readings[2], readings[3], readings[0], readings[1]]
			
		# heading指向右 --> 以右為top重定方位
        else:
            readings = readings[-3:] + readings[:-3] #readings = [readings[1], readings[2], readings[3], readings[0]]

		# 檢查 readings(與四周牆壁的距離)有沒有超過 sensor_limit
        if self.sensor_limit is not None:
            for i in range(len(readings)):
                if readings[i] > self.sensor_limit:
                    readings[i] = self.sensor_limit
        return readings

	# Particle的移動
    def try_move(self, speed, maze, noisy = False):
        heading = self.heading
        heading_rad = np.radians(heading) # radians() --> 把角度(heading)轉換成弧度(heading_rad)

		# 算要移動的(垂直/水平)距離
        dx = np.sin(heading_rad) * speed # x --> speed*sin(heading_rad)
        dy = np.cos(heading_rad) * speed # y --> speed*cos(heading_rad)

		# 算移動後的座標
        x = self.x + dx
        y = self.y + dy

		#原位置 -> 判斷座標位於第幾行/列
        gj1 = int(self.x // maze.grid_width)
        gi1 = int(self.y // maze.grid_height)
		#原位置 -> 判斷座標位於第幾行/列
	    gj2 = int(x // maze.grid_width)
        gi2 = int(y // maze.grid_height)

        # Check if the particle is still in the maze確認移動後粒子是否還在迷宮中
        if gi2 < 0 or gi2 >= maze.num_rows or gj2 < 0 or gj2 >= maze.num_cols:
            return False

        # Move in the same grid # 如果是在同一網格中的移動
        if gi1 == gi2 and gj1 == gj2:
            self.x = x
            self.y = y
            return True
        # Move across one grid vertically # 如果是要垂直的移動到不同網格
        elif abs(gi1 - gi2) == 1 and abs(gj1 - gj2) == 0:
            if maze.maze[min(gi1, gi2), gj1] & 4 != 0: # 旁邊有牆 --> 不能移
                return False
            else: # 旁邊有牆 --> 可以移
                self.x = x
                self.y = y
                return True
        # Move across one grid horizonally # 如果是要水平的移動到不同網格
        elif abs(gi1 - gi2) == 0 and abs(gj1 - gj2) == 1:
            if maze.maze[gi1, min(gj1, gj2)] & 2 != 0: # 旁邊有牆 --> 不能移
                return False
            else: # 旁邊有牆 --> 可以移
                self.x = x
                self.y = y
                return True
        # Move across grids both vertically and horizonally # 如果是要水平+垂直的移動到不同網格
        elif abs(gi1 - gi2) == 1 and abs(gj1 - gj2) == 1:
			# 算要移動的水平距離
            x0 = max(gj1, gj2) * maze.grid_width
            y0 = (y - self.y) / (x - self.x) * (x0 - self.x) + self.y
            if maze.maze[int(y0 // maze.grid_height), min(gj1, gj2)] & 2 != 0: # 旁邊有牆 --> 不能移
                return False
			
			# 算要移動的垂直距離
            y0 = max(gi1, gi2) * maze.grid_height
            x0 = (x - self.x) / (y - self.y) * (y0 - self.y) + self.x
            if maze.maze[min(gi1, gi2), int(x0 // maze.grid_width)] & 4 != 0: # 旁邊有牆 --> 不能移
                return False

            self.x = x
            self.y = y
            return True

        else: # error
            raise Exception('Unexpected collision detection.')


class Robot(Particle):
    def __init__(self, x, y, maze, heading = None, speed = 1.0, sensor_limit = None, noisy = True):
        super(Robot, self).__init__(x = x, y = y, maze = maze, heading = heading, sensor_limit = sensor_limit, noisy = noisy)
        # 調用 --> class Particle(object): def __init__
		
		self.step_count = 0
        self.noisy = noisy
        self.time_step = 0
        self.speed = speed

	# 決定初始的前進方向(隨機)
    def choose_random_direction(self):
        self.heading = np.random.uniform(0, 360)

	# 把雜訊 error 加入 Robot reading
    def add_sensor_noise(self, x, z = 0.05):
        readings = list(x)
        for i in range(len(readings)):
            std = readings[i] * z / 2
            readings[i] = readings[i] + np.random.normal(0, std)
			#將readings + np.random.normal(0, std) --> 加上雜訊誤差
        return readings

	# 回傳 sensor readings
    def read_sensor(self, maze):
        # Robot has error in reading the sensor while particles do not.
        readings = super(Robot, self).read_sensor(maze = maze) # super()：用於調用parent class的一個方法 --> 解決多重繼承問題的
		
		# 把雜訊 error 加入 Robot reading
        if self.noisy == True:
            readings = self.add_sensor_noise(x = readings)
         
        return readings#回傳加上誤差的readings

    def move(self, maze):
        while True:
            self.time_step += 1 # 進入下個時間步
            if self.try_move(speed = self.speed, maze = maze, noisy = False):
                break
            self.choose_random_direction() # 決定初始的前進方向(隨機)

# 算weight的機率分布 --> Resampling會用到
class WeightedDistribution(object):
    def __init__(self, particles):
        accum = 0.0#初始化accum，計算weight總和
        self.particles = particles
        self.distribution = list()
        for particle in self.particles:
            accum += particle.weight #計算weight總和
            self.distribution.append(accum) # 儲存weight distribution到list

    def random_select(self):
		# 根據weight distribution挑選particle
        try:
			return self.particles[bisect.bisect_left(self.distribution, np.random.uniform(0, 1))]
		except IndexError:
            # When all particles have weights zero
            return None
			
def euclidean_distance(x1, x2):
	# 計算euclidean_distance
	# ex.(x1,y1)與(x2,y2) = [(x1-x2)^2+(y1-y2)^2]^1/2
    return np.linalg.norm(np.asarray(x1) - np.asarray(x2))

def weight_gaussian_kernel(x1, x2, std = 10):

    distance = euclidean_distance(x1 = x1, x2 = x2)# --> 計算在m維空間中兩個點之間的真實距離。
    return np.exp(-distance ** 2 / (2 * std))  # 取指數返回







