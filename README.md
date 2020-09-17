# Probabilistic Reasoning and Making Decisions
#### particle filter 推理報告
#### 參考資料 https://github.com/leimao/Particle_Filter


- 報告重點
  - 程式流程
  - 初始化如何建立場景
  - Particle權重計算與散布的程式實作
  - 機器人位置預估的實作
  - Particle重新取樣的程式實作
  - 改變程式的參數對於收斂的影響，如num_particles、wall_prob。 python main.py --help 可看到提供的參數設定。


- 程式流程
  > 1.設定sensor偵測範圍的上限 & 設定畫布視窗大小
  
  > 2.建立迷宮 : 初始化如何建立場景
  
  > 3.設定Robot初始位置 : 建構Robot
  
  > 4.撒下particles(x, y 座標位置隨機) = Particle 散布的程式實作
  
  > 5.畫出建構的迷宮
  
  > 6.robot觀察環境資訊 : robot sensor測量到與周圍的距離
  
  > 7.根據particle與robot的距離遠近，來賦予particle對應weight值，加總每個particle的weight (以便後續 Particle 的權重計算)
  
  > 8.在畫布上顯示particles以及robot (綠色) & 用particles參數取平均值，算出robot可能的位置(橘色) : 機器人位置預估的實作
  
  > 9.將每個particle.weight做標準化
  
  > 10.算weight的機率分布
  
  > 11.做Resampling = Particle 重新取樣的程式實作
  
  > 12.robot 移動
  
  > 13.Particle移動

### 其他觀察： 改變程式的參數對於收斂的影響 

| Parameter                      | Effect                                                                         |
| ------------------------------ | ------------------------------------------------------------------------------ |
| window_width_default           | 影響不大                                                                        |
| window_height_default          | 影響不大                                                                        |
| num_particles_default          | 如果粒子數量不夠大，對於某些複雜的環境會花更長的時間收斂，預測出的robot位置也不夠準確。|
| sensor_limit_ratio_default     | 傳感器越完美收斂越快                                                              |
| grid_height/width_default      | 參數越小，增加particle移動難度，越慢收斂，預測robot位置不準                         |
| num_cols/rows_default          | cols & rows數越多，增加particle移動難度，越慢收斂，預測robot位置不準確              |
| wall_prob_default              | 牆壁越多，移動難度變大，越慢收斂                                                   |
| random_seed_default            | 有給數字的話， 固定的迷宮形狀                                                     |
|                                | 沒給數字的話，每次迷宮的樣子都不一樣                                               |
| robot_speed_default            | 移動速度調快，加快機器人走迷宮的速度，不會影響收斂。                                 |
| kernel_sigma_default           | 調整kernel_sigma標準偏差可能會收斂更快。                                           |
| particle_show_frequency_default| 只是調整顯示particle數目，不影響收斂。                                             |
