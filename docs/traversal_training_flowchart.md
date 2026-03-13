# 自定义遍历顺序训练流程图

本文档使用 Mermaid 流程图展示整个自定义遍历顺序（Learned T-Shape）的训练流程。

## 完整训练流程图

```mermaid
flowchart TD
    Start([开始训练]) --> Init1["创建四叉树<br/>QuadTreeIndex"]
    Init1 --> Init2["加载轨迹数据<br/>TDrive/合成数据"]
    Init2 --> Init3["分配轨迹到四叉树节点<br/>assign_trajectory"]
    Init3 --> Init4{"是否需要剪枝?"}
    Init4 -->|是| Init5["剪枝稀疏节点<br/>prune_and_mark_active_cells"]
    Init4 -->|否| Init6["生成参考查询集<br/>generate_reference_queries"]
    Init5 --> Init6
    Init6 --> Init7["初始化组件<br/>Encoder, CostEvaluator, Environment"]
    Init7 --> Init8["创建RL智能体<br/>TraversalPolicyAgent<br/>Actor-Critic网络"]
    
    Init8 --> TrainLoop["训练循环<br/>for episode in range num_episodes"]
    
    TrainLoop --> EpReset["环境重置<br/>environment.reset"]
    EpReset --> EpState["获取初始状态<br/>state, action_mask"]
    
    EpState --> StepLoop["Episode步进循环<br/>while not done"]
    
    StepLoop --> RefineMask["精炼动作掩码<br/>Top-K相似度过滤"]
    RefineMask --> SelectAction["智能体选择动作<br/>agent.select_action<br/>Actor网络输出"]
    SelectAction --> EnvStep["环境执行动作<br/>environment.step action"]
    
    EnvStep --> CalcLocalReward["计算局部奖励<br/>step_reward<br/>邻近性 + 相似性"]
    CalcLocalReward --> CheckDone{"是否访问完所有节点?"}
    
    CheckDone -->|否| CalcGlobalReward1["全局奖励 = 0"]
    CheckDone -->|是| CalcGlobalReward2["计算全局奖励<br/>baseline_cost - learned_cost<br/>归一化 + 缩放 + 裁剪"]
    
    CalcGlobalReward1 --> CombineReward["组合奖励<br/>local_weight × local_reward<br/>+ global_weight × global_reward"]
    CalcGlobalReward2 --> CombineReward
    
    CombineReward --> StoreTransition["存储经验<br/>agent.store_transition<br/>state, action, reward, log_prob, value"]
    StoreTransition --> UpdateState["更新状态<br/>next_state, next_mask"]
    UpdateState --> CheckDone2{"是否done?"}
    
    CheckDone2 -->|否| StepLoop
    CheckDone2 -->|是| AgentUpdate["智能体更新<br/>agent.update<br/>计算折扣回报<br/>更新Actor-Critic网络"]
    
    AgentUpdate --> RecordEpisode["记录Episode统计<br/>episode_rewards<br/>episode_lengths"]
    RecordEpisode --> CheckEval{"是否评估周期?<br/>episode % eval_interval == 0"}
    
    CheckEval -->|否| CheckEarlyStop1["跳过评估"]
    CheckEval -->|是| EvalRollout["策略rollout<br/>rollout_policy_order<br/>生成当前学习顺序"]
    
    EvalRollout --> EvalCompare["性能评估<br/>evaluate_final_order<br/>对比baseline和学习顺序"]
    EvalCompare --> EvalRecord["记录改进率<br/>improvement_history<br/>improvement_episodes"]
    EvalRecord --> CheckEarlyStop2["检查早停条件<br/>连续N个评估周期<br/>改进率 < 阈值?"]
    
    CheckEarlyStop1 --> CheckNextEp{"是否达到最大episodes?"}
    CheckEarlyStop2 -->|触发早停| EarlyStop["记录早停episode<br/>early_stop_episode"]
    CheckEarlyStop2 -->|未触发| CheckNextEp
    
    EarlyStop --> PlotCurves["绘制训练曲线<br/>奖励、步数、改进率"]
    CheckNextEp -->|否| TrainLoop
    CheckNextEp -->|是| PlotCurves
    
    PlotCurves --> FinalEval["最终评估<br/>使用test_queries<br/>生成测试查询集"]
    FinalEval --> FinalMetrics["计算最终指标<br/>baseline_avg_cost<br/>learned_avg_cost<br/>improvement_percent"]
    FinalMetrics --> SaveModel["保存最终模型<br/>final_model.pth"]
    SaveModel --> SaveOrder["保存学习顺序<br/>learned_order.txt"]
    SaveOrder --> End([训练完成])
    
    style Start fill:#90EE90
    style End fill:#FFB6C1
    style TrainLoop fill:#87CEEB
    style StepLoop fill:#DDA0DD
    style CalcGlobalReward2 fill:#FFD700
    style AgentUpdate fill:#FFA500
    style EarlyStop fill:#FF6347
    style FinalEval fill:#98FB98
```

## 核心组件交互图

```mermaid
graph TB
    subgraph "初始化阶段"
        QT[QuadTreeIndex<br/>四叉树索引]
        TRAJ[轨迹数据<br/>Trajectories]
        ENC[TraversalOrderEncoder<br/>遍历顺序编码器]
        COST[TraversalCostEvaluator<br/>成本评估器]
        ENV[TraversalEnvironment<br/>RL环境]
        AGENT[TraversalPolicyAgent<br/>Actor-Critic智能体]
        QUERIES[参考查询集<br/>Reference Queries]
    end
    
    subgraph "训练阶段"
        STATE[状态特征<br/>State Features<br/>12维向量]
        ACTION[动作选择<br/>Action Selection<br/>单元格索引]
        REWARD[奖励计算<br/>Reward Calculation]
        UPDATE[网络更新<br/>Network Update]
    end
    
    subgraph "评估阶段"
        EVAL[性能评估器<br/>PerformanceEvaluator]
        METRICS[性能指标<br/>Metrics]
    end
    
    TRAJ --> QT
    QT --> ENC
    QT --> COST
    QT --> QUERIES
    ENC --> ENV
    COST --> ENV
    QUERIES --> ENV
    
    ENV --> STATE
    STATE --> AGENT
    AGENT --> ACTION
    ACTION --> ENV
    ENV --> REWARD
    REWARD --> AGENT
    AGENT --> UPDATE
    
    ENC --> EVAL
    COST --> EVAL
    QUERIES --> EVAL
    EVAL --> METRICS
    
    style QT fill:#E6F3FF
    style ENV fill:#FFE6F3
    style AGENT fill:#E6FFE6
    style EVAL fill:#FFF3E6
```

## 奖励计算详细流程

```mermaid
flowchart TD
    subgraph Local["局部奖励计算"]
        A1["当前单元格<br/>current_cell"] --> A2["下一单元格<br/>next_cell"]
        A2 --> A3["计算距离<br/>max_distance归一化"]
        A3 --> A4["邻近性奖励<br/>proximity_reward<br/>权重: 0.5"]
        A3 --> A5["相似性奖励<br/>similarity_reward<br/>Jaccard相似度<br/>权重: 0.5"]
        A4 --> A6["局部奖励<br/>local_reward"]
        A5 --> A6
    end
    
    subgraph Global["全局奖励计算"]
        B1["完整访问顺序<br/>visited_order"] --> B2["计算学习顺序成本<br/>learned_avg_cost"]
        B3["基线顺序<br/>Z-curve baseline"] --> B4["计算基线成本<br/>baseline_avg_cost"]
        B2 --> B5["改进值<br/>baseline - learned"]
        B4 --> B5
        B5 --> B6["改进百分比<br/>improvement_percent"]
        B6 --> B7["归一化<br/>normalized = percent / 100"]
        B7 --> B8["缩放<br/>× global_reward_scale"]
        B8 --> B9["裁剪<br/>clip -2.0 to 2.0"]
        B9 --> B10["全局奖励<br/>global_reward"]
    end
    
    subgraph Final["最终奖励"]
        A6 --> C1["加权组合<br/>local_weight × local_reward<br/>+ global_weight × global_reward"]
        B10 --> C1
        C1 --> C2["最终奖励<br/>reward"]
    end
    
    style A6 fill:#90EE90
    style B10 fill:#FFD700
    style C2 fill:#FF6347
```

## 状态特征构建

```mermaid
flowchart TD
    S1["当前单元格<br/>current_cell"] --> F1["特征1-2: 单元格坐标<br/>normalized x, y"]
    S1 --> F2["特征3-4: 单元格大小<br/>width, height"]
    S1 --> F3["特征5-6: 轨迹数量<br/>trajectory_count<br/>log归一化"]
    S1 --> F4["特征7-8: 到起点的距离<br/>distance_to_start<br/>max_distance归一化"]
    S1 --> F5["特征9-10: 到最近已访问单元格的距离<br/>min_distance_to_visited"]
    S1 --> F6["特征11: 已访问单元格数量<br/>visited_count / total_cells"]
    S1 --> F7["特征12: 剩余单元格数量<br/>remaining_count / total_cells"]
    
    F1 --> STATE["12维状态向量<br/>State Vector"]
    F2 --> STATE
    F3 --> STATE
    F4 --> STATE
    F5 --> STATE
    F6 --> STATE
    F7 --> STATE
    
    STATE --> AGENT["Actor-Critic网络<br/>输入状态特征"]
    
    style STATE fill:#87CEEB
    style AGENT fill:#FFA500
```

## 早停机制流程（收敛判断）

```mermaid
flowchart TD
    E1["评估周期触发<br/>episode % eval_interval == 0"] --> E2["策略rollout生成顺序"]
    E2 --> E3["性能评估<br/>计算改进率"]
    E3 --> E4["记录改进率<br/>improvement_history.append"]
    E4 --> E5{"历史记录数量<br/>>= k (窗口大小)?"}
    
    E5 -->|否| E6["继续训练"]
    E5 -->|是| E7["获取最近k个评估周期的改进率<br/>recent_improvements = history[-k:]"]
    
    E7 --> E8["计算最大差<br/>max_diff = max - min"]
    E8 --> E9{"最大差<br/>< 收敛阈值?"}
    
    E9 -->|否| E6
    E9 -->|是| E10["触发早停（收敛）<br/>early_stop_episode = episode"]
    E10 --> E11["保存模型和结果"]
    
    E6 --> E12["继续下一个episode"]
    
    style E10 fill:#FF6347
    style E11 fill:#90EE90
```

## 关键参数说明

### 训练参数
- **num_episodes**: 最大训练回合数
- **eval_interval**: 评估间隔（每N个episode评估一次）
- **early_stopping_convergence_window**: 收敛判断窗口大小（k，默认4）
- **early_stopping_convergence_threshold**: 收敛判断阈值（默认0.5%）。最近k个评估周期的改进率最大差 < 此值时停止

### 奖励参数
- **local_reward_weight**: 局部奖励权重（默认0.6）
- **global_reward_weight**: 全局奖励权重（默认0.4）
- **global_reward_scale**: 全局奖励缩放倍数（内部默认2.0）
- **global_reward_clip_range**: 全局奖励裁剪范围（-2.0 到 2.0）

### 网络参数
- **lr_actor**: Actor学习率
- **lr_critic**: Critic学习率
- **gamma**: 折扣因子
- **hidden_dims**: 隐藏层维度

### 环境参数
- **alpha, beta**: Z曲线参数
- **tau_loc, tau_scan**: 成本评估参数
- **exclude_muted_cells**: 是否排除哑节点
- **min_cell_trajs**: 单元格最小轨迹数（用于剪枝）

## 输出文件

训练完成后会生成以下文件：

1. **模型文件**:
   - `models/final_model.pth`: 最终训练好的模型
   - `models/model_episode_N.pth`: 定期保存的检查点

2. **结果文件**:
   - `output/learned_order.txt`: 学习到的遍历顺序
   - `output/training_curves.png`: 训练曲线图（奖励、步数、改进率）

3. **评估结果**:
   - 控制台输出：每个评估周期的性能指标
   - 最终评估：使用test_queries计算的最终性能

