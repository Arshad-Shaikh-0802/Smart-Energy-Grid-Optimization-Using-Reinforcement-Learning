# ⚡ Smart Energy Grid Optimization — DQN + Real-Time Dashboard

A complete end-to-end Reinforcement Learning system for smart microgrid energy management. Uses Deep Q-Networks (DQN) to optimize energy flow across solar, wind, battery, and grid components — with a professional real-time web dashboard.

---

## Project Structure

```
smart_grid_rl/
├── grid_env.py        ← Custom Gymnasium environment (microgrid simulation)
├── dqn_agent.py       ← DQN agent (Double DQN, replay buffer, target network)
├── train.py           ← Training pipeline (500 episodes, logs, model saves)
├── app.py             ← Flask backend API + SSE streaming
├── requirements.txt
├── static/
│   └── index.html     ← Real-time dashboard (Chart.js, pure HTML/JS)
└── results/           ← Created after training
    ├── training_log.csv
    ├── best_model.pth
    └── final_model.pth
```

---

## Quickstart

```bash
pip install -r requirements.txt

# Train the agent first:
python train.py --episodes 500

# Start the dashboard:
python app.py
# Then open: http://localhost:5000
```

---

## Components

### Grid Environment (`grid_env.py`)
- **Solar**: Bell-curve generation peaking at noon (0–10 kW)
- **Wind**: Sinusoidal pattern, stronger at night (0–8 kW)
- **Demand**: Morning and evening peaks (0–12 kW)
- **Battery**: 20 kWh capacity, 92% round-trip efficiency
- **Pricing**: Dynamic tariff — cheap off-peak (8¢), expensive peak (32¢)

State space: `[solar, wind, demand, battery_soc, price, sin_t, cos_t, day_of_week]`  
Actions: `Idle | Charge | Discharge | Buy Grid | Sell Grid`

### DQN Agent (`dqn_agent.py`)
- Neural net: `8 → 128 → 128 → 64 → 5`
- **Double DQN** to reduce Q-value overestimation
- **Experience Replay** (50k buffer)
- **Target Network** (synced every 200 steps)
- Epsilon-greedy: 1.0 → 0.05 with multiplicative decay
- Huber loss + gradient clipping

### Training (`train.py`)
```bash
python train.py --episodes 500 --eval-every 25 --noise 0.1
```
Saves `best_model.pth` whenever evaluation reward improves.

### Dashboard Features
- ⚡ **Live Energy Flow** chart (solar, wind, demand over 24h)
- 🔋 **Battery gauge** with SoC visualization
- 💰 **Grid price indicator** with color-coded tariff zones
- 📊 **Reward accumulation** bar + line chart
- 🤖 **Agent reasoning** with Q-value bars
- 📋 **Action log** with color-coded entries + explanations
- 📈 **Training curve** with smoothing (reward / renewable / epsilon tabs)
- ▶ **Step-by-step** or full **episode animation** (SSE streaming)
