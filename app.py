import os
import json
import time
import math
import csv
import threading
import traceback
import numpy as np
from flask import Flask, jsonify, request, Response, send_from_directory

# Lazy import of grid modules to avoid crash if torch not installed
try:
    from grid_env import SmartGridEnv
    from dqn_agent import DQNAgent
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch not available. Running heuristic-only mode.")


# ------------------------------------------------------------------
# Flask App Setup
# ------------------------------------------------------------------

app = Flask(__name__, static_folder="static")

RESULTS_DIR = "results"
BEST_MODEL  = os.path.join(RESULTS_DIR, "best_model.pth")
FINAL_MODEL = os.path.join(RESULTS_DIR, "final_model.pth")
LOG_FILE    = os.path.join(RESULTS_DIR, "training_log.csv")

ACTION_NAMES = ["Idle", "Charge Battery", "Discharge Battery", "Buy from Grid", "Sell to Grid"]
ACTION_COLORS = ["#94a3b8", "#3b82f6", "#f97316", "#ef4444", "#22c55e"]
ACTION_ICONS  = ["⏸", "🔋↑", "🔋↓", "🏭→", "🏭←"]

# Thread lock for simulation state
sim_lock = threading.Lock()

# Global simulation state
sim = {
    "env": None,
    "agent": None,
    "model_loaded": False,
    "state": None,
    "hour": 0,
    "done": False,
    "history": [],  # list of step dicts
    "cumulative": {
        "total_reward": 0.0,
        "total_cost": 0.0,
        "total_renewable": 0.0,
        "total_demand": 0.0,
        "blackout_energy": 0.0,
    }
}


# ------------------------------------------------------------------
# Heuristic Policy (Fallback)
# ------------------------------------------------------------------

# Dynamic grid price schedule (mirrors grid_env.py)
def _grid_price_schedule(hour):
    if hour >= 23 or hour <= 6:
        return 0.08
    elif 7 <= hour <= 11:
        return 0.18
    elif 12 <= hour <= 16:
        return 0.22
    elif 17 <= hour <= 21:
        return 0.32
    else:
        return 0.14

def heuristic_action(state, hour):
    """
    Rule-based fallback policy.
    State: [solar, wind, demand, battery_soc, price, sin_t, cos_t, dow]
    Returns (action_int, reasoning_str)
    """
    solar, wind, demand, soc, price_norm, *_ = state
    price = _grid_price_schedule(hour)

    renewable = solar + wind   # normalized
    net = renewable - demand   # positive = surplus

    is_peak_price   = price >= 0.28
    is_cheap_price  = price <= 0.10
    battery_low     = soc < 0.25
    battery_high    = soc > 0.80
    battery_mid     = 0.25 <= soc <= 0.80
    surplus         = net > 0.1

    reasoning_parts = []

    if solar > 0.5:
        reasoning_parts.append("strong solar")
    elif solar > 0.15:
        reasoning_parts.append("moderate solar")

    if wind > 0.5:
        reasoning_parts.append("strong wind")

    if demand > 0.7:
        reasoning_parts.append("high demand")
    elif demand < 0.3:
        reasoning_parts.append("low demand")

    if battery_low:
        reasoning_parts.append("battery low")
    elif battery_high:
        reasoning_parts.append("battery full")

    if is_peak_price:
        reasoning_parts.append("peak tariff")
    elif is_cheap_price:
        reasoning_parts.append("cheap tariff")

    # Decision logic
    if surplus and battery_high and is_peak_price:
        action = 4  # Sell to grid — surplus + full battery + high price
        reasoning_parts.append("→ Sell to grid (max revenue)")

    elif surplus and not battery_high:
        action = 1  # Charge battery — surplus available
        reasoning_parts.append("→ Charge battery (free renewable)")

    elif not surplus and battery_high and is_peak_price:
        action = 2  # Discharge — high price, battery available
        reasoning_parts.append("→ Discharge battery (avoid peak cost)")

    elif not surplus and battery_mid and not is_peak_price:
        action = 3  # Buy from grid — deficit, moderate price
        reasoning_parts.append("→ Buy from grid (acceptable price)")

    elif not surplus and battery_low:
        action = 3  # Must buy — battery critically low
        reasoning_parts.append("→ Buy from grid (battery critical)")

    elif is_cheap_price and not battery_high:
        action = 3  # Buy cheap to charge
        reasoning_parts.append("→ Buy from grid (cheap tariff opportunity)")

    else:
        action = 0  # Idle
        reasoning_parts.append("→ Idle (balanced)")

    reasoning = ", ".join(reasoning_parts)
    return action, reasoning



def init_simulation():
    """Initialize or re-initialize the environment and agent."""
    with sim_lock:
        if TORCH_AVAILABLE:
            sim["env"] = SmartGridEnv(noise_level=0.05)
        else:
            sim["env"] = None

        sim["agent"] = None
        sim["model_loaded"] = False

        # Try loading best model
        if TORCH_AVAILABLE:
            agent = DQNAgent()
            model_path = BEST_MODEL if os.path.exists(BEST_MODEL) else FINAL_MODEL
            if agent.load(model_path):
                agent.epsilon = 0.0  # greedy for demo
                sim["agent"] = agent
                sim["model_loaded"] = True
                print(f"[app] DQN model loaded: {model_path}")
            else:
                print("[app] No model found — using heuristic policy")

        if sim["env"]:
            state, _ = sim["env"].reset()
            sim["state"] = state.tolist()
        else:
            # Fallback: generate a synthetic state
            sim["state"] = _synthetic_state(hour=0)

        sim["hour"] = 0
        sim["done"] = False
        sim["history"] = []
        sim["cumulative"] = {
            "total_reward": 0.0,
            "total_cost": 0.0,
            "total_renewable": 0.0,
            "total_demand": 0.0,
            "blackout_energy": 0.0,
        }


def _synthetic_state(hour):
    """Generate a plausible state vector without the Gym env."""
    solar = max(0, math.exp(-0.5 * ((hour - 12) / 3.5) ** 2)) if 6 <= hour <= 20 else 0
    wind  = 0.5 + 0.4 * math.cos((hour - 3) * math.pi / 12)
    demand = 0.3 + 0.6 * math.exp(-0.5 * ((hour - 19) / 2) ** 2) + \
             0.5 * math.exp(-0.5 * ((hour - 8) / 1.5) ** 2)
    soc   = 0.5
    price_norm = (_grid_price_schedule(hour) - 0.05) / 0.30
    time_sin = (math.sin(2 * math.pi * hour / 24) + 1) / 2
    time_cos = (math.cos(2 * math.pi * hour / 24) + 1) / 2
    dow = 0.0
    return [
        round(min(solar, 1), 3), round(min(wind, 1), 3), round(min(demand, 1), 3),
        soc, round(price_norm, 3), round(time_sin, 3), round(time_cos, 3), dow
    ]


def _run_step():
    """Run one simulation step and return the step data dict."""
    with sim_lock:
        if sim["done"]:
            return None

        hour = sim["hour"]
        state = sim["state"]

        # Get action and reasoning
        if sim["agent"] and sim["model_loaded"]:
            action = sim["agent"].select_action(state, greedy=True)
            q_values = sim["agent"].get_q_values(state)
            reasoning = f"DQN policy → {ACTION_NAMES[action]} (Q={q_values[action]:.3f})"
        else:
            action, reasoning = heuristic_action(state, hour)
            q_values = [0.0] * 5

        # Step environment or simulate synthetically
        if sim["env"]:
            next_state, reward, terminated, truncated, info = sim["env"].step(action)
            solar   = info["solar"]
            wind    = info["wind"]
            demand  = info["demand"]
            price   = info["price"]
            bat_soc = info["battery_soc"]
            blackout = info["blackout"]
            done    = terminated or truncated
            next_state_list = next_state.tolist()
        else:
            # Synthetic step (no gym env)
            solar   = state[0] * 10
            wind    = state[1] * 8
            demand  = state[2] * 12
            price   = _grid_price_schedule(hour)
            bat_soc = state[3] * 20
            blackout = 0.0
            reward  = float(solar + wind - demand * 0.5 - price * 2)
            done    = hour >= 23
            next_hour = (hour + 1) % 24
            next_state_list = _synthetic_state(next_hour)

        # Update cumulative stats
        c = sim["cumulative"]
        c["total_reward"]    += reward
        c["total_cost"]      += price * max(0, demand - solar - wind) if action == 3 else 0
        c["total_renewable"] += min(solar + wind, demand)
        c["total_demand"]    += demand
        c["blackout_energy"] += blackout

        renewable_ratio = (
            c["total_renewable"] / max(c["total_demand"], 1e-6)
        )

        step_data = {
            "hour":            hour,
            "solar":           round(solar, 3),
            "wind":            round(wind, 3),
            "demand":          round(demand, 3),
            "price":           round(price, 4),
            "battery_soc":     round(bat_soc, 3),
            "action":          action,
            "action_name":     ACTION_NAMES[action],
            "action_color":    ACTION_COLORS[action],
            "action_icon":     ACTION_ICONS[action],
            "reward":          round(reward, 4),
            "blackout":        round(blackout, 3),
            "q_values":        [round(q, 3) for q in q_values],
            "reasoning":       reasoning,
            "cumulative": {
                "total_reward":    round(c["total_reward"], 3),
                "total_cost":      round(c["total_cost"], 3),
                "renewable_ratio": round(renewable_ratio, 4),
                "blackout_energy": round(c["blackout_energy"], 3),
            },
        }

        sim["history"].append(step_data)
        sim["state"] = next_state_list
        sim["hour"] = hour + 1
        sim["done"] = done

        return step_data


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the dashboard HTML file."""
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    return send_from_directory(static_dir, "index.html")


@app.route("/api/status")
def api_status():
    return jsonify({
        "model_loaded":   sim.get("model_loaded", False),
        "torch_available": TORCH_AVAILABLE,
        "hour":           sim.get("hour", 0),
        "done":           sim.get("done", False),
        "history_len":    len(sim.get("history", [])),
    })


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Reset the simulation environment."""
    try:
        init_simulation()
        return jsonify({"status": "ok", "message": "Simulation reset", "model_loaded": sim["model_loaded"]})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/step", methods=["POST"])
def api_step():
    """Simulate one timestep."""
    try:
        if sim["env"] is None and sim["state"] is None:
            init_simulation()
        step_data = _run_step()
        if step_data is None:
            return jsonify({"status": "done", "message": "Episode finished"})
        return jsonify({"status": "ok", "data": step_data})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/run_episode")
def api_run_episode():
    """Stream a full 24-step episode via Server-Sent Events."""
    def generate():
        init_simulation()
        for _ in range(24):
            step_data = _run_step()
            if step_data is None:
                break
            yield f"data: {json.dumps(step_data)}\n\n"
            time.sleep(0.90)  # animation delay
        yield "data: {\"status\": \"episode_complete\"}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        }
    )


@app.route("/api/train_curve")
def api_train_curve():
    """Return training log data for the curve chart."""
    if not os.path.exists(LOG_FILE):
        return jsonify({"status": "demo", "data": _generate_demo_curve()})

    try:
        rows = []
        with open(LOG_FILE, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({
                    "episode":        int(row["episode"]),
                    "total_reward":   float(row["total_reward"]),
                    "renewable_ratio": float(row["renewable_ratio"]),
                    "total_cost":     float(row.get("total_cost", 0)),
                    "epsilon":        float(row["epsilon"]),
                    "eval_reward":    float(row["eval_reward"]) if row["eval_reward"] else None,
                })
        return jsonify({"status": "ok", "data": rows})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


def _generate_demo_curve():
    """Generate a realistic-looking demo training curve."""
    rows = []
    reward = -80
    renewable = 0.35
    for ep in range(1, 201):
        noise = np.random.normal(0, 8)
        reward = reward + 0.3 + noise
        renewable = min(0.92, renewable + 0.002 + np.random.normal(0, 0.01))
        epsilon = max(0.05, 1.0 * (0.995 ** ep))
        rows.append({
            "episode":        ep,
            "total_reward":   round(reward, 2),
            "renewable_ratio": round(renewable, 4),
            "total_cost":     round(max(0, 3.0 - ep * 0.01 + np.random.normal(0, 0.2)), 3),
            "epsilon":        round(epsilon, 4),
            "eval_reward":    round(reward + np.random.normal(0, 5), 2) if ep % 25 == 0 else None,
        })
    return rows


@app.route("/api/history")
def api_history():
    """Return full episode history."""
    with sim_lock:
        return jsonify({"status": "ok", "data": sim["history"]})


# ------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    init_simulation()
    print("\n[app] Smart Grid Dashboard running at http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
