from __future__ import annotations

import os
import sys
import math
import json
import time
import subprocess
import webbrowser
from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces

import requests
import plotly.express as px
import plotly.graph_objects as go

# Streamlit
try:
    import streamlit as st
except Exception:
    st = None  # type: ignore

# Stable Baselines 3 + Vec Env
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
except Exception:
    PPO = None  # type: ignore
    DummyVecEnv = None  # type: ignore


# -----------------------------
# Config
# -----------------------------
@dataclass
class GridConfig:
    horizon: int = 24

    demand_min: float = 40.0
    demand_max: float = 120.0

    renewable_min: float = 0.0
    renewable_max: float = 90.0

    fossil_max: float = 140.0

    fossil_cost_per_kwh: float = 0.22
    co2_kg_per_kwh: float = 0.70

    curtailment_penalty_per_kwh: float = 0.02
    unserved_penalty_per_kwh: float = 5.0
    ramp_penalty_per_kw_change: float = 0.03

    alpha_cost: float = 1.0
    beta_co2: float = 0.6

    seed: int = 42


# -----------------------------
# Environment
# -----------------------------
class SmartGridEnv(gym.Env):
    """
    Obs: [demand_norm, renewable_norm, prev_fossil_norm, time_norm] in [0,1]
    Action: a in [0,1] = renewable fraction preference
    Reward: -(cost + CO2 + unserved penalty + ramp penalty)
    """
    metadata = {"render_modes": []}

    def __init__(self, cfg: GridConfig):
        super().__init__()
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

        self.t = 0
        self.demand = 0.0
        self.renewable = 0.0
        self.prev_fossil = 0.0

        self.ep_cost = 0.0
        self.ep_co2 = 0.0
        self.ep_unserved = 0.0
        self.ep_curtail = 0.0
        self.ep_ramp = 0.0

    def _sample_demand(self, t: int) -> float:
        base = self.rng.uniform(self.cfg.demand_min, self.cfg.demand_max)
        peak_factor = 1.0 + 0.15 * math.sin((2 * math.pi * t) / max(self.cfg.horizon, 1))
        noise = self.rng.normal(0, 5.0)
        return float(np.clip(base * peak_factor + noise, self.cfg.demand_min, self.cfg.demand_max))

    def _sample_renewable(self, t: int) -> float:
        daylight = max(0.0, math.sin((math.pi * t) / max(self.cfg.horizon, 1)))
        base = self.rng.uniform(self.cfg.renewable_min, self.cfg.renewable_max)
        noise = self.rng.normal(0, 4.0)
        gen = base * daylight + noise
        return float(np.clip(gen, self.cfg.renewable_min, self.cfg.renewable_max))

    def _obs(self) -> np.ndarray:
        d_norm = (self.demand - self.cfg.demand_min) / (self.cfg.demand_max - self.cfg.demand_min + 1e-9)
        r_norm = (self.renewable - self.cfg.renewable_min) / (self.cfg.renewable_max - self.cfg.renewable_min + 1e-9)
        f_norm = self.prev_fossil / (self.cfg.fossil_max + 1e-9)
        t_norm = self.t / (self.cfg.horizon - 1 + 1e-9)
        return np.array([d_norm, r_norm, f_norm, t_norm], dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.t = 0
        self.prev_fossil = 0.0

        self.ep_cost = 0.0
        self.ep_co2 = 0.0
        self.ep_unserved = 0.0
        self.ep_curtail = 0.0
        self.ep_ramp = 0.0

        self.demand = self._sample_demand(self.t)
        self.renewable = self._sample_renewable(self.t)
        return self._obs(), {}

    def step(self, action: np.ndarray):
        a = float(np.clip(action[0], 0.0, 1.0))

        demand = self.demand
        renewable = self.renewable

        renewable_target = a * demand
        renewable_used = min(renewable, renewable_target)

        remaining = max(demand - renewable_used, 0.0)
        fossil = min(remaining, self.cfg.fossil_max)
        served = renewable_used + fossil

        unserved = max(demand - served, 0.0)
        curtail = max(renewable - renewable_used, 0.0)

        cost = fossil * self.cfg.fossil_cost_per_kwh + curtail * self.cfg.curtailment_penalty_per_kwh
        co2 = fossil * self.cfg.co2_kg_per_kwh

        ramp = abs(fossil - self.prev_fossil)
        ramp_pen = ramp * self.cfg.ramp_penalty_per_kw_change
        unserved_pen = unserved * self.cfg.unserved_penalty_per_kwh

        reward = -(self.cfg.alpha_cost * cost + self.cfg.beta_co2 * co2 + unserved_pen + ramp_pen)

        self.ep_cost += cost
        self.ep_co2 += co2
        self.ep_unserved += unserved
        self.ep_curtail += curtail
        self.ep_ramp += ramp

        self.prev_fossil = fossil
        self.t += 1
        terminated = self.t >= self.cfg.horizon

        if not terminated:
            self.demand = self._sample_demand(self.t)
            self.renewable = self._sample_renewable(self.t)

        info = {
            "demand": demand,
            "renewable": renewable,
            "renewable_used": renewable_used,
            "fossil": fossil,
            "unserved": unserved,
            "curtail": curtail,
            "cost": cost,
            "co2": co2,
            "ramp": ramp,
            "episode_cost": self.ep_cost,
            "episode_co2": self.ep_co2,
            "episode_unserved": self.ep_unserved,
            "episode_ramp": self.ep_ramp,
        }

        return self._obs(), float(reward), terminated, False, info


# -----------------------------
# Baseline policy
# -----------------------------
def baseline_policy(obs: np.ndarray) -> np.ndarray:
    _, renewable_norm, _, _ = obs
    a = 0.25 + 0.75 * renewable_norm
    return np.array([float(np.clip(a, 0.0, 1.0))], dtype=np.float32)


# -----------------------------
# Helpers
# -----------------------------
def ensure_results_dir():
    os.makedirs("results", exist_ok=True)

def pct_reduction(base: float, rl: float) -> float:
    if base == 0:
        return 0.0
    return 100.0 * (base - rl) / base

def run_episode(env: SmartGridEnv, policy_fn) -> Dict[str, Any]:
    obs, _ = env.reset()
    done = False
    rewards = []
    logs = []
    while not done:
        action = policy_fn(obs)
        obs, r, done, _, info = env.step(action)
        rewards.append(r)
        logs.append(info)
    last = logs[-1]
    return {
        "total_reward": float(np.sum(rewards)),
        "episode_cost": float(last["episode_cost"]),
        "episode_co2": float(last["episode_co2"]),
        "episode_unserved": float(last["episode_unserved"]),
        "episode_ramp": float(last["episode_ramp"]),
    }


# -----------------------------
# Training / Evaluation
# -----------------------------
def train_model(cfg: GridConfig, timesteps: int, model_path: str = "results/rl_agent"):
    if PPO is None or DummyVecEnv is None:
        raise RuntimeError("stable-baselines3 missing. Install: pip install stable-baselines3 shimmy")

    ensure_results_dir()

    def make_env():
        return SmartGridEnv(cfg)

    venv = DummyVecEnv([make_env])

    policy_kwargs = dict(net_arch=[64, 64])  # small = fast

    model = PPO(
        "MlpPolicy",
        env=venv,
        verbose=0,
        n_steps=cfg.horizon * 4,
        batch_size=cfg.horizon * 4,
        learning_rate=3e-4,
        gamma=0.99,
        policy_kwargs=policy_kwargs,
        device="cpu",
    )
    model.learn(total_timesteps=int(timesteps))
    model.save(model_path)

def evaluate_model(cfg: GridConfig, episodes: int, model_path: str = "results/rl_agent") -> Dict[str, Any]:
    if PPO is None:
        raise RuntimeError("stable-baselines3 not installed.")

    ensure_results_dir()

    env = SmartGridEnv(cfg)

    baseline_eps = [run_episode(env, baseline_policy) for _ in range(int(episodes))]
    base_df = pd.DataFrame(baseline_eps)

    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"Model not found: {model_path}.zip (Train first!)")

    model = PPO.load(model_path)

    def rl_policy(obs):
        act, _ = model.predict(obs, deterministic=True)
        return np.array(act, dtype=np.float32)

    rl_eps = [run_episode(env, rl_policy) for _ in range(int(episodes))]
    rl_df = pd.DataFrame(rl_eps)

    base_df.to_csv("results/baseline_episodes.csv", index=False)
    rl_df.to_csv("results/rl_episodes.csv", index=False)

    metrics = {
        "episodes": int(episodes),
        "baseline_mean_cost": float(base_df["episode_cost"].mean()),
        "rl_mean_cost": float(rl_df["episode_cost"].mean()),
        "baseline_mean_co2": float(base_df["episode_co2"].mean()),
        "rl_mean_co2": float(rl_df["episode_co2"].mean()),
        "baseline_mean_unserved": float(base_df["episode_unserved"].mean()),
        "rl_mean_unserved": float(rl_df["episode_unserved"].mean()),
        "cost_reduction_percent": pct_reduction(float(base_df["episode_cost"].mean()), float(rl_df["episode_cost"].mean())),
        "co2_reduction_percent": pct_reduction(float(base_df["episode_co2"].mean()), float(rl_df["episode_co2"].mean())),
    }

    with open("results/metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    _save_matplotlib_plots(base_df, rl_df)
    return metrics

def _save_matplotlib_plots(base_df: pd.DataFrame, rl_df: pd.DataFrame):
    ensure_results_dir()

    plt.figure()
    plt.plot(base_df["episode_cost"].rolling(10, min_periods=1).mean(), label="Baseline")
    plt.plot(rl_df["episode_cost"].rolling(10, min_periods=1).mean(), label="RL")
    plt.title("Cost Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Cost")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/cost_comparison.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(base_df["episode_co2"].rolling(10, min_periods=1).mean(), label="Baseline")
    plt.plot(rl_df["episode_co2"].rolling(10, min_periods=1).mean(), label="RL")
    plt.title("CO₂ Comparison")
    plt.xlabel("Episode")
    plt.ylabel("CO₂")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/emissions_comparison.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(base_df["total_reward"].rolling(10, min_periods=1).mean(), label="Baseline")
    plt.plot(rl_df["total_reward"].rolling(10, min_periods=1).mean(), label="RL")
    plt.title("Reward Convergence")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/reward_comparison.png", dpi=160)
    plt.close()


# -----------------------------
# AQI
# -----------------------------
CITY_COORDS = {
    "Lahore": (31.5204, 74.3587),
    "Karachi": (24.8607, 67.0011),
    "Islamabad": (33.6844, 73.0479),
    "Peshawar": (34.0151, 71.5249),
    "Quetta": (30.1798, 66.9750),
    "Delhi": (28.6139, 77.2090),
    "Dubai": (25.2048, 55.2708),
}

def pm25_to_us_aqi(pm25: float) -> int:
    bps = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    pm25 = max(0.0, float(pm25))
    for cl, ch, il, ih in bps:
        if cl <= pm25 <= ch:
            aqi = (ih - il) / (ch - cl) * (pm25 - cl) + il
            return int(round(aqi))
    return 500

def aqi_category(aqi: int) -> str:
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy (Sensitive)"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "Very Unhealthy"
    return "Hazardous"

def fetch_city_aqi(city: str) -> dict:
    lat, lon = CITY_COORDS[city]
    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=pm2_5&timezone=auto"
    )
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()

    times = data["hourly"]["time"]
    pm25_vals = data["hourly"]["pm2_5"]
    idx = len(pm25_vals) - 1
    while idx >= 0 and pm25_vals[idx] is None:
        idx -= 1
    pm25 = float(pm25_vals[idx])
    aqi = pm25_to_us_aqi(pm25)
    return {"city": city, "pm2_5": pm25, "aqi": aqi, "category": aqi_category(aqi), "time": times[idx]}


# -----------------------------
# Streamlit GUI
# -----------------------------
def gui():
    if st is None:
        raise RuntimeError("streamlit not installed.")

    st.set_page_config(page_title="AI Smart Grid RL Optimizer", layout="wide")
    st.title("⚡ GRID VIBE ")
    st.caption("OPTIMIZING ENERGY.REDUCING EMISSION.TRANSFORMING THE GRID WITH AI.")

    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Train", "Evaluate", "Playback"], index=0)

    st.sidebar.header("Quick Settings")
    cfg = GridConfig(
        horizon=int(st.sidebar.slider("Horizon (steps)", 12, 96, 24, 1)),
        alpha_cost=float(st.sidebar.number_input("Alpha (Cost Weight)", value=1.0)),
        beta_co2=float(st.sidebar.number_input("Beta (CO₂ Weight)", value=0.6)),
        seed=int(st.sidebar.number_input("Seed", value=42, step=1)),
    )

    # FAST defaults
    timesteps = int(st.sidebar.number_input("Training Timesteps", value=5000, step=1000))
    episodes = int(st.sidebar.number_input("Evaluation Episodes", value=50, step=10))
    model_path = "results/rl_agent"

    if page == "Dashboard":
        st.subheader("🌍 AQI ")
        enable_aqi = st.checkbox("Fetch live AQI ", value=True)
        if enable_aqi:
            city = st.selectbox("City", list(CITY_COORDS.keys()), index=0)

            if hasattr(st, "cache_data"):
                @st.cache_data(ttl=900)
                def cached(c):
                    return fetch_city_aqi(c)
            else:
                def cached(c):
                    return fetch_city_aqi(c)

            try:
                aqi = cached(city)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("City", aqi["city"])
                c2.metric("AQI", aqi["aqi"])
                c3.metric("PM2.5", f'{aqi["pm2_5"]:.1f} µg/m³')
                c4.metric("Category", aqi["category"])
                st.caption(f"Last updated: {aqi['time']}")
            except Exception as e:
                st.warning(f"AQI fetch failed: {e}")
        else:
            st.info("AQI disabled for speed.")

        st.subheader("✅ Metrics Summary")
        if os.path.exists("results/metrics_summary.json"):
            with open("results/metrics_summary.json", "r", encoding="utf-8") as f:
                m = json.load(f)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Cost Reduction (%)", f'{m["cost_reduction_percent"]:.2f}')
            k2.metric("CO₂ Reduction (%)", f'{m["co2_reduction_percent"]:.2f}')
            k3.metric("Baseline Mean Cost", f'{m["baseline_mean_cost"]:.3f}')
            k4.metric("RL Mean Cost", f'{m["rl_mean_cost"]:.3f}')
        else:
            st.info("No metrics yet. Run Evaluate first.")

        st.subheader("📊 Interactive Results")
        if os.path.exists("results/baseline_episodes.csv") and os.path.exists("results/rl_episodes.csv"):
            base_df = pd.read_csv("results/baseline_episodes.csv")
            rl_df = pd.read_csv("results/rl_episodes.csv")
            base_df["episode"] = range(1, len(base_df) + 1)
            rl_df["episode"] = range(1, len(rl_df) + 1)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=base_df["episode"], y=base_df["episode_cost"], mode="lines", name="Baseline"))
            fig.add_trace(go.Scatter(x=rl_df["episode"], y=rl_df["episode_cost"], mode="lines", name="RL"))
            fig.update_layout(title="Cost per Episode", xaxis_title="Episode", yaxis_title="Cost")
            st.plotly_chart(fig, use_container_width=True)

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=base_df["episode"], y=base_df["episode_co2"], mode="lines", name="Baseline"))
            fig2.add_trace(go.Scatter(x=rl_df["episode"], y=rl_df["episode_co2"], mode="lines", name="RL"))
            fig2.update_layout(title="CO₂ per Episode", xaxis_title="Episode", yaxis_title="CO₂")
            st.plotly_chart(fig2, use_container_width=True)

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=base_df["episode"], y=base_df["total_reward"], mode="lines", name="Baseline"))
            fig3.add_trace(go.Scatter(x=rl_df["episode"], y=rl_df["total_reward"], mode="lines", name="RL"))
            fig3.update_layout(title="Reward Convergence", xaxis_title="Episode", yaxis_title="Reward")
            st.plotly_chart(fig3, use_container_width=True)

            merged = pd.concat([base_df.assign(controller="Baseline"), rl_df.assign(controller="RL")], ignore_index=True)
            colA, colB = st.columns(2)
            with colA:
                st.plotly_chart(px.line(merged, x="episode", y="episode_unserved", color="controller",
                                        title="Stability: Unserved Energy"), use_container_width=True)
            with colB:
                st.plotly_chart(px.line(merged, x="episode", y="episode_ramp", color="controller",
                                        title="Stability: Ramp"), use_container_width=True)
        else:
            st.info("No evaluation CSVs found. Run Evaluate page first.")

    elif page == "Train":
        st.subheader("🧠 Train PPO RL Agent ")
        st.write("Start with 5000 timesteps. Increase later if you want better results.")
        if st.button("Train Now"):
            try:
                with st.spinner("Training..."):
                    train_model(cfg, timesteps=timesteps, model_path=model_path)
                st.success("Training done! Model saved in results/rl_agent.zip")
            except Exception as e:
                st.error(str(e))

    elif page == "Evaluate":
        st.subheader("📈 Evaluate RL vs Baseline ")
        if st.button("Evaluate Now"):
            try:
                with st.spinner("Evaluating..."):
                    m = evaluate_model(cfg, episodes=episodes, model_path=model_path)
                st.success("Evaluation complete! Results saved in results/")
                st.json(m)
            except Exception as e:
                st.error(str(e))

    else:
        st.subheader("▶️ One-Day Playback (Baseline)")
        if st.button("Run Playback"):
            env = SmartGridEnv(cfg)
            obs, _ = env.reset()
            rows = []
            for t in range(cfg.horizon):
                act = baseline_policy(obs)
                obs, r, done, _, info = env.step(act)
                rows.append({
                    "t": t,
                    "demand": info["demand"],
                    "renewable": info["renewable"],
                    "renewable_used": info["renewable_used"],
                    "fossil": info["fossil"],
                    "unserved": info["unserved"],
                })
                if done:
                    break

            df = pd.DataFrame(rows)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["t"], y=df["demand"], mode="lines", name="Demand"))
            fig.add_trace(go.Scatter(x=df["t"], y=df["renewable"], mode="lines", name="Renewable"))
            fig.add_trace(go.Scatter(x=df["t"], y=df["renewable_used"], mode="lines", name="Renewable Used"))
            fig.add_trace(go.Scatter(x=df["t"], y=df["fossil"], mode="lines", name="Fossil"))
            fig.update_layout(title="Playback Dispatch (24 steps)", xaxis_title="Step", yaxis_title="kW (sim)")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df, use_container_width=True)


# -----------------------------
# AUTO-LAUNCH (no loops)
# -----------------------------
def auto_launch():
    # If already inside streamlit run, show GUI
    if os.environ.get("RUNNING_IN_STREAMLIT") == "1":
        gui()
        return

    print("Launching Streamlit GUI...")

    env = os.environ.copy()
    env["RUNNING_IN_STREAMLIT"] = "1"

    cmd = [sys.executable, "-m", "streamlit", "run", sys.argv[0], "--server.headless=true"]
    p = subprocess.Popen(cmd, env=env)

    time.sleep(2)
    try:
        webbrowser.open("http://localhost:8501")
    except Exception:
        print("Open manually: http://localhost:8501")

    p.wait()


if __name__ == "__main__":
    auto_launch()
