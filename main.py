import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from jax import value_and_grad, jit
import optax
import os

# --- 1. 損失ありの物理エンジン (変更なし) ---
def create_lossy_engine(loss_db_per_mzi):
    attenuation = 10.0 ** (-loss_db_per_mzi / 20.0)

    def directional_coupler():
        val = 1.0 / jnp.sqrt(2.0)
        return jnp.array([[val, val * 1j], [val * 1j, val]])

    def pockels_phase_shifter(voltage):
        L = 2000e-6; d = 0.3e-6; wl = 1.55e-6; n = 3.5; r = 100e-12
        E = voltage / d
        dn = 0.5 * (n**3) * r * E
        phi = (2 * jnp.pi / wl) * dn * L
        return jnp.array([[jnp.exp(1j * phi), 0], [0, 1.0 + 0j]])

    def mzi_switch(voltage):
        DC = directional_coupler()
        PS = pockels_phase_shifter(voltage)
        return jnp.dot(DC, jnp.dot(PS, DC)) * attenuation

    @jit
    def simulate_mesh(voltages):
        T0 = mzi_switch(voltages[0]); T1 = mzi_switch(voltages[1])
        L1 = jnp.block([[T0, jnp.zeros((2,2))], [jnp.zeros((2,2)), T1]])
        T2 = mzi_switch(voltages[2])
        L2 = jnp.eye(4, dtype=complex); L2 = L2.at[1:3, 1:3].set(T2)
        T3 = mzi_switch(voltages[3]); T4 = mzi_switch(voltages[4])
        L3 = jnp.block([[T3, jnp.zeros((2,2))], [jnp.zeros((2,2)), T4]])
        T5 = mzi_switch(voltages[5])
        L4 = jnp.eye(4, dtype=complex); L4 = L4.at[1:3, 1:3].set(T5)
        U = jnp.dot(L4, jnp.dot(L3, jnp.dot(L2, L1)))
        return U

    return simulate_mesh

def run_normalized_loss_analysis():
    print("🚀 DiffPhoton: Normalized Loss Analysis (Signal-to-Noise Ratio)...")
    print("   Goal: Find the limit where Noise overwhelms the Signal.")

    img_1 = jnp.array([0.0, 0.70710678, 0.0, 0.70710678]) + 0j
    target_1 = jnp.array([0.0, 1.0, 0.0, 0.0])
    img_0 = jnp.array([0.5, 0.5, 0.5, -0.5]) + 0j
    target_0 = jnp.array([1.0, 0.0, 0.0, 0.0])

    # ノイズレベル (検出器の限界)
    NOISE_FLOOR = 0.001

    # テスト範囲を広げる (0dB 〜 10dBまでいじめてみる)
    loss_candidates = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    final_losses = []
    signal_levels = []

    print(f"   Detector Noise Floor: {NOISE_FLOOR}")

    for loss_val in loss_candidates:
        print(f"   Testing Loss = {loss_val:4} dB ...", end="", flush=True)
        
        mesh_fn = create_lossy_engine(loss_val)

        # ★★★ ここが修正ポイント：出力の正規化 ★★★
        @jit
        def predict_normalized(voltages, input_vec, key):
            U = mesh_fn(voltages)
            pure_out = jnp.abs(jnp.dot(U, input_vec))**2
            
            # 1. ノイズを足す
            noise = jax.random.normal(key, shape=pure_out.shape) * NOISE_FLOOR
            noisy_out = jnp.maximum(pure_out + noise, 0.0) # マイナスは0にする
            
            # 2. 合計値で割って「割合」にする (これが正規化)
            # これにより、光が弱くても [0.9, 0.1, 0, 0] のような比率になればOKとなる
            total_power = jnp.sum(noisy_out) + 1e-9 # ゼロ除算防止
            normalized_out = noisy_out / total_power
            
            return normalized_out, jnp.sum(pure_out) # 信号強度も返す

        @jit
        def loss_fn(params, key):
            k1, k2 = jax.random.split(key)
            p0, _ = predict_normalized(params, img_0, k1)
            p1, _ = predict_normalized(params, img_1, k2)
            return jnp.mean((p0-target_0)**2) + jnp.mean((p1-target_1)**2)

        # 学習実行
        key = jax.random.PRNGKey(42)
        params = jax.random.uniform(key, shape=(6,), minval=-0.1, maxval=0.1)
        optimizer = optax.adam(learning_rate=0.05)
        opt_state = optimizer.init(params)
        
        # ちょっと長めに学習して、粘らせる
        for i in range(1200):
            key, subkey = jax.random.split(key)
            grads = jax.grad(loss_fn)(params, subkey)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
        
        # 評価
        key, subkey = jax.random.split(key)
        final_l, power_val = loss_fn(params, subkey), 0.0 
        # パワー値だけ取り直すためのダミー呼び出し
        _, power_val = predict_normalized(params, img_0, subkey)
        
        final_losses.append(final_l)
        signal_levels.append(power_val)
        
        print(f" -> Loss={final_l:.4f} (Power={power_val:.4f})")

    # --- グラフ作成 ---
    if not os.path.exists('output'): os.makedirs('output')

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Component Loss (dB/MZI)', fontsize=12)
    ax1.set_ylabel('Normalized Classification Error', color=color, fontsize=12)
    ax1.plot(loss_candidates, final_losses, 'o-', color=color, linewidth=3, label="AI Error (Normalized)")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_yscale('log')
    ax1.set_ylim(1e-5, 1.0) # 範囲固定
    
    # 合格ライン
    ax1.axhline(y=0.01, color='gray', linestyle='--', label='Acceptable Limit')

    # 右軸：信号パワー
    ax2 = ax1.twinx() 
    color = 'tab:blue'
    ax2.set_ylabel('Raw Signal Power (Before Norm)', color=color, fontsize=12) 
    ax2.plot(loss_candidates, signal_levels, 'x--', color=color, label="Signal Strength")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yscale('log') # パワーもログスケールで見やすく

    plt.title("Optical Loss Limit: Signal vs Noise (Normalized)", fontsize=14)
    fig.tight_layout()
    
    output_path = "output/loss_analysis_normalized.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"✅ Analysis Complete.")
    print(f"   Graph saved to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    run_normalized_loss_analysis()