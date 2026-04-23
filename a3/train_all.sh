#!/bin/bash
# ============================================================================
#  train_all.sh — Train all 3 SAC configurations with live monitoring
#
#  Usage:
#    chmod +x train_all.sh
#    ./train_all.sh              # Train all 3 configs (1M steps each)
#    ./train_all.sh 500000       # Train all 3 configs (custom timesteps)
#
#  Monitor in a separate terminal:
#    tensorboard --logdir logs/ --port 6006
#    Then open http://localhost:6006 in your browser
# ============================================================================

set -e

TIMESTEPS=${1:-1000000}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        SAC Biped Jump — Full Training Pipeline              ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Timesteps per config : $TIMESTEPS"
echo "║  Configs              : config_3                            ║"
echo "║  GPU                  : $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'CPU')"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Create separate log dirs for each config so TensorBoard shows them side by side
for i in 3; do
    mkdir -p "logs/config_${i}"
    mkdir -p "models/config_${i}_best"
done

TOTAL_START=$(date +%s)

# ── Config 3: Supercharged Jumper ────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [3/3] Training config_3 (Conservative) — $TIMESTEPS steps"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f "models/config_3_final.zip" ]; then
    echo "  ⏭️ Config 3 already completed. Skipping."
else
    START=$(date +%s)
    python3 main.py --mode train --config config_3 --resume --timesteps "$TIMESTEPS"
    END=$(date +%s)
    ELAPSED_3=$((END - START))
    echo "  ✅ Config 3 done in $(($ELAPSED_3 / 60))m $(($ELAPSED_3 % 60))s"

    cp -f models/sac_biped_goal.zip "models/config_3_final.zip"
    cp -f models/sac_best_config_3/best_model.zip "models/config_3_best/best_model.zip" 2>/dev/null || true
    cp -f reward_curve_sac_config_3.png "reward_curve_config_3.png" 2>/dev/null || true
    echo ""
fi

# ── Evaluate all 3 configs ────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Running evaluation (10 episodes)..."

echo ""
echo "── Config 3 (Conservative) ──"
python3 main.py --mode test --model_path "models/config_3_best/best_model" --episodes 10

# ── Summary ───────────────────────────────────────────────────────────────────
TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                   TRAINING COMPLETE                         ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Config 3 (Supercharged Jumper): $((${ELAPSED_3:-0} / 60))m $((${ELAPSED_3:-0} % 60))s"
echo "║  Total wall time            : $(($TOTAL_ELAPSED / 60))m $(($TOTAL_ELAPSED % 60))s"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Reward curves: reward_curve_config_3.png                   ║"
echo "║  Best models : models/config_3_best/best_model.zip          ║"
echo "║  TensorBoard : tensorboard --logdir logs/                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── Auto-shutdown after training ──────────────────────────────────────────────
echo ""
echo "  🔌 Training complete. Shutting down in 60 seconds..."
echo "     (Run 'sudo shutdown -c' to cancel)"
sudo shutdown -h +1 "SAC training finished. Auto-shutdown."
