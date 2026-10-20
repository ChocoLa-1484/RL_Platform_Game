# RL Platform Game

PyGame 平台跳躍遊戲，包含 13 個固定關卡、隨機長路線與爬高關卡。支援按住時間決定跳高、八方向衝刺、落地恢復能量；衝刺觸發時先清除原速度再加速。

兩種 agent 分別保存，方便比較：**狀態 PPO** 讀取環境數值，**視覺 CNN** 只接收連續四張畫面。舊實驗已移出工作目錄；現用模型與歷史結果保持原內容。

## 執行

在專案根目錄執行，常用指令不變：

```powershell
# 自己遊玩
python -B game.py --level 13

# 原本的狀態 agent
python -B game.py --agent --level 13 --trace

# 最近儲存的視覺 agent
python -B play_vision.py --level 11

# 開啟即時畫面並訓練新的 CNN；每輪建立獨立目錄
python -B train_vision.py

# 從最新已完成 CNN 接續訓練，另存新一輪
python -B train_vision.py --init-policy experiments/vision/20261020_134343_080697/policy.pt

# 接續訓練狀態 PPO，指定新的輸出目錄
python -B train_movement.py --output-dir experiments/state/my_run
```

遊玩：A/D 或左右移動、Space 按久跳高、Shift/X 搭配方向鍵衝刺、R 重試、`[` / `]` 換關。CNN 訓練視窗：Space 暫停／繼續，Esc 儲存後停止。

## 資料夾

| 路徑 | 內容 |
| --- | --- |
| `rl_platformer/envs/` | 物理、跳躍／衝刺、像素環境 |
| `rl_platformer/levels/` | 隨機關卡生成器；固定關卡在環境模組 |
| `rl_platformer/agents/` | NumPy PPO 與 PyTorch CNN |
| `rl_platformer/training/` | 收集資料、訓練、示範搜尋與評估 |
| `rl_platformer/ui/` | 遊戲、播放與共用繪圖 |
| `experiments/state/` | 原狀態 agent、示範初始化及比較基準 |
| `experiments/vision/` | 每一輪 CNN 的模型、曲線與比較結果 |
| `tests/state/`、`tests/vision/` | 分開執行的測試 |
| `docs/`、`docs/reference/` | 操作文件、結構說明與原始 PDF |
| `artifacts/state/`、`artifacts/vision/` | 截圖與軌跡 |
| `requirements/` | 選用視覺套件需求 |
| `archive/` | 停用版本的可還原封存 |

根目錄四個 `.py` 只負責啟動，實作集中在 `rl_platformer/`。完整搬移紀錄、保留原因與路徑對照見 [專案結構](docs/PROJECT_STRUCTURE.md)。

## 模型與結果

| 模型 | 保存位置 | 已保存測試成績 |
| --- | --- | --- |
| 狀態 PPO（預設） | `experiments/state/dash_speedrun/policy.json` | 920 / 960，95.83% |
| CNN 第一版 | `experiments/vision/cnn_v1/policy.pt` | 74 / 109，67.89% |
| CNN 後續完成版本 | `experiments/vision/20261020_134343_080697/policy.pt` | 81 / 109，74.31% |

CNN 的同批 109 局比較中，狀態 agent 為 106 / 109（97.25%）。不同案例數的百分比不能直接比較；請使用各輪 `results.json` 中的同批結果與配對耗時。

- [視覺 agent 操作、訓練與限制](docs/VISION.md)
- [狀態 agent 與衝刺訓練](docs/DASH_V2.md)
- [最新 CNN 同批比較](experiments/vision/20261020_134343_080697/results.json)

## 安裝與檢查

```powershell
python -m pip install -r requirements.txt

# 狀態與物理測試，不需要 PyTorch
python -B -m unittest discover -s tests/state -v

# 視覺測試，使用獨立程序避免 NumPy/MKL 與 PyTorch 混用
python -B -m unittest discover -s tests/vision -v

# 只重新評估，不覆寫既有結果
python -B train_movement.py --evaluate-only
python -B train_vision.py --evaluate-only experiments/vision/20261020_134343_080697 --no-preview
```

視覺套件安裝見 [VISION.md](docs/VISION.md)。本機 `.deps/` 與 `.vision_deps/` 是仍在使用的套件，保留於根目錄並由 Git 忽略。
