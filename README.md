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

# 消融：完全從零開始，或從不會衝刺的 baseline_v1 加寬後自行探索衝刺
python -B train_movement.py --from-scratch --phases 1 2 3 --updates-per-phase 1067 --output-dir experiments/state/my_scratch
python -B train_movement.py --init-policy experiments/state/baseline_v1/policy.json --extra-hidden 64 --refresh-dash --output-dir experiments/state/my_no_demo

# 失敗分析：重播 960 個測試案例，分類每次死亡；再畫報告用圖（需 requirements/analysis.txt）
python -B analyze_failures.py --agent state  --output-dir experiments/analysis/my_state
python -B analyze_failures.py --agent vision --output-dir experiments/analysis/my_vision
python -B analyze_failures.py --figures experiments/analysis/my_state experiments/analysis/my_vision --output-dir experiments/analysis/my_figures
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
| `rl_platformer/analysis/` | 失敗分析與報告用圖（不訓練） |
| `experiments/state/` | 原狀態 agent、示範初始化及比較基準 |
| `experiments/state/ablation/` | 消融與多 seed 實驗（E0–E3） |
| `experiments/vision/` | 每一輪 CNN 的模型、曲線與比較結果 |
| `experiments/analysis/` | 兩種 agent 的失敗分析與圖 |
| `tests/state/`、`tests/vision/` | 分開執行的測試 |
| `docs/`、`docs/reference/` | 操作文件、結構說明與原始 PDF |
| `artifacts/state/`、`artifacts/vision/` | 截圖與軌跡 |
| `requirements/` | 選用套件：`vision.txt`（CNN）、`analysis.txt`（畫圖） |
| `archive/` | 停用版本的可還原封存 |

根目錄五個 `.py` 只負責啟動，實作集中在 `rl_platformer/`。完整搬移紀錄、保留原因與路徑對照見 [專案結構](docs/PROJECT_STRUCTURE.md)。

## 模型與結果

兩種 agent 都在**同一組 960 個測試案例**（700 張程序生成地圖、13 個固定關卡各 20 個起點）上評估：

| 模型 | 保存位置 | 通關 | 成功局平均耗時 |
| --- | --- | ---: | ---: |
| 狀態 PPO（預設） | `experiments/state/dash_speedrun/policy.json` | 920 / 960，95.83% | 2.486 秒 |
| CNN 後續完成版本 | `experiments/vision/20261020_134343_080697/policy.pt` | 651 / 960，67.81% | 2.339 秒 |

雙方皆成功的 646 局，CNN 平均 2.336 秒、狀態 agent 2.321 秒；差距集中在窄平台與長路線（full 第 3／4 階段 CNN 13、6 局，狀態 agent 90、89 局）。CNN 以狀態 agent 為老師做行為複製與 DAgger，再跑 200 次 PPO 更新；但選中的 checkpoint 只在 PPO 第 10 次更新後（之後驗證未再提升），實質上是模仿學習的結果。因此這是完整系統的比較，不能單獨歸因於輸入形式。完整報告見 [eval_960_hw3_suite](experiments/vision/eval_960_hw3_suite/results.json)。

較早的 CNN 評估只用 109 局（每組 10 張地圖、每關 3 個起點，是上述 960 局的子集）：第一版 74 / 109、後續版本 81 / 109，同批狀態 agent 106 / 109。

**消融與多 seed**（[DASH_V2.md](docs/DASH_V2.md#消融與多-seed)）：示範初始化的三個 seed 為 923、935、925 / 960；不用示範或從零開始也能通關 93–95%，但成功局需 4.0–4.6 秒，示範初始化約 2.5 秒。

**失敗分析**（[experiments/analysis](experiments/analysis/)）：狀態 agent 97.6% 的爬升使用水平衝刺，成功落地時腳部餘裕中位數 10 px，13 次爬升失誤全部低於平台 0–15 px；CNN 的死亡以「跳不夠遠」為主，中位數差 33.5 px。

- [視覺 agent 操作、訓練與限制](docs/VISION.md)
- [狀態 agent、衝刺訓練、消融](docs/DASH_V2.md)
- [CNN 與狀態 agent 的 960 局同批比較](experiments/vision/eval_960_hw3_suite/results.json)

## 安裝與檢查

```powershell
python -m pip install -r requirements.txt

# 選用：只有 analyze_failures.py --figures 需要
python -m pip install -r requirements/analysis.txt

# 狀態與物理測試，不需要 PyTorch
python -B -m unittest discover -s tests/state -v

# 視覺測試，使用獨立程序避免 NumPy/MKL 與 PyTorch 混用
python -B -m unittest discover -s tests/vision -v

# 只重新評估，不覆寫既有結果
python -B train_movement.py --evaluate-only
python -B train_vision.py --evaluate-only experiments/vision/20261020_134343_080697 --no-preview

# CNN 改用狀態 agent 的 960 個測試案例；結果寫入新的目錄
python -B train_vision.py --evaluate-only experiments/vision/20261020_134343_080697 `
    --test-layouts 100 --fixed-starts 20 --report-dir experiments/vision/my_eval_960 --no-preview --device cpu
```

**重現性**：評估是確定性的，任何 NumPy 版本重跑都與保存結果相同。**訓練**則受 NumPy 版本的浮點差異影響：HW3 用 NumPy 1.26.4 得到 920 / 960，同設定在 NumPy 2.4.6 重跑為 923 / 960（`experiments/state/ablation/e0_reproduce_seed42`），選中的 checkpoint 也不同。消融比較以同環境重跑的 E0 為基準。

視覺套件安裝見 [VISION.md](docs/VISION.md)。本機 `.deps/` 與 `.vision_deps/` 是仍在使用的套件，保留於根目錄並由 Git 忽略。
