# 專案分類與舊版清理

常用的四個入口保留在根目錄，原本的 `python -B game.py --agent --level 13` 仍可直接執行。入口內只呼叫套件的 `main()`，程式實作已依用途分類。

```text
RL_PlatGame/
├─ game.py                    手動遊玩／狀態 agent 播放
├─ play_vision.py             CNN 播放
├─ train_movement.py          狀態 PPO 訓練
├─ train_vision.py            CNN 訓練與即時預覽
├─ rl_platformer/
│  ├─ envs/                  環境與物理、像素觀察
│  ├─ levels/                三種隨機地形生成器
│  ├─ agents/                NumPy PPO、PyTorch CNN
│  ├─ training/              訓練、評估、示範搜尋、共用 rollout
│  ├─ ui/                    遊戲、播放、共用繪圖
│  ├─ utils/                 JSON／CSV 輸出
│  └─ paths.py               根目錄與舊 manifest 路徑相容
├─ tests/
│  ├─ state/                 物理與狀態 PPO 測試
│  └─ vision/                需要 PyTorch 的獨立測試
├─ experiments/
│  ├─ state/
│  │  ├─ dash_speedrun/       目前預設狀態 agent 與完整結果
│  │  ├─ dash_demonstrations_v2/  訓練初始化、示範資料與來源
│  │  └─ baseline_v1/         衝刺研究的比較基準
│  └─ vision/
│     ├─ cnn_v1/              第一版 CNN，含中止後 checkpoint
│     ├─ cnn_v1_evaluation/   第一版與狀態 agent 的比較
│     └─ 20261020_134343_080697/  後續完成訓練，含完整結果
├─ artifacts/
│  ├─ state/                 現用狀態 agent 的截圖與軌跡
│  └─ vision/                CNN 截圖
├─ docs/
│  ├─ VISION.md              視覺 agent 操作與結果
│  ├─ DASH_V2.md             衝刺、狀態 agent 訓練與結果
│  └─ reference/             原始 easy.pdf 與 Full_version.pdf
├─ requirements/vision.txt   視覺 agent 選用套件
├─ requirements.txt          基礎套件
└─ archive/
   ├─ retired_20261020.zip   停用版本的可還原備份
   └─ retired_20261020_manifest.json  備份逐檔 SHA256
```

## 路徑對照

| 原路徑 | 現在路徑 |
| --- | --- |
| `movement.py`、`platformer.py`、`vision_env.py` | `rl_platformer/envs/` |
| `ppo.py`、`vision_agent.py` | `rl_platformer/agents/` |
| `procedural.py`、`full_curriculum.py`、`height_curriculum.py` | `rl_platformer/levels/` |
| `bootstrap_dash.py`、`compare_movement.py` | `rl_platformer/training/` |
| `VISION.md`、`DASH_V2.md` | `docs/` |
| `experiments/dash_speedrun/` | `experiments/state/dash_speedrun/` |
| `experiments/dash_demonstrations_v2/` | `experiments/state/dash_demonstrations_v2/` |
| `experiments/movement/` | `experiments/state/baseline_v1/` |
| `tests_vision/` | `tests/vision/` |

示範與狀態比較工具改用套件入口：

```powershell
python -B -m rl_platformer.training.bootstrap_dash --output-dir experiments/state/new_demonstrations
python -B -m rl_platformer.training.compare_movement --policy experiments/state/dash_speedrun/policy.json --output experiments/state/new_comparison.json
```

## 保留與移除的範圍

目前兩種 agent、所有視覺訓練版本、現用狀態 agent、示範資料與比較基準全部保留。`.deps/`、`.vision_deps/` 是本機仍在使用的套件，不屬於無用檔案。

停用的無衝刺／自動跳躍實驗（`easy_pdf`、`full_version`、`height_curriculum`）、早期示範 `dash_demonstrations`、未採用候選 `movement_dash`／`movement_reset`／`movement_speed`／`movement_v2`，以及對應的舊訓練入口、繪圖工具、文件、截圖及暫存驗證資料，已從工作目錄移除。

這些內容先完整備份到 `archive/retired_20261020.zip`，共 **174 個檔案、約 22.5 MB**；所有 ZIP 內容與來源逐檔 SHA256 相符後，才清除解壓縮的重複副本。需要查閱歷史時可解壓到另外的資料夾，避免覆蓋現用程式。備份保留歷史內容，不是目前維護的可執行入口。

現用遊戲入口不再提供 `--classic`／`--auto-jump` 舊實驗選項；共用基礎物理與模型讀取相容性仍保留。`baseline_v1` 是仍用於示範搜尋與配對比較的參考模型，因此沒有刪除。

## 完整性與重現

保留模型、CSV、manifest 與結果 JSON 的內容不變。舊 manifest 的絕對路徑透過 `rl_platformer/paths.py` 對應到新位置；不改寫歷史紀錄中的來源、參數或雜湊。

- [分類搬移清單與原始雜湊](cleanup_manifest.json)
- [保留檔案與連結檢查結果](reorganization_verification.json)
- [ZIP 備份逐檔雜湊](../archive/retired_20261020_manifest.json)

測試分成兩個程序，避免舊 Anaconda 的 NumPy/MKL 與 PyTorch 同時載入產生衝突：

```powershell
python -B -m unittest discover -s tests/state -v
python -B -m unittest discover -s tests/vision -v
```

整理後已通過 49 項狀態／物理／路徑測試與 8 項視覺測試。狀態模型 960 局與後續 CNN 109 局重測均與原本保存結果一致；狀態 agent 第 13 關仍為 283 幀通關。
