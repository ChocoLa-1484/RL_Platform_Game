# easy.pdf 實作對照與實驗紀錄

依據 [easy.pdf](easy.pdf) 的兩頁提案，這次補上未見地圖測試與方法對照。原有五個固定關卡保留為操作展示；新的研究流程用三個階段訓練同一個模型，沒有依測試地圖切換專用策略。

## 文件要求與落實方式

| 文件位置 | 要求 | 實作 |
| --- | --- | --- |
| §2.1 環境 | 控制間距、寬度、高低差 | `procedural.py` 依階段與 seed 產生地形；玩家與代理共用 `PlatformerEnv` |
| §2.2 狀態 | 數值位置、速度、終點與平台幾何 | 18 維觀察，新增終點垂直距離；17 維舊模型相容 |
| §2.3 動作 | 六動作與 automated-jump baseline | 主模型六動作；auto-jump 模型學習不動／左／右三動作，落地即自動跳起 |
| §2.4 PPO | rollout、GAE、clipped objective | 延用 NumPy PPO，增加 entropy、approximate KL、clip fraction、value MSE 診斷 |
| §2.5 獎勵 | 平台 +1、checkpoint +5、goal +20、death −5、stall −0.1，避免重複領獎 | 提供逐項 reward components；平台與 checkpoint 只在首次落地發放；增加重複落地與往返位移測試 |
| §2.6 課程 | 寬平地 → 中等高低差 → 精準窄平台 | 每階段固定 200 次更新；後續階段 25% 回合重練先前階段，選模考量所有已見階段 |
| §3 預期結果 | 塑形 vs sparse、未見 procedural configurations | 同互動預算、三個訓練 seeds、分開 train／validation／test 地圖，記錄通關、死亡、軌跡進度與學習效率 |

提案中的「continuous velocity control」解讀為動量與加減速的連續物理狀態；動作仍遵照 §2.3 的離散動作定義。沒有把主模型改成連續動作演算法。

## 隨機關卡與資料切分

生成器版本為 `precision-v1`，寬 900、高 540，沿用原本重力、跳躍速度與碰撞模型。

| 階段 | 平台數 | 起始平台寬 | 中繼平台寬 | 間隙 | 相鄰高度變化 |
| --- | ---: | --- | --- | --- | --- |
| 1 | 3 | 180～220 | 235～310 | 45～80 | 全部 Y=450 |
| 2 | 4 | 160～180 | 110～150 | 60～90 | −55～55，Y 限於 335～450 |
| 3 | 5 | 140～160 | 45～75 | 75～105 | −85～70，Y 限於 270～450 |

末平台補足到畫面右界，終點 X=850；因此末平台寬度不採中繼平台的範圍。Y 軸向下為正。生成時檢查平台邊界、起終點、重疊與單跳的離散物理上限。**這是幾何與必要可達性篩選，不是每張地圖的完整可解性證明。**沒有將幾何檢查轉成代理的移動規則。

每個階段預先定義 1,000 個訓練 layout seeds、24 個驗證 layout seeds、100 個測試 layout seeds。三種 split 使用分開的 seed 範圍；額外比較完整地形的 SHA-256，確認這次三個 split 之間沒有相同地圖。`manifest.json` 保存生成器版本、全部訓練 seed 池、驗證／測試地形、hash 與實際起點。

所有評估起點 X 在 25～50，初速為零，從第一個平台開始。評估清單與動作 RNG 分開，因此不同模型與隨機基準使用相同地形和起點。訓練依回合結束重新抽地圖，策略不同會造成抽樣順序與看到的地圖數量不同；使用的是相同的訓練地圖池與分布。

## 公平對照設定

三個訓練 seeds：42、43、44。每組均從隨機權重開始，不載入舊五關模型，也不提供示範、動作規則或中途平台起點。

| 參數 | 值 |
| --- | --- |
| 每階段更新 | 200 |
| 環境數 × rollout 長度 | 8 × 256 |
| 每組總環境步數 | 1,228,800 |
| 全部 9 組總環境步數 | 11,059,200 |
| 每次更新 | 4 epochs，minibatch 256 |
| 學習率／entropy 係數 | 0.0005／0.02 |
| gamma／GAE lambda／PPO clip | 0.99／0.95／0.2 |
| 網路 | 18 → tanh(64) → policy／value |
| 課程回顧機率 | 後續階段 25% 回合取先前階段，75% 取目前階段 |
| 驗證頻率 | 每 20 次更新，以及各階段更新前 |

課程按固定預算切換，並非達到某個通關門檻才升級。每一階段以已見階段的平均驗證通關率選最佳 checkpoint；平手依平台覆蓋、最遠進度、步數排序。接續下一階段時重設 Adam 狀態。第三階段最佳 checkpoint 凍結後，才執行測試。

- `ppo_shaped`：六動作，原有塑形獎勵。
- `ppo_auto_jump`：三個方向動作，落地自動跳躍，使用相同塑形獎勵。這是明確指定的實驗基準；主模型沒有自動跳躍。
- `ppo_sparse`：六動作，僅成功 +20、掉落 −5。中間平台、位移、停滯與每步懲罰都為零。

稀疏與塑形獎勵的總 return 不直接比較；主要比較通關率、死亡率與步數。三種策略使用相同互動預算、課程與評估清單。

## 實測結果與解讀

每列為三個訓練 seeds 的平均 ± 樣本標準差。每個 seed、每階段測試 100 張未見地圖，共 2,700 個模型測試回合；另測 300 個隨機策略回合。

| 方法 | 階段 1 通關 | 階段 2 通關 | 階段 3 通關 |
| --- | ---: | ---: | ---: |
| PPO＋塑形 | 67.3% ± 28.3% | 96.3% ± 6.4% | 78.7% ± 25.8% |
| 自動跳躍 PPO＋塑形 | 51.0% ± 0.0% | 96.3% ± 6.4% | 85.0% ± 26.0% |
| PPO＋稀疏 | 33.3% ± 57.7% | 30.0% ± 51.1% | 15.0% ± 26.0% |
| 均勻隨機動作（單一固定 RNG） | 1/100 | 0/100 | 0/100 |

1. 這次預算內，塑形組的平均通關率高於稀疏組，但三個 seeds 仍有很大變異；不能用單次成功代表穩定收斂。
2. 自動跳躍組在階段 3 的平均通關率較高，階段 1 較低。這不支持「手動跳躍一定更好」的結論；需要更多 seeds 與更豐富地圖才能做可靠比較。
3. 階段 1 的部分寬平台仍會讓模型在不合適的位置連跳而掉落，說明簡單地形不保證對已學會的策略容易。課程回顧並未完全解決遺忘與策略偏差。
4. 這是相同生成器參數範圍內的未見地圖測試，不代表能適應任意高度、物理參數或其他遊戲。

![研究圖表](experiments/easy_pdf/comparison.png)

`learning_curve.csv` 記錄每次驗證的目前階段／已見階段通關率、累計環境步數與 PPO 診斷。曲線畫的是當下策略；最終模型使用最佳驗證 checkpoint，所以曲線最後一點不必等於最終測試表現。

`stage_steps_to_first_80pct_validation` 記錄目前階段第一次在驗證集達到 80% 所花的額外步數。0 代表轉移過來的策略已達標，`null` 表示本階段預算內沒達到，並非零成本成功。這是第一次達標，不保證後續維持：第 3 階段達標的訓練 seeds 數量為塑形組 2/3、自動跳躍組 2/3、稀疏組 0/3。

每個 `results.json` 另有死亡數、timeout 數、最遠進度、實際落過的平台比例、平均步數、成功回合平均步數／秒數、跳躍次數與落地水平速度。進度可能在掉落前接近 100%，因此必須與實際通關及平台覆蓋一起看。每個模型的 Wilson 95% 區間只描述測試地圖抽樣，不代替跨訓練 seeds 的變異。

## 播放與重現

```powershell
# 依驗證集選出的預設主模型：同一組權重跨三個階段。
python game.py --agent --procedural --stage 3 --trace

# 精確重播一張地圖並匯出逐幀狀態、動作、獎勵與落地事件。
python game.py --agent --procedural --stage 3 --layout-seed 2300000 --trace --headless --screenshot artifacts/procedural_stage3.png --trajectory artifacts/procedural_stage3_trajectory.json

# 自動跳躍模型會自動使用相符的三動作環境。
python game.py --agent --procedural --stage 3 --model-dir experiments/easy_pdf/ppo_auto_jump --trace

# 手動遊玩隨機關卡；加 --auto-jump 可體驗自動跳躍規則。
python game.py --procedural --stage 3

# 重新執行全部實驗，保留目前資料。
python run_experiments.py --output-dir experiments/my_run

# 重測原始全部九個模型，不重新選 checkpoint 或展示模型。
python run_experiments.py --evaluate-only

# 小預算確認流程；不適合用來下效能結論。
python run_experiments.py --seeds 42 --updates-per-stage 2 --validation-layouts 2 --test-layouts 2 --output-dir experiments/smoke

# 重畫原始圖表與執行回歸測試。
python -m pip install -r requirements-experiments.txt
python plot_experiments.py
python -m unittest -v
```

同一實驗目錄已有 `manifest.json` 時，訓練命令會要求改用新目錄，避免把新參數與舊報告混在一起。`--evaluate-only` 會讀原 manifest 的測試數量，並輸出 `matches_saved_results` 檢查是否一致。

`selected_policies.json` 記錄展示模型的來源。主模型 seed 43 是依最終 checkpoint 的驗證表現選出；測試結果為 100/100、100/100、96/100。展示模型沒有取代三個 seeds 的完整結果。

## 產物

- `experiments/easy_pdf/manifest.json`：參數、seed 池、評估地形及 hash。
- `experiments/easy_pdf/summary.json`、`summary.md`：跨 seeds 統計與原始模型結果。
- `experiments/easy_pdf/comparison.png`：學習曲線與未見地圖比較。
- `experiments/easy_pdf/<variant>/seed<N>/`：三階段權重、最終權重、結果、學習曲線 CSV、逐回合測試 CSV。
- `experiments/easy_pdf/<variant>/policy.json`：只按驗證集挑選的展示權重。
- `artifacts/procedural_stage3.png`、`procedural_stage3_trajectory.json`：實際新地圖通關畫面與逐幀軌跡。

下一步研究可增加訓練 seeds、用更大的驗證集分析 checkpoint 波動、增加長間隙／更窄落點的分布外測試，或加入沒有課程學習的對照。本次已比較獎勵與跳躍控制，尚未隔離課程學習本身的因果效果。
