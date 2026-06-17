❯ 1. 做這個會開始收錢嗎：
  Step 1：選模板

  左側點 Templates → 搜尋框輸入：
  pytorch
  選這個：pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

  ---
  Step 2：篩選機器

  頁面上方的 Filter 設定：
  - GPU Type：RTX 4090
  - # GPUs：1
  - Disk Space：120 GB（很重要，資料很大）
  - Instance Type：On-Demand

  ---
  Step 3：選一台機器

  會出現一排機器列表，選價格最低的那台（通常 $0.35-0.45/hr），按 Rent。

  ---
  租好之後左側 Instances 會看到你的機器，狀態從 Loading 變成 Running 後，截圖給我看，我帶你下一步連線。