# GEMINI.md - 專案概觀與使用說明

這份文件旨在說明 dPSO-TSP 實驗平台的目標、結構、建置與執行方法。

## 1. 專案目標

本專案旨在建立一個 C++ 實驗平台，用於研究與評估**離散粒子群演算法 (Discrete Particle Swarm Optimization, dPSO)** 在解決**對稱旅行推銷員問題 (Symmetric Traveling Salesman Problem, TSP)** 上的效能。

核心目標包含：
- **實作與驗證**：實作一個序列版本的 dPSO 演算法作為基準 (Baseline)。
- **效能評估**：根據標準的 TSPLIB 資料集，從「解的品質 (相對誤差)」與「執行時間」兩個維度評估演算法效能。
- **可擴展性分析**：為未來將 dPSO擴展到平行版本做準備，並評估其加速比 (Speedup) 與效率 (Efficiency)。

## 2. 專案結構

專案目錄結構如下，符合 Phase 0 中規劃的架構：

```

├── main.cpp             # 實驗執行的主程式入口
├── tsp_experiment       # (建置後產生) 編譯後的可執行檔
├── configs/             # (未使用) 預計用於存放實驗設定檔
├── data/
│   └── instances/
│       └── berlin52.tsp # TSPLIB 資料實例
├── results/
│   ├── logs/            # (未使用) 預計用於存放詳細日誌
│   └── tables/          # (未使用) 預計用於存放格式化的結果表格
└── src/
    ├── baseline_nn.h, .cpp  # Baseline #0: 最近鄰居演算法 (Nearest Neighbor)
    ├── dps_tsp.h, .cpp      # Baseline #1: 序列 dPSO 演算法
    ├── tsp_instance.h, .cpp # TSPLIB 資料讀取與問題實例表示
    └── utils.h, .cpp        # 共用工具 (如亂數產生、Tour 操作)
```

## 3. 相依套件

- **C++ 編譯器**：需要支援 C++17 標準的編譯器 (例如 `g++` 或 `clang++`)。
- **標準函式庫**：無任何外部程式庫相依。

## 4. 建置專案 (How to Build)

目前專案未包含 `Makefile` 或 `CMakeLists.txt`。你可以使用以下指令從專案根目錄 (`dPSO_TSP/`) 進行編譯：

```bash
g++ -std=c++17 -O3 -o tsp_experiment main.cpp src/*.cpp
```

- `-std=c++17`：指定使用 C++17 標準。
- `-O3`：啟用最高等級的編譯優化，以取得較準確的執行時間。
- `-o tsp_experiment`：將產生的可執行檔命名為 `tsp_experiment` 並放置於 `tsp-exp` 目錄下。

## 5. 執行實驗 (How to Run)

編譯完成後，可直接執行 `tsp_experiment` 來進行實驗。

```bash
./tsp_experiment
```

**注意**：目前的 `main.cpp` 是硬編碼 (hard-coded) 執行以下設定：
- **TSP 實例**: `data/instances/berlin52.tsp`
- **已知最佳解**: `7542`
- **執行演算法**:
  1. **Nearest Neighbor (NN)**: 從 52 個城市各出發一次，計算最佳與平均結果。
  2. **dPSO_seq**: 連續執行 10 次 (`NUM_RUNS = 10`)，計算統計結果。

若要更換測試實例或調整演算法參數，需要直接修改 `main.cpp` 中的常數。

### 輸出結果

程式會將執行結果以表格形式直接輸出到終端機，格式如下：

```
Instance      Algorithm   Best Cost   Mean Cost   Best Err(%) Mean Err(%) Mean Time(ms) 
-------------------------------------------------------------------------------------
berlin52      NN          8980.00     10034.52    19.07       33.05       0.01          
berlin52      dPSO_seq    7834.00     8142.20     3.87        7.96        154.32        
```

- **Best Cost**: 所有執行中找到的最低路徑長度。
- **Mean Cost**: 平均路徑長度。
- **Best Err(%)**: 找到的最佳解相對於已知最佳解的誤差百分比。
- **Mean Err(%)**: 平均解的誤差百分比。
- **Mean Time(ms)**: 每次執行 (run) 的平均耗時（毫秒）。

## 6. 已實作的演算法

根據你的計畫，目前已完成 Phase 3 和 Phase 4 的 Baseline 實作：

1.  **Nearest Neighbor (`baseline_nn.cpp`)**:
    -   一個貪婪演算法，作為 Sanity Check。
    -   從每個城市出發執行一次，以找出 NN 方法在此實例上的最佳表現。

2.  **Sequential Discrete PSO (`dps_tsp.cpp`)**:
    -   核心的序列 dPSO 演算法。
    -   **粒子表示**: `std::vector<int>` 代表城市排列 (Tour)。
    -   **速度表示**: `std::vector<SwapOp>` 代表一組交換操作。
    -   **更新規則**: 依據慣性 (Inertia)、認知 (Cognitive) 與社會 (Social) 三個分量更新速度，並應用於粒子位置上。

## 7. 下一步建議 (Next Steps)

你的實驗平台基礎已穩固。根據你的原始計畫，後續可以進行：
1.  **建立平行版本 dPSO**: 這是計算 `Speedup` 和 `Efficiency` 指標的前提。
2.  **擴充實驗框架**:
    -   將 `main.cpp` 中的硬編碼設定改為**命令列參數**，方便自動化測試不同實例與參數。
    -   將實驗結果輸出至 `results/tables/` 中的 CSV 檔。
3.  **新增 GA Baseline**: 實作 Phase 6 中提及的基因演算法，以進行更豐富的比較。
4.  **擴充 TSPLIB 資料集**: 加入更多不同規模的 TSP 實例，以驗證演算法的**可擴展性 (Scalability)**。
