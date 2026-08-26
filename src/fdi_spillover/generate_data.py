# src/fdi_spillover/generate_data.py

import numpy as np
import pandas as pd


def generate_fdi_panel_data(
    n_firms: int = 500, n_years: int = 6, seed: int = 42
) -> pd.DataFrame:
    """インドの企業レベル財務データ、サプライチェーン経由のFDIスピルオーバー、

    およびBartik操作変数（Shift-Share IV）を再現する擬似データ生成関数。
    """
    np.random.seed(seed)

    # 1. 基本設定（5地域、4主要産業、2018-2023年）
    states = [f"State_{i}" for i in range(1, 6)]  # 例: Maharashtra, Gujarat等
    industries = [
        "Automotive",
        "Electronics",
        "Chemicals",
        "Textiles",
    ]  # 主要4産業
    years = list(range(2018, 2018 + n_years))

    # 2. 産業連関（Input-Output）投入係数行列（縦: 投入産業, 横: 産出産業）
    # α_jk = 産業jの生産に必要な産業kからの投入割合
    io_matrix = np.array(
        [
            [0.10, 0.25, 0.15, 0.05],  # Automotiveへの投入
            [0.30, 0.10, 0.10, 0.05],  # Electronicsへの投入
            [0.20, 0.15, 0.10, 0.10],  # Chemicalsへの投入
            [0.05, 0.05, 0.20, 0.10],  # Textilesへの投入
        ]
    )

    # 3. 地域×産業別の基準年（2018年）初期シェア（Bartik Share: s_rj0）
    share_dict = {}
    for s in states:
        raw_shares = np.random.dirichlet(np.ones(len(industries)))
        for idx, ind in enumerate(industries):
            share_dict[(s, ind)] = raw_shares[idx]

    # 4. 全国レベルの産業別FDIトレンド（Bartik Shift: 全国成長率）
    national_fdi_shocks = {
        (ind, yr): np.random.normal(loc=0.08, scale=0.03)  # 年率約8%成長
        for ind in industries
        for yr in years
    }

    # 5. 企業マスタの生成
    firm_ids = [f"IND_FIRM_{i:04d}" for i in range(1, n_firms + 1)]
    firm_states = np.random.choice(states, size=n_firms)
    firm_industries = np.random.choice(industries, size=n_firms)
    firm_fe = np.random.normal(0, 1.0, size=n_firms)  # 企業固有効果

    records = []

    # 6. パネルデータの構築
    for t_idx, yr in enumerate(years):
        # 地域×産業別の直接FDIストックを更新
        state_ind_fdi = {}
        for s in states:
            for ind_idx, ind in enumerate(industries):
                # 直近のFDI＝初期シェア × 全国の累積ショック + ノイズ
                base_share = share_dict[(s, ind)]
                national_growth = sum(
                    national_fdi_shocks[(ind, y)]
                    for y in years[: t_idx + 1]
                )
                fdi_val = (
                    10.0 + base_share * 50.0 * (1 + national_growth)
                ) + np.random.normal(0, 1)
                state_ind_fdi[(s, ind)] = max(0.1, fdi_val)

        # 企業ごとに各指標を算出
        for f_idx in range(n_firms):
            f_id = firm_ids[f_idx]
            s = firm_states[f_idx]
            ind = firm_industries[f_idx]
            ind_idx = industries.index(ind)

            # (A) 直接FDI (Direct FDI in same state-industry)
            direct_fdi = state_ind_fdi[(s, ind)]

            # (B) Backward Spillover (買い手産業からのFDI波及効果)
            # BS = Σ (α_jk * Direct_FDI_k)
            backward_spillover = sum(
                io_matrix[ind_idx, k_idx] * state_ind_fdi[(s, industries[k_idx])]
                for k_idx in range(len(industries))
                if k_idx != ind_idx
            )

            # (C) Forward Spillover (売り手産業からのFDI波及効果)
            # FS = Σ (α_kj * Direct_FDI_k)
            forward_spillover = sum(
                io_matrix[k_idx, ind_idx] * state_ind_fdi[(s, industries[k_idx])]
                for k_idx in range(len(industries))
                if k_idx != ind_idx
            )

            # (D) Bartik IV (Shift-Share Instrument)
            # 初期地域シェア × 全国の他地域FDI成長率の合成
            bartik_iv = share_dict[(s, ind)] * national_fdi_shocks[(ind, yr)]

            # (E) 被説明変数：借入金利（Interest Cost Ratio %）
            # 外資流入や金融スピルオーバーが進むと、信用アクセスが改善し金利が下がる関係
            true_beta_direct = -0.05
            true_beta_back = -0.12  # サプライチェーン経由の金融波及効果
            true_beta_fwd = -0.08
            unobserved_credit_demand = np.random.normal(0, 0.5)  # 内生性を生む未観測ショック

            # 金利 = 定数項 - 効果 + 企業個別要因 + 経年低下トレンド + 内生性ノイズ
            interest_rate = (
                8.5
                + true_beta_direct * direct_fdi
                + true_beta_back * backward_spillover
                + true_beta_fwd * forward_spillover
                + firm_fe[f_idx]
                - 0.2 * (yr - 2018)
                + 0.3 * unobserved_credit_demand
                + np.random.normal(0, 0.3)
            )

            # 企業レベルの財務コントロール変数（レバレッジ・資産規模）
            log_assets = (
                5.0
                + 0.1 * (yr - 2018)
                + firm_fe[f_idx] * 0.5
                + np.random.normal(0, 0.2)
            )
            leverage = (
                0.4
                + 0.02 * unobserved_credit_demand
                + np.random.normal(0, 0.05)
            )

            records.append(
                {
                    "firm_id": f_id,
                    "state": s,
                    "industry": ind,
                    "year": yr,
                    "interest_rate": round(interest_rate, 3),
                    "direct_fdi": round(direct_fdi, 3),
                    "backward_spillover": round(backward_spillover, 3),
                    "forward_spillover": round(forward_spillover, 3),
                    "bartik_iv": round(bartik_iv, 4),
                    "log_assets": round(log_assets, 3),
                    "leverage": round(leverage, 3),
                }
            )

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    df = generate_fdi_panel_data()
    print("=== 生成されたパネルデータのプレビュー ===")
    print(df.head(10))
    print("\n=== データ形状 ===")
    print(df.shape)

    # CSVとして保存
    output_path = "data/india_fdi_panel.csv"
    df.to_csv(output_path, index=False)
    print(f"\nデータセットを保存しました: {output_path}")