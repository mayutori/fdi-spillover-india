# src/fdi_spillover/analyzer.py

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS


class FDISpilloverAnalyzer:
    """FDIの金融スピルオーバー効果（直接効果、Backward/Forward波及）を
    OLSおよびBartik IVを用いた2SLS（2段階最小二乗法）で推計するクラス。
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def estimate_ols(self) -> dict:
        """【ベンチマーク OLS 回帰】"""
        formula = (
            "interest_rate ~ backward_spillover + forward_spillover + "
            "direct_fdi + log_assets + leverage + C(state) + C(year)"
        )
        model = smf.ols(formula=formula, data=self.df).fit(
            cov_type="cluster", cov_kwds={"groups": self.df["state"]}
        )

        return {
            "method": "OLS (State-Clustered SE)",
            "backward_coef": round(model.params["backward_spillover"], 4),
            "backward_se": round(model.bse["backward_spillover"], 4),
            "backward_pvalue": round(model.pvalues["backward_spillover"], 4),
            "forward_coef": round(model.params["forward_spillover"], 4),
            "direct_coef": round(model.params["direct_fdi"], 4),
            "n_obs": int(model.nobs),
            "r2": round(model.rsquared, 4),
        }

    def estimate_2sls(self) -> dict:
        """【Bartik IV 2SLS (2段階最小二乗法) 回帰】"""
        formula = (
            "interest_rate ~ 1 + forward_spillover + direct_fdi + log_assets + leverage + "
            "C(state) + C(year) + [backward_spillover ~ bartik_iv]"
        )

        model = IV2SLS.from_formula(formula, data=self.df).fit(
            cov_type="clustered", clusters=self.df["state"]
        )

        coef = model.params["backward_spillover"]
        se = model.std_errors["backward_spillover"]
        pvalue = model.pvalues["backward_spillover"]

        # Weak IV（弱操作変数検定）用の第1段階 F値を安全に取得
        try:
            diag = model.first_stage.diagnostics
            f_col = [c for c in diag.columns if "f" in str(c).lower() and "stat" in str(c).lower()]
            if f_col:
                first_stage_f = float(diag[f_col[0]].iloc[0])
            else:
                first_stage_f = float(diag.iloc[0, 0])
        except Exception:
            first_stage_f = 0.0

        return {
            "method": "2SLS (Bartik IV)",
            "backward_coef": round(coef, 4),
            "backward_se": round(se, 4),
            "backward_pvalue": round(pvalue, 4),
            "first_stage_f": round(first_stage_f, 2),
            "n_obs": int(model.nobs),
        }

    def get_summary_table(self) -> pd.DataFrame:
        """OLS と 2SLS の推計結果を学術標準の比較テーブルにまとめる。"""
        ols_res = self.estimate_ols()
        iv_res = self.estimate_2sls()

        summary = pd.DataFrame(
            [
                {
                    "Method": ols_res["method"],
                    "Backward Spillover Beta": f"{ols_res['backward_coef']} (se: {ols_res['backward_se']})",
                    "p-value": ols_res["backward_pvalue"],
                    "First-Stage F-Stat": "N/A",
                    "N": ols_res["n_obs"],
                },
                {
                    "Method": iv_res["method"],
                    "Backward Spillover Beta": f"{iv_res['backward_coef']} (se: {iv_res['backward_se']})",
                    "p-value": iv_res["backward_pvalue"],
                    "First-Stage F-Stat": iv_res["first_stage_f"],
                    "N": iv_res["n_obs"],
                },
            ]
        )
        return summary


if __name__ == "__main__":
    try:
        from fdi_spillover.generate_data import generate_fdi_panel_data
    except ImportError:
        from generate_data import generate_fdi_panel_data

    print("1. Generating FDI Panel Data...")
    df = generate_fdi_panel_data()

    print("2. Running OLS & Bartik 2SLS Econometric Estimation...")
    analyzer = FDISpilloverAnalyzer(df)

    print("\n=== OLS vs 2SLS (Bartik IV) 推計結果の比較 ===")
    summary_df = analyzer.get_summary_table()
    print(summary_df.to_string(index=False))