# src/fdi_spillover/cli.py

from fdi_spillover.generate_data import generate_fdi_panel_data
from fdi_spillover.analyzer import FDISpilloverAnalyzer
from fdi_spillover.reporter import SpilloverReporter

def main():
    print("1. Generating FDI Panel Data...")
    df = generate_fdi_panel_data()

    print("2. Running OLS & Bartik 2SLS Estimation...")
    analyzer = FDISpilloverAnalyzer(df)
    ols_res = analyzer.estimate_ols()
    iv_res = analyzer.estimate_2sls()

    print("3. Generating Executive PDF Report...")
    reporter = SpilloverReporter(ols_res, iv_res)
    reporter.generate_pdf("fdi_spillover_report.pdf")
    print("Pipeline execution complete! Output saved to fdi_spillover_report.pdf")

if __name__ == "__main__":
    main()