import os
import sys
from pathlib import Path
import numpy as np

# Add project root and analysis folder to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))

from calculate_performance import calculate_all_performance, OUTPUT_DIR

def fmt(val, format_str="{:.2f}", default="---"):
    if val is None:
        return default
    try:
        if isinstance(val, (float, np.floating, int, np.integer)):
            if np.isnan(val) or np.isinf(val):
                return default
            return format_str.format(val)
    except Exception:
        pass
    return str(val)

def generate_report():
    results = calculate_all_performance()
    
    lines = []
    lines.append("==================================================================================")
    lines.append("                        PERFORMANCE INDEX SUMMARY REPORT")
    lines.append("==================================================================================")
    lines.append("")
    
    # ----------------------------------------------------------------------------------
    # TIC-100 (Servo & Regulatory for Disturbance Tuning Formula)
    # ----------------------------------------------------------------------------------
    lines.append("1. TIC-100: TEMPERATURE CONTROLLER PERFORMANCE")
    lines.append("   (Scenario: Servo & Regulatory for Disturbance Tuning Formula)")
    lines.append("   -------------------------------------------------------------------------------")
    
    # Python
    lines.append("   [PYTHON]")
    # Servo
    sp_syn = results["tic_dist_py_sp_si"]["Syn"].metrics
    sp_qdr = results["tic_dist_py_sp_si"]["QDR"].metrics
    sp_iae = results["tic_dist_py_sp_si"]["IAE"].metrics
    
    lines.append("     Scenario: Servo (Setpoint Tracking)")
    lines.append("     " + f"{'Performance Index':<25} | {'Syn':<15} | {'QDR':<15} | {'IAE':<15}")
    lines.append("     " + "-" * 78)
    lines.append("     " + f"{'Decay Ratio':<25} | {fmt(sp_syn.DecayRatio, '{:.3f}'):<15} | {fmt(sp_qdr.DecayRatio, '{:.3f}'):<15} | {fmt(sp_iae.DecayRatio, '{:.3f}'):<15}")
    lines.append("     " + f"{'Rise Time':<25} | {fmt(sp_syn.RiseTime, '{:.1f} s'):<15} | {fmt(sp_qdr.RiseTime, '{:.1f} s'):<15} | {fmt(sp_iae.RiseTime, '{:.1f} s'):<15}")
    lines.append("     " + f"{'Settling Time':<25} | {fmt(sp_syn.SettlingTime, '{:.1f} s'):<15} | {fmt(sp_qdr.SettlingTime, '{:.1f} s'):<15} | {fmt(sp_iae.SettlingTime, '{:.1f} s'):<15}")
    lines.append("     " + f"{'Overshoot':<25} | {fmt(sp_syn.Overshoot, '{:.2f} %'):<15} | {fmt(sp_qdr.Overshoot, '{:.2f} %'):<15} | {fmt(sp_iae.Overshoot, '{:.2f} %'):<15}")
    lines.append("     " + f"{'IAE':<25} | {fmt(sp_syn.IAE, '{:.1f}'):<15} | {fmt(sp_qdr.IAE, '{:.1f}'):<15} | {fmt(sp_iae.IAE, '{:.1f}'):<15}")
    lines.append("")
    
    # Regulatory
    dist_syn = results["tic_dist_py_si"]["Syn"].metrics
    dist_qdr = results["tic_dist_py_si"]["QDR"].metrics
    dist_iae = results["tic_dist_py_si"]["IAE"].metrics
    
    lines.append("     Scenario: Regulatory (Disturbance Rejection)")
    lines.append("     " + f"{'Performance Index':<25} | {'Syn':<15} | {'QDR':<15} | {'IAE':<15}")
    lines.append("     " + "-" * 78)
    lines.append("     " + f"{'Decay Ratio':<25} | {fmt(dist_syn.DecayRatio, '{:.3f}'):<15} | {fmt(dist_qdr.DecayRatio, '{:.3f}'):<15} | {fmt(dist_iae.DecayRatio, '{:.3f}'):<15}")
    lines.append("     " + f"{'Rise Time':<25} | {fmt(dist_syn.RiseTime, '{:.1f} s'):<15} | {fmt(dist_qdr.RiseTime, '{:.1f} s'):<15} | {fmt(dist_iae.RiseTime, '{:.1f} s'):<15}")
    lines.append("     " + f"{'Settling Time':<25} | {fmt(dist_syn.SettlingTime, '{:.1f} s'):<15} | {fmt(dist_qdr.SettlingTime, '{:.1f} s'):<15} | {fmt(dist_iae.SettlingTime, '{:.1f} s'):<15}")
    lines.append("     " + f"{'Overshoot':<25} | {fmt(dist_syn.Overshoot, '{:.2f} %'):<15} | {fmt(dist_qdr.Overshoot, '{:.2f} %'):<15} | {fmt(dist_iae.Overshoot, '{:.2f} %'):<15}")
    lines.append("     " + f"{'IAE':<25} | {fmt(dist_syn.IAE, '{:.1f}'):<15} | {fmt(dist_qdr.IAE, '{:.1f}'):<15} | {fmt(dist_iae.IAE, '{:.1f}'):<15}")
    lines.append("")
    
    # HYSYS
    lines.append("   [HYSYS]")
    # Servo
    sp_syn = results["tic_dist_hy_sp_si"]["Syn"].metrics
    sp_qdr = results["tic_dist_hy_sp_si"]["QDR"].metrics
    sp_iae = results["tic_dist_hy_sp_si"]["IAE"].metrics
    
    lines.append("     Scenario: Servo (Setpoint Tracking)")
    lines.append("     " + f"{'Performance Index':<25} | {'Syn':<15} | {'QDR':<15} | {'IAE':<15}")
    lines.append("     " + "-" * 78)
    lines.append("     " + f"{'Decay Ratio':<25} | {fmt(sp_syn.DecayRatio, '{:.3f}'):<15} | {fmt(sp_qdr.DecayRatio, '{:.3f}'):<15} | {fmt(sp_iae.DecayRatio, '{:.3f}'):<15}")
    lines.append("     " + f"{'Rise Time':<25} | {fmt(sp_syn.RiseTime, '{:.1f} s'):<15} | {fmt(sp_qdr.RiseTime, '{:.1f} s'):<15} | {fmt(sp_iae.RiseTime, '{:.1f} s'):<15}")
    lines.append("     " + f"{'Settling Time':<25} | {fmt(sp_syn.SettlingTime, '{:.1f} s'):<15} | {fmt(sp_qdr.SettlingTime, '{:.1f} s'):<15} | {fmt(sp_iae.SettlingTime, '{:.1f} s'):<15}")
    lines.append("     " + f"{'Overshoot':<25} | {fmt(sp_syn.Overshoot, '{:.2f} %'):<15} | {fmt(sp_qdr.Overshoot, '{:.2f} %'):<15} | {fmt(sp_iae.Overshoot, '{:.2f} %'):<15}")
    lines.append("     " + f"{'IAE':<25} | {fmt(sp_syn.IAE, '{:.1f}'):<15} | {fmt(sp_qdr.IAE, '{:.1f}'):<15} | {fmt(sp_iae.IAE, '{:.1f}'):<15}")
    lines.append("")
    
    # Regulatory
    dist_syn = results["tic_dist_hy_si"]["Syn"].metrics
    dist_qdr = results["tic_dist_hy_si"]["QDR"].metrics
    dist_iae = results["tic_dist_hy_si"]["IAE"].metrics
    
    lines.append("     Scenario: Regulatory (Disturbance Rejection)")
    lines.append("     " + f"{'Performance Index':<25} | {'Syn':<15} | {'QDR':<15} | {'IAE':<15}")
    lines.append("     " + "-" * 78)
    lines.append("     " + f"{'Decay Ratio':<25} | {fmt(dist_syn.DecayRatio, '{:.3f}'):<15} | {fmt(dist_qdr.DecayRatio, '{:.3f}'):<15} | {fmt(dist_iae.DecayRatio, '{:.3f}'):<15}")
    lines.append("     " + f"{'Rise Time':<25} | {fmt(dist_syn.RiseTime, '{:.1f} s'):<15} | {fmt(dist_qdr.RiseTime, '{:.1f} s'):<15} | {fmt(dist_iae.RiseTime, '{:.1f} s'):<15}")
    lines.append("     " + f"{'Settling Time':<25} | {fmt(dist_syn.SettlingTime, '{:.1f} s'):<15} | {fmt(dist_qdr.SettlingTime, '{:.1f} s'):<15} | {fmt(dist_iae.SettlingTime, '{:.1f} s'):<15}")
    lines.append("     " + f"{'Overshoot':<25} | {fmt(dist_syn.Overshoot, '{:.2f} %'):<15} | {fmt(dist_qdr.Overshoot, '{:.2f} %'):<15} | {fmt(dist_iae.Overshoot, '{:.2f} %'):<15}")
    lines.append("     " + f"{'IAE':<25} | {fmt(dist_syn.IAE, '{:.1f}'):<15} | {fmt(dist_qdr.IAE, '{:.1f}'):<15} | {fmt(dist_iae.IAE, '{:.1f}'):<15}")
    lines.append("")
    lines.append("==================================================================================")
    lines.append("")
    
    # ----------------------------------------------------------------------------------
    # LIC-100 (Setpoint Change / Servo for TLC and ALC)
    # ----------------------------------------------------------------------------------
    lines.append("2. LIC-100: LEVEL CONTROLLER PERFORMANCE")
    lines.append("   (Scenario: Setpoint Change / Servo Only)")
    lines.append("   -------------------------------------------------------------------------------")
    
    # Python
    lines.append("   [PYTHON]")
    py_tlc = results["lic_sp_si"]["Tight"].metrics
    py_alc = results["lic_sp_si"]["Averaging"].metrics
    
    lines.append("     " + f"{'Performance Index':<25} | {'TLC (Tight)':<20} | {'ALC (Averaging)':<20}")
    lines.append("     " + "-" * 71)
    lines.append("     " + f"{'Decay Ratio':<25} | {fmt(py_tlc.DecayRatio, '{:.3f}'):<20} | {fmt(py_alc.DecayRatio, '{:.3f}'):<20}")
    lines.append("     " + f"{'Rise Time':<25} | {fmt(py_tlc.RiseTime, '{:.1f} s'):<20} | {fmt(py_alc.RiseTime, '{:.1f} s'):<20}")
    lines.append("     " + f"{'Settling Time':<25} | {fmt(py_tlc.SettlingTime, '{:.1f} s'):<20} | {fmt(py_alc.SettlingTime, '{:.1f} s'):<20}")
    lines.append("     " + f"{'Overshoot':<25} | {fmt(py_tlc.Overshoot, '{:.2f} %'):<20} | {fmt(py_alc.Overshoot, '{:.2f} %'):<20}")
    lines.append("     " + f"{'IAE':<25} | {fmt(py_tlc.IAE, '{:.1f}'):<20} | {fmt(py_alc.IAE, '{:.1f}'):<20}")
    lines.append("")
    
    # HYSYS
    lines.append("   [HYSYS]")
    hy_tlc = results["lic_sp_hy_si"]["Tight"].metrics
    hy_alc = results["lic_sp_hy_si"]["Averaging"].metrics
    
    lines.append("     " + f"{'Performance Index':<25} | {'TLC (Tight)':<20} | {'ALC (Averaging)':<20}")
    lines.append("     " + "-" * 71)
    lines.append("     " + f"{'Decay Ratio':<25} | {fmt(hy_tlc.DecayRatio, '{:.3f}'):<20} | {fmt(hy_alc.DecayRatio, '{:.3f}'):<20}")
    lines.append("     " + f"{'Rise Time':<25} | {fmt(hy_tlc.RiseTime, '{:.1f} s'):<20} | {fmt(hy_alc.RiseTime, '{:.1f} s'):<20}")
    lines.append("     " + f"{'Settling Time':<25} | {fmt(hy_tlc.SettlingTime, '{:.1f} s'):<20} | {fmt(hy_alc.SettlingTime, '{:.1f} s'):<20}")
    lines.append("     " + f"{'Overshoot':<25} | {fmt(hy_tlc.Overshoot, '{:.2f} %'):<20} | {fmt(hy_alc.Overshoot, '{:.2f} %'):<20}")
    lines.append("     " + f"{'IAE':<25} | {fmt(hy_tlc.IAE, '{:.1f}'):<20} | {fmt(hy_alc.IAE, '{:.1f}'):<20}")
    lines.append("")
    lines.append("==================================================================================")
    lines.append("")
    
    # ----------------------------------------------------------------------------------
    # FIC-100 / FIC-101 / FIC-102 (Setpoint Change)
    # ----------------------------------------------------------------------------------
    lines.append("3. FIC-100 / 101 / 102: FLOW CONTROLLER PERFORMANCE")
    lines.append("   (Scenario: Setpoint Change Only)")
    lines.append("   -------------------------------------------------------------------------------")
    
    # Python
    lines.append("   [PYTHON]")
    py_fic100 = results["si_fic100_py"].metrics
    py_fic101 = results["si_fic101_py"].metrics
    py_fic102 = results["si_fic102_py"].metrics
    
    lines.append("     " + f"{'Performance Index':<25} | {'FIC-100':<15} | {'FIC-101':<15} | {'FIC-102':<15}")
    lines.append("     " + "-" * 78)
    lines.append("     " + f"{'Decay Ratio':<25} | {fmt(py_fic100.DecayRatio, '{:.3f}'):<15} | {fmt(py_fic101.DecayRatio, '{:.3f}'):<15} | {fmt(py_fic102.DecayRatio, '{:.3f}'):<15}")
    lines.append("     " + f"{'Rise Time':<25} | {fmt(py_fic100.RiseTime, '{:.1f} s'):<15} | {fmt(py_fic101.RiseTime, '{:.1f} s'):<15} | {fmt(py_fic102.RiseTime, '{:.1f} s'):<15}")
    lines.append("     " + f"{'Settling Time':<25} | {fmt(py_fic100.SettlingTime, '{:.1f} s'):<15} | {fmt(py_fic101.SettlingTime, '{:.1f} s'):<15} | {fmt(py_fic102.SettlingTime, '{:.1f} s'):<15}")
    lines.append("     " + f"{'Overshoot':<25} | {fmt(py_fic100.Overshoot, '{:.2f} %'):<15} | {fmt(py_fic101.Overshoot, '{:.2f} %'):<15} | {fmt(py_fic102.Overshoot, '{:.2f} %'):<15}")
    lines.append("     " + f"{'IAE':<25} | {fmt(py_fic100.IAE, '{:.1f}'):<15} | {fmt(py_fic101.IAE, '{:.1f}'):<15} | {fmt(py_fic102.IAE, '{:.1f}'):<15}")
    lines.append("")
    
    # HYSYS
    lines.append("   [HYSYS]")
    hy_fic100 = results["si_fic100_hy"].metrics
    hy_fic101 = results["si_fic101_hy"].metrics
    hy_fic102 = results["si_fic102_hy"].metrics
    
    lines.append("     " + f"{'Performance Index':<25} | {'FIC-100':<15} | {'FIC-101':<15} | {'FIC-102':<15}")
    lines.append("     " + "-" * 78)
    lines.append("     " + f"{'Decay Ratio':<25} | {fmt(hy_fic100.DecayRatio, '{:.3f}'):<15} | {fmt(hy_fic101.DecayRatio, '{:.3f}'):<15} | {fmt(hy_fic102.DecayRatio, '{:.3f}'):<15}")
    lines.append("     " + f"{'Rise Time':<25} | {fmt(hy_fic100.RiseTime, '{:.1f} s'):<15} | {fmt(hy_fic101.RiseTime, '{:.1f} s'):<15} | {fmt(hy_fic102.RiseTime, '{:.1f} s'):<15}")
    lines.append("     " + f"{'Settling Time':<25} | {fmt(hy_fic100.SettlingTime, '{:.1f} s'):<15} | {fmt(hy_fic101.SettlingTime, '{:.1f} s'):<15} | {fmt(hy_fic102.SettlingTime, '{:.1f} s'):<15}")
    lines.append("     " + f"{'Overshoot':<25} | {fmt(hy_fic100.Overshoot, '{:.2f} %'):<15} | {fmt(hy_fic101.Overshoot, '{:.2f} %'):<15} | {fmt(hy_fic102.Overshoot, '{:.2f} %'):<15}")
    lines.append("     " + f"{'IAE':<25} | {fmt(hy_fic100.IAE, '{:.1f}'):<15} | {fmt(hy_fic101.IAE, '{:.1f}'):<15} | {fmt(hy_fic102.IAE, '{:.1f}'):<15}")
    lines.append("")
    lines.append("==================================================================================")
    
    report_content = "\n".join(lines)
    
    # Ensure output dir exists
    os.makedirs(os.path.join(OUTPUT_DIR, "reports"), exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, "reports", "report_summary.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"Report saved to: {report_path}")
    print("\nReport Content:")
    print(report_content)

if __name__ == "__main__":
    generate_report()
