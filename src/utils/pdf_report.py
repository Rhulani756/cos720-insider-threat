"""
src/utils/pdf_report.py
-----------------------
PDF report generator for ThreatSense forensic investigation reports.
Uses reportlab to produce a professional PDF matching the UI style.
"""

import os
import sys
from datetime import datetime
from io import BytesIO

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT


# ── Colour palette ────────────────────────────────────────────────────────────
DARK_BG     = colors.HexColor("#0f1117")
CARD_BG     = colors.HexColor("#161b22")
BORDER      = colors.HexColor("#21262d")
ACCENT      = colors.HexColor("#06b6d4")
TEXT_PRI    = colors.HexColor("#e6edf3")
TEXT_SEC    = colors.HexColor("#8b949e")
RED         = colors.HexColor("#ef4444")
GREEN       = colors.HexColor("#22c55e")
ORANGE      = colors.HexColor("#f97316")
WHITE       = colors.white

RISK_COLOURS = {
    "Critical": RED,
    "High":     ORANGE,
    "Medium":   ACCENT,
    "Low":      GREEN,
}

LIGHT_BG       = colors.HexColor("#f8fafc")
LIGHT_CARD     = colors.white
LIGHT_BORDER   = colors.HexColor("#e2e8f0")
LIGHT_TEXT_PRI = colors.HexColor("#0f172a")
LIGHT_TEXT_SEC = colors.HexColor("#64748b")


def generate_pdf_report(
    result: dict,
    local_df,
    breakdown,
    record: dict,
    theme: str = "dark",
) -> bytes:
    """
    Generate a professional PDF forensic investigation report.

    Parameters:
        result:    prediction result dict from predict_record()
        local_df:  local explanation DataFrame
        breakdown: risk score breakdown DataFrame
        record:    raw employee record dict
        theme:     "dark" or "light"

    Returns:
        PDF as bytes
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=20*mm, bottomMargin=20*mm,
    )

    is_dark  = theme == "dark"
    bg       = DARK_BG    if is_dark else LIGHT_BG
    card     = CARD_BG    if is_dark else LIGHT_CARD
    border   = BORDER     if is_dark else LIGHT_BORDER
    t_pri    = TEXT_PRI   if is_dark else LIGHT_TEXT_PRI
    t_sec    = TEXT_SEC   if is_dark else LIGHT_TEXT_SEC
    is_mal   = result["prediction"] == 1
    risk     = result["risk_level"]
    risk_col = RISK_COLOURS.get(risk, ACCENT)

    # ── Styles ────────────────────────────────────────────────────────────────
    def style(name, **kwargs):
        base = {"fontName": "Helvetica", "fontSize": 10, "textColor": t_pri,
                "leading": 14, "spaceAfter": 0, "spaceBefore": 0}
        base.update(kwargs)
        return ParagraphStyle(name, **base)

    S = {
        "title":    style("title",   fontSize=20, fontName="Helvetica-Bold", textColor=t_pri, leading=24),
        "subtitle": style("subtitle",fontSize=10, textColor=t_sec, leading=14),
        "section":  style("section", fontSize=8,  fontName="Helvetica-Bold", textColor=t_sec,
                          leading=12, spaceAfter=4, letterSpacing=1),
        "body":     style("body",    fontSize=9,  textColor=t_sec, leading=14),
        "value":    style("value",   fontSize=9,  fontName="Helvetica-Bold", textColor=t_pri),
        "mono":     style("mono",    fontSize=8,  fontName="Courier", textColor=ACCENT, leading=12),
        "alert":    style("alert",   fontSize=11, fontName="Helvetica-Bold",
                          textColor=RED if is_mal else GREEN),
        "label":    style("label",   fontSize=8,  textColor=t_sec, leading=12),
    }

    story = []
    W = doc.width

    def hr(col=border, thickness=0.5):
        return HRFlowable(width="100%", thickness=thickness, color=col, spaceAfter=8, spaceBefore=8)

    def sp(h=4):
        return Spacer(1, h*mm)

    # ── Header ────────────────────────────────────────────────────────────────
    header_data = [[
        Paragraph("🛡 ThreatSense", style("hd", fontSize=18, fontName="Helvetica-Bold", textColor=ACCENT)),
        Paragraph(f"Generated: {datetime.now().strftime('%d %b %Y %H:%M')}", style("ts", fontSize=8, textColor=t_sec, alignment=TA_RIGHT)),
    ]]
    header_tbl = Table(header_data, colWidths=[W*0.6, W*0.4])
    header_tbl.setStyle(TableStyle([
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("BACKGROUND",    (0,0), (-1,-1), card),
        ("ROUNDEDCORNERS",(0,0), (-1,-1), [6,6,6,6]),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ("RIGHTPADDING",  (0,0), (-1,-1), 12),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("LINEBELOW",     (0,0), (-1,0), 0.5, risk_col),
    ]))
    story += [header_tbl, sp(3)]

    # ── Subheader ─────────────────────────────────────────────────────────────
    story += [
        Paragraph("INSIDER THREAT FORENSIC INVESTIGATION REPORT", style("sh", fontSize=8, fontName="Helvetica-Bold", textColor=t_sec, letterSpacing=1)),
        Paragraph("Hybrid Detection Engine · University of Pretoria COS720 · ISO/IEC 27043 Aligned", S["subtitle"]),
        sp(2), hr(), sp(2),
    ]

    # ── Verdict banner ────────────────────────────────────────────────────────
    verdict_text = "⚠ THREAT DETECTED" if is_mal else "✓ NO THREAT DETECTED"
    verdict_col  = RED if is_mal else GREEN
    sub_text     = "Immediate escalation to security team recommended." if is_mal else "Behaviour consistent with normal departmental activity."

    verdict_data = [[
        Paragraph(verdict_text, style("v", fontSize=14, fontName="Helvetica-Bold", textColor=verdict_col)),
        Paragraph(risk, style("r", fontSize=11, fontName="Helvetica-Bold", textColor=risk_col, alignment=TA_RIGHT)),
    ],[
        Paragraph(sub_text, style("vs", fontSize=9, textColor=t_sec)),
        Paragraph(f"Threshold: {result.get('threshold_used', 0.65)}", style("rt", fontSize=8, textColor=t_sec, alignment=TA_RIGHT)),
    ]]
    verdict_tbl = Table(verdict_data, colWidths=[W*0.7, W*0.3])
    verdict_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), card),
        ("LEFTPADDING",   (0,0), (-1,-1), 14),
        ("RIGHTPADDING",  (0,0), (-1,-1), 14),
        ("TOPPADDING",    (0,0), (- 1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("LINEBEFORE",    (0,0), (0,-1), 3, verdict_col),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))
    story += [verdict_tbl, sp(3)]

    # ── Score cards ───────────────────────────────────────────────────────────
    scores = [
        ("Composite Risk Score", f"{result['composite_risk_score']}/100", risk_col),
        ("XGBoost Confidence",   f"{result['rf_confidence']}%",           ACCENT),
        ("Malicious Probability",f"{result['prob_malicious']}%",          RED if is_mal else t_sec),
        ("Normal Probability",   f"{result['prob_normal']}%",             GREEN if not is_mal else t_sec),
    ]
    card_data = [[Paragraph(label, S["label"]) for label, _, _ in scores]]
    card_data.append([Paragraph(val, style("cv", fontSize=14, fontName="Helvetica-Bold", textColor=col)) for _, val, col in scores])

    score_tbl = Table(card_data, colWidths=[W/4]*4)
    score_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), card),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ("RIGHTPADDING",  (0,0), (-1,-1), 10),
        ("TOPPADDING",    (0,0), (0,-1), 8),
        ("TOPPADDING",    (0,1), (-1,1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LINEAFTER",     (0,0), (2,-1), 0.5, border),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))
    story += [score_tbl, sp(3)]

    # IF status
    if_text  = "Anomaly detected — statistically unusual behaviour" if result["isolation_forest_anomaly"] else "Within normal range — no statistical anomaly"
    if_col   = ORANGE if result["isolation_forest_anomaly"] else GREEN
    story += [Paragraph(f"◆ Isolation Forest: {if_text}", style("if", fontSize=9, textColor=if_col)), sp(3), hr()]

    # ── Phase 1: Readiness ────────────────────────────────────────────────────
    story.append(sp(2))
    story.append(Paragraph("PHASE 1 — READINESS (Evidence Collection)", style("ph", fontSize=8, fontName="Helvetica-Bold", textColor=t_sec, letterSpacing=1)))
    story.append(sp(2))

    phase1_data = [
        [Paragraph("Behavioural record analysed",  S["body"]), Paragraph("YES", style("yes", fontSize=9, fontName="Helvetica-Bold", textColor=GREEN))],
        [Paragraph("Department baseline computed",  S["body"]), Paragraph("YES", style("yes", fontSize=9, fontName="Helvetica-Bold", textColor=GREEN))],
        [Paragraph("Deviation features engineered", S["body"]), Paragraph("YES", style("yes", fontSize=9, fontName="Helvetica-Bold", textColor=GREEN))],
        [Paragraph("Features evaluated",            S["body"]), Paragraph("25",  S["value"])],
    ]
    ph1_tbl = Table(phase1_data, colWidths=[W*0.6, W*0.4])
    ph1_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), card),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ("RIGHTPADDING",  (0,0), (-1,-1), 12),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LINEBELOW",     (0,0), (-1,-2), 0.3, border),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))
    story += [ph1_tbl, sp(3), hr()]

    # ── Phase 2: Detection ────────────────────────────────────────────────────
    story.append(sp(2))
    story.append(Paragraph("PHASE 2 — DETECTION (Anomaly Identification)", style("ph2", fontSize=8, fontName="Helvetica-Bold", textColor=t_sec, letterSpacing=1)))
    story.append(sp(2))

    # Top risk indicators table
    top_risks = local_df[
        ~local_df["feature"].isin({"employee_department_enc","employee_campus_enc","employee_position_enc","employee_origin_country_enc"})
    ].head(8)

    risk_rows = [[
        Paragraph("Indicator", style("th", fontSize=8, fontName="Helvetica-Bold", textColor=t_sec)),
        Paragraph("Value",     style("th", fontSize=8, fontName="Helvetica-Bold", textColor=t_sec, alignment=TA_CENTER)),
        Paragraph("Risk",      style("th", fontSize=8, fontName="Helvetica-Bold", textColor=t_sec, alignment=TA_CENTER)),
    ]]
    for _, row in top_risks.iterrows():
        val = row["value"]
        val_str = f"{val:.2f}" if isinstance(val, float) and val != int(val) else str(int(val)) if isinstance(val, (int, float)) else str(val)
        rc  = RISK_COLOURS.get(row["risk_level"], t_sec)
        risk_rows.append([
            Paragraph(row["description"], S["body"]),
            Paragraph(val_str, style("rv", fontSize=9, textColor=t_pri, alignment=TA_CENTER)),
            Paragraph(row["risk_level"], style("rl", fontSize=8, fontName="Helvetica-Bold", textColor=rc, alignment=TA_CENTER)),
        ])

    risk_tbl = Table(risk_rows, colWidths=[W*0.6, W*0.2, W*0.2])
    risk_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  border),
        ("BACKGROUND",    (0,1), (-1,-1), card),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ("RIGHTPADDING",  (0,0), (-1,-1), 10),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LINEBELOW",     (0,0), (-1,-2), 0.3, border),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))
    story += [risk_tbl, sp(3), hr()]

    # ── Phase 3: Investigative ────────────────────────────────────────────────
    story.append(sp(2))
    story.append(Paragraph("PHASE 3 — INVESTIGATIVE (Interpretation & Recommendation)", style("ph3", fontSize=8, fontName="Helvetica-Bold", textColor=t_sec, letterSpacing=1)))
    story.append(sp(2))

    if is_mal:
        finding = ("This employee's behavioural profile contains multiple indicators consistent "
                   "with insider threat activity. Both the supervised classifier (XGBoost) and "
                   "the unsupervised anomaly detector (Isolation Forest) flag this record as "
                   "deviating significantly from normal departmental behaviour patterns.")
        rec = ("Escalate to the security team for immediate review. Preserve all associated "
               "activity logs for forensic investigation. Do not alert the employee until "
               "the investigation is complete. Initiate chain-of-custody documentation "
               "in accordance with ISO/IEC 27043 guidelines.")
    elif result["isolation_forest_anomaly"]:
        finding = ("The supervised classifier classifies this record as Normal, however the "
                   "Isolation Forest has flagged statistically unusual behaviour. This may "
                   "indicate an emerging or novel threat pattern not yet captured in labelled training data.")
        rec = ("Monitor this employee closely over the next review cycle. Flag for secondary "
               "review if anomalous behaviour persists. No immediate escalation required.")
    else:
        finding = ("No significant threat indicators detected. This employee's behavioural "
                   "profile is consistent with normal departmental activity patterns. "
                   "Both the supervised and unsupervised models agree on this classification.")
        rec = ("No immediate action required. Continue routine monitoring as per "
               "organisational security policy.")

    ph3_data = [
        [Paragraph("Finding",        style("fl", fontSize=8, fontName="Helvetica-Bold", textColor=t_sec)),
         Paragraph(finding,          S["body"])],
        [Paragraph("Recommendation", style("fl", fontSize=8, fontName="Helvetica-Bold", textColor=t_sec)),
         Paragraph(rec,              S["body"])],
    ]
    ph3_tbl = Table(ph3_data, colWidths=[W*0.22, W*0.78])
    ph3_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), card),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ("RIGHTPADDING",  (0,0), (-1,-1), 12),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LINEBELOW",     (0,0), (-1,-2), 0.3, border),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ("LINEBEFORE",    (1,0), (1,-1), 2, risk_col),
    ]))
    story += [ph3_tbl, sp(3), hr()]

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(sp(2))
    footer_data = [[
        Paragraph("ThreatSense · Hybrid Detection Engine · COS720 University of Pretoria", S["label"]),
        Paragraph(f"Classification threshold: {result.get('threshold_used', 0.65)} · ISO/IEC 27043 aligned", style("fl2", fontSize=8, textColor=t_sec, alignment=TA_RIGHT)),
    ]]
    footer_tbl = Table(footer_data, colWidths=[W*0.6, W*0.4])
    footer_tbl.setStyle(TableStyle([
        ("LINEABOVE",     (0,0), (-1,0), 0.5, border),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(footer_tbl)

    doc.build(story)
    return buf.getvalue()