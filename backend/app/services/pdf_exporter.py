import io
from collections import defaultdict
from datetime import datetime
from typing import Dict, List
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.models.schemas import AnalysisResponse, PriorityTier, QuestionCluster
from app.services.clustering import chronological_paper_key


class CleanBookCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_footer(num_pages)
            super().showPage()
        super().save()

    def _draw_footer(self, page_count: int):
        self.saveState()
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(18 * mm, 15 * mm, A4[0] - 18 * mm, 15 * mm)
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(18 * mm, 10.5 * mm, "RepPYQ by Pushkar Shinde · pushkarez.vercel.app")
        self.drawRightString(
            A4[0] - 18 * mm, 10.5 * mm, f"Page {self._pageNumber} of {page_count}"
        )
        self.restoreState()


def _unit_sort_key(unit: str) -> tuple:
    if unit.startswith("Unit "):
        try:
            num = int(unit.split()[1])
            return (0, num, unit)
        except Exception:
            pass
    return (1, 99, unit)


def build_revision_pdf(analysis: AnalysisResponse) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=21 * mm,
        title=f"{analysis.subject_name} - RepPYQ Revision Notes",
    )

    styles = getSampleStyleSheet()
    ink = colors.HexColor("#0F172A")
    body = colors.HexColor("#1E293B")
    slate = colors.HexColor("#475569")
    muted = colors.HexColor("#64748B")
    border_soft = colors.HexColor("#E2E8F0")
    card_bg = colors.HexColor("#F8FAFC")
    accent = colors.HexColor("#1E3A8A")
    high_col = colors.HexColor("#BE123C")
    med_col = colors.HexColor("#0369A1")

    title_style = ParagraphStyle(
        "TitleClean", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=22, leading=27, textColor=ink, spaceAfter=5,
    )
    meta_style = ParagraphStyle(
        "MetaClean", parent=styles["Normal"], fontName="Helvetica",
        fontSize=9.5, leading=14.5, textColor=slate, spaceAfter=12,
    )
    h2_style = ParagraphStyle(
        "H2Clean", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=12.5, leading=16, textColor=ink, spaceBefore=10, spaceAfter=7,
    )
    unit_title_style = ParagraphStyle(
        "UnitTitle", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=16, leading=20, textColor=ink,
    )
    unit_sub_style = ParagraphStyle(
        "UnitSub", parent=styles["Normal"], fontName="Helvetica",
        fontSize=9, leading=12, textColor=muted, alignment=2,
    )
    topic_style = ParagraphStyle(
        "TopicClean", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=11.2, leading=15, textColor=ink, spaceAfter=2,
    )
    rep_high_style = ParagraphStyle(
        "RepHigh", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=8.5, leading=11, textColor=high_col, spaceAfter=6,
    )
    rep_med_style = ParagraphStyle(
        "RepMed", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=8.5, leading=11, textColor=med_col, spaceAfter=6,
    )
    rep_low_style = ParagraphStyle(
        "RepLow", parent=styles["Normal"], fontName="Helvetica",
        fontSize=8.5, leading=11, textColor=muted, spaceAfter=6,
    )
    q_meta_style = ParagraphStyle(
        "QMeta", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=8.3, leading=11, textColor=slate, spaceAfter=2.5,
    )
    q_text_style = ParagraphStyle(
        "QBody", parent=styles["Normal"], fontName="Helvetica",
        fontSize=9.5, leading=14, textColor=body,
    )

    story = []

    # --- PAGE 1: CLEAN STUDY OVERVIEW ---
    sorted_papers = sorted(analysis.papers, key=lambda p: chronological_paper_key(p.paper_name))
    story.append(Paragraph(escape(analysis.subject_name), title_style))
    paper_list = "  ·  ".join(f"{p.paper_name} ({p.exam_type})" for p in sorted_papers)
    story.append(
        Paragraph(
            f"<b>Papers Analyzed ({analysis.total_papers}):</b> {escape(paper_list)}<br/>"
            f"<b>Exact Questions:</b> {analysis.total_questions}   ·   "
            f"<b>Generated:</b> {datetime.now().strftime('%d %b %Y')}",
            meta_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=ink, spaceAfter=14))

    unit_groups: Dict[str, List[QuestionCluster]] = defaultdict(list)
    for cl in analysis.clusters:
        u = (cl.unit_label or "General / Unmapped").split("·")[0].strip()
        unit_groups[u].append(cl)

    sorted_units = sorted(unit_groups.keys(), key=_unit_sort_key)

    story.append(Paragraph("Unit-Wise Question Summary", h2_style))
    overview_rows = [["Unit", "Total Topics", "Repeated (2+ Papers)", "Asked Once"]]
    for u in sorted_units:
        cls = unit_groups[u]
        rep = sum(1 for c in cls if c.unique_paper_count >= 2)
        once = len(cls) - rep
        overview_rows.append([u, str(len(cls)), str(rep), str(once)])

    ov_table = Table(overview_rows, colWidths=[56 * mm, 32 * mm, 48 * mm, 38 * mm])
    ov_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), ink),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, card_bg]),
            ("GRID", (0, 0), (-1, -1), 0.5, border_soft),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ])
    )
    story.append(ov_table)
    story.append(Spacer(1, 12))

    repeated_all = [c for c in analysis.clusters if c.unique_paper_count >= 2]
    if repeated_all:
        story.append(Paragraph("Top Repeated Topics (Must Revise First)", h2_style))
        chk_rows = [["#", "Unit", "Topic", "Frequency"]]
        for i, cl in enumerate(repeated_all[:15], start=1):
            u_short = (cl.unit_label or "-").split("·")[0].strip()
            chk_rows.append([
                str(i),
                u_short,
                Paragraph(escape(cl.topic_title), ParagraphStyle("ChkT", fontSize=8.8, leading=12)),
                f"{cl.unique_paper_count} / {cl.total_papers} papers",
            ])
        chk_table = Table(chk_rows, colWidths=[10 * mm, 24 * mm, 108 * mm, 32 * mm])
        chk_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.4, border_soft),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ])
        )
        story.append(chk_table)

    # --- UNIT BY UNIT CHAPTERS ---
    for unit_name in sorted_units:
        story.append(PageBreak())

        unit_clusters = sorted(
            unit_groups[unit_name],
            key=lambda c: (c.unique_paper_count, c.total_occurrences),
            reverse=True,
        )
        total_u_qs = sum(len(c.questions) for c in unit_clusters)

        # Clean Editorial Unit Header
        hdr_table = Table(
            [[
                Paragraph(escape(unit_name.upper()), unit_title_style),
                Paragraph(
                    f"{len(unit_clusters)} Topics  ·  {total_u_qs} Questions",
                    unit_sub_style,
                ),
            ]],
            colWidths=[104 * mm, 70 * mm],
        )
        hdr_table.setStyle(
            TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LINEBELOW", (0, 0), (-1, -1), 1.8, accent),
            ])
        )
        story.append(hdr_table)
        story.append(Spacer(1, 14))

        for idx, cl in enumerate(unit_clusters, start=1):
            if cl.priority == PriorityTier.HIGH:
                b_style = rep_high_style
                tag = "MUST PREPARE"
                bar_col = high_col
            elif cl.priority == PriorityTier.MEDIUM:
                b_style = rep_med_style
                tag = "IMPORTANT"
                bar_col = med_col
            else:
                b_style = rep_low_style
                tag = "ASKED ONCE" if cl.unique_paper_count == 1 else "LOW FREQUENCY"
                bar_col = colors.HexColor("#94A3B8")

            topic_p = Paragraph(f"{idx}. {escape(cl.topic_title)}", topic_style)
            badge_p = Paragraph(
                f"{tag}   ·   Appeared in {cl.unique_paper_count} of {cl.total_papers} papers",
                b_style,
            )

            ordered_qs = sorted(
                cl.questions,
                key=lambda q: chronological_paper_key(q.paper_name, q.question_number),
            )

            q_cards: List[Table] = []
            for q in ordered_qs:
                marks_str = f"   ·   {q.marks} Marks" if q.marks is not None else ""
                meta_p = Paragraph(
                    f"{escape(q.paper_name)}   ·   {escape(q.question_number)}{marks_str}",
                    q_meta_style,
                )
                body_p = Paragraph(escape(q.exact_text), q_text_style)

                card = Table([[meta_p], [body_p]], colWidths=[174 * mm])
                card.setStyle(
                    TableStyle([
                        ("BACKGROUND", (0, 0), (-1, -1), card_bg),
                        ("LINEBEFORE", (0, 0), (0, -1), 2.2, bar_col),
                        ("BOX", (0, 0), (-1, -1), 0.4, border_soft),
                        ("LEFTPADDING", (0, 0), (-1, -1), 10),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                        ("TOPPADDING", (0, 0), (-1, 0), 7),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 1),
                        ("TOPPADDING", (0, 1), (-1, 1), 1),
                        ("BOTTOMPADDING", (0, 1), (-1, 1), 7),
                    ])
                )
                q_cards.append(card)

            # Keep topic header + first question card together so title is never orphaned
            first_group = [topic_p, badge_p]
            if q_cards:
                first_group.append(q_cards[0])
            story.append(KeepTogether(first_group))

            for extra_card in q_cards[1:]:
                story.append(Spacer(1, 5))
                story.append(extra_card)

            # Generous spacing between different topics
            story.append(Spacer(1, 15))

    doc.build(story, canvasmaker=CleanBookCanvas)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes
