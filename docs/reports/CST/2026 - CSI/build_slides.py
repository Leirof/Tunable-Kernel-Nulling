"""Script to generate the PowerPoint presentation for CST / CSI 2026.
Creates a minimalist, modern, sleek 16:9 presentation with keyword-focused slides.
"""

from __future__ import annotations

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from PIL import Image

# Output path
OUTPUT_DIR = r"e:\PhD-Theory\docs\events\2026 - CSI"
OUTPUT_PPTX = os.path.join(OUTPUT_DIR, "Slides.pptx")
ASSETS_DIR = os.path.join(OUTPUT_DIR, "assets")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# Color Palette (Minimalist, Slate / Indigo executive theme)
BG_LIGHT = RGBColor(248, 250, 252)       # Slate 50
CARD_BG = RGBColor(255, 255, 255)        # Pure White
CARD_BORDER = RGBColor(226, 232, 240)    # Slate 200
TEXT_DARK = RGBColor(15, 23, 42)         # Slate 900
TEXT_MUTED = RGBColor(71, 85, 105)       # Slate 600
TEXT_SUBTLE = RGBColor(148, 163, 184)    # Slate 400
PRIMARY = RGBColor(37, 99, 235)          # Blue 600
PRIMARY_DARK = RGBColor(30, 64, 175)     # Blue 800
ACCENT_BG = RGBColor(239, 246, 255)      # Blue 50
ACCENT_BORDER = RGBColor(191, 219, 254)  # Blue 200
PILL_BG = RGBColor(224, 231, 255)        # Indigo 100
PILL_TEXT = RGBColor(67, 56, 202)        # Indigo 700

DARK_BG = RGBColor(15, 23, 42)           # Slate 900
DARK_TITLE = RGBColor(248, 250, 252)     # White
DARK_SUBTITLE = RGBColor(56, 189, 248)   # Sky 400

FONT_NAME = "Segoe UI"


def create_presentation() -> Presentation:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    return prs


def add_header(slide, category: str, title: str, slide_num: int, total_slides: int = 14):
    """Add standardized clean header and footer to a content slide."""
    # Category badge / pill
    pill = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.8), Inches(0.42), Inches(3.2), Inches(0.32)
    )
    pill.fill.solid()
    pill.fill.fore_color.rgb = PILL_BG
    pill.line.color.rgb = PILL_BG
    tf_pill = pill.text_frame
    tf_pill.word_wrap = True
    tf_pill.margin_left = tf_pill.margin_right = tf_pill.margin_top = tf_pill.margin_bottom = 0
    p_pill = tf_pill.paragraphs[0]
    p_pill.text = category.upper()
    p_pill.alignment = PP_ALIGN.CENTER
    p_pill.font.name = FONT_NAME
    p_pill.font.size = Pt(9.5)
    p_pill.font.bold = True
    p_pill.font.color.rgb = PILL_TEXT

    # Title
    title_box = slide.shapes.add_textbox(Inches(0.78), Inches(0.78), Inches(11.75), Inches(0.6))
    tf_title = title_box.text_frame
    tf_title.word_wrap = True
    tf_title.margin_left = tf_title.margin_right = tf_title.margin_top = tf_title.margin_bottom = 0
    p_title = tf_title.paragraphs[0]
    p_title.text = title
    p_title.font.name = FONT_NAME
    p_title.font.size = Pt(22)
    p_title.font.bold = True
    p_title.font.color.rgb = TEXT_DARK

    # Footer Left
    footer_left = slide.shapes.add_textbox(Inches(0.8), Inches(7.08), Inches(8.0), Inches(0.3))
    tf_fl = footer_left.text_frame
    tf_fl.margin_left = tf_fl.margin_right = tf_fl.margin_top = tf_fl.margin_bottom = 0
    p_fl = tf_fl.paragraphs[0]
    p_fl.text = "Comité de Suivi Individuel 2026 — Vincent Foriel"
    p_fl.font.name = FONT_NAME
    p_fl.font.size = Pt(9)
    p_fl.font.color.rgb = TEXT_SUBTLE

    # Footer Right
    footer_right = slide.shapes.add_textbox(Inches(10.53), Inches(7.08), Inches(2.0), Inches(0.3))
    tf_fr = footer_right.text_frame
    tf_fr.margin_left = tf_fr.margin_right = tf_fr.margin_top = tf_fr.margin_bottom = 0
    p_fr = tf_fr.paragraphs[0]
    p_fr.text = f"{slide_num} / {total_slides}"
    p_fr.alignment = PP_ALIGN.RIGHT
    p_fr.font.name = FONT_NAME
    p_fr.font.size = Pt(9)
    p_fr.font.color.rgb = TEXT_SUBTLE


def add_bottom_takeaway(slide, takeaway_text: str):
    """Add bottom highlighted takeaway strip for CSI members."""
    banner = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.8), Inches(6.38), Inches(11.733), Inches(0.58)
    )
    banner.fill.solid()
    banner.fill.fore_color.rgb = ACCENT_BG
    banner.line.color.rgb = ACCENT_BORDER
    banner.line.width = Pt(1)

    tf = banner.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.2)
    tf.margin_top = Inches(0.12)
    tf.margin_right = Inches(0.2)
    tf.margin_bottom = Inches(0.1)

    p = tf.paragraphs[0]
    p.text = "💡 Message clé pour le CSI : "
    p.font.name = FONT_NAME
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_DARK

    run = p.add_run()
    run.text = takeaway_text
    run.font.name = FONT_NAME
    run.font.size = Pt(11)
    run.font.bold = False
    run.font.color.rgb = TEXT_DARK


def add_content_card(slide, left: float, top: float, width: float, height: float):
    """Add a card shape for content grouping."""
    card = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    card.fill.solid()
    card.fill.fore_color.rgb = CARD_BG
    card.line.color.rgb = CARD_BORDER
    card.line.width = Pt(1)
    return card


def populate_bullets(text_frame, bullet_list: list[dict]):
    """Populate bullet points with bold keywords and concise text."""
    text_frame.word_wrap = True
    text_frame.margin_left = Inches(0.25)
    text_frame.margin_top = Inches(0.25)
    text_frame.margin_right = Inches(0.25)
    text_frame.margin_bottom = Inches(0.25)

    for i, item in enumerate(bullet_list):
        p = text_frame.paragraphs[0] if i == 0 else text_frame.add_paragraph()
        p.space_after = Pt(item.get("space_after", 10))
        p.level = item.get("level", 0)

        # Keyword
        if "keyword" in item:
            run_kw = p.add_run()
            run_kw.text = item["keyword"] + (" " if not item["keyword"].endswith(" ") else "")
            run_kw.font.name = FONT_NAME
            run_kw.font.bold = True
            run_kw.font.size = Pt(item.get("size", 14.5 if p.level == 0 else 12.5))
            run_kw.font.color.rgb = item.get("kw_color", TEXT_DARK if p.level == 0 else TEXT_MUTED)

        # Body text
        if "text" in item:
            run_txt = p.add_run()
            run_txt.text = item["text"]
            run_txt.font.name = FONT_NAME
            run_txt.font.bold = False
            run_txt.font.size = Pt(item.get("size", 14 if p.level == 0 else 12))
            run_txt.font.color.rgb = TEXT_MUTED if p.level == 0 else TEXT_SUBTLE


def add_image_or_placeholder(slide, img_path: str | None, left: float, top: float, width: float, height: float, caption: str, placeholder_desc: str = ""):
    """Insert an image centered within the bounds or draw a styled placeholder box."""
    if img_path and os.path.exists(img_path):
        try:
            with Image.open(img_path) as im:
                img_w, img_h = im.size

            box_w_px = width * 100
            box_h_px = (height - 0.4) * 100
            scale = min(box_w_px / img_w, box_h_px / img_h)

            final_w = Inches((img_w * scale) / 100)
            final_h = Inches((img_h * scale) / 100)

            # Center image horizontally and vertically
            pos_x = Inches(left + (width - (img_w * scale) / 100) / 2)
            pos_y = Inches(top + (height - 0.35 - (img_h * scale) / 100) / 2)

            slide.shapes.add_picture(img_path, pos_x, pos_y, width=final_w, height=final_h)

            # Caption
            cap_box = slide.shapes.add_textbox(Inches(left), Inches(top + height - 0.35), Inches(width), Inches(0.35))
            tf_cap = cap_box.text_frame
            tf_cap.word_wrap = True
            tf_cap.margin_left = tf_cap.margin_right = tf_cap.margin_top = tf_cap.margin_bottom = 0
            p_cap = tf_cap.paragraphs[0]
            p_cap.text = caption
            p_cap.alignment = PP_ALIGN.CENTER
            p_cap.font.name = FONT_NAME
            p_cap.font.size = Pt(10)
            p_cap.font.italic = True
            p_cap.font.color.rgb = TEXT_MUTED
            return
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

    # Fallback to placeholder box
    ph_box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    ph_box.fill.solid()
    ph_box.fill.fore_color.rgb = RGBColor(241, 245, 249) # Slate 100
    ph_box.line.color.rgb = RGBColor(203, 213, 225)      # Slate 300
    ph_box.line.width = Pt(1.5)

    tf = ph_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(0.3)

    p0 = tf.paragraphs[0]
    p0.text = "📷 [PLACEHOLDER FIGURE]"
    p0.alignment = PP_ALIGN.CENTER
    p0.font.name = FONT_NAME
    p0.font.bold = True
    p0.font.size = Pt(13)
    p0.font.color.rgb = PRIMARY_DARK
    p0.space_after = Pt(8)

    p1 = tf.add_paragraph()
    p1.text = caption
    p1.alignment = PP_ALIGN.CENTER
    p1.font.name = FONT_NAME
    p1.font.bold = True
    p1.font.size = Pt(11)
    p1.font.color.rgb = TEXT_DARK
    p1.space_after = Pt(12)

    if placeholder_desc:
        p2 = tf.add_paragraph()
        p2.text = "Instructions d'insertion :"
        p2.font.name = FONT_NAME
        p2.font.bold = True
        p2.font.size = Pt(10)
        p2.font.color.rgb = TEXT_MUTED
        p2.space_after = Pt(4)

        p3 = tf.add_paragraph()
        p3.text = placeholder_desc
        p3.font.name = FONT_NAME
        p3.font.size = Pt(9.5)
        p3.font.color.rgb = TEXT_MUTED


# ==============================================================================
# SLIDE BUILDERS
# ==============================================================================

def build_slide_1_title(prs: Presentation):
    """Slide 1 : Titre & Présentation"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    # Background
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = DARK_BG
    bg.line.color.rgb = DARK_BG

    # Header category pill
    pill = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(1.0), Inches(1.1), Inches(4.8), Inches(0.38)
    )
    pill.fill.solid()
    pill.fill.fore_color.rgb = RGBColor(30, 41, 59)
    pill.line.color.rgb = RGBColor(56, 189, 248)
    pill.line.width = Pt(1)
    tf_p = pill.text_frame
    p_p = tf_p.paragraphs[0]
    p_p.text = "THÈSE DE DOCTORAT — COMITÉ DE SUIVI (CSI 2026)"
    p_p.alignment = PP_ALIGN.CENTER
    p_p.font.name = FONT_NAME
    p_p.font.size = Pt(10.5)
    p_p.font.bold = True
    p_p.font.color.rgb = DARK_SUBTITLE

    # Main Title
    tb_title = slide.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(11.3), Inches(2.2))
    tf_title = tb_title.text_frame
    tf_title.word_wrap = True
    tf_title.margin_left = tf_title.margin_right = tf_title.margin_top = tf_title.margin_bottom = 0
    p_t = tf_title.paragraphs[0]
    p_t.text = "Caractérisation, calibration et modélisation d'un interféromètre d'annulation photonique actif pour la détection directe d'exoplanètes"
    p_t.font.name = FONT_NAME
    p_t.font.size = Pt(28)
    p_t.font.bold = True
    p_t.font.color.rgb = DARK_TITLE

    # Candidate Name
    tb_cand = slide.shapes.add_textbox(Inches(1.0), Inches(4.1), Inches(8.0), Inches(0.5))
    tf_cand = tb_cand.text_frame
    tf_cand.margin_left = tf_cand.margin_right = tf_cand.margin_top = tf_cand.margin_bottom = 0
    p_c = tf_cand.paragraphs[0]
    p_c.text = "Vincent Foriel"
    p_c.font.name = FONT_NAME
    p_c.font.size = Pt(20)
    p_c.font.bold = True
    p_c.font.color.rgb = RGBColor(255, 255, 255)

    # Supervisors & Mentors
    tb_sup = slide.shapes.add_textbox(Inches(1.0), Inches(4.7), Inches(11.0), Inches(1.0))
    tf_sup = tb_sup.text_frame
    tf_sup.word_wrap = True
    tf_sup.margin_left = tf_sup.margin_right = tf_sup.margin_top = tf_sup.margin_bottom = 0

    p_s1 = tf_sup.paragraphs[0]
    p_s1.text = "Direction : Frantz Martinache & David Mary (Laboratoire Lagrange, OCA, CNRS, UCA)"
    p_s1.font.name = FONT_NAME
    p_s1.font.size = Pt(13)
    p_s1.font.color.rgb = RGBColor(203, 213, 225)
    p_s1.space_after = Pt(4)

    p_s2 = tf_sup.add_paragraph()
    p_s2.text = "Collaborations : Marc-Antoine Martinod, Nick Cvetojevic, Roxanne Ligi, Sylvie Robbe-Dubois"
    p_s2.font.name = FONT_NAME
    p_s2.font.size = Pt(12)
    p_s2.font.color.rgb = RGBColor(148, 163, 184)
    p_s2.space_after = Pt(4)

    p_s3 = tf_sup.add_paragraph()
    p_s3.text = "Membres du Comité : Sylvie Robbe-Dubois & Jean-Marc Petit"
    p_s3.font.name = FONT_NAME
    p_s3.font.size = Pt(12)
    p_s3.font.color.rgb = RGBColor(56, 189, 248)

    # Logos image at bottom
    logo_path = os.path.join(ASSETS_DIR, "image1.png")
    if os.path.exists(logo_path):
        slide.shapes.add_picture(logo_path, Inches(1.0), Inches(6.0), width=Inches(6.2))


def build_slide_2_context(prs: Presentation):
    """Slide 2 : Rappels & Enjeux"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, "Contexte & Problématique", "Le Défi du Nulling Actif : De la Photonique aux Observables Robustes", 2)

    # Left Column Card
    left_card = add_content_card(slide, 0.8, 1.55, 5.4, 4.65)
    bullets = [
        {"keyword": "Contraste extrême :", "text": "Séparation angulaire infime et étoile 10⁴ à 10⁶ fois plus brillante en proche IR."},
        {"keyword": "Nulling interférométrique :", "text": "Extinction destructive axiale de l'étoile hôte ; transmission du signal planétaire hors axe."},
        {"keyword": "Sensibilité aux pistons :", "text": "Les perturbations de phase dégradent immédiatement l'annulation."},
        {"keyword": "Observable Kernel-Null :", "text": "Combinaison linéaire de sorties insensible au 1er ordre aux erreurs de phase."},
        {"keyword": "Puce active MMI 4x4 :", "text": "Recombinaison intégrée compacte + 4 actionneurs thermo-optiques (TOPAs) en entrée."},
    ]
    populate_bullets(left_card.text_frame, bullets)

    # Right Column Card
    right_card = add_content_card(slide, 6.45, 1.55, 6.08, 4.65)
    img_path = r"e:\PhD-Publications\Papers\4x4 MMI characterization\img\mmi_scheme.png"
    add_image_or_placeholder(
        slide, img_path, 6.55, 1.65, 5.88, 4.45,
        caption="Schéma conceptuel du composant actif MMI 4x4 avec 4 shifters thermiques (TOPAs)."
    )

    add_bottom_takeaway(
        slide,
        "Associer la stabilité de l'optique intégrée active à l'immunité aux pistons du Kernel-Null pour franchir les limites instrumentales."
    )


def build_slide_3_roadmap(prs: Presentation):
    """Slide 3 : Feuille de route & Bilan CSI 2025"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, "Bilan d'Avancement", "Continuité des Travaux depuis le CSI 2025", 3)

    # Left Column Card
    left_card = add_content_card(slide, 0.8, 1.55, 5.4, 4.65)
    bullets = [
        {"keyword": "CSI 2025 → CSI 2026 :", "text": "Passage complet de la théorie numérique au banc expérimental réel."},
        {"keyword": "Modélisation statistique :", "text": "Démonstration de la quasi-optimalité de la médiane (Neyman-Pearson)."},
        {"keyword": "Dispersion chromatique :", "text": "Implémentation dans PHISE ; validation jusqu'à 100 nm de bande passante."},
        {"keyword": "Banc & PHOBos :", "text": "Développement de l'OS de pilotage et calibration du MMI 4x4."},
        {"keyword": "Confrontation modèle/mesures :", "text": "Modèle matriciel CMPCE et jumeau numérique réaliste (caméra C-RED 3)."},
        {"keyword": "Valorisation :", "text": "Article de caractérisation en phase de révision finale."},
    ]
    populate_bullets(left_card.text_frame, bullets)

    # Right Column Card (Roadmap / Table summary)
    right_card = add_content_card(slide, 6.45, 1.55, 6.08, 4.65)
    tf_r = right_card.text_frame
    tf_r.word_wrap = True
    tf_r.margin_left = tf_r.margin_right = tf_r.margin_top = tf_r.margin_bottom = Inches(0.3)

    p0 = tf_r.paragraphs[0]
    p0.text = "🎯 Jalons Engagés vs Réalisations"
    p0.font.name = FONT_NAME
    p0.font.bold = True
    p0.font.size = Pt(15)
    p0.font.color.rgb = PRIMARY_DARK
    p0.space_after = Pt(14)

    milestones = [
        ("Théorie & Décision", "Médiane validée vs Neyman-Pearson sous turbulence", "✅ Terminé"),
        ("Dispersion spectrale", "Réponse chromatique modélisée dans PHISE", "✅ Terminé"),
        ("Banc d'optique", "Logiciel PHOBos opérationnel & acquisition fluide", "✅ En service"),
        ("Calibration labo", "Null depth franchissant le seuil des 10⁻³", "✅ Réalisé"),
        ("Modélisation fine", "Modèle CMPCE & simulation détecteur C-RED 3", "✅ Validé"),
        ("Publication A&A", "Manuscrit 4x4 MMI en relecture co-auteurs", "⏳ En cours"),
    ]

    for domain, detail, status in milestones:
        p = tf_r.add_paragraph()
        p.space_after = Pt(8)
        run_d = p.add_run()
        run_d.text = f"• {domain} : "
        run_d.font.name = FONT_NAME
        run_d.font.bold = True
        run_d.font.size = Pt(12.5)
        run_d.font.color.rgb = TEXT_DARK

        run_det = p.add_run()
        run_det.text = f"{detail} "
        run_det.font.name = FONT_NAME
        run_det.font.size = Pt(11.5)
        run_det.font.color.rgb = TEXT_MUTED

        run_st = p.add_run()
        run_st.text = f"({status})"
        run_st.font.name = FONT_NAME
        run_st.font.bold = True
        run_st.font.size = Pt(11.5)
        run_st.font.color.rgb = RGBColor(16, 185, 129) if "✅" in status else RGBColor(245, 158, 11)

    add_bottom_takeaway(
        slide,
        "Tous les objectifs annoncés l'an passé ont été tenus, transformant les simulations théoriques en résultats de laboratoire."
    )


def build_slide_4_stats(prs: Presentation):
    """Slide 4 : Modélisation statistique Neyman-Pearson vs Médiane"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, "Théorie & Données", "Données Kernel : Test Optimal de Neyman-Pearson vs Médiane", 4)

    # Left Column Card
    left_card = add_content_card(slide, 0.8, 1.55, 5.4, 4.65)
    bullets = [
        {"keyword": "Distributions asymétriques :", "text": "Les résidus de phase atmosphériques induisent des lois fortement non-gaussiennes."},
        {"keyword": "Borne théorique ultime :", "text": "Application du lemme de Neyman-Pearson (rapport de vraisemblance optimal)."},
        {"keyword": "Résultat clé :", "text": "La médiane empirique est quasiment indiscernable du test optimal théorique."},
        {"keyword": "Robustesse aux pistons RMS :", "text": "Performance maintenue quel que soit le niveau de perturbation (labo vs atmosphère)."},
        {"keyword": "Avantage pratique :", "text": "Pas besoin d'ajustement paramétrique complexe en temps réel."},
    ]
    populate_bullets(left_card.text_frame, bullets)

    # Right Column Card
    right_card = add_content_card(slide, 6.45, 1.55, 6.08, 4.65)
    img_path = r"e:\PhD-Theory\src\analysis\distrib_test_statistics\generated\thesis\statistics\test_power.png"
    add_image_or_placeholder(
        slide, img_path, 6.55, 1.65, 5.88, 4.45,
        caption="Puissance du test : Comparaison du test optimal Neyman-Pearson et de la médiane."
    )

    add_bottom_takeaway(
        slide,
        "La médiane empirique constitue un estimateur à la fois simple, robuste et quasi-optimal pour exploiter les observables Kernel."
    )


def build_slide_5_chromatic(prs: Presentation):
    """Slide 5 : Réponse chromatique dans PHISE"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, "Simulation Numérique", "Réponse Chromatique : Robustesse en Bande Étroite", 5)

    # Left Column Card
    left_card = add_content_card(slide, 0.8, 1.55, 5.4, 4.65)
    bullets = [
        {"keyword": "Au-delà du monochromatique :", "text": "Implémentation complète de la dispersion chromatique dans PHISE."},
        {"keyword": "Bande étroite (Δλ ≤ 100 nm) :", "text": "La médiane conserve intégralement ses performances de réjection."},
        {"keyword": "Bande large (~ 1 µm) :", "text": "Déformations notables des distributions statistiques en sortie de Kernel."},
        {"keyword": "Bilan pratique :", "text": "Une largeur de bande de 100 nm est largement suffisante pour les observations astronomiques visées."},
    ]
    populate_bullets(left_card.text_frame, bullets)

    # Right Column Card
    right_card = add_content_card(slide, 6.45, 1.55, 6.08, 4.65)
    img_path = r"e:\PhD-Theory\src\analysis\wavelength_scan\generated\thesis\spectral\wavelength_scan.png"
    add_image_or_placeholder(
        slide, img_path, 6.55, 1.65, 5.88, 4.45,
        caption="Scan spectral dans PHISE : Dépendance de l'extinction et stabilité en bande étroite."
    )

    add_bottom_takeaway(
        slide,
        "La technique Kernel-Null reste parfaitement robuste aux effets chromatiques jusqu'à au moins 100 nm de largeur de bande."
    )


def build_slide_6_phobos(prs: Presentation):
    """Slide 6 : Contrôle du banc optique & PHOBos"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, "Instrumentation & Logiciel", "Contrôle du Banc Optique : Écosystème Logiciel PHOBos", 6)

    # Left Column Card
    left_card = add_content_card(slide, 0.8, 1.55, 5.4, 4.65)
    bullets = [
        {"keyword": "Banc optique photonique :", "text": "Montage et alignement réalisés en grande partie grâce à Marc-Antoine Martinod."},
        {"keyword": "PHOBos (Python OOP) :", "text": "Architecture logicielle modulaire, orientée objet et hautement flexible."},
        {"keyword": "Pilotes matériels unifiés :", "text": "Caméra C-RED 3, actionneurs piézo, miroir déformable BMC, shifters thermiques."},
        {"keyword": "Automatisation complète :", "text": "Calibration, acquisition haute cadence, pré-traitement et archivage standardisé."},
        {"keyword": "Utilitaire PltEdit :", "text": "Outil transverse pour éditer et mettre en forme les tracés matplotlib post-génération."},
    ]
    populate_bullets(left_card.text_frame, bullets)

    # Right Column Card
    right_card = add_content_card(slide, 6.45, 1.55, 6.08, 4.65)
    img_path = r"e:\PhD-Publications\Papers\4x4 MMI characterization\img\bench.png"
    add_image_or_placeholder(
        slide, img_path, 6.55, 1.65, 5.88, 4.45,
        caption="Le banc de test optique PHOBos piloté par l'interface logicielle dédiée en Python."
    )

    add_bottom_takeaway(
        slide,
        "PHOBos offre un environnement logiciel robuste, pérenne et documenté pour automatiser toutes les acquisitions de laboratoire."
    )


def build_slide_7_calibration(prs: Presentation):
    """Slide 7 : Calibration active Hooke & Jeeves"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, "Résultats de Laboratoire", "Calibration sur MMI 4x4 : Franchissement du Seuil des 10⁻³", 7)

    # Left Column Card
    left_card = add_content_card(slide, 0.8, 1.55, 5.4, 4.65)
    bullets = [
        {"keyword": "Architecture testée :", "text": "MMI 4x4 équipé de 4 déphaseurs thermiques en entrée."},
        {"keyword": "Algorithme Hooke & Jeeves :", "text": "Méthode d'exploration directe adaptée et testée expérimentalement."},
        {"keyword": "Null depth obtenu :", "text": "Extinctions régulières comprises entre 10⁻² et 10⁻³."},
        {"keyword": "Progrès notable :", "text": "Amélioration d'un ordre de grandeur par rapport aux travaux de Peter Chingaipe (limités à 10⁻²)."},
        {"keyword": "Stabilité :", "text": "Convergence rapide de l'optimisation en boucle fermée sur le banc."},
    ]
    populate_bullets(left_card.text_frame, bullets)

    # Right Column Card
    right_card = add_content_card(slide, 6.45, 1.55, 6.08, 4.65)
    img_path = r"e:\PhD-Publications\Papers\4x4 MMI characterization\img\calibration.png"
    add_image_or_placeholder(
        slide, img_path, 6.55, 1.65, 5.88, 4.45,
        caption="Convergence de la calibration Hooke & Jeeves sur le banc : franchissement régulier de 10⁻³."
    )

    add_bottom_takeaway(
        slide,
        "L'optimisation active par Hooke & Jeeves permet de franchir le cap des 10⁻³ de null depth brut en conditions de laboratoire."
    )


def build_slide_8_cmpce(prs: Presentation):
    """Slide 8 : Modélisation CMPCE & Scan systématique"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, "Caractérisation Physique", "Modèle Matriciel CMPCE : Scan Systématique de la Puce", 8)

    # Left Column Card
    left_card = add_content_card(slide, 0.8, 1.55, 5.4, 4.65)
    bullets = [
        {"keyword": "Scan systématique :", "text": "Acquisition de 256 configurations d'entrées (isolées, paires, triplets, toutes)."},
        {"keyword": "Modèle matriciel CMPCE :", "text": "I = |C_out · M · P · C_in · E|² intégrant les fuites optiques (cross-talk)."},
        {"keyword": "Phases retrouvées :", "text": "Excellente cohérence avec les corrections appliquées par Hooke & Jeeves."},
        {"keyword": "Modulations d'interférence :", "text": "Très bonne reproduction des franges observées en sortie."},
        {"keyword": "Limite identifiée :", "text": "Difficulté à découpler la phase exacte des termes de cross-talk hors diagonale."},
    ]
    populate_bullets(left_card.text_frame, bullets)

    # Right Column Card
    right_card = add_content_card(slide, 6.45, 1.55, 6.08, 4.65)
    img_path = r"e:\PhD-Publications\Papers\4x4 MMI characterization\img\phasors_calibrated_day_after.png"
    add_image_or_placeholder(
        slide, img_path, 6.55, 1.65, 5.88, 4.45,
        caption="Phaseurs de sortie reconstruits par le modèle CMPCE après optimisation (voie sombre + quadratures)."
    )

    add_bottom_takeaway(
        slide,
        "Le modèle CMPCE reconstitue avec fidélité les phases d'entrée et met en évidence la présence de termes de cross-talk parasites."
    )


def build_slide_9_cred3(prs: Presentation):
    """Slide 9 : Simulateur PHISE & Détecteur C-RED 3"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, "Jumeau Numérique", "Simulateur PHISE : Intégration Réaliste de la Caméra C-RED 3", 9)

    # Left Column Card
    left_card = add_content_card(slide, 0.8, 1.55, 5.4, 4.65)
    bullets = [
        {"keyword": "Physique du capteur :", "text": "Dispersion spatiale des faisceaux gaussiens réimagés sur les pixels C-RED 3."},
        {"keyword": "Bruit réaliste :", "text": "Intégration du bruit de lecture constructeur et soustraction de dark théorique."},
        {"keyword": "Injection du modèle CMPCE :", "text": "Matrice ajustée en laboratoire intégrée directement dans PHISE."},
        {"keyword": "Concordance spectaculaire :", "text": "Simulations et mesures de banc réelles se superposent de manière robuste."},
        {"keyword": "Jumeau numérique :", "text": "Permet d'isoler numériquement l'impact individuel de chaque imperfection."},
    ]
    populate_bullets(left_card.text_frame, bullets)

    # Right Column Card
    right_card = add_content_card(slide, 6.45, 1.55, 6.08, 4.65)
    img_path = r"e:\PhD-Publications\Papers\4x4 MMI characterization\img\outputs.png"
    add_image_or_placeholder(
        slide, img_path, 6.55, 1.65, 5.88, 4.45,
        caption="Profils d'intensité des sorties réimagées et distributions associées simulées dans PHISE."
    )

    add_bottom_takeaway(
        slide,
        "PHISE est devenu un véritable jumeau numérique du banc, reproduisant fidèlement les distributions expérimentales observées."
    )


def build_slide_10_limits(prs: Presentation):
    """Slide 10 : Limites fondamentales (Nick's plot)"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, "Limites Physiques", "Limites Fondamentales : Détecteur, Cross-Talk & Rôle de H&J", 10)

    # Left Column Card
    left_card = add_content_card(slide, 0.8, 1.55, 5.4, 4.65)
    bullets = [
        {"keyword": "Limite douce détecteur :", "text": "Le bruit de la caméra C-RED 3 borne généralement le null brut vers 10⁻³."},
        {"keyword": "Limite dure cross-talk :", "text": "Les termes de cross-talk fixent un plancher fondamental infranchissable à 10⁻³."},
        {"keyword": "Stabilité temporelle :", "text": "Cross-talk stable sur plusieurs jours et à différentes températures (remarque F. Martinache) → interne à la puce."},
        {"keyword": "Limitation de Hooke & Jeeves :", "text": "L'algo compense les déséquilibres d'amplitude, mais PAS les phases de cross-talk."},
        {"keyword": "Plot de Nick Cvetojevic :", "text": "Lien direct entre taux de cross-talk fonderie (coût) et limite d'extinction attendue."},
    ]
    populate_bullets(left_card.text_frame, bullets)

    # Right Column Card
    right_card = add_content_card(slide, 6.45, 1.55, 6.08, 4.65)
    img_path = os.path.join(ASSETS_DIR, "crosstalk_vs_null.png")
    add_image_or_placeholder(
        slide, img_path, 6.55, 1.65, 5.88, 4.45,
        caption="Plot de Nick : Null depth en fonction du cross-talk (avant vs après calibration Hooke & Jeeves)."
    )

    add_bottom_takeaway(
        slide,
        "Le cross-talk interne constitue la limite physique fondamentale à 10⁻³ ; Hooke & Jeeves ne permet pas de le contourner."
    )


def build_slide_11_activities(prs: Presentation):
    """Slide 11 : Valorisation, Conférences & Formations"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, "Activités Doctorales", "Activités Scientifiques : Conférences & Formations", 11)

    # Left Column Card
    left_card = add_content_card(slide, 0.8, 1.55, 5.4, 4.65)
    bullets = [
        {"keyword": "Conférences internationales :", "text": "Participation active aux rendez-vous majeurs de la communauté :"},
        {"keyword": "• Workshop WITSO (ESA) :", "text": "Octobre 2025 — Échanges sur les architectures spatiales de nulling.", "level": 1},
        {"keyword": "• Conférence LIFE :", "text": "Novembre 2025 — Présentation des méthodes statistiques et Kernel-Nulling.", "level": 1},
        {"keyword": "Aléa médical :", "text": "Annulation contrainte de la conférence SPIE (juillet 2026)."},
        {"keyword": "Formations doctorales :", "text": "Quota d'heures de formation requis atteint (validations finales en cours sur ADUM)."},
    ]
    populate_bullets(left_card.text_frame, bullets)

    # Right Column Card (Structured visual summary)
    right_card = add_content_card(slide, 6.45, 1.55, 6.08, 4.65)
    tf_r = right_card.text_frame
    tf_r.word_wrap = True
    tf_r.margin_left = tf_r.margin_right = tf_r.margin_top = tf_r.margin_bottom = Inches(0.3)

    p0 = tf_r.paragraphs[0]
    p0.text = "📌 Bilan des Engagements Doctoraux"
    p0.font.name = FONT_NAME
    p0.font.bold = True
    p0.font.size = Pt(15)
    p0.font.color.rgb = PRIMARY_DARK
    p0.space_after = Pt(14)

    items = [
        ("Conférence WITSO (ESA)", "Octobre 2025", "Mission spatiale & interférométrie"),
        ("Conférence LIFE", "Novembre 2025", "Communauté détection exoplanètes"),
        ("Conférence SPIE", "Juillet 2026", "Annulée pour raison médicale"),
        ("Formations transversales", "UCA / EDSFA", "Heures complètes effectuées"),
        ("Formations scientifiques", "Ateliers & Séminaires", "Quota global atteint"),
    ]

    for title, date, desc in items:
        p = tf_r.add_paragraph()
        p.space_after = Pt(8)
        run_t = p.add_run()
        run_t.text = f"• {title} ({date}) : "
        run_t.font.name = FONT_NAME
        run_t.font.bold = True
        run_t.font.size = Pt(12)
        run_t.font.color.rgb = TEXT_DARK

        run_d = p.add_run()
        run_d.text = desc
        run_d.font.name = FONT_NAME
        run_d.font.size = Pt(11.5)
        run_d.font.color.rgb = TEXT_MUTED

    add_bottom_takeaway(
        slide,
        "Présence assurée auprès de la communauté européenne du nulling et obligations de formation doctorale validées."
    )


def build_slide_12_paper_adhd(prs: Presentation):
    """Slide 12 : Rédaction Article 4x4 & Diagnostic TDAH"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, "Avancement & Rédaction", "Rédaction de l'Article 4x4 : Diagnostic TDAH & Dynamique de Fin", 12)

    # Left Column Card
    left_card = add_content_card(slide, 0.8, 1.55, 5.4, 4.65)
    bullets = [
        {"keyword": "Article A&A en cours :", "text": "Calibration and characterization of an active photonic nulling interferometer."},
        {"keyword": "État du manuscrit :", "text": "Structure complète, figures générées, phase de révision avec les co-auteurs."},
        {"keyword": "Diagnostic TDAH récent :", "text": "Identification clinique de la source des lenteurs et difficultés attentionnelles."},
        {"keyword": "Prise en charge médicale :", "text": "Mise en place d'un protocole adapté (Ritaline) et structuration du cadre de travail."},
        {"keyword": "Perspective positive :", "text": "Soulagement et sérénité pour aborder le sprint final d'écriture du manuscrit."},
    ]
    populate_bullets(left_card.text_frame, bullets)

    # Right Column Card
    right_card = add_content_card(slide, 6.45, 1.55, 6.08, 4.65)
    img_path = os.path.join(ASSETS_DIR, "paper_page1.png")
    add_image_or_placeholder(
        slide, img_path, 6.55, 1.65, 5.88, 4.45,
        caption="Manuscrit de l'article de caractérisation du MMI 4x4 (soumission A&A)."
    )

    add_bottom_takeaway(
        slide,
        "Le diagnostic médical récent apporte une réponse concrète pour accélérer sereinement la finalisation du papier et du manuscrit."
    )


def build_slide_13_perspectives(prs: Presentation):
    """Slide 13 : Perspectives scientifiques & transmission"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, "Perspectives & Pérennité", "Perspectives Scientifiques & Transmission aux Successeurs", 13)

    # Left Column Card
    left_card = add_content_card(slide, 0.8, 1.55, 5.4, 4.65)
    bullets = [
        {"keyword": "Extension Kernel-Null :", "text": "Appliquer l'étude cross-talk de Nick directement à l'observable Kernel (au-delà du null brut)."},
        {"keyword": "Passerelle PHOBos ↔ PHISE :", "text": "Outil automatique pour extraire la matrice d'un composant sur banc et l'injecter dans PHISE."},
        {"keyword": "Documentation & Packaging :", "text": "Nettoyage, modularisation et guides d'utilisation complets pour le laboratoire."},
        {"keyword": "Transmission pérenne :", "text": "Permettre aux futurs doctorants et chercheurs de reprendre les travaux sans redondance."},
    ]
    populate_bullets(left_card.text_frame, bullets)

    # Right Column Card (Workflow architecture)
    right_card = add_content_card(slide, 6.45, 1.55, 6.08, 4.65)
    tf_r = right_card.text_frame
    tf_r.word_wrap = True
    tf_r.margin_left = tf_r.margin_right = tf_r.margin_top = tf_r.margin_bottom = Inches(0.3)

    p0 = tf_r.paragraphs[0]
    p0.text = "🔄 Écosystème Pérenne pour le Laboratoire"
    p0.font.name = FONT_NAME
    p0.font.bold = True
    p0.font.size = Pt(15)
    p0.font.color.rgb = PRIMARY_DARK
    p0.space_after = Pt(14)

    steps = [
        ("1. Banc PHOBos", "Scan systématique automatisé & acquisition des données"),
        ("2. Outil Matriciel", "Extraction automatique du modèle CMPCE sur banc"),
        ("3. Passerelle directe", "Export direct vers la classe Context de PHISE"),
        ("4. Jumeau Numérique PHISE", "Simulations astrophysiques & optimisation des algorithmes"),
        ("5. PltEdit", "Édition graphique standardisée des résultats scientifiques"),
    ]

    for step, desc in steps:
        p = tf_r.add_paragraph()
        p.space_after = Pt(8)
        run_s = p.add_run()
        run_s.text = f"• {step} : "
        run_s.font.name = FONT_NAME
        run_s.font.bold = True
        run_s.font.size = Pt(12)
        run_s.font.color.rgb = TEXT_DARK

        run_d = p.add_run()
        run_d.text = desc
        run_d.font.name = FONT_NAME
        run_d.font.size = Pt(11.5)
        run_d.font.color.rgb = TEXT_MUTED

    add_bottom_takeaway(
        slide,
        "Laisser une suite logicielle documentée, éprouvée et interopérable garantissant une transmission immédiate au laboratoire."
    )


def build_slide_14_schedule(prs: Presentation):
    """Slide 14 : Calendrier & Rétroplanning vers la soutenance"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, "Planning de Fin de Thèse", "Rétroplanning vers la Soutenance de Thèse", 14)

    # Left Column Card
    left_card = add_content_card(slide, 0.8, 1.55, 5.4, 4.65)
    bullets = [
        {"keyword": "Automne 2026 :", "text": "Soumission article A&A 4x4 MMI + finalisation de la passerelle PHOBos/PHISE."},
        {"keyword": "Hiver 2026-2027 :", "text": "Rédaction intensive du manuscrit de thèse (3 piliers : théorie, banc, modélisation)."},
        {"keyword": "Printemps 2027 :", "text": "Relectures des directeurs, ajustements et dépôt officiel aux rapporteurs."},
        {"keyword": "Mi-2027 :", "text": "Soutenance de thèse à Nice."},
        {"keyword": "Après-thèse :", "text": "Projet professionnel orienté vers l'ingénierie de recherche / R&D."},
    ]
    populate_bullets(left_card.text_frame, bullets)

    # Right Column Card (Gantt-like visual milestones)
    right_card = add_content_card(slide, 6.45, 1.55, 6.08, 4.65)
    tf_r = right_card.text_frame
    tf_r.word_wrap = True
    tf_r.margin_left = tf_r.margin_right = tf_r.margin_top = tf_r.margin_bottom = Inches(0.3)

    p0 = tf_r.paragraphs[0]
    p0.text = "📅 Calendrier Prévisionnel Séquencé"
    p0.font.name = FONT_NAME
    p0.font.bold = True
    p0.font.size = Pt(15)
    p0.font.color.rgb = PRIMARY_DARK
    p0.space_after = Pt(14)

    phases = [
        ("Sept. – Nov. 2026", "Finalisation & Soumission Article A&A", RGBColor(37, 99, 235)),
        ("Oct. – Déc. 2026", "Pont Cross-talk/Kernel & Outils PHOBos/PHISE", RGBColor(14, 165, 233)),
        ("Nov. 2026 – Fév. 2027", "Rédaction Manuscrit de Thèse", RGBColor(99, 102, 241)),
        ("Mars – Avril 2027", "Relectures & Dépôt aux Rapporteurs", RGBColor(168, 85, 247)),
        ("Mai – Juin 2027", "Préparation de la Soutenance", RGBColor(236, 72, 153)),
        ("Mi-2027", "🎓 Soutenance de Thèse", RGBColor(16, 185, 129)),
    ]

    for dates, label, color in phases:
        p = tf_r.add_paragraph()
        p.space_after = Pt(8)
        run_d = p.add_run()
        run_d.text = f"• {dates} : "
        run_d.font.name = FONT_NAME
        run_d.font.bold = True
        run_d.font.size = Pt(12)
        run_d.font.color.rgb = color

        run_l = p.add_run()
        run_l.text = label
        run_l.font.name = FONT_NAME
        run_l.font.size = Pt(11.5)
        run_l.font.color.rgb = TEXT_DARK

    add_bottom_takeaway(
        slide,
        "Un rétroplanning structuré et réaliste pour finaliser la publication, le manuscrit et soutenir la thèse mi-2027."
    )


def main():
    print("Building presentation...")
    prs = create_presentation()

    build_slide_1_title(prs)
    build_slide_2_context(prs)
    build_slide_3_roadmap(prs)
    build_slide_4_stats(prs)
    build_slide_5_chromatic(prs)
    build_slide_6_phobos(prs)
    build_slide_7_calibration(prs)
    build_slide_8_cmpce(prs)
    build_slide_9_cred3(prs)
    build_slide_10_limits(prs)
    build_slide_11_activities(prs)
    build_slide_12_paper_adhd(prs)
    build_slide_13_perspectives(prs)
    build_slide_14_schedule(prs)

    prs.save(OUTPUT_PPTX)
    print(f"Presentation saved successfully to: {OUTPUT_PPTX}")

    # Also make a copy in docs/reports/CST/Slides.pptx for convenience
    cst_copy = r"e:\PhD-Theory\docs\reports\CST\Slides.pptx"
    prs.save(cst_copy)
    print(f"Convenience copy saved to: {cst_copy}")


if __name__ == "__main__":
    main()
