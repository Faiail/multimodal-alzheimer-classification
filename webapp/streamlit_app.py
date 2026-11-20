"""
Multimodal Alzheimer's Disease Classification - Streamlit Web Application
Features:
- 3D Neuroimaging Slice Viewer (sMRI, 18F-FDG PET, Dual View, Alpha Fusion Overlay)
- Tri-planar slicing (Axial, Coronal, Sagittal) with colormaps and slice slider (0-127)
- Patient Clinical Profile & Biomarkers (PTID, Age, Gender, Education, MMSE, APOE4, Diagnosis)
- Prediction Dashboard across all 11 tested models (KGLassoNet, MMGNN, HiGNN, BrainGNN,
  PopulationGCN, DenseNet121, ResNet18, ResNet34, ResNet50, ViT, SwinViT)
- 5-Fold Cross Validation Benchmark Analysis & Dataset Distribution
"""

import sys
from pathlib import Path
import io

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from PIL import Image

from webapp.data_manager import DataManager
from webapp.prediction_engine import PredictionEngine

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Alzheimer Multimodal AI Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Custom Styling (Dark Medical UI)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #1f2937;
        padding: 12px 16px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.8rem;
        color: #9ca3af;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.3rem;
        color: #f3f4f6;
        font-weight: 700;
    }
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    /* Custom container borders */
    .med-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 14px;
    }
    .pred-card {
        background-color: #0b0f19;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Backend Services
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Caricamento dataset e indici volumetrici...")
def get_data_manager() -> DataManager:
    return DataManager(PROJECT_ROOT)

@st.cache_resource(show_spinner="Inizializzazione motore di predizione (11 modelli)...")
def get_prediction_engine() -> PredictionEngine:
    return PredictionEngine(PROJECT_ROOT)

data_mgr = get_data_manager()
pred_engine = get_prediction_engine()


# -----------------------------------------------------------------------------
# Curated Showcase Presets
# -----------------------------------------------------------------------------
SHOWCASE_SCANS = [
    {"label": "🟢 CN (Paziente Sano) - I238627", "scan_id": "I238627", "dx": "CN"},
    {"label": "🟡 MCI (Decadimento Lieve) - I241470", "scan_id": "I241470", "dx": "MCI"},
    {"label": "🔴 AD (Alzheimer Avanzato) - I233777", "scan_id": "I233777", "dx": "AD"},
    {"label": "🟣 AD (Stadio Grave, MMSE 14) - I229272", "scan_id": "I229272", "dx": "AD"},
    {"label": "🔵 Paired sMRI + PET - I115006", "scan_id": "I115006", "dx": "CN"},
    {"label": "⚪ sMRI-only Normal - I241416", "scan_id": "I241416", "dx": "AD"},
]


# -----------------------------------------------------------------------------
# SIDEBAR: Dataset Search, Filters & Selection
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧠 Alzheimer AI Explorer")
    st.caption("Multimodal sMRI & 18F-FDG PET Deep Classification")

    # Quick Presets
    st.markdown("##### 📌 Casi Dimostrativi:")
    preset_choice = st.selectbox(
        "Seleziona un caso clinico:",
        options=[s["label"] for s in SHOWCASE_SCANS],
        index=0,
        key="preset_select"
    )
    selected_preset_scan = next(s["scan_id"] for s in SHOWCASE_SCANS if s["label"] == preset_choice)

    st.divider()

    # Search and Filter
    st.markdown("##### 🔍 Ricerca su 5,002 Scansioni:")
    search_query = st.text_input("Cerca Scan ID o PTID", placeholder="es. I238627, 002_S_0295", key="search_input")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        diag_filter = st.selectbox("Diagnosi:", ["ALL", "CN", "MCI", "AD"], index=0)
    with col_f2:
        modal_filter = st.selectbox("Modalità:", ["ALL", "PAIRED", "MRI_ONLY"], index=0)

    # Filter scans in dataframe
    filtered_res = data_mgr.filter_scans(
        query=search_query,
        diagnosis=diag_filter,
        modality=modal_filter,
        limit=100
    )
    matching_scans = filtered_res["scans"]
    total_found = filtered_res["total"]

    st.caption(f"Trovate **{total_found:,}** scansioni (prime {len(matching_scans)} mostrate)")

    # Build options list
    scan_options = {}
    for s in matching_scans:
        label = f"{'🟢' if s['diagnosis']=='CN' else ('🟡' if s['diagnosis']=='MCI' else '🔴')} {s['scan_id']} ({s['ptid']}) - {s['diagnosis']}"
        scan_options[label] = s["scan_id"]

    # Selection logic: prioritize search/filter or preset
    default_index = 0
    option_labels = list(scan_options.keys())
    
    # Check if preset scan is in list
    matching_index = 0
    if not search_query and selected_preset_scan:
        for idx, lbl in enumerate(option_labels):
            if selected_preset_scan in lbl:
                matching_index = idx
                break

    if option_labels:
        selected_label = st.selectbox(
            "Seleziona Scansione Attiva:",
            options=option_labels,
            index=matching_index,
            key="scan_selector"
        )
        current_scan_id = scan_options[selected_label]
    else:
        # Fallback to preset
        current_scan_id = selected_preset_scan

    st.divider()

    # Dataset Summary Metrics Expander
    stats = data_mgr.get_stats()
    with st.expander("📊 Statistiche Dataset Globale", expanded=False):
        st.write(f"**Totale Scansioni:** {stats.get('total_scans', 5003):,}")
        st.write(f"**Pazienti Unici:** {stats.get('unique_patients', 1092):,}")
        st.write(f"**Distribuzione Classi:**")
        cls_dict = stats.get('classes', {})
        st.write(f"- MCI: {cls_dict.get('MCI', 2811):,} ({cls_dict.get('MCI', 2811)/stats.get('total_scans', 5003):.1%})")
        st.write(f"- CN:  {cls_dict.get('CN', 1485):,} ({cls_dict.get('CN', 1485)/stats.get('total_scans', 5003):.1%})")
        st.write(f"- AD:  {cls_dict.get('AD', 707):,} ({cls_dict.get('AD', 707)/stats.get('total_scans', 5003):.1%})")
        st.write(f"**Scansioni con PET:** {stats.get('has_pet_count', 2564):,}")
        st.write(f"**Età Media:** {stats.get('age_mean', 73.5)} anni")
        st.write(f"**MMSE Medio:** {stats.get('mmse_mean', 27.4)} / 30")


# -----------------------------------------------------------------------------
# Fetch Active Scan Data & Model Predictions
# -----------------------------------------------------------------------------
scan_record = data_mgr.get_scan_by_id(current_scan_id)
if not scan_record:
    st.error(f"Scansione '{current_scan_id}' non trovata.")
    st.stop()

# Retrieve raw dataframe row for exact inference
meta_row = data_mgr.df[data_mgr.df["scan_id"] == scan_record["scan_id"]].iloc[0]
predictions_data = pred_engine.compute_scan_predictions(meta_row)
all_preds = predictions_data["predictions"]
consensus = predictions_data["consensus"]


# -----------------------------------------------------------------------------
# APP HEADER
# -----------------------------------------------------------------------------
col_h1, col_h2 = st.columns([8, 4])
with col_h1:
    st.title("🧠 Alzheimer Multimodal AI Explorer")
    st.markdown(
        "Esplorazione volumetrica 3D di **sMRI (T1-w)**, **$^{18}\\text{F}$-FDG PET** e predizioni cliniche su **11 modelli di Deep Learning**."
    )
with col_h2:
    st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)
    gt_dx = scan_record["diagnosis"]
    badge_bg = "#064e3b" if gt_dx == "CN" else ("#78350f" if gt_dx == "MCI" else "#7f1d1d")
    badge_color = "#34d399" if gt_dx == "CN" else ("#fbbf24" if gt_dx == "MCI" else "#f87171")
    st.markdown(f"""
    <div style="background-color: {badge_bg}; border: 1px solid {badge_color}; border-radius: 8px; padding: 10px 16px; text-align: center;">
        <span style="font-size: 0.75rem; color: #d1d5db; text-transform: uppercase; font-weight: 600;">Diagnosi Reale (Ground Truth)</span><br/>
        <span style="font-size: 1.4rem; color: {badge_color}; font-weight: 800; letter-spacing: 0.05em;">{gt_dx} &bull; {'Sano (CN)' if gt_dx=='CN' else ('Decadimento Lieve (MCI)' if gt_dx=='MCI' else 'Alzheimer (AD)')}</span>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# MAIN APP TABS
# -----------------------------------------------------------------------------
tab_explorer, tab_benchmarks, tab_dataset = st.tabs([
    "🔬 Visualizzatore Scansione & Predizioni Modelli",
    "📊 Benchmark 5-Fold Cross Validation (11 Modelli)",
    "📈 Statistiche & Distribuzioni Dataset"
])


# =============================================================================
# TAB 1: EXPLORER (Viewer + Patient Profile + 11 Models Predictions)
# =============================================================================
with tab_explorer:
    col_viewer, col_info = st.columns([7, 5], gap="large")

    # -------------------------------------------------------------------------
    # LEFT COLUMN: 3D SLICE VIEWER
    # -------------------------------------------------------------------------
    with col_viewer:
        st.markdown("#### 🖼️ Visualizzatore 3D Multi-Planare")

        # Modality and Plane controls
        row_c1, row_c2 = st.columns(2)
        with row_c1:
            modality_option = st.selectbox(
                "Modalità Imaging:",
                options=["sMRI (T1-w)", "18F-FDG PET", "Dual View (MRI + PET)", "Alpha Fusion Overlay"],
                index=0,
                key="modality_select"
            )
        with row_c2:
            plane_option = st.selectbox(
                "Piano Anatomico:",
                options=["Assiale (Z)", "Coronale (Y)", "Sagittale (X)"],
                index=0,
                key="plane_select"
            )

        plane_map = {"Assiale (Z)": "axial", "Coronale (Y)": "coronal", "Sagittale (X)": "sagittal"}
        current_plane = plane_map[plane_option]

        # Slider & Navigation controls
        row_s1, row_s2 = st.columns([8, 4])
        with row_s1:
            slice_idx = st.slider(
                "Indice Slice (0 - 127):",
                min_value=0,
                max_value=127,
                value=64,
                step=1,
                key="slice_slider"
            )
        with row_s2:
            cmap_choice = st.selectbox(
                "Colormap:",
                options=["gray", "inferno", "hot", "jet", "viridis", "plasma"],
                index=0 if modality_option == "sMRI (T1-w)" else 1,
                key="cmap_select"
            )

        # Alpha blending slider for Fusion
        alpha_val = 0.55
        if modality_option == "Alpha Fusion Overlay":
            alpha_val = st.slider(
                "Opacità PET Overlay (Alpha):",
                min_value=0.0,
                max_value=1.0,
                value=0.55,
                step=0.05,
                key="alpha_slider"
            )

        # Render Slice(s)
        safetensor_file = scan_record["safetensor_file"]

        if modality_option == "Dual View (MRI + PET)":
            col_mri, col_pet = st.columns(2)
            with col_mri:
                mri_bytes = data_mgr.render_slice(
                    safetensor_file=safetensor_file,
                    modality="mri",
                    plane=current_plane,
                    slice_idx=slice_idx,
                    colormap="gray"
                )
                mri_img = Image.open(io.BytesIO(mri_bytes))
                st.image(mri_img, caption=f"sMRI T1-w • {plane_option} Slice {slice_idx}", use_container_width=True)
            with col_pet:
                pet_bytes = data_mgr.render_slice(
                    safetensor_file=safetensor_file,
                    modality="pet",
                    plane=current_plane,
                    slice_idx=slice_idx,
                    colormap=cmap_choice if cmap_choice != "gray" else "hot"
                )
                pet_img = Image.open(io.BytesIO(pet_bytes))
                st.image(pet_img, caption=f"18F-FDG PET • {plane_option} Slice {slice_idx}", use_container_width=True)

        elif modality_option == "Alpha Fusion Overlay":
            fused_bytes = data_mgr.render_slice(
                safetensor_file=safetensor_file,
                modality="fused",
                plane=current_plane,
                slice_idx=slice_idx,
                colormap=cmap_choice if cmap_choice != "gray" else "inferno",
                alpha=alpha_val
            )
            fused_img = Image.open(io.BytesIO(fused_bytes))
            st.image(
                fused_img,
                caption=f"PET / sMRI Fusion (Alpha: {int(alpha_val*100)}%) • {plane_option} Slice {slice_idx}",
                use_container_width=True
            )

        else:
            # Single modality: MRI or PET
            req_mod = "mri" if "MRI" in modality_option else "pet"
            single_bytes = data_mgr.render_slice(
                safetensor_file=safetensor_file,
                modality=req_mod,
                plane=current_plane,
                slice_idx=slice_idx,
                colormap=cmap_choice
            )
            single_img = Image.open(io.BytesIO(single_bytes))
            st.image(
                single_img,
                caption=f"{modality_option} • {plane_option} Slice {slice_idx}/127",
                use_container_width=True
            )

        # Slice Download button
        st.download_button(
            label="💾 Scarica Slice PNG",
            data=single_bytes if modality_option not in ["Dual View (MRI + PET)", "Alpha Fusion Overlay"] else (fused_bytes if modality_option == "Alpha Fusion Overlay" else mri_bytes),
            file_name=f"{scan_record['scan_id']}_{current_plane}_slice{slice_idx}.png",
            mime="image/png"
        )


    # -------------------------------------------------------------------------
    # RIGHT COLUMN: PATIENT PROFILE & MODEL PREDICTIONS
    # -------------------------------------------------------------------------
    with col_info:
        st.markdown("#### 📋 Dati Clinici del Paziente")

        # Patient Metrics Cards in 4 columns
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Paziente ID", scan_record["ptid"])
        m2.metric("Scan ID", scan_record["scan_id"])
        m3.metric("Età", f"{scan_record['age']:.1f} a")
        m4.metric("Sesso", "Maschio" if scan_record["gender"] in ["M", "Male"] else "Femmina")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Istruzione", f"{int(scan_record['education'])} anni")
        
        # MMSE with clinical status
        mmse_val = scan_record["mmse"]
        mmse_status = "Normale" if mmse_val >= 27 else ("Lieve" if mmse_val >= 21 else "Grave")
        m6.metric("MMSE (0-30)", f"{mmse_val:.0f}", delta=mmse_status, delta_color="normal" if mmse_val >= 27 else "inverse")

        # APOE4 Alleles
        apoe_val = int(scan_record["apoe4"])
        apoe_desc = "Basso" if apoe_val == 0 else ("Medio (1 allele)" if apoe_val == 1 else "Alto (2 alleli)")
        m7.metric("APOE4 Rischio", f"{apoe_val} allel{'e' if apoe_val==1 else 'i'}", delta=apoe_desc, delta_color="off" if apoe_val==0 else "inverse")

        m8.metric("Volumi 3D", "sMRI + PET" if scan_record["has_both"] else "Solo sMRI")

        st.divider()

        # ---------------------------------------------------------------------
        # ENSEMBLE CONSENSUS METER
        # ---------------------------------------------------------------------
        st.markdown("#### 🤖 Consenso d'Insieme (11 Modelli)")
        cons_class = consensus["consensus_class"]
        agreement_pct = consensus["agreement_pct"]
        votes = consensus["votes"]

        c_col1, c_col2 = st.columns([7, 5])
        with c_col1:
            st.write(f"Verdetto d'Insieme: **{cons_class}** &bull; **{votes.get(cons_class, 0)}/11 Modelli d'accordo** ({agreement_pct}%)")
            st.progress(agreement_pct / 100.0)
        with c_col2:
            st.caption(f"Voti per classe:")
            st.write(f"🟢 CN: **{votes.get('CN', 0)}** | 🟡 MCI: **{votes.get('MCI', 0)}** | 🔴 AD: **{votes.get('AD', 0)}**")

        st.divider()

        # ---------------------------------------------------------------------
        # PREDICTIONS FOR ALL 11 TESTED MODELS
        # ---------------------------------------------------------------------
        st.markdown("#### ⚡ Predizioni Dettagliate per Modello")

        # Filter model category
        cat_filter = st.radio(
            "Filtra per Famiglia Architetturale:",
            options=["Tutti (11 Modelli)", "GNN (Graph Neural Networks)", "3D CNN & Vision Transformers"],
            horizontal=True
        )

        for model_id, pred in all_preds.items():
            model_type = pred["model_type"]
            
            # Apply category filter
            if "GNN" in cat_filter and ("Graph" not in model_type and "GCN" not in model_type and "GNN" not in model_type):
                continue
            if "CNN" in cat_filter and ("CNN" not in model_type and "ResNet" not in model_type and "Dense" not in model_type and "Transformer" not in model_type and "ViT" not in model_type):
                continue

            p_class = pred["predicted_class"]
            is_match = pred["is_correct"]
            conf = pred["confidence"]
            probs = pred["probabilities"]

            p_color = "#10b981" if p_class == "CN" else ("#f59e0b" if p_class == "MCI" else "#ef4444")
            match_icon = "✅ Accordo" if is_match else "⚠️ Disaccordo"

            with st.container():
                st.markdown(f"""
                <div class="pred-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong style="color: #f3f4f6; font-size: 0.95rem;">{pred['model_name']}</strong>
                            <span style="color: #9ca3af; font-size: 0.75rem; margin-left: 6px;">({model_type})</span>
                        </div>
                        <div>
                            <span style="font-size: 0.75rem; font-weight: 600; color: {'#34d399' if is_match else '#f87171'}; margin-right: 8px;">{match_icon}</span>
                            <span style="background-color: {p_color}22; color: {p_color}; border: 1px solid {p_color}66; border-radius: 4px; padding: 2px 8px; font-weight: 700; font-size: 0.85rem;">
                                {p_class} ({conf:.1f}%)
                            </span>
                        </div>
                    </div>
                    <div style="margin-top: 6px; font-size: 0.75rem; color: #9ca3af; font-family: monospace; display: flex; justify-content: space-between;">
                        <span>Probabilità &bull; CN: <b style="color: #34d399;">{probs['CN']*100:.1f}%</b> | MCI: <b style="color: #fbbf24;">{probs['MCI']*100:.1f}%</b> | AD: <b style="color: #f87171;">{probs['AD']*100:.1f}%</b></span>
                        <span style="color: #cbd5e1;">5-Fold CV Acc: <b>{pred['benchmark_acc']}</b></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# =============================================================================
# TAB 2: BENCHMARK SUMMARY (5-Fold Cross Validation Across 11 Models)
# =============================================================================
with tab_benchmarks:
    st.markdown("### 🏆 5-Fold Cross Validation Benchmark Summary")
    st.markdown(
        "Risultati sperimentali medi &plusmn; deviazione standard sui 5 fold per tutti gli **11 modelli testati** sui vari task di classificazione."
    )

    models_overview = pred_engine.get_models_overview()
    table_rows = []
    chart_data = []

    for m in models_overview:
        cn_ad = m["metrics"].get("cn_ad", {})
        mci_ad = m["metrics"].get("mci_ad", {})
        cn_mci = m["metrics"].get("cn_mci", {})
        multi = m["metrics"].get("cn_mci_ad", {})

        # Extract numeric accuracy for chart
        try:
            acc_val = float(cn_ad.get("acc", "0").split("±")[0].replace("%", "").strip())
        except Exception:
            acc_val = 0.0

        try:
            auc_val = float(cn_ad.get("auc", "0").split("±")[0].strip())
        except Exception:
            auc_val = 0.0

        table_rows.append({
            "Modello": m["name"],
            "Paradigma": m["type"],
            "Modalità": m["modality"],
            "CN vs AD Acc": cn_ad.get("acc", "N/A"),
            "CN vs AD AUC": cn_ad.get("auc", "N/A"),
            "MCI vs AD Acc": mci_ad.get("acc", "N/A"),
            "CN vs MCI vs AD Acc": multi.get("acc", "N/A"),
            "Descrizione": m["description"]
        })

        chart_data.append({
            "Modello": m["name"],
            "Accuracy": acc_val,
            "AUC": auc_val,
            "Tipo": m["type"]
        })

    df_table = pd.DataFrame(table_rows)
    st.dataframe(df_table, use_container_width=True, hide_index=True)

    st.markdown("#### 📊 Confronto Grafico: CN vs AD (Accuratezza e AUC)")
    df_chart = pd.DataFrame(chart_data)

    col_ch1, col_ch2 = st.columns(2)
    with col_ch1:
        chart_acc = (
            alt.Chart(df_chart)
            .mark_bar(cornerRadius=4)
            .encode(
                x=alt.X("Accuracy:Q", title="Accuracy (%)", scale=alt.Scale(domain=[40, 75])),
                y=alt.Y("Modello:N", sort="-x", title="Modello"),
                color=alt.Color("Tipo:N", legend=alt.Legend(title="Paradigma")),
                tooltip=["Modello", "Tipo", "Accuracy", "AUC"]
            )
            .properties(height=380, title="Accuratezza CN vs AD")
        )
        st.altair_chart(chart_acc, use_container_width=True)

    with col_ch2:
        chart_auc = (
            alt.Chart(df_chart)
            .mark_bar(cornerRadius=4)
            .encode(
                x=alt.X("AUC:Q", title="Area Under ROC (AUC)", scale=alt.Scale(domain=[0.45, 0.8])),
                y=alt.Y("Modello:N", sort="-x", title="Modello"),
                color=alt.Color("Tipo:N", legend=alt.Legend(title="Paradigma")),
                tooltip=["Modello", "Tipo", "Accuracy", "AUC"]
            )
            .properties(height=380, title="AUC-ROC CN vs AD")
        )
        st.altair_chart(chart_auc, use_container_width=True)


# =============================================================================
# TAB 3: DATASET STATISTICS & DISTRIBUTIONS
# =============================================================================
with tab_dataset:
    st.markdown("### 📈 Statistiche e Distribuzione del Dataset ADNI")
    
    col_d1, col_d2, col_d3 = st.columns(3)
    col_d1.metric("Totale Scansioni", f"{len(data_mgr.df):,}")
    col_d2.metric("Pazienti Unici", f"{data_mgr.df['PTID'].nunique():,}")
    col_d3.metric("Scansioni Paired (sMRI + PET)", f"{int(data_mgr.df['has_both'].sum()):,}")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("##### Distribuzione delle Diagnosi Cliniche:")
        dx_counts = data_mgr.df["original_class"].value_counts().reset_index()
        dx_counts.columns = ["Diagnosi", "Conteggio"]
        pie_chart = (
            alt.Chart(dx_counts)
            .mark_arc(innerRadius=50)
            .encode(
                theta=alt.Theta("Conteggio:Q"),
                color=alt.Color("Diagnosi:N", scale=alt.Scale(domain=["CN", "MCI", "AD"], range=["#10b981", "#f59e0b", "#ef4444"])),
                tooltip=["Diagnosi", "Conteggio"]
            )
            .properties(height=300)
        )
        st.altair_chart(pie_chart, use_container_width=True)

    with col_g2:
        st.markdown("##### Distribuzione dell'Età per Diagnosi:")
        age_chart = (
            alt.Chart(data_mgr.df)
            .mark_boxplot(extent="min-max")
            .encode(
                x=alt.X("original_class:N", title="Diagnosi"),
                y=alt.Y("AGE:Q", title="Età (anni)", scale=alt.Scale(zero=False)),
                color=alt.Color("original_class:N", scale=alt.Scale(domain=["CN", "MCI", "AD"], range=["#10b981", "#f59e0b", "#ef4444"]))
            )
            .properties(height=300)
        )
        st.altair_chart(age_chart, use_container_width=True)

    st.markdown("##### Distribuzione Punteggio MMSE per Diagnosi:")
    mmse_chart = (
        alt.Chart(data_mgr.df)
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X("original_class:N", title="Diagnosi"),
            y=alt.Y("MMSE:Q", title="Punteggio Cognitivo MMSE (0 - 30)", scale=alt.Scale(domain=[10, 30])),
            color=alt.Color("original_class:N", scale=alt.Scale(domain=["CN", "MCI", "AD"], range=["#10b981", "#f59e0b", "#ef4444"]))
        )
        .properties(height=260)
    )
    st.altair_chart(mmse_chart, use_container_width=True)

    st.divider()

    # -------------------------------------------------------------------------
    # ADNIMERGE INTEGRATION & FAILURE ANALYSIS SECTION
    # -------------------------------------------------------------------------
    st.markdown("### 🔬 Indagine Diagnostica ADNIMERGE: Perché i Modelli Hanno Risultati Bassi?")
    st.caption("Integrazione con `data/raw/ADNIMERGE_31Dec2025.csv` (16.421 record, 116 biomarcatori) e analisi dei fattori limitanti.")

    # KPI summary cards
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Pazienti Convertitori", "284 (26.0%)", delta="Rumore Etichetta", delta_color="inverse")
    a2.metric("PET Mancante / Zeri", "2,439 (48.8%)", delta="Zero-Padding", delta_color="inverse")
    a3.metric("Follow-up Medio", "4.9 anni", delta="Max: 16.9 anni", delta_color="off")
    a4.metric("Scala Multi-Atlas sMRI", "4.3 × 10⁷", delta="vs PET: 0.0 - 2.5", delta_color="inverse")

    st.markdown("""
    #### 🚨 Le 6 Cause Primarie Identificate:
    1. **Rumore di Etichetta (Label Contradiction) sui Dati Longitudinali**:
       - Il **26.0% dei pazienti converte** nel tempo (**224 MCI &rarr; AD**, **41 CN &rarr; MCI**, **19 CN &rarr; AD**).
       - Nel dataset di training, a tutte le scansioni di un paziente è stata assegnata la diagnosi statica baseline (`original_class`). Un paziente MCI convertito ad Alzheimer conclamato dopo 3 anni viene addestrato/valutato con etichetta obsoleta *"MCI"*, penalizzando il modello quando rileva la neurodegenerazione reale.
    2. **Mancanza di PET nel 48.8% dei Campioni e Zero-Padding**:
       - Quasi metà del dataset non dispone di PET. Il codice inietta canali di **zeri costanti** (`mri * 0.0`). Questo squilibra le statistiche di *Batch Normalization* delle 3D-CNN e costringe le reti ad apprendere se la PET è presente invece del metabolismo cerebrale.
    3. **Catastrofe di Scala nelle Feature Multi-Atlas ($10^7$ vs $1.0$)**:
       - Per `KGLassoNet`, le feature sMRI non sono standardizzate (somma voxel = fino a $43.594.032$), mentre la PET è SUVR ($0.0 - 2.5$). La matrice di pesi viene dominata dalla risonanza di un fattore $10^6$, annullando il segnale metabolico.
    4. **Indistinguibilità Biologica tra CN ed Early MCI (EMCI)**:
       - Il volume medio dell'ippocampo in EMCI ($7.327\\text{ mm}^3$) è identico ai sani CN ($7.352\\text{ mm}^3$), e il metabolismo FDG è $1.30$ per entrambi. Senza traccianti molecolari dell'amiloide, la sMRI strutturale non contiene segnali differenziali in fase prodromica precoce.
    5. **Eterogeneità Scanner e Protocolli (ADNI1 1.5T vs ADNI2/3 3.0T)**:
       - Le scansioni uniscono protocolli a 1.5T e 3.0T di 50+ centri clinici senza armonizzazione (es. ComBat).
    6. **Sbilanciamento di Classe**:
       - La classe AD rappresenta solo il 14.1% del dataset totale (707 scansioni su 5.003).

    #### 🛠️ Roadmap Raccomandata per Incrementare le Performance:
    - ✅ **Etichettatura Dinamica Visita-per-Visita**: associare a ogni singola scansione la diagnosi clinica `DX` registrata alla data esatta di quella visita in `ADNIMERGE`.
    - ✅ **Standardizzazione Z-Score su Multi-Atlas**: normalizzare $z = (x - \\mu) / \\sigma$ separatamente sMRI e PET prima dell'ingresso in LassoNet.
    - ✅ **Valutazione Specializzata sul Sottoinsieme Paired**: addestrare/testare i modelli multimodali sulle 2.564 scansioni con entrambi i volumi reali.
    - ✅ **Scomposizione MCI**: testare CN vs Late-MCI (LMCI) o predire la conversione a 24 mesi (*MCI-Stable vs MCI-Converter*).
    """)
