import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import ttest_1samp, ttest_ind, chi2_contingency
PALETTE = [
    "#93c5fd",  # light blue
    "#60a5fa",
    "#3b82f6",  # positive bars
    "#ef4444",  # negative bars
]
# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Sleep Health Analysis",
    page_icon="🛌",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>

/* =====================================================
   APP
===================================================== */

.stApp {
    background-color: #fcfcfd;
    color: #1e293b;
}

/* =====================================================
   SIDEBAR
===================================================== */

section[data-testid="stSidebar"] {
    background-color: #f8fafc;
    border-right: 1px solid #e5e7eb;
    padding-top: 10px;
}

/* Sidebar radio spacing */
.stRadio > div {
    gap: 6px;
}

/* Sidebar items */
.stRadio label {
    padding: 10px 12px;
    border-radius: 10px;
    transition: 0.2s ease;
    color: #334155 !important;
}

/* Hover */
.stRadio label:hover {
    background: #eef2ff;
}

/* Selected text */
.stRadio input[type="radio"]:checked + div {
    color: #4f6ef7 !important;
    font-weight: 700 !important;
}

/* Streamlit/BaseWeb selected states */
[data-baseweb="radio"] [aria-checked="true"] > div:first-child,
[data-baseweb="checkbox"] [aria-checked="true"] > div:first-child {
    background-color: #4f6ef7 !important;
    border-color: #4f6ef7 !important;
}

/* =====================================================
   SECTION TITLES
===================================================== */

.sec {
    font-size: 18px;
    font-weight: 700;
    color: #1e293b;
    border-left: 4px solid #4f6ef7;
    padding-left: 10px;
    margin: 22px 0 14px 0;
}

/* =====================================================
   INFO BOXES
===================================================== */

.box {
    background: white;
    border: 1px solid #e5e7eb;
    border-left: 4px solid #4f6ef7;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 12px;
    color: #334155;
    line-height: 1.6;
}

.badge-blue,
.badge-green,
.badge-red {
    background: #eef2ff;
    border: 1px solid #bfdbfe;
    border-left: 4px solid #4f6ef7;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 12px 0;
    color: #1d4ed8;
    font-weight: 700;
}

/* =====================================================
   METRICS
===================================================== */

[data-testid="metric-container"] {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 14px;
}

/* =====================================================
   TABS
===================================================== */

.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    margin-bottom: 10px;
}

.stTabs [data-baseweb="tab"] {
    background: #f1f5f9;
    border-radius: 10px;
    padding: 10px 18px;
    color: #475569;
    font-weight: 600;
}

/* Selected tab */
.stTabs [aria-selected="true"] {
    background: #4f6ef7 !important;
    color: white !important;
}

/* Remove red underline */
.stTabs [data-baseweb="tab-highlight"] {
    background-color: transparent !important;
}

/* ===== Predict Button ===== */

.stButton > button {
    width: 100%;

    background: linear-gradient(
        135deg,
        #4f6ef7,
        #3b82f6
    );

    color: white;
    border: none;

    border-radius: 14px;

    padding: 10px 18px;

    font-size: 16px;
    font-weight: 600;

    transition: 0.2s ease;

    box-shadow:
        0 6px 18px rgba(79,110,247,0.18);

    margin-top: 8px;
}

/* Hover */
.stButton > button:hover {
    transform: translateY(-1px);

    box-shadow:
        0 8px 22px rgba(79,110,247,0.24);
}
/* =====================================================
   MULTISELECT TAGS
===================================================== */

.stMultiSelect [data-baseweb="tag"] {
    border-radius: 8px;
    background-color: #eef2ff !important;
    color: #1d4ed8 !important;
}

.stSlider [data-baseweb="slider"] [role="slider"] {
    background-color: #4f6ef7 !important;
    border-color: #4f6ef7 !important;
}

.stSlider [data-baseweb="slider"] div {
    border-color: #4f6ef7 !important;
}

/* =====================================================
   REMOVE STREAMLIT RED FOCUS
===================================================== */

*:focus {
    box-shadow: none !important;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Matplotlib — clean light style
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#f8f9fb",
    "axes.edgecolor":    "#d0d9f5",
    "axes.labelcolor":   "#1a1a2e",
    "xtick.color":       "#4a4a6a",
    "ytick.color":       "#4a4a6a",
    "text.color":        "#1a1a2e",
    "grid.color":        "#e2e8f0",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})
PAL = ["#4f6ef7", "#7c9cff", "#34d399", "#2563eb", "#fbbf24", "#60a5fa"]


# ─────────────────────────────────────────────
#  Load data
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/Sleep_health_and_lifestyle_dataset.csv")
    df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")
    return df

df = load_data()

# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛌 Sleep Health")
    st.markdown("**Data Science Project**")
    st.markdown("---")

    page = st.radio("Navigate", [
        "Overview",
        "Data Collection",
        "Preprocessing",
        "Inferential Statistics",
        "EDA",
        "Modeling",
        "Predictor",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### Filters")
    gender_filter = st.multiselect(
        "Gender", df["Gender"].unique(), default=list(df["Gender"].unique()))
    bmi_filter = st.multiselect(
        "BMI Category", df["BMI Category"].unique(),
        default=list(df["BMI Category"].unique()))
    df_f = df[df["Gender"].isin(gender_filter) & df["BMI Category"].isin(bmi_filter)]

    st.markdown("---")
    st.caption(f"**Records shown:** {len(df_f)} / {len(df)}")
    st.caption("**Dataset:** Sleep Health & Lifestyle")
    st.caption("**Source:** Kaggle – Laksika Tharmalingam")


# ══════════════════════════════════════════════
#  PAGE 1 – OVERVIEW
# ══════════════════════════════════════════════
if page == "Overview":
    st.title("🛌 Sleep Health & Lifestyle Analysis")
    st.markdown("End-to-end Data Science Pipeline — from collection to prediction.")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",      len(df))
    c2.metric("Features",           13)
    c3.metric("Avg Sleep Duration", f"{df['Sleep Duration'].mean():.2f} h")
    c4.metric("Avg Sleep Quality",  f"{df['Quality of Sleep'].mean():.2f} / 10")

    st.divider()
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown('<div class="sec">Project Pipeline</div>', unsafe_allow_html=True)
        steps = [
            ("📦 Data Collection",        "Loaded 374-row CSV from Kaggle with 13 lifestyle & health features."),
            ("🧹 Preprocessing",          "Handled NaNs, encoded categoricals, scaled features, removed irrelevant columns."),
            ("📐 Inferential Statistics", "Probability, t-tests, chi-squared, confidence intervals, CLT."),
            ("📊 EDA",                    "Distributions, correlations, Q→Q scatter & C→Q boxplots."),
            ("📈 Modeling",               "OLS, Ridge & Lasso regression — R² = 0.9155 on test set."),
        ]
        for title, desc in steps:
            st.markdown(f'<div class="box"><b>{title}</b><br><span style="color:#555;font-size:13px;">{desc}</span></div>',
                        unsafe_allow_html=True)

    with right:
        st.markdown('<div class="sec">Key Findings</div>', unsafe_allow_html=True)
        findings = [
            ("Stress Level",   "strongest negative predictor (r = −0.90)"),
            ("Sleep Duration", "strongest positive predictor (r = +0.88)"),
            ("Heart Rate",     "moderate negative effect (r = −0.66)"),
            ("Model R²",       "91.5% variance explained on test set"),
            ("Sleep Disorder", "41.4% of individuals have a diagnosis"),
            ("No Overfitting", "Test R² slightly higher than Train R²"),
        ]
        for key, val in findings:
            st.markdown(f'<div class="box" style="padding:10px 16px;"><b>{key}</b> — <span style="color:#555;">{val}</span></div>',
                        unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="sec">Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)


# ══════════════════════════════════════════════
#  PAGE 2 – DATA COLLECTION
# ══════════════════════════════════════════════
elif page == "Data Collection":

    st.title("📦 Data Collection")
    st.markdown("Understanding the raw dataset before any processing.")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing values", df.isnull().sum().sum())
    c4.metric("Duplicate rows", df.duplicated().sum())

    st.divider()

    # ===== Column Types =====
    st.markdown('<div class="sec">Column Types</div>', unsafe_allow_html=True)

    type_data = {
        "Column": [
            "Gender", "Occupation", "Sleep Disorder",
            "BMI Category",
            "Quality of Sleep", "Stress Level",
            "Age", "Sleep Duration", "Heart Rate",
            "Physical Activity Level", "Daily Steps",
            "Blood Pressure", "Person ID"
        ],

        "Type": (
            ["Qualitative Nominal"] * 3 +
            ["Qualitative Ordinal"] +
            ["Quantitative Discrete"] * 2 +
            ["Quantitative Continuous"] * 5 +
            ["Dropped"] * 2
        ),
    }

    type_df = pd.DataFrame(type_data)

    color_map = {
        "Qualitative Nominal": "#a78bfa",
        "Qualitative Ordinal": "#fbbf24",
        "Quantitative Discrete": "#34d399",
        "Quantitative Continuous": "#4f6ef7",
        "Dropped": "#2563eb",
    }

    def highlight(row):
        c = color_map.get(row["Type"], "#ffffff")
        return [f"background-color:{c}22; color:{c}" for _ in row]

    st.dataframe(
        type_df.style.apply(highlight, axis=1),
        use_container_width=True,
        height=430
    )

    # ===== Descriptive Statistics =====
    st.markdown('<div class="sec">Descriptive Statistics</div>', unsafe_allow_html=True)

    st.dataframe(
        df.describe().round(2),
        use_container_width=True
    )

    # ===== Key Observations =====
    st.markdown('<div class="sec">Key Observations</div>', unsafe_allow_html=True)

    for k, v in [
        ("Average age", "42 years (range 27–59)"),
        ("Average sleep", "7.13 hours close to recommended"),
        ("Avg stress level", "5.4 / 8 moderate"),
        ("Daily steps range", "3 000 – 10 000"),
        ("Heart rate range", "65 – 86 bpm normal"),
        ("Blood Pressure", "Stored as string '126/83' dropped"),
    ]:

        st.markdown(
            f'''
            <div class="box" style="padding:9px 14px;">
                <b>{k}</b><br>
                <span style="color:#555;font-size:13px;">{v}</span>
            </div>
            ''',
            unsafe_allow_html=True
        )

    # ===== Raw Data =====
    st.divider()

    st.markdown(
        '<div class="sec">Raw Data (filtered)</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        df_f,
        use_container_width=True,
        height=280
    )

# ══════════════════════════════════════════════
#  PAGE 3 – PREPROCESSING
# ══════════════════════════════════════════════
elif page == "Preprocessing":

    st.title("🧹 Data Preprocessing")
    st.markdown("Cleaning, encoding, scaling and preparing the dataset before modeling.")
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Cleaning",
        "Outliers",
        "Encoding",
        "Scaling"
    ])

    with tab1:

        st.markdown(
            '<div class="sec">Missing Values & Duplicates</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)

        c1.metric("Missing Before", "219")
        c2.metric("Missing After", "0")
        c3.metric("Duplicate Rows", "0")

        st.markdown("""
<div class="box">
The dataset did not contain true missing or corrupted data.<br><br>
In the <b>Sleep Disorder</b> column, 219 rows were stored as
<b>NaN</b>. In this context, NaN meant that the individual had
<b>no diagnosed sleep disorder</b>, not that information was missing.<br><br>
To avoid confusion during analysis and visualization,
these NaN values were replaced with <b>'None'</b>.
</div>
""", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(7, 3.5))

        ax.bar(
            ["Before Cleaning", "After Cleaning"],
            [219, 0],
            color=[PAL[0], PAL[2]],
            width=0.5
        )

        ax.set_ylabel("Missing Values")
        ax.set_title("Missing Values Before vs After Cleaning")

        for i, v in enumerate([219, 0]):
            ax.text(i, v + 5, str(v), ha="center", fontsize=11)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("""
        <div class="box">
            <b>No duplicate rows</b> were found.
            The dataset remained at <b>374 rows</b>
            after preprocessing.
        </div>
        """, unsafe_allow_html=True)

    with tab2:

        st.markdown(
            '<div class="sec">Outlier Detection — IQR Method</div>',
            unsafe_allow_html=True
        )

        st.markdown("""
        <div class="box">
            The IQR method detects outliers using:
            <b>Q1 − 1.5 × IQR</b> and
            <b>Q3 + 1.5 × IQR</b>.
            Values outside this interval are considered outliers.
        </div>
        """, unsafe_allow_html=True)

        num_cols = [
            "Sleep Duration",
            "Heart Rate",
            "Daily Steps",
            "Stress Level"
        ]

        outlier_counts = {}

        for col in num_cols:

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)

            IQR = Q3 - Q1

            outlier_counts[col] = int(
                (
                    (df[col] < Q1 - 1.5 * IQR) |
                    (df[col] > Q3 + 1.5 * IQR)
                ).sum()
            )

        cols_m = st.columns(4)

        for i, (col, cnt) in enumerate(outlier_counts.items()):
            cols_m[i].metric(col, f"{cnt}")

        fig, axes = plt.subplots(1, 4, figsize=(14, 4))

        for ax, col in zip(axes, num_cols):

            ax.boxplot(
                df[col],
                patch_artist=True,

                boxprops=dict(
                    facecolor=PAL[0] + "33",
                    color=PAL[0]
                ),

                medianprops=dict(
                    color=PAL[4],
                    linewidth=2
                ),

                whiskerprops=dict(color="#888"),
                capprops=dict(color="#888"),

                flierprops=dict(
                    marker='o',
                    color=PAL[3],
                    markersize=5
                )
            )

            ax.set_title(col, fontsize=10)
            ax.set_xticks([])

        plt.suptitle(
            "Outlier Visualization",
            fontsize=13,
            fontweight="bold"
        )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("""
        <div class="box">
            Only <b>Heart Rate</b> contained noticeable outliers
            (values above ~83 bpm).
            Sleep Duration, Daily Steps and Stress Level
            showed very few or no extreme observations.
        </div>
        """, unsafe_allow_html=True)

    with tab3:

        st.markdown(
            '<div class="sec">Categorical Encoding</div>',
            unsafe_allow_html=True
        )

        st.markdown("""
        <div class="box">
            Machine learning models require numerical inputs.
            Categorical variables were therefore converted into numbers
            using different encoding techniques.
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("""
            <div class="box">
                <b>Gender → Label Encoding</b><br><br>
                Female = 0<br>
                Male = 1
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown("""
            <div class="box">
                <b>BMI Category → One-Hot Encoding</b><br><br>
                4 binary columns created
                to avoid false numeric ordering.
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown("""
            <div class="box">
                <b>Sleep Disorder → Ordinal Encoding</b><br><br>
                None = 0<br>
                Insomnia = 1<br>
                Sleep Apnea = 2
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Encoded Sample")

        enc_df = df[
            ["Gender", "BMI Category", "Sleep Disorder"]
        ].head(8).copy()

        enc_df["Gender_enc"] = (
            enc_df["Gender"] == "Male"
        ).astype(int)

        enc_df["Disorder_enc"] = enc_df["Sleep Disorder"].map({
            "None": 0,
            "Insomnia": 1,
            "Sleep Apnea": 2
        })

        st.dataframe(
            enc_df,
            use_container_width=True
        )

    with tab4:

        st.markdown(
            '<div class="sec">Feature Scaling</div>',
            unsafe_allow_html=True
        )

        st.markdown("""
        <div class="box">
            Features have different numerical ranges.
            Scaling prevents large-value variables from dominating the model.
        </div>
        """, unsafe_allow_html=True)

        scale_cols = [
            "Age",
            "Sleep Duration",
            "Physical Activity Level",
            "Stress Level",
            "Heart Rate",
            "Daily Steps"
        ]

        mm_df = pd.DataFrame(
            MinMaxScaler().fit_transform(df[scale_cols]),
            columns=scale_cols
        )

        zs_df = pd.DataFrame(
            StandardScaler().fit_transform(df[scale_cols]),
            columns=scale_cols
        )

        c1, c2 = st.columns(2)

        with c1:

            st.markdown("### Min-Max Scaling")

            st.markdown("""
            Rescales values between <b>0 and 1</b>.
            """, unsafe_allow_html=True)

            st.dataframe(
                mm_df.describe().round(3),
                use_container_width=True
            )

        with c2:

            st.markdown("### Z-Score Standardization")

            st.markdown("""
            Centers data around:
            <b>mean = 0</b> and <b>std = 1</b>.
            """, unsafe_allow_html=True)

            st.dataframe(
                zs_df.describe().round(3),
                use_container_width=True
            )

        st.markdown("""
        <div class="box">
            <b>Final dataset shape:</b> 374 rows × 15 columns.<br><br>
            Removed columns:
            <b>Person ID</b> and <b>Blood Pressure</b>.
            Blood Pressure was stored as text ("126/83")
            and required additional parsing.
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  PAGE 4 – INFERENTIAL STATISTICS
# ══════════════════════════════════════════════
elif page == "Inferential Statistics":
    st.title("📐 Inferential Statistics")
    st.markdown("Probability, hypothesis testing, confidence intervals and the Central Limit Theorem.")
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Probability", "Distributions & CLT", "Hypothesis Tests", "Chi-Squared Test"
    ])

    with tab1:
        st.markdown('<div class="sec">Basic Probability</div>', unsafe_allow_html=True)
        total      = len(df)
        disorder   = (df["Sleep Disorder"] != "None").sum()
        apnea      = (df["Sleep Disorder"] == "Sleep Apnea").sum()
        insomnia   = (df["Sleep Disorder"] == "Insomnia").sum()
        p_disorder = disorder / total

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total individuals", total)
        c2.metric("With a disorder",   disorder)
        c3.metric("Sleep Apnea",       apnea)
        c4.metric("Insomnia",          insomnia)

        st.markdown(f"""<div class="box">
            <b>P(having a sleep disorder)</b> = {disorder} / {total} = <b>{p_disorder:.1%}</b><br><br>
            Out of 374 individuals, 155 have a diagnosed sleep disorder
            (78 Sleep Apnea + 77 Insomnia). The remaining 219 have none.
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec">Conditional Probability</div>', unsafe_allow_html=True)
        females_all     = df[df["Gender"] == "Female"]
        female_disorder = (females_all["Sleep Disorder"] != "None").sum()
        p_f_and_d       = female_disorder / total

        st.markdown(f"""<div class="box">
            <b>P(Female AND has a disorder)</b><br>
            = Females with disorder ({female_disorder}) / Total ({total})
            = <b>{p_f_and_d:.1%}</b><br><br>
            We used the <b>AND rule</b> — both conditions must be true simultaneously.
        </div>""", unsafe_allow_html=True)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].pie(
            [disorder, total-disorder],
            labels=["Has disorder", "No disorder"],
            colors=["#4f6ef7", "#93c5fd"],
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops=dict(width=0.6, edgecolor="white")
        )
        axes[0].set_title("Sleep Disorder Prevalence")

        dc = df[df["Sleep Disorder"] != "None"]["Sleep Disorder"].value_counts()
        axes[1].pie(dc, labels=dc.index,
                    colors=[PAL[0], PAL[1]], autopct="%1.1f%%",
                    startangle=90, wedgeprops=dict(width=0.6))
        axes[1].set_title("Disorder Type Breakdown")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with tab2:
        st.markdown('<div class="sec">Sleep Duration Distribution & Z-scores</div>', unsafe_allow_html=True)
        mu_s  = df["Sleep Duration"].mean()
        std_s = df["Sleep Duration"].std()

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        axes[0].hist(df["Sleep Duration"], bins=25, color=PAL[0],
                     edgecolor="white", alpha=0.85)
        x_n = np.linspace(df["Sleep Duration"].min(), df["Sleep Duration"].max(), 200)
        axes[0].plot(x_n,
                     stats.norm.pdf(x_n, mu_s, std_s) * len(df) *
                     (df["Sleep Duration"].max()-df["Sleep Duration"].min())/25,
                     color=PAL[3], lw=2, label="Normal fit")
        axes[0].axvline(mu_s, color=PAL[2], lw=1.5, linestyle="--", label=f"Mean={mu_s:.2f}")
        axes[0].set_title("Sleep Duration (slightly left-skewed)")
        axes[0].set_xlabel("Hours"); axes[0].legend()

        z_stress = (df["Stress Level"]-df["Stress Level"].mean())/df["Stress Level"].std()
        axes[1].hist(z_stress, bins=20, color=PAL[1], edgecolor="white", alpha=0.85)
        axes[1].axvline(0,  color=PAL[2], lw=1.5, linestyle="--", label="Mean = 0")
        axes[1].axvline(2,  color=PAL[3], lw=1,   linestyle=":",  label="|z|>2 outlier")
        axes[1].axvline(-2, color=PAL[3], lw=1,   linestyle=":")
        axes[1].set_title("Z-scores — Stress Level")
        axes[1].set_xlabel("Z-score"); axes[1].legend()
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("""<div class="box">
            Sleep Duration is <b>slightly left-skewed</b> — most people sleep between 6.5 and 8 hours.
            Z-scores on Stress Level show <b>very few extreme outliers</b>, meaning stress values
            are consistent across the sample.
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec">Central Limit Theorem</div>', unsafe_allow_html=True)
        np.random.seed(42)
        sample    = df["Sleep Duration"].sample(100)
        pop_mean  = df["Sleep Duration"].mean()
        samp_mean = sample.mean()
        se        = std_s / np.sqrt(len(df))
        ci_low    = mu_s - 1.96 * se
        ci_high   = mu_s + 1.96 * se

        c1, c2, c3 = st.columns(3)
        c1.metric("Population Mean",     f"{pop_mean:.4f} h")
        c2.metric("Sample Mean (n=100)", f"{samp_mean:.4f} h")
        c3.metric("Difference",          f"{abs(pop_mean-samp_mean):.4f} h")

        st.markdown(f"""<div class="box">
            <b>95% Confidence Interval</b> for mean sleep duration:
            [{ci_low:.4f}, {ci_high:.4f}] hours.<br><br>
            The true mean ({mu_s:.2f} h) falls <b>inside</b> the interval — sample is representative.
            A sample of 100 gives a mean very close to the true mean → <b>CLT confirmed</b>.
        </div>""", unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="sec">One-Sample T-test</div>', unsafe_allow_html=True)
        st.markdown("""<div class="box">
            <b>Question:</b> Is average sleep duration equal to 8 hours?<br>
            H₀ : mean = 8 h &nbsp;|&nbsp; H₁ : mean ≠ 8 h
        </div>""", unsafe_allow_html=True)

        t1, p1 = ttest_1samp(df["Sleep Duration"], 8.0)
        c1, c2 = st.columns(2)
        c1.metric("T-statistic", f"{t1:.4f}")
        c2.metric("P-value",     f"{p1:.6f}")
        badge1 = "badge-blue"
        msg1   = "H₀ Rejected — people sleep significantly less than 8 hours." if p1 < 0.05 else "H₀ Not rejected."
        st.markdown(f'<div class="{badge1}">{msg1}</div>', unsafe_allow_html=True)

        st.markdown('<div class="sec">Two-Sample T-test</div>', unsafe_allow_html=True)
        st.markdown("""<div class="box">
            <b>Question:</b> Do men and women sleep the same amount?<br>
            H₀ : no difference &nbsp;|&nbsp; H₁ : significant difference exists
        </div>""", unsafe_allow_html=True)

        males_s   = df[df["Gender"] == "Male"]["Sleep Duration"]
        females_s = df[df["Gender"] == "Female"]["Sleep Duration"]
        t2, p2    = ttest_ind(males_s, females_s)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Female avg",   f"{females_s.mean():.2f} h")
        c2.metric("Male avg",     f"{males_s.mean():.2f} h")
        c3.metric("T-statistic",  f"{t2:.4f}")
        c4.metric("P-value",      f"{p2:.4f}")

        badge2 = "badge-blue"
        msg2   = ("H₀ Rejected — significant gender difference." if p2 < 0.05
                  else "H₀ Not rejected — difference is not statistically significant.")
        st.markdown(f'<div class="{badge2}">{msg2}</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(6, 3.5))
        gm = df.groupby("Gender")["Sleep Duration"].mean()
        ax.bar(gm.index, gm.values, color=[PAL[0], PAL[2]], alpha=0.85, width=0.5)
        ax.axhline(8, color=PAL[3], lw=1.5, linestyle="--", label="Recommended 8h")
        ax.set_ylabel("Avg Sleep Duration (h)")
        ax.set_title("Sleep Duration by Gender"); ax.legend()
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with tab4:
        st.markdown('<div class="sec">Chi-Squared Test</div>', unsafe_allow_html=True)
        st.markdown("""<div class="box">
            <b>Question:</b> Is BMI Category related to Sleep Disorder?<br>
            H₀ : BMI and Sleep Disorder are independent<br>
            H₁ : there is a relationship between the two
        </div>""", unsafe_allow_html=True)

        contingency = pd.crosstab(df["BMI Category"], df["Sleep Disorder"])
        chi2, p_chi, dof, _ = chi2_contingency(contingency)

        c1, c2, c3 = st.columns(3)
        c1.metric("Chi² statistic",     f"{chi2:.4f}")
        c2.metric("P-value",            f"{p_chi:.6f}")
        c3.metric("Degrees of freedom", str(dof))

        badge3 = "badge-blue"
        msg3   = "H₀ Rejected — BMI and Sleep Disorder are NOT independent." if p_chi < 0.05 else "H₀ Not rejected."
        st.markdown(f'<div class="{badge3}">{msg3}</div>', unsafe_allow_html=True)

        st.markdown("**Contingency Table:**")
        st.dataframe(contingency, use_container_width=True)

        bmi_sleep = df.groupby("BMI Category")["Sleep Duration"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.barh(bmi_sleep.index, bmi_sleep.values, color=PAL[:len(bmi_sleep)])
        ax.set_xlabel("Avg Sleep Duration (h)")
        ax.set_title("Average Sleep Duration by BMI Category")
        for i, v in enumerate(bmi_sleep.values):
            ax.text(v+0.02, i, f"{v:.2f}h", va="center", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("""<div class="box">
            P-value &lt; 0.05 → BMI and Sleep Disorder are <b>NOT independent</b>.
            Heavier individuals tend to have more sleep disorders and shorter sleep duration.<br>
            Normal BMI → 7.39h average vs Overweight → 6.77h average.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  PAGE 5 – EDA
# ══════════════════════════════════════════════
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("Understanding patterns, distributions, and relationships in the data.")
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Distributions", "Correlations", "Q → Q Analysis", "C → Q Analysis"
    ])

    with tab1:
        st.markdown('<div class="sec">Feature Distributions</div>', unsafe_allow_html=True)
        num_features = ["Age", "Sleep Duration", "Quality of Sleep",
                        "Physical Activity Level", "Stress Level",
                        "Heart Rate", "Daily Steps"]
        selected = st.selectbox("Select a feature to inspect", num_features)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(df_f[selected], bins=25, color=PAL[0], edgecolor="white", alpha=0.85)
        mu_f, sd_f = df_f[selected].mean(), df_f[selected].std()
        x_fit = np.linspace(df_f[selected].min(), df_f[selected].max(), 200)
        axes[0].plot(x_fit,
                     stats.norm.pdf(x_fit, mu_f, sd_f) * len(df_f) *
                     (df_f[selected].max()-df_f[selected].min())/25,
                     color=PAL[3], lw=2, label="Normal fit")
        axes[0].set_title(f"Distribution of {selected}")
        axes[0].set_xlabel(selected); axes[0].legend()

        axes[1].boxplot(df_f[selected], patch_artist=True,
                        boxprops=dict(facecolor=PAL[0]+"44", color=PAL[0]),
                        medianprops=dict(color=PAL[2], linewidth=2),
                        whiskerprops=dict(color="#888"),
                        capprops=dict(color="#888"),
                        flierprops=dict(marker='o', color=PAL[3], markersize=5))
        axes[1].set_title(f"Box Plot — {selected}"); axes[1].set_xticks([])
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean",     f"{df_f[selected].mean():.2f}")
        c2.metric("Median",   f"{df_f[selected].median():.2f}")
        c3.metric("Std",      f"{df_f[selected].std():.2f}")
        c4.metric("Skewness", f"{df_f[selected].skew():.3f}")

        st.markdown('<div class="sec">All Numerical Features</div>', unsafe_allow_html=True)
        fig2, axes2 = plt.subplots(2, 4, figsize=(16, 7))
        for ax, feat in zip(axes2.flat, num_features):
            ax.hist(df_f[feat], bins=20, color=PAL[0], edgecolor="white", alpha=0.85)
            ax.set_title(feat, fontsize=9)
        axes2.flat[-1].set_visible(False)
        plt.suptitle("All Numerical Features Distribution", fontsize=13, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig2); plt.close()

    with tab2:
        st.markdown('<div class="sec">Correlation Heatmap</div>', unsafe_allow_html=True)
        num_cols2 = ["Age", "Sleep Duration", "Physical Activity Level",
                     "Stress Level", "Heart Rate", "Daily Steps", "Quality of Sleep"]
        corr = df_f[num_cols2].corr()

        fig, ax = plt.subplots(figsize=(9, 7))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                    cmap="RdYlGn", center=0, vmin=-1, vmax=1,
                    square=True, linewidths=0.5,
                    annot_kws={"size": 10}, ax=ax)
        ax.set_title("Correlation Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown('<div class="sec">Correlations with Quality of Sleep</div>', unsafe_allow_html=True)
        corr_t = corr["Quality of Sleep"].drop("Quality of Sleep").sort_values()
        fig2, ax2 = plt.subplots(figsize=(8, 3.5))
        bc = [PAL[3] if v < 0 else PAL[2] for v in corr_t]
        ax2.barh(corr_t.index, corr_t.values, color=bc)
        ax2.axvline(0, color="#888", lw=1)
        ax2.set_title("Feature Correlation with Quality of Sleep")
        ax2.set_xlabel("Pearson r")
        plt.tight_layout()
        st.pyplot(fig2); plt.close()

        st.markdown("""<div class="box">
            <b>Stress Level (r = −0.90)</b> — strongest negative predictor.<br>
            <b>Sleep Duration (r = +0.88)</b> — strongest positive predictor.<br>
            <b>Heart Rate (r = −0.66)</b> — moderate negative relationship.
        </div>""", unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="sec">Quantitative → Quantitative (Scatter + Regression Line)</div>',
                    unsafe_allow_html=True)
        features_qq = ["Stress Level", "Sleep Duration", "Heart Rate",
                       "Age", "Daily Steps", "Physical Activity Level"]
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for ax, feat in zip(axes.flat, features_qq):
            ax.scatter(df_f[feat], df_f["Quality of Sleep"],
                       alpha=0.4, color=PAL[0], s=20)
            m_c, b_c = np.polyfit(df_f[feat], df_f["Quality of Sleep"], 1)
            x_l = np.linspace(df_f[feat].min(), df_f[feat].max(), 100)
            ax.plot(x_l, m_c*x_l+b_c, color=PAL[3], lw=2)
            r_v = df_f[[feat, "Quality of Sleep"]].corr().iloc[0, 1]
            ax.set_title(f"{feat}  (r = {r_v:.2f})", fontsize=9)
            ax.set_xlabel(feat, fontsize=8)
            ax.set_ylabel("Quality of Sleep", fontsize=8)
        plt.suptitle("Feature vs Quality of Sleep — Regression Plots",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("""<div class="box">
            <b>Stress Level</b> → clear negative linear trend.<br>
            <b>Sleep Duration</b> → strong positive linear trend.<br>
            <b>Heart Rate</b> → moderate negative trend.
        </div>""", unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="sec">Categorical → Quantitative (Boxplots)</div>',
                    unsafe_allow_html=True)
        target_cq = st.selectbox("Target variable", ["Quality of Sleep", "Sleep Duration"])
        cat_cols  = ["Gender", "BMI Category", "Sleep Disorder", "Occupation"]

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        for ax, col in zip(axes.flat, cat_cols):
            cats = df_f[col].unique()
            data = [df_f[df_f[col] == c][target_cq].dropna() for c in cats]
            bp = ax.boxplot(data, patch_artist=True, labels=cats,
                            medianprops=dict(color=PAL[4], lw=2))
            for patch, color in zip(bp["boxes"], PAL):
                patch.set_facecolor(color+"44"); patch.set_edgecolor(color)
            ax.set_title(f"{col} vs {target_cq}", fontsize=10)
            ax.set_xlabel(col, fontsize=8); ax.set_ylabel(target_cq, fontsize=8)
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("""<div class="box">
            <b>Gender:</b> Females 7.23h vs Males 7.04h — small difference.<br>
            <b>BMI:</b> Normal → 7.39h | Overweight → 6.77h.<br>
            <b>Sleep Disorder:</b> Insomnia → 6.59h | None → 7.36h | Sleep Apnea → 7.03h.<br>
            <b>Occupation:</b> Nurses and Doctors most represented.
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec">Categorical Distributions</div>', unsafe_allow_html=True)
        fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8))
        for ax, col in zip(axes2.flat, cat_cols):
            counts = df_f[col].value_counts()
            ax.bar(counts.index, counts.values, color=PAL[:len(counts)], alpha=0.85)
            ax.set_title(f"{col} Distribution", fontsize=10)
            ax.set_xlabel(col, fontsize=8); ax.set_ylabel("Count", fontsize=8)
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig2); plt.close()


# ══════════════════════════════════════════════
#  PAGE 6 – MODELING
# ══════════════════════════════════════════════
elif page == "Modeling":
    st.title("📈 Linear Regression Modeling")
    st.markdown("Predicting **Quality of Sleep** using 6 lifestyle & health features.")
    st.divider()

    FEATURES = ["Age", "Sleep Duration", "Physical Activity Level",
                "Stress Level", "Heart Rate", "Daily Steps"]
    X = df[FEATURES]; y = df["Quality of Sleep"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    models_dict = {
        "OLS":           LinearRegression(),
        "Ridge (α=1)":   Ridge(alpha=1.0),
        "Lasso (α=0.1)": Lasso(alpha=0.1, max_iter=10000),
    }
    fitted, results = {}, []
    for name, mdl in models_dict.items():
        mdl.fit(X_tr_sc, y_train)
        yp = mdl.predict(X_te_sc)
        fitted[name] = (mdl, yp)
        results.append({
            "Model": name,
            "R²":    round(r2_score(y_test, yp), 4),
            "RMSE":  round(np.sqrt(mean_squared_error(y_test, yp)), 4),
            "MAE":   round(mean_absolute_error(y_test, yp), 4),
        })
    results_df   = pd.DataFrame(results)
    ols_model    = fitted["OLS"][0]
    y_pred_test  = fitted["OLS"][1]
    y_pred_train = ols_model.predict(X_tr_sc)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Model Overview", "Coefficients", "Residuals", "Model Comparison"
    ])

    with tab1:
        st.markdown('<div class="sec">OLS Performance — Train vs Test</div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("R² Train",   f"{r2_score(y_train, y_pred_train):.4f}")
        c2.metric("R² Test",    f"{r2_score(y_test,  y_pred_test):.4f}")
        c3.metric("RMSE Train", f"{np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}")
        c4.metric("RMSE Test",  f"{np.sqrt(mean_squared_error(y_test,  y_pred_test)):.4f}")
        c5.metric("MAE Train",  f"{mean_absolute_error(y_train, y_pred_train):.4f}")
        c6.metric("MAE Test",   f"{mean_absolute_error(y_test,  y_pred_test):.4f}")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(y_test, y_pred_test, color=PAL[0], alpha=0.6, s=40,
                   label="Test predictions")
        lim = [min(y_test.min(), y_pred_test.min())-0.3,
               max(y_test.max(), y_pred_test.max())+0.3]
        ax.plot(lim, lim, color=PAL[2], lw=2, linestyle="--", label="Perfect prediction")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Actual Quality of Sleep")
        ax.set_ylabel("Predicted Quality of Sleep")
        ax.set_title("Actual vs Predicted — OLS", fontweight="bold")
        ax.legend(); plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("""<div class="box">
            The model explains <b>91.5% of the variance</b> in Quality of Sleep on unseen data.
            Predictions are off by less than <b>0.3 points</b> on a 1–10 scale.
            Test R² is slightly higher than Train R² → <b>no overfitting</b>.
        </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="sec">Feature Coefficients (Standardized)</div>', unsafe_allow_html=True)
        coef_df = pd.DataFrame({
            "Feature":     FEATURES,
            "Coefficient": ols_model.coef_.round(4),
        }).sort_values("Coefficient", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        bc = [PAL[3] if v < 0 else PAL[2] for v in coef_df["Coefficient"]]
        ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=bc)
        ax.axvline(0, color="#888", lw=1)
        ax.set_title(f"OLS Coefficients  (intercept = {ols_model.intercept_:.4f})",
                     fontweight="bold")
        ax.set_xlabel("Coefficient value")
        ax.legend(handles=[
            mpatches.Patch(color=PAL[2], label="Positive effect"),
            mpatches.Patch(color=PAL[3], label="Negative effect"),
        ])
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.dataframe(coef_df.sort_values("Coefficient", ascending=False),
                     use_container_width=True)
        st.markdown("""<div class="box">
            <b>Stress Level (−0.569)</b> and <b>Sleep Duration (+0.523)</b> dominate.<br>
            Physical Activity Level and Daily Steps contribute very little
            once other factors are controlled.
        </div>""", unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="sec">Residual Analysis</div>', unsafe_allow_html=True)
        residuals = y_test - y_pred_test

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].scatter(y_pred_test, residuals, alpha=0.6, color=PAL[0], s=40)
        axes[0].axhline(0, color=PAL[3], lw=1.5, linestyle="--")
        axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residual")
        axes[0].set_title("Residuals vs Predicted", fontweight="bold")

        axes[1].hist(residuals, bins=20, color=PAL[0], edgecolor="white", alpha=0.85)
        axes[1].axvline(0, color=PAL[3], lw=1.5, linestyle="--")
        axes[1].set_xlabel("Residual")
        axes[1].set_title("Residual Distribution", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        c1, c2, c3 = st.columns(3)
        c1.metric("Mean residual",  f"{residuals.mean():.4f}")
        c2.metric("Std residual",   f"{residuals.std():.4f}")
        c3.metric("Max |residual|", f"{residuals.abs().max():.4f}")

        st.markdown("""<div class="box">
            Residuals are small (within ±0.8) and centered near zero — a good sign.<br>
            Slight pattern: over-predicts low scores, under-predicts high ones.
            This is expected because <b>Quality of Sleep is a discrete integer</b>
            while linear regression outputs continuous values.
        </div>""", unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="sec">OLS vs Ridge vs Lasso</div>', unsafe_allow_html=True)
        st.dataframe(results_df, use_container_width=True)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, metric in zip(axes, ["R²", "RMSE", "MAE"]):
            clrs = [PAL[i] for i in range(len(results_df))]
            ax.bar(results_df["Model"], results_df[metric], color=clrs, alpha=0.85)
            ax.set_title(metric, fontweight="bold"); ax.set_ylabel(metric)
            for i, v in enumerate(results_df[metric]):
                ax.text(i, v+0.001, str(v), ha="center", va="bottom", fontsize=9)
        plt.suptitle("Model Comparison", fontsize=13, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("""<div class="box">
            All three models perform nearly identically — regularization adds little benefit here.
            <b>OLS</b> is the preferred choice for its simplicity and interpretability.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  PAGE 7 – PREDICTOR
# ══════════════════════════════════════════════
elif page == "Predictor":

    st.title("🛌 Sleep Quality Predictor")

    st.markdown(
        "Enter your lifestyle details to predict your **Quality of Sleep** score."
    )

    st.divider()

    FEATURES = [
        "Age",
        "Sleep Duration",
        "Physical Activity Level",
        "Stress Level",
        "Heart Rate",
        "Daily Steps"
    ]

    X = df[FEATURES]
    y = df["Quality of Sleep"]

    scaler2 = StandardScaler()
    scaler2.fit(X)

    ols2 = LinearRegression()
    ols2.fit(scaler2.transform(X), y)

    # =====================================================
    # INPUTS
    # =====================================================

    c1, c2 = st.columns(2)

    with c1:

        age = st.slider("Age (years)", 27, 59, 35)
        sleep_dur = st.slider("Sleep Duration (hours)", 5.0, 9.0, 7.0, 0.1)
        phys_act = st.slider("Physical Activity Level", 30, 90, 60)

    with c2:

        stress = st.slider("Stress Level (1–10)", 1, 10, 5)
        heart_rate = st.slider("Heart Rate (bpm)", 60, 100, 72)
        daily_steps = st.slider("Daily Steps", 3000, 10000, 6000, 100)

    # =====================================================
    # PREDICTION BUTTON
    # =====================================================

    if st.button("Predict Quality of Sleep", use_container_width=True):

        inp = np.array([[age, sleep_dur, phys_act, stress, heart_rate, daily_steps]])

        pred = np.clip(ols2.predict(scaler2.transform(inp))[0], 1, 10)

        st.divider()

        # =====================================================
        # RESULT CARD  — FIX: use <p> tags, single-line styles
        # =====================================================

        _, col_b, _ = st.columns([1, 2, 1])

        with col_b:

            if pred >= 7:
                color = "#22c55e"
                bg    = "#f0fdf4"
                label = "Good"
            elif pred >= 5:
                color = "#fbbf24"
                bg    = "#fefce8"
                label = "Moderate"
            else:
                color = "#ef4444"
                bg    = "#fef2f2"
                label = "Poor"

            st.markdown(
                f'<div style="text-align:center; background:{bg}; border:2px solid {color}; border-radius:16px; padding:30px;">'
                f'<p style="font-size:64px; font-weight:800; color:{color}; margin:0;">{pred:.1f}</p>'
                f'<p style="font-size:22px; font-weight:700; color:{color}; margin:8px 0 0 0;">{label} Sleep Quality</p>'
                f'<p style="font-size:13px; color:#64748b; margin:10px 0 0 0;">Score out of 10</p>'
                f'</div>',
                unsafe_allow_html=True
            )

        # =====================================================
        # FEATURE CONTRIBUTIONS CHART
        # =====================================================

        contributions = ols2.coef_ * scaler2.transform(inp)[0]

        contrib_df = pd.DataFrame({
            "Feature": FEATURES,
            "Contribution": contributions,
        }).sort_values("Contribution")

        fig, ax = plt.subplots(figsize=(8, 3.5))

        clrs_c = [PAL[3] if v < 0 else PAL[2] for v in contrib_df["Contribution"]]

        ax.barh(contrib_df["Feature"], contrib_df["Contribution"], color=clrs_c)
        ax.axvline(0, color="#888", lw=1)
        ax.set_title("How each feature contributes to your predicted score", fontweight="bold")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # =====================================================
        # PERSONALISED TIPS
        # =====================================================

        st.divider()

        st.markdown('<div class="sec">Personalised Tips</div>', unsafe_allow_html=True)

        if stress >= 7:
            st.markdown(
                '<div class="box">⚠️ <b>High Stress Detected</b> — Stress is the #1 negative factor. Try relaxation, exercise or mindfulness.</div>',
                unsafe_allow_html=True
            )

        if sleep_dur < 7:
            st.markdown(
                '<div class="box">⚠️ <b>Low Sleep Duration</b> — You sleep less than 7 hours. Aim for 7–9 hours to improve your score.</div>',
                unsafe_allow_html=True
            )

        if heart_rate > 80:
            st.markdown(
                '<div class="box">⚠️ <b>Elevated Heart Rate</b> — Higher resting HR is linked to lower sleep quality. Regular aerobic exercise can help.</div>',
                unsafe_allow_html=True
            )

        if pred >= 7:
            st.markdown(
                '<div class="box">✅ Your predicted sleep quality is <b>good</b>! Keep your current habits.</div>',
                unsafe_allow_html=True
            )
        elif pred >= 5:
            st.markdown(
                '<div class="box">⚠️ Your sleep quality is <b>moderate</b>. Some lifestyle habits could still be improved.</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="box">❌ Your predicted sleep quality is <b>low</b>. Improving sleep duration and reducing stress may significantly improve your score.</div>',
                unsafe_allow_html=True
            )