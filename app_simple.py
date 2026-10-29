"""
app_simple.py  --  FID demo: Fidelity and Diversity

Run it with:   streamlit run app_simple.py

It does NOT calculate FID. It only SHOWS scores that run_fid.py saved earlier,
so it never freezes in front of your audience.

Needs, next to this file (made by make_data.py and run_fid.py):
    results/   damage_ladder.csv, mode_collapse.csv, sample_size.csv
    samples/   picture grids
    data/      (optional) the image folders, used only to count the images
"""
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="FID demo", layout="centered")

TEAL, RASP, VIOLET = "#2A9D8F", "#D1495B", "#7C5CFF"

# Big text and a narrow page: readable on a projector without zooming.
st.markdown("""
<style>
html { font-size: 20px; }
.block-container { max-width: 960px; padding-top: 2rem; }
[data-testid="stCaptionContainer"] { font-size: 1rem; }
[data-testid="stMetricValue"] { font-size: 2.6rem; }
[data-testid="stMetricLabel"] p { font-size: 1rem; }
.txt { font-size: 1.05rem; line-height: 1.55; margin: .3rem 0 .8rem 0; }
.box { border-left: 6px solid #2A9D8F; background: rgba(42,157,143,.12); padding: .7rem 1rem;
       border-radius: 0 8px 8px 0; margin: .9rem 0; font-size: 1.05rem; line-height: 1.5; }
.box.warn { border-left-color: #D1495B; background: rgba(209,73,91,.12); }
.box.idea { border-left-color: #7C5CFF; background: rgba(124,92,255,.12); }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers
def text(html):
    st.markdown(f'<div class="txt">{html}</div>', unsafe_allow_html=True)


def box(html, kind=""):
    st.markdown(f'<div class="box {kind}">{html}</div>', unsafe_allow_html=True)


def load(csv, command="python run_fid.py"):
    path = f"results/{csv}"
    if not os.path.exists(path):
        box(f"No scores for this page yet. In the terminal, run: <b>{command}</b>", "warn")
        st.stop()
    return pd.read_csv(path)


def picture(name, caption):
    path = f"samples/{name}.png"
    if os.path.exists(path):
        st.image(path, caption=caption, width=430)


def count_images(folder):
    if not os.path.isdir(folder):
        return None
    return len([f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))])


def sample_info():
    """Images per set, and a factor if an old run_fid.py counted every image twice."""
    real = count_images("data/real_a")
    p = "results/sample_size.csv"
    biggest = int(pd.read_csv(p)["n_images"].max()) if os.path.exists(p) else None
    if real and biggest and abs(biggest / real - 2) < 0.01:
        return real, 2
    return (real or biggest), 1


def floor_value():
    p = "results/damage_ladder.csv"
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    return float(d[d["level"] == 0]["fid"].iloc[0])


def score_row(fid):
    """Show the FID score together with the floor and the gap above it."""
    floor = floor_value()
    a, b, c = st.columns(3)
    a.metric("FID score", f"{fid:.1f}")
    if floor is not None:
        b.metric("Floor (real vs real)", f"{floor:.1f}")
        c.metric("Above the floor", f"{max(fid - floor, 0):.1f}")


# ---------------------------------------------------------------- menu
st.sidebar.title("FID demo")
page = st.sidebar.radio("Choose a page", [
    "1. About and the floor",
    "2. Fidelity and diversity",
    "3. Damage ladder",
    "4. Mode collapse",
    "5. Sample size",
])
st.sidebar.markdown("**Lower FID = closer to real.** 0 = identical.")

# ================================================================ 1
if page == "1. About and the floor":
    st.title("What is this demo about?")
    text("AI programs can make images. <b>FID</b> (Fréchet Inception Distance) is <b>one number</b> that says how "
         "close a set of AI-made images is to a set of real images.<br>"
         "It looks at two things:<br>"
         "&bull; <b>Fidelity</b>: does each image look real?<br>"
         "&bull; <b>Diversity</b>: does the set have the same variety as real images?")
    box("<b>Lower FID is better. 0 means identical.</b> This demo uses simple dummy pictures (shapes) so we can change "
        "one thing at a time and watch the FID score react.")

    st.subheader("Parameters of this demo")
    n, factor = sample_info()
    params = pd.DataFrame([
        ["Pictures", "Dummy shapes: circle, square, triangle, ring, stripes"],
        ["Images in each set", f"{n}" if n else "run make_data.py"],
        ["Sets used", "17 (2 real, 3 with fewer shapes, 12 damaged)"],
        ["Network that reads the images", "Inception-v3"],
        ["Numbers per image", "2,048"],
        ["Image size given to the network", "299 x 299"],
        ["Score", "FID (lower is better)"],
    ], columns=["Parameter", "Value"])
    st.table(params.set_index("Parameter"))

    st.subheader("Real set A and real set B")
    a, b = st.columns(2)
    with a:
        picture("real_a", "Real set A")
    with b:
        picture("real_b", "Real set B")

    st.subheader("What is the floor?")
    f = floor_value()
    if f is not None:
        st.metric("FID between real set A and real set B", f"{f:.1f}")
    text("<b>The floor is the FID between two sets of REAL images.</b><br>"
         "Both sets are real, yet the score is not 0. Each set is a random sample, like two rounds of coin flips: "
         "they never match exactly.")
    box("<b>Why it matters:</b> the floor is the <b>best score any generator can reach</b>. "
        "Compare your generator with the floor, not with 0. The floor is bigger when the sets are small.", "idea")

# ================================================================ 2
elif page == "2. Fidelity and diversity":
    st.title("Fidelity and diversity")
    text("FID is the sum of two parts. Real images are fixed as the reference. Move the two sliders and watch each part.")
    text("&bull; <b>Fidelity</b>: how close the generated images are to the real ones (the <i>centre</i> gap).<br>"
         "&bull; <b>Diversity</b>: how much variety the generated images have (the <i>spread</i>).")

    fid_gap = st.slider("Fidelity gap (0 = as real as real images)", 0.0, 3.0, 1.0, 0.1)
    div = st.slider("Diversity of generated images (1.0 = same variety as real)", 0.2, 3.0, 0.5, 0.1)

    fid_part = fid_gap ** 2
    div_part = (1.0 - div) ** 2
    a, b, c = st.columns(3)
    a.metric("Fidelity part", f"{fid_part:.2f}")
    b.metric("Diversity part", f"{div_part:.2f}")
    c.metric("FID score", f"{fid_part + div_part:.2f}")

    x = np.linspace(-7, 7, 300)
    bell = lambda m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    st.line_chart(pd.DataFrame({"Real images": bell(0, 1.0), "Generated images": bell(fid_gap, div)}, index=x))
    text("FID (one feature) = <b>(fidelity gap)&sup2; + (diversity gap)&sup2;</b>. "
         "Real FID does the same with 2,048 features: the first part is the <i>mean</i> term, "
         "the second part is the <i>covariance</i> term.")
    box("Fidelity to 0 and diversity to 1.0 gives FID = 0. Make the pink curve narrower and the diversity part grows. "
        "Slide it sideways and the fidelity part grows.")

# ================================================================ 3
elif page == "3. Damage ladder":
    st.title("Damage ladder")
    text("<b>What it is:</b> we take real pictures and make them worse in 4 steps (level 1 to 4). "
         "Level 0 means no damage.<br>"
         "<b>What it tests:</b> <b>fidelity</b>. The shapes are the same, so diversity stays the same, "
         "but the pictures look less real.")
    df = load("damage_ladder.csv")
    kind = st.selectbox("Type of damage", ["blur", "noise", "jpeg"])
    level = st.slider("Damage level (0 = none, 4 = worst)", 0, 4, 0)
    sub = df[df["damage"] == kind].sort_values("level")
    now = float(sub[sub["level"] == level]["fid"].iloc[0])

    score_row(now)
    picture("real_b" if level == 0 else f"{kind}_{level}", f"{kind}, level {level}")
    st.line_chart(sub.set_index("level")["fid"])
    box("<b>What to notice:</b> as damage rises, fidelity falls and the FID score rises. "
        "At level 0 the score is the floor.")

# ================================================================ 4
elif page == "4. Mode collapse":
    st.title("Mode collapse")
    text("<b>What it is:</b> the generator keeps drawing the <b>same few kinds</b> of picture. "
         "Every picture looks fine, so <b>fidelity is fine</b>, but the set has little variety, so "
         "<b>diversity is lost</b>.")
    st.subheader("What do 1, 2, 3 and 5 shapes mean?")
    st.table(pd.DataFrame([
        ["5 shapes", "circle, square, triangle, ring, stripes", "Full variety (like real images)"],
        ["3 shapes", "circle, square, triangle", "Some variety lost"],
        ["2 shapes", "circle, square", "More variety lost"],
        ["1 shape", "circle only", "No variety: complete mode collapse"],
    ], columns=["Set", "Shapes drawn", "Diversity"]).set_index("Set"))

    df = load("mode_collapse.csv").sort_values("shapes_kept", ascending=False)
    df["label"] = df["shapes_kept"].apply(lambda k: f"{k} shape" + ("" if k == 1 else "s"))
    choice = st.selectbox("Choose a generated set", df["label"])
    row = df[df["label"] == choice].iloc[0]
    grid = {5: "real_b", 3: "collapse_3class", 2: "collapse_2class", 1: "collapse_1class"}[int(row["shapes_kept"])]

    score_row(float(row["fid"]))
    picture(grid, choice)
    st.bar_chart(df.sort_values("shapes_kept").set_index("label")["fid"])
    box("<b>What to notice:</b> fewer shapes means less diversity, and the FID score rises, "
        "even though no single picture is bad.")

    st.subheader("Floor vs mode collapse: what is the difference?")
    st.table(pd.DataFrame([
        ["Compared sets", "Real vs real", "Real vs a generated set with fewer shapes"],
        ["Why FID is above 0", "Random sampling only", "Diversity is missing"],
        ["What it means", "Best score possible", "A problem to fix"],
    ], columns=["", "Floor", "Mode collapse"]).set_index(""))

# ================================================================ 5
elif page == "5. Sample size":
    st.title("Sample size")
    text("<b>Sample size = the number of images in each set used to calculate FID.</b><br>"
         "This page is not about fidelity or diversity. It is about <b>how reliable the FID number is</b>.")
    text("Everyday example: asking 5 people their height gives a shaky average for a city. Asking 5,000 gives a "
         "steady one. FID is the same: with few images the score is noisy and too high.")
    df = load("sample_size.csv")
    n, factor = sample_info()
    if factor == 2:
        box("Your results were made with an older run_fid.py that counted each image twice. "
            f"The image counts below are corrected (your sets have {n} images). "
            "Run the updated run_fid.py to fix the scores too.", "warn")
        df["n_images"] = (df["n_images"] / factor).round().astype(int)

    st.subheader("What the chart shows")
    text("For each number of images, we calculate FID again using only the <b>first n images</b> of each set. "
         "There are two lines:<br>"
         "&bull; <b>Real vs real (the floor)</b>: two real sets. No fidelity or diversity problem.<br>"
         "&bull; <b>Real vs blurred</b>: a blurred copy. Fidelity is lost.")

    names = {"real vs real": "Real vs real (floor)", "real vs blur_2": "Real vs blurred (fidelity lost)"}
    df["Line"] = df["pair"].map(names).fillna(df["pair"])
    order = [str(v) for v in sorted(df["n_images"].unique())]
    df["Images per set"] = df["n_images"].astype(str)
    chart = (alt.Chart(df).mark_line(point=alt.OverlayMarkDef(size=110), strokeWidth=4).encode(
        x=alt.X("Images per set:O", sort=order, axis=alt.Axis(labelAngle=0, labelFontSize=15, titleFontSize=15)),
        y=alt.Y("fid:Q", title="FID score", scale=alt.Scale(zero=True), axis=alt.Axis(labelFontSize=15, titleFontSize=15)),
        color=alt.Color("Line:N", scale=alt.Scale(range=[RASP, TEAL]),
                        legend=alt.Legend(title=None, orient="bottom", labelFontSize=14)),
        tooltip=["Images per set", "Line", alt.Tooltip("fid:Q", format=".1f")])
        .properties(width=620, height=320))
    st.altair_chart(chart)

    wide = df.pivot(index="n_images", columns="Line", values="fid").round(1)
    wide.index.name = "Images per set"
    st.table(wide)

    lo, hi = wide.index.min(), wide.index.max()
    fl = "Real vs real (floor)"
    if fl in wide.columns:
        box(f"<b>What to notice:</b> the floor is <b>{wide.loc[lo, fl]}</b> with {lo} images and "
            f"<b>{wide.loc[hi, fl]}</b> with {hi} images. The pictures did not change, only how many we used. "
            "The floor changes with the number of images, but it never reaches 0.")
    box("<b>Rule:</b> only compare FID scores that use the <b>same number of images</b>. "
        "For a real report, use 10,000 or more, ideally 50,000.", "idea")
