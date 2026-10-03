import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="WanderAI | Personalized Travel Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 8% 10%,
                rgba(255, 214, 165, 0.55),
                transparent 28%
            ),
            radial-gradient(
                circle at 92% 15%,
                rgba(186, 230, 253, 0.65),
                transparent 30%
            ),
            radial-gradient(
                circle at 75% 90%,
                rgba(167, 243, 208, 0.50),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #fff7ed 0%,
                #e0f2fe 35%,
                #ecfeff 65%,
                #f0fdf4 100%
            );

        min-height: 100vh;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 0.6rem;
        padding-bottom: 1rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    .nav-card {
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid rgba(255, 255, 255, 0.95);
        border-radius: 17px;
        padding: 13px 18px;
        min-height: 55px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 25px rgba(30, 64, 175, 0.10);
    }

    .nav-brand {
        color: #12304a;
        font-size: 25px;
        font-weight: 800;
    }

    .nav-item {
        color: #12304a;
        font-size: 15px;
        font-weight: 700;
        text-align: center;
        padding-top: 7px;
    }

    .hero-card {
        background:
            linear-gradient(
                135deg,
                rgba(14, 165, 233, 0.88),
                rgba(79, 70, 229, 0.88)
            );

        border: 1px solid rgba(255,255,255,0.35);
        border-radius: 24px;
        padding: 25px 34px;
        margin-top: 15px;
        margin-bottom: 15px;

        box-shadow:
            0 18px 45px rgba(30, 64, 175, 0.18);
    }

    .hero-small {
        color: #e0f2fe;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
    }

    .hero-title {
        color: white;
        font-size: 42px;
        font-weight: 850;
        line-height: 1.05;
        margin-top: 7px;
    }

    .hero-subtitle {
        color: #e0f2fe;
        font-size: 15px;
        margin-top: 7px;
    }

    .planner-card {
        background: rgba(255,255,255,0.97);
        border-radius: 24px;
        padding: 20px 27px 18px 27px;
        margin-top: 10px;

        box-shadow:
            0 18px 50px rgba(30, 64, 175, 0.14);

        border: 1px solid rgba(255,255,255,0.9);
    }

    .planner-title {
        color: #102a43;
        font-size: 25px;
        font-weight: 800;
    }

    .planner-subtitle {
        color: #627d98;
        font-size: 13px;
        margin-top: 2px;
        margin-bottom: 8px;
    }

    label {
        color: #243b53 !important;
        font-weight: 700 !important;
    }

    .stSelectbox > div > div,
    .stNumberInput > div > div {
        background: white !important;
        border-radius: 11px !important;
        border: 1px solid #cbd5e1 !important;
    }

    .stButton > button {
        width: 100%;
        min-height: 49px;
        border-radius: 12px;
        border: none;

        background:
            linear-gradient(
                90deg,
                #0891b2,
                #2563eb,
                #4f46e5
            );

        color: white;
        font-size: 15px;
        font-weight: 800;

        box-shadow:
            0 8px 22px rgba(37,99,235,0.30);

        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow:
            0 12px 28px rgba(37,99,235,0.40);
    }

    [data-testid="stMetric"] {
        background: white;
        border-radius: 15px;
        padding: 14px;
        box-shadow:
            0 7px 22px rgba(30, 64, 175, 0.08);
    }

    .stDownloadButton > button {
        width: 100%;
        border-radius: 12px;
        background: #0f172a;
        color: white;
        font-weight: 700;
    }

    .result-card {
        background: white;
        border-radius: 17px;
        padding: 18px;
        margin-bottom: 12px;
        box-shadow:
            0 7px 22px rgba(30, 64, 175, 0.08);
    }

    hr {
        border-color: rgba(30, 64, 175, 0.12);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("travel_data.csv")


try:
    df = load_data()

except FileNotFoundError:
    st.error(
        "travel_data.csv was not found. "
        "Please keep travel_data.csv in the same folder as app.py."
    )
    st.stop()


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Destination",
    "Place",
    "Cost"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        "Missing columns in travel_data.csv: "
        + ", ".join(missing_columns)
    )
    st.stop()


# ============================================================
# OPTIONAL COLUMNS
# ============================================================

if "Category" not in df.columns:
    df["Category"] = "Tourist Attraction"

if "Activities" not in df.columns:
    df["Activities"] = "Sightseeing"

if "Description" not in df.columns:
    df["Description"] = "Popular place to visit"

if "Rating" not in df.columns:
    df["Rating"] = 4.5


# ============================================================
# CLEAN DATA
# ============================================================

df["Cost"] = pd.to_numeric(
    df["Cost"],
    errors="coerce"
).fillna(0)

df["Rating"] = pd.to_numeric(
    df["Rating"],
    errors="coerce"
).fillna(4.0)

df = df.dropna(
    subset=["Destination", "Place"]
)


# ============================================================
# AI RECOMMENDATION FUNCTION
# ============================================================

def generate_recommendations(
    destination,
    interest,
    activity,
    budget
):

    destination_df = df[
        df["Destination"]
        .astype(str)
        .str.lower()
        == destination.lower()
    ].copy()

    if destination_df.empty:
        destination_df = df.copy()

    destination_df["AI_Text"] = (
        destination_df["Destination"].astype(str)
        + " "
        + destination_df["Place"].astype(str)
        + " "
        + destination_df["Category"].astype(str)
        + " "
        + destination_df["Activities"].astype(str)
        + " "
        + destination_df["Description"].astype(str)
    )

    user_preferences = (
        str(destination)
        + " "
        + str(interest)
        + " "
        + str(activity)
    )

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        vectors = vectorizer.fit_transform(
            [user_preferences]
            + destination_df["AI_Text"].tolist()
        )

        similarity_scores = cosine_similarity(
            vectors[0:1],
            vectors[1:]
        ).flatten()

    except ValueError:

        similarity_scores = [0.5] * len(destination_df)

    destination_df["AI_Score"] = similarity_scores

    affordable = destination_df[
        destination_df["Cost"] <= budget
    ].copy()

    if affordable.empty:
        affordable = destination_df.copy()

    affordable = affordable.sort_values(
        by=["AI_Score", "Rating"],
        ascending=False
    )

    affordable = affordable.drop_duplicates(
        subset=["Place"]
    )

    return affordable


# ============================================================
# NAVIGATION
# ============================================================

nav1, nav2, nav3, nav4 = st.columns(
    [2.3, 1, 1, 1]
)

with nav1:
    st.markdown(
        """
        <div class="nav-card">
            <div class="nav-brand">✈️ WanderAI</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with nav2:
    st.markdown(
        """
        <div class="nav-card">
            <div class="nav-item">🏠 Home</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with nav3:
    st.markdown(
        """
        <div class="nav-card">
            <div class="nav-item">🧭 Plan Trip</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with nav4:
    st.markdown(
        """
        <div class="nav-card">
            <div class="nav-item">🌍 Explore</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    '<div class="hero-card"><div class="hero-small">✈️ AI-POWERED TRAVEL PLANNER</div><div class="hero-title">Explore more. Travel smarter.</div><div class="hero-subtitle">Your personalized journey, planned around you.</div></div>',
    unsafe_allow_html=True
)


# ============================================================
# PLANNER SECTION
# ============================================================

st.markdown(
    '<div class="planner-card"><div class="planner-title">🎯 Plan Your Perfect Trip</div><div class="planner-subtitle">Choose your preferences and create your personalized itinerary.</div></div>',
    unsafe_allow_html=True
)


# ============================================================
# INPUT SECTION
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    destinations = sorted(
        df["Destination"]
        .astype(str)
        .unique()
        .tolist()
    )

    destination = st.selectbox(
        "📍 Destination",
        destinations
    )

with col2:

    duration = st.selectbox(
        "📅 Duration",
        [
            "1 Day",
            "2 Days",
            "3 Days",
            "4 Days",
            "5 Days",
            "6 Days",
            "7 Days"
        ],
        index=2
    )

with col3:

    budget = st.number_input(
        "💰 Budget (₹)",
        min_value=1000,
        max_value=1000000,
        value=15000,
        step=1000
    )


col4, col5, col6 = st.columns(3)

with col4:

    interest = st.selectbox(
        "❤️ Interest",
        [
            "Beaches",
            "Nature",
            "Adventure",
            "Historical",
            "Culture",
            "Shopping",
            "Food",
            "Sightseeing",
            "Relaxation"
        ]
    )

with col5:

    activity = st.selectbox(
        "🎯 Activity",
        [
            "Sightseeing",
            "Water Activity",
            "Adventure",
            "Shopping",
            "Relaxation",
            "Food",
            "Cultural Experience",
            "Nature Exploration"
        ]
    )

with col6:

    st.markdown(
        "<div style='height:27px'></div>",
        unsafe_allow_html=True
    )

    generate = st.button(
        "✨ Generate Itinerary"
    )


# ============================================================
# GENERATE RESULTS
# ============================================================

if generate:

    with st.spinner(
        "✈️ Creating your personalized itinerary..."
    ):

        recommendations = generate_recommendations(
            destination,
            interest,
            activity,
            budget
        )

    if recommendations.empty:

        st.warning(
            "No suitable recommendations were found."
        )

        st.stop()

    number_of_days = int(
        duration.split()[0]
    )

    recommendations = recommendations.head(
        number_of_days * 2
    ).reset_index(drop=True)


    # ========================================================
    # SUCCESS MESSAGE
    # ========================================================

    st.success(
        "✨ Your personalized itinerary is ready!"
    )


    # ========================================================
    # TRIP SUMMARY
    # ========================================================

    st.markdown("## 🧳 Trip Summary")

    summary1, summary2, summary3, summary4 = st.columns(4)

    with summary1:
        st.metric(
            "📍 Destination",
            destination
        )

    with summary2:
        st.metric(
            "📅 Duration",
            f"{number_of_days} Days"
        )

    with summary3:
        st.metric(
            "💰 Budget",
            f"₹{budget:,}"
        )

    with summary4:
        st.metric(
            "❤️ Interest",
            interest
        )


    # ========================================================
    # RECOMMENDED PLACES
    # ========================================================

    st.markdown(
        "## 🤖 Recommended Places"
    )

    st.write(
        "Places selected according to your destination, "
        "interests, activity preference and budget."
    )

    recommendation_columns = st.columns(2)

    for index, row in recommendations.iterrows():

        with recommendation_columns[index % 2]:

            st.markdown(
                f"### 📍 {row['Place']}"
            )

            st.caption(
                f"{row['Category']}  •  ⭐ {row['Rating']:.1f}"
            )

            st.write(
                row["Description"]
            )

            st.write(
                f"🎯 {row['Activities']}"
            )

            st.write(
                f"💰 Estimated cost: ₹{row['Cost']:,.0f}"
            )

            score = max(
                0.0,
                min(
                    float(row["AI_Score"]),
                    1.0
                )
            )

            st.progress(score)

            st.caption(
                f"AI personalization match: {score * 100:.1f}%"
            )


    # ========================================================
    # PERSONALIZED ITINERARY
    # ========================================================

    st.markdown(
        "## 🗓️ Personalized Itinerary"
    )

    itinerary_text = ""

    itinerary_text += (
        "WANDERAI - PERSONALIZED TRAVEL ITINERARY\n"
    )

    itinerary_text += (
        "==========================================\n\n"
    )

    itinerary_text += (
        f"Destination: {destination}\n"
        f"Duration: {number_of_days} Days\n"
        f"Budget: ₹{budget:,}\n"
        f"Interest: {interest}\n"
        f"Activity: {activity}\n\n"
    )

    place_index = 0

    for day in range(
        1,
        number_of_days + 1
    ):

        st.markdown(
            f"### 🌅 Day {day}"
        )

        itinerary_text += (
            f"DAY {day}\n"
            "------\n"
        )

        day_places = recommendations.iloc[
            place_index:place_index + 2
        ]

        if day_places.empty:

            st.info(
                "Enjoy local sightseeing and explore nearby attractions."
            )

            itinerary_text += (
                "Enjoy local sightseeing and explore nearby attractions.\n\n"
            )

        else:

            for _, row in day_places.iterrows():

                st.markdown(
                    f"**📍 {row['Place']}**"
                )

                st.write(
                    row["Description"]
                )

                st.write(
                    f"🎯 Activity: {row['Activities']}"
                )

                st.write(
                    f"💰 Estimated cost: ₹{row['Cost']:,.0f}"
                )

                itinerary_text += (
                    f"Place: {row['Place']}\n"
                    f"Category: {row['Category']}\n"
                    f"Activity: {row['Activities']}\n"
                    f"Description: {row['Description']}\n"
                    f"Estimated Cost: ₹{row['Cost']:,.0f}\n\n"
                )

                place_index += 1

        st.divider()


    # ========================================================
    # BUDGET SUMMARY
    # ========================================================

    estimated_cost = recommendations[
        "Cost"
    ].sum()

    remaining_budget = budget - estimated_cost

    st.markdown(
        "## 💳 Budget Overview"
    )

    budget1, budget2 = st.columns(2)

    with budget1:

        st.metric(
            "Estimated Attraction Cost",
            f"₹{estimated_cost:,.0f}"
        )

    with budget2:

        st.metric(
            "Remaining Budget",
            f"₹{max(remaining_budget, 0):,.0f}"
        )


    # ========================================================
    # DOWNLOAD ITINERARY
    # ========================================================

    st.markdown(
        "## 📥 Save Your Itinerary"
    )

    st.download_button(
        label="📥 Download My Itinerary",
        data=itinerary_text,
        file_name=(
            f"{destination}_Personalized_Itinerary.txt"
        ),
        mime="text/plain"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        color:#475569;
        font-size:11px;
        padding:10px 0 2px 0;
    ">
        ✈️ WanderAI • AI-Based Personalized Travel Itinerary Planning System
    </div>
    """,
    unsafe_allow_html=True
)