
import streamlit as st
import pandas as pd
import numpy as np
import textwrap
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from streamlit_cookies_controller import CookieController
from src.upload_service import (
    read_uploaded_file,
    validate_dataset,
    analyse_uploaded_dataset,
    REQUIRED_COLUMNS,
    load_analysis_results
)
import jwt

from auth.service import register_user, login_user
from auth.security import decode_access_token
from database.repository import (
    initialize_database,
    get_user_by_id,
    get_user_analysis_history,
    get_user_analysis_run,
    get_user_borrower_predictions
)
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix
)

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CreditLens AI | Risk Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. DARK THEME AND ANIMATIONS
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

/* MAIN ANIMATED BACKGROUND */
.stApp {
    background: linear-gradient(
        -45deg,
        #080E1B,
        #0D1729,
        #111B35,
        #17152F,
        #0B1B2E
    );
    background-size: 400% 400%;
    animation: backgroundFlow 22s ease infinite;
    color: #E8EEF8;
    font-family: 'DM Sans', sans-serif;
}

@keyframes backgroundFlow {
    0%   { background-position: 0% 50%; }
    25%  { background-position: 50% 100%; }
    50%  { background-position: 100% 50%; }
    75%  { background-position: 50% 0%; }
    100% { background-position: 0% 50%; }
}

/* AMBIENT BACKGROUND GLOW */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background:
        radial-gradient(
            circle at 12% 25%,
            rgba(37, 99, 235, 0.13),
            transparent 30%
        ),
        radial-gradient(
            circle at 85% 20%,
            rgba(124, 58, 237, 0.12),
            transparent 28%
        ),
        radial-gradient(
            circle at 65% 85%,
            rgba(14, 165, 233, 0.10),
            transparent 32%
        );
    animation: ambientGlow 14s ease-in-out infinite alternate;
}

@keyframes ambientGlow {
    0% {
        opacity: 0.45;
        transform: scale(1);
    }
    100% {
        opacity: 1;
        transform: scale(1.08);
    }
}

/* KEEP APPLICATION CONTENT ABOVE BACKGROUND */
.stApp > header,
.stApp > div {
    position: relative;
    z-index: 1;
}

header[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101C30, #0A1220);
    border-right: 1px solid #24344D;
}

[data-testid="stSidebar"] * {
    color: #DCE6F5;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

h1, h2, h3 {
    color: #F1F5FC !important;
    font-weight: 700 !important;
}

p, label {
    color: #A8B8D0;
}

div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #17263D, #111D30);
    border: 1px solid #293C59;
    padding: 19px;
    border-radius: 14px;
    box-shadow: 0 6px 22px rgba(0,0,0,0.18);
    transition: all 0.25s ease;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    border-color: #3989D9;
    box-shadow: 0 8px 25px rgba(45,130,220,0.15);
}

div[data-testid="stMetricValue"] {
    color: #F4F8FF;
    font-weight: 800;
}

div[data-testid="stMetricLabel"] {
    color: #A5B8D1;
}

div[data-testid="stTabs"] button {
    color: #AABBD2;
    font-weight: 600;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #62B6FF;
    border-bottom: 2px solid #399BFF;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #293C59;
    border-radius: 12px;
    overflow: hidden;
}

.stSelectbox, .stTextInput, .stMultiSelect {
    background-color: transparent;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

button[kind="primary"] {
    background: linear-gradient(90deg, #2563EB, #7C3AED);
    border: none;
    border-radius: 9px;
}

/* HERO */
.hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(115deg, #172C4B, #182448, #25234C);
    border: 1px solid #344D72;
    border-radius: 20px;
    padding: 30px;
    margin-bottom: 22px;
}

.hero:before {
    content: "";
    position: absolute;
    width: 280px;
    height: 280px;
    right: -60px;
    top: -160px;
    border-radius: 50%;
    background: rgba(56, 156, 255, 0.20);
    filter: blur(35px);
    animation: pulse 5s ease-in-out infinite alternate;
}

.hero:after {
    content: "";
    position: absolute;
    width: 200px;
    height: 200px;
    right: 180px;
    bottom: -150px;
    border-radius: 50%;
    background: rgba(139, 92, 246, 0.20);
    filter: blur(35px);
    animation: pulse 7s ease-in-out infinite alternate;
}

.hero-content {
    position: relative;
    z-index: 2;
}

.hero-title {
    font-size: 32px;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -1px;
}

.hero-subtitle {
    color: #B7C9E2;
    font-size: 14px;
    margin-top: 7px;
}

.status {
    display: inline-block;
    color: #54E5A5;
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(52,211,153,0.35);
    border-radius: 30px;
    padding: 5px 12px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}

/* ANIMATED TICKER */
.ticker {
    display: flex;
    overflow: hidden;
    white-space: nowrap;
    background: #111E32;
    border: 1px solid #273B57;
    border-radius: 10px;
    padding: 12px 0;
    margin-bottom: 25px;
}

.ticker-track {
    display: inline-flex;
    gap: 55px;
    min-width: max-content;
    animation: ticker 32s linear infinite;
    color: #AFC2DC;
    font-size: 12px;
}

.ticker-track span {
    display: inline-block;
}

.ticker-track strong {
    color: #55D6A2;
}

@keyframes ticker {
    from { transform: translateX(0); }
    to { transform: translateX(-50%); }
}

@keyframes pulse {
    from { transform: scale(0.9); opacity: 0.5; }
    to { transform: scale(1.25); opacity: 1; }
}

/* SECTION AND INSIGHT CARDS */
.section-label {
    color: #65B9FF;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.insight-card {
    background: linear-gradient(140deg, #182941, #111D30);
    border: 1px solid #2A405F;
    padding: 20px;
    border-radius: 14px;
    height: 100%;
}

.insight-card h3 {
    font-size: 16px;
    margin-bottom: 8px;
}

.insight-card p {
    font-size: 13px;
    line-height: 1.7;
}

div[data-testid="stPlotlyChart"] {
    background: #111D30;
    border: 1px solid #263A56;
    border-radius: 13px;
    padding: 8px;
}

.small-note {
    color: #8298B5;
    font-size: 11px;
}

hr {
    border-color: #263A56;
}

/* FLOATING BANKING FUN FACT POPUP */
.funfact-popup {
    position: fixed;
    right: 25px;
    bottom: 25px;
    width: 310px;
    z-index: 99999;
    background: linear-gradient(145deg, #172C4B, #111D30);
    border: 1px solid #3989D9;
    border-radius: 15px;
    padding: 17px 18px;
    box-shadow: 0 10px 35px rgba(0,0,0,0.45);
    animation: funFactSlideIn 0.7s ease-out;
}

.funfact-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
}

.funfact-title {
    color: #65B9FF;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1px;
}

.funfact-icon {
    font-size: 23px;
    margin-right: 8px;
}

.funfact-text {
    color: #DCE6F5;
    font-size: 12px;
    line-height: 1.6;
}

.funfact-close {
    position: absolute;
    top: 8px;
    right: 11px;
    color: #A8B8D0;
    font-size: 19px;
    cursor: pointer;
    background: transparent;
    border: none;
}

.funfact-close:hover {
    color: white;
}

@keyframes funFactSlideIn {
    from {
        opacity: 0;
        transform: translateX(35px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

</style>
""", unsafe_allow_html=True)






# =========================================================
# 2A. USER AUTHENTICATION
# =========================================================

initialize_database()

# Browser-persistent authentication cookie
cookie_controller = CookieController(key="creditlens_auth")
AUTH_COOKIE_NAME = "creditlens_access_token"

# Restore token from Streamlit session or browser cookie.
access_token = st.session_state.get("access_token")

if not access_token:
    cookie_token = cookie_controller.get(AUTH_COOKIE_NAME)

    if cookie_token:
        access_token = cookie_token
        st.session_state["access_token"] = access_token


def clear_login_session():
    """Clear the current authentication session."""

    try:
        cookie_controller.remove(
            AUTH_COOKIE_NAME,
            path="/",
            same_site="lax",
            secure=False
        )
    except (KeyError, Exception):
        pass

    for key in [
        "access_token",
        "auth_user_id",
        "auth_email",
        "auth_role",
        "latest_upload_analysis",
        "switch_to_uploaded_dataset",
        "active_dataset_selector"
    ]:
        st.session_state.pop(key, None)


# Validate token and retrieve current role from database.
current_user = None

if access_token:
    try:
        token_payload = decode_access_token(access_token)
        authenticated_user_id = token_payload["sub"]

        current_user = get_user_by_id(authenticated_user_id)

        if current_user is None:
            raise jwt.InvalidTokenError("User account not found.")

        if current_user["role"] not in ["employee", "client"]:
            raise jwt.InvalidTokenError("Invalid account role.")

        st.session_state["auth_user_id"] = current_user["user_id"]
        st.session_state["auth_email"] = current_user["email"]
        st.session_state["auth_role"] = current_user["role"]

    except (jwt.InvalidTokenError, KeyError, TypeError):
        clear_login_session()
        current_user = None


def establish_login_session(result):
    """Save the authenticated user session and browser cookie."""

    token = result["access_token"]

    st.session_state["access_token"] = token
    st.session_state["auth_user_id"] = result["user_id"]
    st.session_state["auth_email"] = result["email"]
    st.session_state["auth_role"] = result["role"]

    cookie_controller.set(
        AUTH_COOKIE_NAME,
        token,
        path="/",
        expires=datetime.now(timezone.utc) + timedelta(days=7),
        same_site="lax",
        secure=False
    )



# =========================================================
# AUTHENTICATION SCREEN
# =========================================================

if current_user is None:

    # -----------------------------------------------------
    # AUTHENTICATION-SPECIFIC STYLING
    # Scoped to the login screen; dashboard styling remains
    # unchanged after authentication.
    # -----------------------------------------------------

    st.markdown("""
    <style>

    /* Identify the authentication screen */
    div[data-testid="stMainBlockContainer"]:has(.auth-marker) {
        max-width: 1180px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    /* Left branding panel */
    .auth-brand-card {
        position: relative;
        overflow: hidden;
        min-height: 535px;
        padding: 38px 32px;
        border-radius: 24px;
        border: 1px solid rgba(103, 169, 255, 0.25);
        background:
            radial-gradient(
                circle at 90% 10%,
                rgba(87, 119, 255, 0.25),
                transparent 35%
            ),
            radial-gradient(
                circle at 5% 90%,
                rgba(0, 194, 255, 0.16),
                transparent 35%
            ),
            linear-gradient(
                145deg,
                #172C4B 0%,
                #111D35 55%,
                #201B43 100%
            );
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.25);
        animation: authEnter 0.8s ease-out;
    }

    .auth-brand-card:before {
        content: "";
        position: absolute;
        width: 230px;
        height: 230px;
        top: -110px;
        right: -80px;
        border-radius: 50%;
        background: rgba(66, 153, 255, 0.22);
        filter: blur(35px);
        animation: authFloat 7s ease-in-out infinite alternate;
    }

    .auth-brand-card:after {
        content: "";
        position: absolute;
        width: 170px;
        height: 170px;
        bottom: -90px;
        left: -60px;
        border-radius: 50%;
        background: rgba(139, 92, 246, 0.22);
        filter: blur(30px);
        animation: authFloat 9s ease-in-out infinite alternate-reverse;
    }

    .auth-brand-content {
        position: relative;
        z-index: 2;
    }

    .auth-brand-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 48px;
        height: 48px;
        border-radius: 14px;
        background: rgba(80, 160, 255, 0.16);
        border: 1px solid rgba(117, 184, 255, 0.35);
        color: #8ACBFF;
        font-size: 25px;
        margin-bottom: 28px;
    }

    .auth-brand-name {
        color: #FFFFFF;
        font-size: 30px;
        font-weight: 800;
        letter-spacing: -1px;
    }

    .auth-brand-tagline {
        color: #8EC9FF;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 2px;
        margin-top: 9px;
    }

    .auth-brand-heading {
        color: #FFFFFF;
        font-size: 27px;
        font-weight: 700;
        line-height: 1.35;
        margin-top: 48px;
        max-width: 380px;
    }

    .auth-brand-description {
        color: #B5C7DF;
        font-size: 13px;
        line-height: 1.8;
        margin-top: 14px;
        max-width: 390px;
    }

    .auth-feature {
        display: flex;
        align-items: center;
        gap: 13px;
        margin-top: 25px;
        color: #DCE9FA;
        font-size: 13px;
    }

    .auth-feature-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        width: 34px;
        height: 34px;
        border-radius: 10px;
        background: rgba(76, 155, 255, 0.15);
        border: 1px solid rgba(100, 174, 255, 0.22);
        color: #80C5FF;
        font-size: 16px;
    }

    .auth-brand-footer {
        margin-top: 42px;
        padding-top: 17px;
        border-top: 1px solid rgba(174, 204, 241, 0.16);
        color: #829BB9;
        font-size: 10px;
        letter-spacing: 1px;
    }

    /* Right side heading */
    .auth-heading {
        color: #F4F8FF;
        font-size: 27px;
        font-weight: 800;
        letter-spacing: -0.7px;
        margin-top: 8px;
    }

    .auth-subheading {
        color: #94A9C5;
        font-size: 13px;
        margin-top: 7px;
        margin-bottom: 20px;
    }

    .auth-eyebrow {
        color: #65B9FF;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 2px;
    }

    /* Tabs */
    div[data-testid="stMainBlockContainer"]:has(.auth-marker)
    div[data-testid="stTabs"] {
        margin-top: 12px;
    }

    div[data-testid="stMainBlockContainer"]:has(.auth-marker)
    div[data-testid="stTabs"] button {
        font-size: 13px;
        font-weight: 700;
        padding-bottom: 13px;
    }

    /* Glass-style forms */
    div[data-testid="stMainBlockContainer"]:has(.auth-marker)
    [data-testid="stForm"] {
        background: linear-gradient(
            145deg,
            rgba(25, 43, 68, 0.88),
            rgba(15, 28, 47, 0.92)
        );
        border: 1px solid rgba(94, 139, 193, 0.25);
        border-radius: 18px;
        padding: 24px 22px;
        box-shadow: 0 14px 40px rgba(0, 0, 0, 0.20);
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }

    div[data-testid="stMainBlockContainer"]:has(.auth-marker)
    [data-testid="stForm"]:hover {
        border-color: rgba(82, 163, 255, 0.45);
        box-shadow: 0 16px 45px rgba(26, 104, 190, 0.12);
    }

    /* Input fields */
    div[data-testid="stMainBlockContainer"]:has(.auth-marker)
    div[data-testid="stTextInput"] input {
        background: rgba(7, 17, 32, 0.65);
        border: 1px solid #304663;
        border-radius: 10px;
        color: #F2F7FF;
        min-height: 44px;
        transition: all 0.25s ease;
    }

    div[data-testid="stMainBlockContainer"]:has(.auth-marker)
    div[data-testid="stTextInput"] input:focus {
        border-color: #52A9FF;
        box-shadow: 0 0 0 3px rgba(59, 150, 255, 0.13);
    }

    /* Primary button */
    div[data-testid="stMainBlockContainer"]:has(.auth-marker)
    button[kind="primary"] {
        background: linear-gradient(100deg, #2675EB, #7048E8);
        border: none;
        border-radius: 10px;
        min-height: 45px;
        font-weight: 700;
        box-shadow: 0 5px 18px rgba(53, 111, 235, 0.22);
        transition: all 0.25s ease;
    }

    div[data-testid="stMainBlockContainer"]:has(.auth-marker)
    button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 9px 25px rgba(67, 132, 255, 0.35);
        filter: brightness(1.12);
    }

    /* Role selection */
    div[data-testid="stMainBlockContainer"]:has(.auth-marker)
    [data-testid="stRadio"] {
        background: rgba(10, 24, 43, 0.45);
        border: 1px solid rgba(92, 132, 179, 0.22);
        border-radius: 10px;
        padding: 10px 13px;
    }

    

    @keyframes authEnter {
        from {
            opacity: 0;
            transform: translateY(18px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes authFloat {
        from {
            transform: translate(0, 0) scale(0.9);
        }
        to {
            transform: translate(-18px, 20px) scale(1.15);
        }
    }

    /* Respect reduced-motion preferences */
    @media (prefers-reduced-motion: reduce) {
        .auth-brand-card,
        .auth-brand-card:before,
        .auth-brand-card:after {
            animation: none;
        }
    }

    </style>
    """, unsafe_allow_html=True)

    # Marker used to scope the authentication CSS
    st.markdown(
        '<div class="auth-marker"></div>',
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # TWO-PANEL AUTHENTICATION LAYOUT
    # -----------------------------------------------------

    left_panel, right_panel = st.columns(
        [1.05, 0.95],
        gap="large"
    )

    # LEFT: BRANDING AND PRODUCT HIGHLIGHTS

        # LEFT: BRANDING AND PRODUCT HIGHLIGHTS

    with left_panel:
        st.title("CreditLens AI")

        st.subheader("See beyond the numbers.")

        st.write(
            "AI-powered credit risk intelligence "
            "for smarter lending decisions."
        )

        st.divider()

        st.markdown("### What you can explore")

        st.markdown("📊 **Credit Risk Prediction**")
        st.caption("Understand borrower risk using machine learning.")

        st.markdown("🔍 **Explainable AI**")
        st.caption("Explore the factors influencing risk predictions.")

        st.markdown("🤖 **AI Credit Copilot**")
        st.caption("Ask questions and uncover borrower insights.")
        
    # RIGHT: LOGIN AND REGISTRATION

    with right_panel:

        st.markdown("""
        <div class="auth-eyebrow">WELCOME TO CREDITLENS</div>
        <div class="auth-heading">Your workspace awaits.</div>
        <div class="auth-subheading">
            Sign in or create an account to continue.
        </div>
        """, unsafe_allow_html=True)

        login_tab, register_tab = st.tabs(
            ["Sign in", "Create account"]
        )

        # ---------------- LOGIN ----------------

        with login_tab:

            with st.form("creditlens_login_form"):

                st.markdown("### Welcome back")

                st.caption(
                    "Enter your credentials to access your account."
                )

                login_email = st.text_input(
                    "Email address",
                    key="login_email"
                )

                login_password = st.text_input(
                    "Password",
                    type="password",
                    key="login_password"
                )

                login_submitted = st.form_submit_button(
                    "Sign in to CreditLens",
                    type="primary",
                    use_container_width=True
                )

            if login_submitted:

                if not login_email.strip() or not login_password:
                    st.error(
                        "Please enter your email and password."
                    )

                else:
                    result = login_user(
                        email=login_email,
                        password=login_password
                    )

                    if result is None:
                        st.error("Invalid email or password.")

                    else:
                        establish_login_session(result)
                        st.success("Login successful.")
                        st.rerun()

        # ---------------- REGISTRATION ----------------

        with register_tab:

            with st.form("creditlens_register_form"):

                st.markdown("### Create your account")

                st.caption(
                    "Choose the account type that describes "
                    "your access needs."
                )

                selected_role = st.radio(
                    "Account type",
                    options=["Client", "Employee"],
                    horizontal=True,
                    key="register_role"
                )

                if selected_role == "Client":
                    st.caption(
                        "Client access: view your dashboard, permitted "
                        "analysis reports, and AI Copilot."
                    )
                else:
                    st.caption(
                        "Employee access: full access to CreditLens AI "
                        "including uploads, analytics, and model tools."
                    )

                register_email = st.text_input(
                    "Email address",
                    key="register_email"
                )

                register_password = st.text_input(
                    "Password",
                    type="password",
                    key="register_password"
                )

                confirm_password = st.text_input(
                    "Confirm password",
                    type="password",
                    key="confirm_password"
                )

                register_submitted = st.form_submit_button(
                    "Create my account",
                    type="primary",
                    use_container_width=True
                )

            if register_submitted:

                if not register_email.strip():
                    st.error("Please enter an email address.")

                elif len(register_password) < 8:
                    st.error(
                        "Password must contain at least 8 characters."
                    )

                elif register_password != confirm_password:
                    st.error("Passwords do not match.")

                else:
                    try:
                        role_value = selected_role.lower()

                        register_user(
                            email=register_email,
                            password=register_password,
                            role=role_value
                        )

                        result = login_user(
                            email=register_email,
                            password=register_password
                        )

                        if result is None:
                            st.error(
                                "Account created, but automatic login "
                                "failed. Please sign in."
                            )
                        else:
                            establish_login_session(result)
                            st.success(
                                f"{selected_role} account created "
                                "successfully!"
                            )
                            st.rerun()

                    except ValueError as error:
                        st.error(str(error))

    st.stop()

# =========================================================
# 3. LOAD MODEL AND DEMO DATA
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(
        "models/creditlens_random_forest_tuned.joblib"
    )


@st.cache_data
def load_data():
    return pd.read_csv(
        "data/processed/test_data.csv"
    )


@st.cache_data
def load_model_comparison():
    path = Path("reports/model_comparison.csv")
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


model = load_model()

# Preserve the original evaluation dataset
demo_df = load_data().copy()

# Score the demo dataset
demo_X = demo_df.drop(columns=["DEFAULT_NEXT_MONTH"])
demo_probabilities = model.predict_proba(demo_X)[:, 1]
demo_predictions = (demo_probabilities >= 0.5).astype(int)

demo_df["PREDICTED_DEFAULT_PROBABILITY"] = demo_probabilities
demo_df["PREDICTED_DEFAULT"] = demo_predictions

demo_df["BORROWER_REFERENCE"] = [
    f"CL-{i:05d}" for i in range(len(demo_df))
]

demo_df["RISK_CATEGORY"] = pd.cut(
    demo_probabilities,
    bins=[-0.001, 0.30, 0.60, 1.001],
    labels=["Low", "Medium", "High"]
)

model_comparison = load_model_comparison()

# =========================================================
# DEMO MODEL EVALUATION — KEEP SEPARATE FROM UPLOADS
# =========================================================

demo_y = demo_df["DEFAULT_NEXT_MONTH"].astype(int)

demo_metrics = {
    "accuracy": accuracy_score(demo_y, demo_predictions),
    "precision": precision_score(
        demo_y, demo_predictions, zero_division=0
    ),
    "recall": recall_score(
        demo_y, demo_predictions, zero_division=0
    ),
    "f1": f1_score(
        demo_y, demo_predictions, zero_division=0
    ),
    "roc_auc": roc_auc_score(
        demo_y, demo_probabilities
    ),
    "pr_auc": average_precision_score(
        demo_y, demo_probabilities
    )
}

demo_confusion_matrix = confusion_matrix(
    demo_y, demo_predictions
)
# =========================================================
# 4. SHARED PLOT THEME
# =========================================================

def dark_layout(fig, title=None):
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#111D30",
        plot_bgcolor="#111D30",
        font=dict(
            family="DM Sans",
            color="#DCE7F7"
        ),
        margin=dict(l=30, r=25, t=65, b=35),
        legend=dict(
            bgcolor="rgba(0,0,0,0)"
        ),
        hoverlabel=dict(
            bgcolor="#1E3350",
            font_color="white"
        )
    )
    fig.update_xaxes(
        gridcolor="#253750",
        zerolinecolor="#253750"
    )
    fig.update_yaxes(
        gridcolor="#253750",
        zerolinecolor="#253750"
    )
    return fig



# =========================================================
# 5. SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:

    st.markdown("""
    <div style="padding:15px 0 25px 0;">
        <div style="font-size:27px;font-weight:800;color:#65B9FF;">
            ◈ CREDITLENS
        </div>
        <div style="font-size:11px;color:#849BB9;letter-spacing:2px;">
            AI RISK INTELLIGENCE
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Upload & Analyse",
            "Analysis History",
            "Customer Intelligence",
            "Data Analytics",
            "Model Lab",
            "Explainability",
            "AI Copilot",
            "Profile"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.markdown(f"**Signed in as:** {current_user['email']}")
    st.caption(f"Account type: {current_user['role'].title()}")

    if st.button(
        "Log out",
        key="sidebar_logout",
        use_container_width=True
    ):
        clear_login_session()
        st.rerun()

    st.divider()

    st.markdown("**ACTIVE DATASET**")

    available_datasets = ["Demo dataset"]

    if st.session_state.get("latest_upload_analysis") is not None:
        available_datasets.append("Latest uploaded dataset")

    # Automatically switch to the newly uploaded dataset
    if st.session_state.pop("switch_to_uploaded_dataset", False):
        st.session_state["active_dataset_selector"] = (
            "Latest uploaded dataset"
        )

    active_dataset = st.selectbox(
        "Choose dataset",
        available_datasets,
        key="active_dataset_selector"
    )

    st.divider()

    st.markdown("**SYSTEM INFORMATION**")
    st.caption("Model: Tuned Random Forest")
    st.caption("Dataset: UCI Taiwan Credit Card Default")
    st.caption("Prediction horizon: Next month")
    st.caption("Environment: Portfolio prototype")

    st.divider()

    st.markdown(
        '<span class="status">◉ SYSTEM ONLINE</span>',
        unsafe_allow_html=True
    )

# =========================================================
# 5A. ACTIVE DATASET AND DYNAMIC METRICS
# =========================================================

if active_dataset == "Latest uploaded dataset":

    latest_analysis = st.session_state["latest_upload_analysis"]
    df = latest_analysis["results"].copy()

    dataset_label = "Latest uploaded dataset"

else:

    df = demo_df.copy()
    dataset_label = "UCI Taiwan demo evaluation dataset"


# Check whether actual outcomes are available
has_actuals = (
    "DEFAULT_NEXT_MONTH" in df.columns
    and df["DEFAULT_NEXT_MONTH"].notna().all()
)

# Shared prediction fields
probabilities = df["PREDICTED_DEFAULT_PROBABILITY"].to_numpy()
predictions = df["PREDICTED_DEFAULT"].to_numpy()

total_borrowers = len(df)
predicted_defaults = int(predictions.sum())
average_predicted_risk = float(probabilities.mean())

# Calculate actual performance only when outcomes exist
if has_actuals:

    y = df["DEFAULT_NEXT_MONTH"].astype(int)

    actual_defaults = int(y.sum())
    actual_non_defaults = total_borrowers - actual_defaults
    default_rate = float(y.mean())

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions, zero_division=0)
    recall = recall_score(y, predictions, zero_division=0)
    f1 = f1_score(y, predictions, zero_division=0)

    # AUC requires both outcome classes
    if y.nunique() == 2:
        roc_auc = roc_auc_score(y, probabilities)
        pr_auc = average_precision_score(y, probabilities)
    else:
        roc_auc = None
        pr_auc = None

else:

    y = None
    actual_defaults = None
    actual_non_defaults = None
    default_rate = None

    accuracy = None
    precision = None
    recall = None
    f1 = None
    roc_auc = None
    pr_auc = None

st.sidebar.caption(f"Active: {dataset_label}")
st.sidebar.caption(f"Records: {total_borrowers:,}")


# =========================================================
# 6. HEADER AND ANIMATED TICKER
# =========================================================

st.markdown("""
<div class="hero">
    <div class="hero-content">
        <div class="status">● RISK INTELLIGENCE ACTIVE</div>
        <div class="hero-title">CreditLens AI</div>
        <div class="hero-subtitle">
            Credit Risk Intelligence & Early Warning Platform
            <br>
            Turning borrower data into explainable risk insights.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


ticker_html = """
<div class="ticker">
    <div class="ticker-track">
        <span>PORTFOLIO STATUS <strong>● ANALYTICS ACTIVE</strong></span>
        <span>MODEL <strong>RANDOM FOREST</strong></span>
        <span>ACTIVE DATASET <strong>DATASET_PLACEHOLDER</strong></span>
        <span>RECORDS <strong>RECORDS_PLACEHOLDER</strong></span>
        <span>RISK MONITORING <strong>ENABLED</strong></span>
        <span>MARKET MOVEMENT <strong>+2.0% SIMULATED</strong></span>
        <span>PORTFOLIO MOMENTUM <strong>+1.4% SIMULATED</strong></span>

        <span>PORTFOLIO STATUS <strong>● ANALYTICS ACTIVE</strong></span>
        <span>MODEL <strong>RANDOM FOREST</strong></span>
        <span>ACTIVE DATASET <strong>DATASET_PLACEHOLDER</strong></span>
        <span>RECORDS <strong>RECORDS_PLACEHOLDER</strong></span>
        <span>RISK MONITORING <strong>ENABLED</strong></span>
        <span>MARKET MOVEMENT <strong>+2.0% SIMULATED</strong></span>
        <span>PORTFOLIO MOMENTUM <strong>+1.4% SIMULATED</strong></span>
    </div>
</div>

<style>
body {
    margin: 0;
    background: transparent;
    overflow: hidden;
}
.ticker {
    width: 100%;
    overflow: hidden;
    white-space: nowrap;
}
.ticker-track {
    display: inline-flex;
    gap: 45px;
    width: max-content;
    animation: tickerScroll 35s linear infinite;
}
.ticker-track span {
    color: #AAB9D0;
    font-size: 12px;
    letter-spacing: 1px;
}
.ticker-track strong {
    color: #55D6BE;
    margin-left: 6px;
}
@keyframes tickerScroll {
    from { transform: translateX(0); }
    to { transform: translateX(-50%); }
}
</style>
"""

ticker_html = ticker_html.replace("DATASET_PLACEHOLDER", str(dataset_label))
ticker_html = ticker_html.replace("RECORDS_PLACEHOLDER", f"{total_borrowers:,}")

st.components.v1.html(
    ticker_html,
    height=42,
    scrolling=False
)



# =========================================================
# 7. OVERVIEW
# =========================================================

if page == "Overview":

    st.markdown(
        '<div class="section-label">PORTFOLIO INTELLIGENCE</div>',
        unsafe_allow_html=True
    )

    st.header("Portfolio Overview")

    st.caption(
        f"Active dataset: {dataset_label}. "
        "A consolidated view of borrower risk and model predictions."
    )

    # -----------------------------------------------------
    # PORTFOLIO KPI CARDS
    # -----------------------------------------------------

    if has_actuals:

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("Borrowers Analysed", f"{total_borrowers:,}")
        c2.metric("Predicted Defaults", f"{predicted_defaults:,}")
        c3.metric("Average Predicted Risk", f"{average_predicted_risk:.2%}")
        c4.metric("Historical Defaults", f"{actual_defaults:,}")
        c5.metric("Historical Default Rate", f"{default_rate:.2%}")

    else:

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Borrowers Analysed", f"{total_borrowers:,}")
        c2.metric("Predicted Defaults", f"{predicted_defaults:,}")
        c3.metric("Average Predicted Risk", f"{average_predicted_risk:.2%}")
        c4.metric("Actual Outcomes", "Not supplied")

    st.divider()

    # -----------------------------------------------------
    # RISK DISTRIBUTION AND MODEL HEALTH
    # -----------------------------------------------------

    left, right = st.columns([1.5, 1])

    with left:

        st.subheader("Borrower Risk Distribution")

        risk_counts = (
            df["RISK_CATEGORY"]
            .astype(str)
            .value_counts()
            .reindex(["Low", "Medium", "High"], fill_value=0)
            .reset_index()
        )

        risk_counts.columns = ["Risk Category", "Borrowers"]

        fig = px.pie(
            risk_counts,
            names="Risk Category",
            values="Borrowers",
            hole=0.65,
            color="Risk Category",
            color_discrete_map={
                "Low": "#36CFA0",
                "Medium": "#F5B942",
                "High": "#F16C79"
            }
        )

        fig.update_traces(
            textposition="outside",
            textinfo="percent+label"
        )

        st.plotly_chart(
            dark_layout(fig, "Predicted Risk Bands"),
            use_container_width=True
        )

    with right:

        st.subheader("Portfolio Risk Snapshot")

        risk_summary = (
            df.groupby(
                df["RISK_CATEGORY"].astype(str),
                observed=True
            )["PREDICTED_DEFAULT_PROBABILITY"]
            .agg(["count", "mean"])
            .reindex(["Low", "Medium", "High"])
            .fillna(0)
            .reset_index()
        )

        risk_summary.columns = [
            "Risk Category",
            "Borrowers",
            "Average Probability"
        ]

        st.dataframe(
            risk_summary.style.format({
                "Average Probability": "{:.2%}"
            }),
            use_container_width=True,
            hide_index=True
        )

        if has_actuals:

            st.caption(
                "Historical outcomes are available for retrospective analysis."
            )

        else:

            st.info(
                "This dataset contains predictions only. "
                "Actual model performance cannot be measured without "
                "known outcomes."
            )

    st.divider()

    # -----------------------------------------------------
    # HIGH-RISK WATCHLIST
    # -----------------------------------------------------

    st.subheader("High-Risk Borrower Watchlist")

    high_risk = df.nlargest(
        10,
        "PREDICTED_DEFAULT_PROBABILITY"
    )

    watchlist_columns = [
        "BORROWER_REFERENCE",
        "PREDICTED_DEFAULT_PROBABILITY",
        "RISK_CATEGORY",
        "LIMIT_BAL",
        "AGE",
        "PAY_0"
    ]

    if has_actuals:
        watchlist_columns.append("DEFAULT_NEXT_MONTH")

    watchlist = high_risk[watchlist_columns].rename(
        columns={
            "BORROWER_REFERENCE": "Borrower Reference",
            "PREDICTED_DEFAULT_PROBABILITY": "Predicted Probability",
            "RISK_CATEGORY": "Risk Band",
            "LIMIT_BAL": "Credit Limit",
            "AGE": "Age",
            "PAY_0": "Latest Repayment Status",
            "DEFAULT_NEXT_MONTH": "Historical Outcome"
        }
    )

    st.dataframe(
        watchlist.style.format({
            "Predicted Probability": "{:.2%}",
            "Credit Limit": "{:,.0f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Risk bands use illustrative probability cutoffs of 30% and 60%. "
        "They are not approved bank credit policy thresholds."
    )

    if has_actuals:

        st.info(
            f"Historical outcomes are available in {dataset_label}. "
            "The results support retrospective analysis, not independent "
            "model validation."
        )

    else:

        st.info(
            "This upload has no historical outcome column. "
            "The dashboard displays predicted risk only."
        )


# =========================================================
# 7A. UPLOAD & ANALYSE
# =========================================================

elif page == "Upload & Analyse":

    st.markdown(
        '<div class="section-label">DATA INGESTION & RISK SCORING</div>',
        unsafe_allow_html=True
    )

    st.header("Upload & Analyse")
    st.caption(
        "Upload a borrower portfolio, validate its structure, "
        "and generate AI-assisted credit risk predictions."
    )

    st.info(
        "Supported format: CSV, XLSX and XLS. "
        "The uploaded dataset must contain all 23 CreditLens input columns."
    )

    uploaded_file = st.file_uploader(
        "Upload borrower dataset",
        type=["csv", "xlsx", "xls"],
        key="creditlens_upload"
    )

    if uploaded_file is not None:

        # ---------------------------------------------
        # 1. READ UPLOADED FILE
        # ---------------------------------------------

        try:
            uploaded_df = read_uploaded_file(uploaded_file)

            st.success(
                f"File loaded successfully: {uploaded_file.name}"
            )

        except Exception as error:
            st.error(f"Unable to read file: {error}")
            uploaded_df = None

        if uploaded_df is not None:

            # ---------------------------------------------
            # 2. FILE SUMMARY
            # ---------------------------------------------

            st.divider()
            st.subheader("Dataset Overview")

            info1, info2, info3 = st.columns(3)

            info1.metric(
                "Total Records",
                f"{len(uploaded_df):,}"
            )

            info2.metric(
                "Total Columns",
                f"{len(uploaded_df.columns):,}"
            )

            info3.metric(
                "Duplicate Rows",
                f"{int(uploaded_df.duplicated().sum()):,}"
            )

            # ---------------------------------------------
            # 3. VALIDATION
            # ---------------------------------------------

            st.divider()
            st.subheader("Dataset Validation")

            validation = validate_dataset(uploaded_df)

            if validation["valid"]:
                st.success("Dataset structure and required fields validated.")
            else:
                st.error("Dataset validation failed.")

                for error in validation["errors"]:
                    st.error(error)

            for warning in validation.get("warnings", []):
                st.warning(warning)

            # ---------------------------------------------
            # 4. DATA PREVIEW
            # ---------------------------------------------

            st.divider()
            st.subheader("Data Preview")

            st.dataframe(
                uploaded_df.head(10),
                use_container_width=True,
                hide_index=True
            )

            with st.expander("View all dataset columns"):
                st.write(list(uploaded_df.columns))

            # ---------------------------------------------
            # 5. RUN ANALYSIS
            # ---------------------------------------------

            st.divider()
            st.subheader("Credit Risk Analysis")

            st.caption(
                "The tuned Random Forest model generates next-month "
                "default probabilities using the CreditLens feature pipeline."
            )

            if st.button(
                "Analyse Dataset",
                type="primary",
                disabled=not validation["valid"],
                use_container_width=True,
                key="run_creditlens_analysis"
            ):

                try:
                    with st.spinner(
                        "Engineering features and analysing borrower risk..."
                    ):

                        analysis = analyse_uploaded_dataset(
                            uploaded_df,
                            uploaded_file.name,
                            owner_id=current_user["user_id"]
                        )

                    st.session_state["latest_upload_analysis"] = analysis

                    st.success("Portfolio analysis completed successfully!")

                except Exception as error:
                    st.error(f"Analysis failed: {error}")

    # -------------------------------------------------
    # 6. DISPLAY SAVED ANALYSIS RESULTS
    # -------------------------------------------------

    analysis = st.session_state.get("latest_upload_analysis")

    if analysis is not None:

        results = analysis["results"]

        st.divider()
        st.header("Portfolio Risk Results")

        st.caption(
            f"Analysis Run ID: {analysis['run_id']}"
        )

        total_records = analysis["record_count"]
        predicted_defaults = analysis["predicted_defaults"]
        average_probability = analysis["average_probability"]

        predicted_rate = (
            predicted_defaults / total_records
            if total_records > 0
            else 0
        )

        r1, r2, r3, r4 = st.columns(4)

        r1.metric(
            "Borrowers Analysed",
            f"{total_records:,}"
        )

        r2.metric(
            "Predicted Defaults",
            f"{predicted_defaults:,}"
        )

        r3.metric(
            "Average Default Probability",
            f"{average_probability:.2%}"
        )

        r4.metric(
            "Predicted Default Rate",
            f"{predicted_rate:.2%}"
        )

        if analysis["target_available"]:
            st.info(
                f"Historical default rate in the uploaded data: "
                f"{analysis['actual_default_rate']:.2%}. "
                "This is a descriptive historical outcome, not a "
                "measure of independent model performance."
            )
        else:
            st.info(
                "No historical outcome column was provided. "
                "Results represent model predictions only."
            )

        # ---------------------------------------------
        # 7. RISK DISTRIBUTION
        # ---------------------------------------------

        st.divider()
        st.subheader("Borrower Risk Distribution")

        risk_order = ["Low Risk", "Medium Risk", "High Risk"]

        risk_counts = (
            results["RISK_CATEGORY"]
            .value_counts()
            .reindex(risk_order, fill_value=0)
            .reset_index()
        )

        risk_counts.columns = ["Risk Category", "Borrowers"]

        chart_col, summary_col = st.columns([1.5, 1])

        with chart_col:

            fig = px.pie(
                risk_counts,
                names="Risk Category",
                values="Borrowers",
                hole=0.62,
                color="Risk Category",
                color_discrete_map={
                    "Low Risk": "#36CFA0",
                    "Medium Risk": "#F5B942",
                    "High Risk": "#F06B78"
                }
            )

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#E8EEF8",
                legend_title_text=""
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with summary_col:

            for _, row in risk_counts.iterrows():

                st.metric(
                    row["Risk Category"],
                    f"{int(row['Borrowers']):,} borrowers"
                )

        st.caption(
            "Risk categories use illustrative probability bands. "
            "They are not approved bank credit policy thresholds."
        )

        # ---------------------------------------------
        # 8. BORROWER LEVEL RESULTS
        # ---------------------------------------------

        st.divider()
        st.subheader("Borrower-Level Predictions")

        display_columns = [
            "BORROWER_REFERENCE",
            "PREDICTED_DEFAULT_PROBABILITY",
            "PREDICTED_DEFAULT",
            "RISK_CATEGORY"
        ]

        display_results = results[display_columns].copy()

        display_results["PREDICTED_DEFAULT_PROBABILITY"] = (
            display_results["PREDICTED_DEFAULT_PROBABILITY"]
            .map(lambda value: f"{value:.2%}")
        )

        display_results["PREDICTED_DEFAULT"] = (
            display_results["PREDICTED_DEFAULT"]
            .map({
                0: "No default predicted",
                1: "Default predicted"
            })
        )

        st.dataframe(
            display_results,
            use_container_width=True,
            hide_index=True
        )

        # ---------------------------------------------
        # 9. DOWNLOAD RESULTS
        # ---------------------------------------------

        st.download_button(
            label="Download Complete Prediction Report",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="creditlens_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.caption(
            "The complete report includes the original model features, "
            "engineered features, borrower references and predictions. "
            "The analysis run and borrower predictions are also stored "
            "in the local SQLite database."
        )

# =========================================================
# 8. CUSTOMER INTELLIGENCE
# =========================================================

elif page == "Customer Intelligence":

    st.markdown(
        '<div class="section-label">BORROWER INVESTIGATION</div>',
        unsafe_allow_html=True
    )

    st.header("Customer Intelligence")

    st.caption(
        f"Investigate borrower profiles and risk predictions "
        f"from {dataset_label}."
    )

    col1, col2 = st.columns(2)

    with col1:

        risk_filter = st.multiselect(
            "Filter by predicted risk band",
            ["Low", "Medium", "High"],
            default=["Low", "Medium", "High"]
        )

    with col2:

        min_probability = st.slider(
            "Minimum predicted default probability",
            0, 100, 0
        )

    filtered = df[
        df["RISK_CATEGORY"].astype(str).isin(risk_filter)
        & (
            df["PREDICTED_DEFAULT_PROBABILITY"]
            >= min_probability / 100
        )
    ]

    st.write(
        f"**{len(filtered):,} borrowers** match your filters."
    )

    display_columns = [
        "BORROWER_REFERENCE",
        "PREDICTED_DEFAULT_PROBABILITY",
        "RISK_CATEGORY",
        "LIMIT_BAL",
        "AGE",
        "PAY_0"
    ]

    if has_actuals:
        display_columns.append("DEFAULT_NEXT_MONTH")

    display_df = filtered[display_columns].rename(
        columns={
            "BORROWER_REFERENCE": "Reference",
            "PREDICTED_DEFAULT_PROBABILITY": "Predicted Risk",
            "RISK_CATEGORY": "Risk Band",
            "LIMIT_BAL": "Credit Limit",
            "AGE": "Age",
            "PAY_0": "Latest Repayment Status",
            "DEFAULT_NEXT_MONTH": "Actual Outcome"
        }
    )

    st.dataframe(
        display_df.style.format({
            "Predicted Risk": "{:.2%}",
            "Credit Limit": "{:,.0f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Investigate an Individual Borrower")

    if len(filtered) > 0:

        selected_reference = st.selectbox(
            "Select borrower",
            filtered["BORROWER_REFERENCE"].tolist()
        )

        borrower = df[
            df["BORROWER_REFERENCE"] == selected_reference
        ].iloc[0]

        b1, b2, b3 = st.columns(3)

        b1.metric(
            "Predicted Default Probability",
            f"{borrower['PREDICTED_DEFAULT_PROBABILITY']:.2%}"
        )

        b2.metric(
            "Predicted Risk Band",
            str(borrower["RISK_CATEGORY"])
        )

        if has_actuals:

            outcome_label = (
                "Default"
                if borrower["DEFAULT_NEXT_MONTH"] == 1
                else "No Default"
            )

            b3.metric(
                "Historical Default Outcome",
                outcome_label
            )

        else:

            b3.metric(
                "Historical Outcome",
                "Not available"
            )

        st.markdown("**Borrower repayment profile**")

        profile_cols = [
            "LIMIT_BAL", "AGE",
            "PAY_0", "PAY_2", "PAY_3",
            "PAY_4", "PAY_5", "PAY_6",
            "BILL_AMT1", "BILL_AMT2",
            "PAY_AMT1", "PAY_AMT2",
            "DELAY_MONTH_COUNT_6M",
            "MAX_DELAY_MONTHS_6M",
            "LATEST_DELAY_MONTHS"
        ]

        st.dataframe(
            pd.DataFrame({
                "Attribute": profile_cols,
                "Value": [borrower[c] for c in profile_cols]
            }),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Predicted probabilities are model outputs, not validated "
            "regulatory PD estimates."
        )

    else:

        st.warning(
            "No borrowers match the selected filters. "
            "Adjust the risk band or probability threshold."
        )


# =========================================================
# 9. DATA ANALYTICS
# =========================================================

elif page == "Data Analytics":

    st.markdown(
        '<div class="section-label">DATA EXPLORATION</div>',
        unsafe_allow_html=True
    )

    st.header("Portfolio Data Analytics")

    st.caption(
        f"Explore borrower characteristics and repayment patterns "
        f"from {dataset_label}."
    )

    if not has_actuals:

        st.info(
            "Actual outcomes were not supplied. "
            "Historical default comparisons are disabled; "
            "the charts below analyse borrower and prediction patterns."
        )

    tab1, tab2, tab3, tab4 = st.tabs([
        "Default Analysis",
        "Repayment Behaviour",
        "Customer Demographics",
        "Credit & Payments"
    ])

    # -----------------------------------------------------
    # TAB 1: DEFAULT ANALYSIS
    # -----------------------------------------------------

    with tab1:

        if has_actuals:

            st.subheader("Default vs Non-Default Customers")

            outcome = (
                df["DEFAULT_NEXT_MONTH"]
                .value_counts()
                .rename(index={
                    0: "No Default",
                    1: "Default"
                })
                .reset_index()
            )

            outcome.columns = ["Outcome", "Borrowers"]

            fig = px.bar(
                outcome,
                x="Outcome",
                y="Borrowers",
                color="Outcome",
                color_discrete_map={
                    "Default": "#F16C79",
                    "No Default": "#36CFA0"
                },
                text_auto=True
            )

            st.plotly_chart(
                dark_layout(fig, "Historical Default Distribution"),
                use_container_width=True
            )

            st.subheader("Default Rate by Repayment Status")

            repayment = (
                df.groupby("PAY_0")["DEFAULT_NEXT_MONTH"]
                .mean()
                .reset_index()
            )

            repayment.columns = ["PAY_0", "Default Rate"]

            fig = px.bar(
                repayment,
                x="PAY_0",
                y="Default Rate",
                text="Default Rate"
            )

            fig.update_traces(
                marker_color="#7B8CFF",
                texttemplate="%{text:.1%}",
                textposition="outside"
            )

            st.plotly_chart(
                dark_layout(fig, "Default Rate by Latest Repayment Status"),
                use_container_width=True
            )

        else:

            st.subheader("Predicted Risk Distribution")

            risk_counts = (
                df["RISK_CATEGORY"]
                .astype(str)
                .value_counts()
                .reindex(["Low", "Medium", "High"], fill_value=0)
                .reset_index()
            )

            risk_counts.columns = ["Risk Category", "Borrowers"]

            fig = px.bar(
                risk_counts,
                x="Risk Category",
                y="Borrowers",
                color="Risk Category",
                color_discrete_map={
                    "Low": "#36CFA0",
                    "Medium": "#F5B942",
                    "High": "#F16C79"
                },
                text_auto=True
            )

            st.plotly_chart(
                dark_layout(fig, "Predicted Risk Band Distribution"),
                use_container_width=True
            )

            st.subheader("Predicted Probability Distribution")

            fig = px.histogram(
                df,
                x="PREDICTED_DEFAULT_PROBABILITY",
                nbins=25
            )

            fig.update_traces(marker_color="#59B7FF")

            st.plotly_chart(
                dark_layout(fig, "Predicted Default Probability"),
                use_container_width=True
            )

    # -----------------------------------------------------
    # TAB 2: REPAYMENT BEHAVIOUR
    # -----------------------------------------------------

    with tab2:

        st.subheader("Six-Month Repayment History")

        pay_columns = [
            "PAY_0", "PAY_2", "PAY_3",
            "PAY_4", "PAY_5", "PAY_6"
        ]

        delay_summary = pd.DataFrame({
            "Month": pay_columns,
            "Average Repayment Status": [
                df[c].mean() for c in pay_columns
            ]
        })

        fig = px.line(
            delay_summary,
            x="Month",
            y="Average Repayment Status",
            markers=True
        )

        fig.update_traces(
            line_color="#59B7FF",
            line_width=3,
            marker_size=10
        )

        st.plotly_chart(
            dark_layout(fig, "Average Repayment Status Across Periods"),
            use_container_width=True
        )

        st.subheader("Repayment Delay Count")

        if has_actuals:

            fig = px.histogram(
                df,
                x="DELAY_MONTH_COUNT_6M",
                color="DEFAULT_NEXT_MONTH",
                barmode="group",
                color_discrete_map={
                    0: "#36CFA0",
                    1: "#F16C79"
                },
                labels={
                    "DEFAULT_NEXT_MONTH": "Actual Default"
                }
            )

            chart_title = "Delay Count by Historical Default Outcome"

        else:

            fig = px.histogram(
                df,
                x="DELAY_MONTH_COUNT_6M",
                nbins=7
            )

            fig.update_traces(marker_color="#7B8CFF")

            chart_title = "Delay Count Distribution"

        st.plotly_chart(
            dark_layout(fig, chart_title),
            use_container_width=True
        )

    # -----------------------------------------------------
    # TAB 3: CUSTOMER DEMOGRAPHICS
    # -----------------------------------------------------

    with tab3:

        st.subheader("Borrower Age Distribution")

        if has_actuals:

            fig = px.histogram(
                df,
                x="AGE",
                color="DEFAULT_NEXT_MONTH",
                nbins=25,
                barmode="overlay",
                opacity=0.75,
                color_discrete_map={
                    0: "#36CFA0",
                    1: "#F16C79"
                }
            )

            age_title = "Age Distribution by Default Outcome"

        else:

            fig = px.histogram(
                df,
                x="AGE",
                nbins=25
            )

            fig.update_traces(marker_color="#59B7FF")

            age_title = "Borrower Age Distribution"

        st.plotly_chart(
            dark_layout(fig, age_title),
            use_container_width=True
        )

        st.subheader("Credit Limit Distribution")

        if has_actuals:

            fig = px.box(
                df,
                x="DEFAULT_NEXT_MONTH",
                y="LIMIT_BAL",
                color="DEFAULT_NEXT_MONTH",
                color_discrete_map={
                    0: "#36CFA0",
                    1: "#F16C79"
                }
            )

            credit_title = "Credit Limits by Default Outcome"

        else:

            fig = px.histogram(
                df,
                x="LIMIT_BAL",
                nbins=25
            )

            fig.update_traces(marker_color="#8B78F6")

            credit_title = "Credit Limit Distribution"

        st.plotly_chart(
            dark_layout(fig, credit_title),
            use_container_width=True
        )

    # -----------------------------------------------------
    # TAB 4: CREDIT AND PAYMENTS
    # -----------------------------------------------------

    with tab4:

        st.subheader("Historical Bill Amounts")

        bill_columns = [
            "BILL_AMT1", "BILL_AMT2",
            "BILL_AMT3", "BILL_AMT4",
            "BILL_AMT5", "BILL_AMT6"
        ]

        bill_summary = pd.DataFrame({
            "Period": bill_columns,
            "Average Bill Amount": [
                df[c].mean() for c in bill_columns
            ]
        })

        fig = px.bar(
            bill_summary,
            x="Period",
            y="Average Bill Amount"
        )

        fig.update_traces(marker_color="#8B78F6")

        st.plotly_chart(
            dark_layout(fig, "Average Historical Bill Amount"),
            use_container_width=True
        )

        st.subheader("Historical Payments")

        payment_columns = [
            "PAY_AMT1", "PAY_AMT2",
            "PAY_AMT3", "PAY_AMT4",
            "PAY_AMT5", "PAY_AMT6"
        ]

        payment_summary = pd.DataFrame({
            "Period": payment_columns,
            "Average Payment": [
                df[c].mean() for c in payment_columns
            ]
        })

        fig = px.bar(
            payment_summary,
            x="Period",
            y="Average Payment"
        )

        fig.update_traces(marker_color="#36CFA0")

        st.plotly_chart(
            dark_layout(fig, "Average Historical Payment Amount"),
            use_container_width=True
        )



# =========================================================
# 10. MODEL LAB
# =========================================================

elif page == "Model Lab":

    st.markdown(
        '<div class="section-label">MODEL GOVERNANCE & PERFORMANCE</div>',
        unsafe_allow_html=True
    )
    st.header("Model Lab")

    st.caption(
        "Understand the models evaluated, their performance "
        "and the final candidate selected for this prototype."
    )

    st.subheader("Model Development Journey")

    st.markdown("""
    <div class="insight-card">
        <h3>01 · Logistic Regression</h3>
        <p>Baseline classification model used to establish an initial
        performance benchmark.</p>
        <h3>02 · Decision Tree</h3>
        <p>Introduced non-linear decision rules and interpretable
        branching logic.</p>
        <h3>03 · Random Forest</h3>
        <p>Combined multiple decision trees to improve predictive
        performance and capture complex relationships.</p>
        <h3>04 · Hyperparameter Tuning</h3>
        <p>Used randomized search with three-fold stratified
        cross-validation and average precision scoring.</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.subheader("Model Comparison")

    if not model_comparison.empty:
        st.dataframe(
            model_comparison,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "These are the earlier internal validation results "
            "from model_comparison.csv."
        )
    else:
        st.warning("Model comparison report was not found.")

    st.divider()

    st.subheader("Current Tuned Random Forest")

    mc1, mc2, mc3 = st.columns(3)

    mc1.metric("Trees", "100")
    mc2.metric("Maximum Depth", "8")
    mc3.metric("Cross-validation PR-AUC", "0.5631")

    st.markdown("**Selected hyperparameters**")

    st.code("""
n_estimators      = 100
max_depth         = 8
min_samples_split = 20
min_samples_leaf  = 10
max_features      = sqrt

Cross-validation:
StratifiedKFold(n_splits=3)
Scoring = Average Precision
    """)

    st.divider()

    st.subheader("Exploratory Test Evaluation")

    metrics = pd.DataFrame({
        "Metric": [
            "Accuracy", "Precision", "Recall",
            "F1 Score", "ROC-AUC", "PR-AUC"
        ],
        "Score": [
            demo_metrics["accuracy"],
            demo_metrics["precision"],
            demo_metrics["recall"],
            demo_metrics["f1"],
            demo_metrics["roc_auc"],
            demo_metrics["pr_auc"]
        ]
    })

    st.dataframe(
        metrics.style.format({"Score": "{:.4f}"}),
        use_container_width=True,
        hide_index=True
    )

    cm = demo_confusion_matrix

    fig = px.imshow(
        cm,
        text_auto=True,
        x=["Predicted Non-default", "Predicted Default"],
        y=["Actual Non-default", "Actual Default"],
        color_continuous_scale="Blues",
        aspect="auto"
    )

    st.plotly_chart(
        dark_layout(fig, "Demo Evaluation Confusion Matrix"),
        use_container_width=True
    )

    st.caption(
        "Model evaluation is based on the original 5,993-record demo "
        "test partition, regardless of the active uploaded dataset."
    )

    st.warning(
        "Evaluation caveat: the test partition was inspected during "
        "development and threshold exploration. These metrics are "
        "exploratory, not a final untouched benchmark. The model has "
        "not undergone independent validation or probability calibration."
    )


# =========================================================
# 11. EXPLAINABILITY
# =========================================================

elif page == "Explainability":

    st.info(
        "The SHAP reports currently displayed are generated from "
        "the original UCI demo evaluation dataset. They do not "
        "represent explanations for the active uploaded dataset."
    )
    st.markdown('<div class="section-label">MODEL TRANSPARENCY</div>',
                unsafe_allow_html=True)
    st.header("Explainable AI")
    st.caption(
        "Understand which variables influence the model's "
        "portfolio-level and individual predictions."
    )

    tab1, tab2, tab3 = st.tabs([
        "Global SHAP",
        "Feature Importance",
        "Individual Borrower"
    ])

    with tab1:

        path = Path("reports/shap_summary.png")

        if path.exists():
            st.image(
                str(path),
                use_container_width=True
            )
            st.info(
                "Red represents higher feature values and blue represents "
                "lower values. SHAP values to the right push the model "
                "output towards default; values to the left push it away."
            )
        else:
            st.warning("SHAP summary not found.")

    with tab2:

        path = Path("reports/shap_feature_importance.png")

        if path.exists():
            st.image(
                str(path),
                use_container_width=True
            )

        csv_path = Path(
            "reports/random_forest_feature_importance.csv"
        )

        if csv_path.exists():
            importance = pd.read_csv(csv_path)
            st.dataframe(
                importance.head(20),
                use_container_width=True,
                hide_index=True
            )

    with tab3:

        path = Path("reports/individual_borrower_shap.png")

        if path.exists():
            st.image(
                str(path),
                use_container_width=True
            )

        contribution_path = Path(
            "reports/individual_borrower_contributions.csv"
        )

        if contribution_path.exists():

            contributions = pd.read_csv(contribution_path)

            st.subheader("Factors increasing model output")

            increasing = contributions[
                contributions["SHAP_CONTRIBUTION"] > 0
            ].sort_values(
                "SHAP_CONTRIBUTION",
                ascending=False
            )

            st.dataframe(
                increasing.head(10),
                use_container_width=True,
                hide_index=True
            )

            st.subheader("Factors reducing model output")

            reducing = contributions[
                contributions["SHAP_CONTRIBUTION"] < 0
            ].sort_values(
                "SHAP_CONTRIBUTION"
            )

            st.dataframe(
                reducing.head(10),
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "These saved individual explanations refer to the "
                "previously selected borrower, dataset row 5775."
            )


# =========================================================
# 12. AI COPILOT PLACEHOLDER
# =========================================================


elif page == "AI Copilot":

    import sys
    from pathlib import Path

    # Make the src folder accessible to Streamlit
    SRC_DIR = Path(__file__).resolve().parent / "src"

    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    from borrower_context_service import (
        get_borrower_context,
        data as copilot_data
    )

    from ai_copilot_service import generate_copilot_response

    st.markdown(
        '<div class="section-label">GENERATIVE AI INTELLIGENCE</div>',
        unsafe_allow_html=True
    )

    st.header("CreditLens AI Copilot")

    st.markdown("""
    <div class="hero">
        <div class="hero-content">
            <div class="hero-title">Your Credit Risk Assistant</div>
            <div class="hero-subtitle">
                Explore borrower risk, understand model predictions,
                and investigate repayment behaviour using AI.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "AI explanations are grounded in CreditLens model outputs "
        "and SHAP evidence. This is a decision-support prototype."
    )

    # --------------------------------------------------
    # 1. BORROWER SELECTION
    # --------------------------------------------------

    st.markdown("### Select a borrower")

    borrower_positions = list(range(len(copilot_data)))

    selected_position = st.selectbox(
        "Choose borrower",
        options=borrower_positions,
        format_func=lambda x: (
            f"DEMO-{x + 1:04d} | Evaluation record {x + 1}"
        ),
        key="copilot_borrower_selection"
    )

    # --------------------------------------------------
    # 2. RETRIEVE BORROWER CONTEXT
    # --------------------------------------------------

    if "copilot_context_cache" not in st.session_state:
        st.session_state.copilot_context_cache = {}

    if selected_position not in st.session_state.copilot_context_cache:

        with st.spinner("Analysing borrower and calculating SHAP evidence..."):

            st.session_state.copilot_context_cache[selected_position] = (
                get_borrower_context(selected_position)
            )

    borrower_context = (
        st.session_state.copilot_context_cache[selected_position]
    )

    probability = borrower_context["default_probability"]

    # Illustrative display bands only, not lending thresholds
    if probability < 0.30:
        display_band = "Lower"
    elif probability < 0.60:
        display_band = "Moderate"
    else:
        display_band = "Higher"

    # --------------------------------------------------
    # 3. TWO-PANEL LAYOUT
    # --------------------------------------------------

    left_panel, right_panel = st.columns([0.95, 1.35], gap="large")

    # ==================================================
    # LEFT: BORROWER INTELLIGENCE
    # ==================================================

    with left_panel:

        st.markdown("### Borrower Intelligence")

        st.metric(
            "Predicted next-month default probability",
            f"{probability:.2%}"
        )

        st.markdown(
            f"**Illustrative probability band:** `{display_band}`"
        )

        st.caption(
            "Display bands are for exploration only. "
            "They are not validated lending decision thresholds."
        )

        st.divider()

        st.markdown("#### Borrower Financial Profile")

        profile_df = pd.DataFrame(
            list(borrower_context["profile"].items()),
            columns=["Feature", "Value"]
        )

        st.dataframe(
            profile_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.markdown("#### Model Explanation — SHAP")

        shap_df = borrower_context["shap_contributions"].copy()

        shap_df["absolute_contribution"] = (
            shap_df["shap_contribution"].abs()
        )

        top_shap = (
            shap_df.sort_values(
                "absolute_contribution",
                ascending=False
            )
            .head(8)
            .sort_values("shap_contribution")
        )

        st.bar_chart(
            top_shap.set_index("feature")["shap_contribution"],
            horizontal=True
        )

        st.caption(
            "Positive values push the model output toward default; "
            "negative values push it away. These are SHAP contributions, "
            "not percentages or causal effects."
        )

        with st.expander("View complete SHAP contribution table"):

            st.dataframe(
                shap_df[
                    ["feature", "shap_contribution"]
                ].sort_values(
                    "shap_contribution",
                    ascending=False
                ),
                use_container_width=True,
                hide_index=True
            )

    # ==================================================
    # RIGHT: AI CHAT INTERFACE
    # ==================================================

    with right_panel:

        st.markdown("### Ask CreditLens AI")

        st.markdown("""
        <div style="
            padding:14px;
            border-radius:12px;
            background:rgba(80,120,190,0.12);
            border:1px solid rgba(100,150,220,0.25);
            margin-bottom:15px;
        ">
            <strong>Active borrower:</strong>
            Your selected borrower is automatically included
            in the AI analysis.
        </div>
        """, unsafe_allow_html=True)

        # Keep separate conversations for each borrower
        if "creditlens_chat_history" not in st.session_state:
            st.session_state.creditlens_chat_history = {}

        borrower_key = selected_position

        if borrower_key not in st.session_state.creditlens_chat_history:
            st.session_state.creditlens_chat_history[borrower_key] = []

        chat_history = st.session_state.creditlens_chat_history[borrower_key]

        # Suggested questions
        suggested_questions = [
            "Why does this borrower have this default probability?",
            "What are the strongest factors increasing this borrower's risk?",
            "Which factors are reducing the predicted risk?",
            "Explain the repayment behaviour of this borrower.",
            "What should a credit analyst investigate further?"
        ]

        # Display previous conversation
        chat_container = st.container(height=390)

        with chat_container:

            if not chat_history:

                st.markdown("""
                **Welcome to CreditLens AI.**

                I can help you investigate this borrower's
                model prediction and supporting evidence.

                Start with one of the suggested questions below.
                """)

            for message in chat_history:

                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Question input
        with st.form(
            key=f"copilot_form_{selected_position}",
            clear_on_submit=True
        ):

            selected_question = st.selectbox(
                "Suggested questions",
                ["Choose a question..."] + suggested_questions
            )

            custom_question = st.text_area(
                "Your question",
                placeholder="Ask something about this borrower...",
                height=90
            )

            submitted = st.form_submit_button(
                "Generate Insight",
                use_container_width=True,
                type="primary"
            )

        if submitted:

            question = (
                custom_question.strip()
                if custom_question.strip()
                else selected_question
            )

            if question == "Choose a question..." or not question:

                st.warning("Please enter or select a question.")

            else:

                chat_history.append({
                    "role": "user",
                    "content": question
                })

                with st.spinner("CreditLens AI is analysing the evidence..."):

                    answer = generate_copilot_response(
                        question=question,
                        context=borrower_context["context"]
                    )

                chat_history.append({
                    "role": "assistant",
                    "content": answer
                })

                st.rerun()

        if chat_history:

            if st.button(
                "Clear this borrower's conversation",
                key=f"clear_chat_{selected_position}"
            ):

                st.session_state.creditlens_chat_history[borrower_key] = []
                st.rerun()

    # --------------------------------------------------
    # 4. IMPORTANT MODEL LIMITATIONS
    # --------------------------------------------------

    st.divider()

    with st.expander("About this AI assessment and its limitations"):

        st.markdown("""
        - The underlying dataset contains historical Taiwan credit
          card records.
        - The model predicts next-month default.
        - Model probability is not a validated regulatory Probability
          of Default (PD).
        - SHAP explains model behaviour, not causality.
        - The selected record is a historical evaluation example,
          not a live customer.
        - AI output is decision support only. It must not independently
          approve or reject credit.
        """)


elif page == "Analysis History":

    st.markdown("## Analysis History")
    st.caption(
        "Review previously uploaded datasets and retrieve their saved credit risk assessments."
    )

    # Retrieve all saved analysis runs
    history = get_user_analysis_history(
    current_user["user_id"]
)

    if not history:

        st.info(
            "No saved analyses found. Upload a dataset and run an analysis first."
        )

    else:

        history_df = pd.DataFrame(history)

        # Display historical analysis overview
        st.markdown("### Previously analysed datasets")

        display_columns = [
            "original_filename",
            "uploaded_at",
            "record_count",
            "predicted_defaults",
            "average_probability",
            "model_version"
        ]

        st.dataframe(
            history_df[display_columns],
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # Select a historical analysis
        st.markdown("### Retrieve an analysis")

        history_by_id = {
            item["run_id"]: item
            for item in history
        }

        run_ids = list(history_by_id.keys())

        selected_run_id = st.selectbox(
            "Select an analysis to inspect",
            options=run_ids,
            format_func=lambda run_id: (
                f"{history_by_id[run_id]['original_filename']} "
                f"— {history_by_id[run_id]['uploaded_at'][:19].replace('T', ' ')}"
            )
        )

        # Retrieve selected analysis
        analysis = get_user_analysis_run(
        selected_run_id,
        current_user["user_id"]
)

        if analysis:

            st.markdown("### Analysis summary")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Total records",
                    f"{analysis['record_count']:,}"
                )

            with col2:
                st.metric(
                    "Predicted defaults",
                    f"{analysis['predicted_defaults']:,}"
                )

            with col3:
                st.metric(
                    "Average predicted risk",
                    f"{analysis['average_probability']:.2%}"
                )

            with col4:
                st.metric(
                    "Model version",
                    analysis["model_version"]
                )

            st.caption(
                f"Analysis ID: {analysis['run_id']}"
            )

            st.caption(
                f"Original dataset: {analysis['original_filename']}"
            )

            st.divider()

            # Retrieve borrower-level predictions from SQLite
            st.markdown("### Borrower-level predictions")

            predictions = get_user_borrower_predictions(
    selected_run_id,
    current_user["user_id"]
)

            if not predictions.empty:

                st.dataframe(
                    predictions,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info("No borrower-level predictions found for this run.")

            # Retrieve complete saved result CSV
            results = load_analysis_results(
    selected_run_id,
    current_user["user_id"]
)

            st.download_button(
                label="Download complete analysis results",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name=analysis["result_filename"],
                mime="text/csv",
                key=f"download_history_{selected_run_id}"
            )


elif page == "Profile":

    st.markdown("## My Profile")
    st.caption(
        "Manage your CreditLens AI account and review your account information."
    )

    st.divider()

    # Account information
    st.markdown("### Account information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Registered email**")
        st.write(current_user["email"])

    with col2:
        st.markdown("**Account created**")
        st.write(current_user["created_at"])

    st.markdown("**Account ID**")
    st.code(current_user["user_id"])

    st.divider()

    # Security information
    st.markdown("### Security")

    st.success("Your account is authenticated.")
    st.caption(
        "Your password is securely stored as a hash and is never displayed."
    )

    st.divider()

    # Account preferences placeholder
    st.markdown("### Preferences")
    st.info(
        "Profile preferences and notification settings will be added "
        "in a future update."
    )

    st.divider()

    # Logout
    st.markdown("### Sign out")

    if st.button(
        "Log out",
        key="profile_logout",
        type="primary"
    ):
        try:
            cookie_controller.remove(
                AUTH_COOKIE_NAME,
                path="/",
                same_site="lax",
                secure=False
            )
        except KeyError:
            pass

        st.session_state.pop("access_token", None)
        st.session_state.pop("auth_user_id", None)
        st.session_state.pop("auth_email", None)

        st.session_state["logout_pending"] = True

        st.success("Your session has been cleared.")

    if st.session_state.get("logout_pending"):

        st.info("Click below to return to the login page.")

        if st.button("Continue to login", key="continue_to_login"):
            st.session_state.pop("logout_pending", None)
            st.rerun()

# =========================================================
# FLOATING BANKING FUN FACT POPUP
# =========================================================

if "show_funfact" not in st.session_state:
    st.session_state.show_funfact = True

if st.session_state.show_funfact:

    st.markdown("""
    <div class="funfact-popup">
        <div class="funfact-header">
            <div>
                <span class="funfact-icon">🤖</span>
                <span class="funfact-title">BANKING INSIGHT</span>
            </div>
        </div>
        <div class="funfact-text">
            <strong>Did you know?</strong><br>
            A borrower's repayment history can reveal early
            signs of credit stress even before a payment
            becomes overdue.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("×", key="close_funfact", help="Close banking insight"):
        st.session_state.show_funfact = False
        st.rerun()

# =========================================================
# 13. FOOTER
# =========================================================

st.divider()

st.markdown("""
<div style="text-align:center;padding:15px;color:#7187A5;font-size:11px;">
    <strong style="color:#65B9FF;">CREDITLENS AI</strong>
    &nbsp; | &nbsp; Credit Risk Intelligence Platform
    <br><br>
    Portfolio prototype · UCI Taiwan Credit Card Default Dataset
    <br>
    Next-month credit-card default prediction · Decision support only
</div>
""", unsafe_allow_html=True)
