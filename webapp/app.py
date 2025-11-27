import streamlit as st
import base64
from pathlib import Path
import plotly.graph_objects as go
import numpy as np
from utils.styles import apply_cognivise_theme
import utils.bloom_analyzer_complete as bloom_analyzer_complete
from utils.pdf_extract import extract_questions_from_pdf, extract_all_text_from_pdf
from utils.image_extract import extract_questions_from_image, extract_all_text_from_image
from utils.rag_exam_generator import RAGExamGenerator, search_web_content
from utils.auth import (
    init_auth, verify_user, register_user, save_user_history, get_user_history,
    save_user_book, get_user_books, get_user_book_path
)
import tempfile
import os
import json
import shutil

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)  # override=True ensures .env values take precedence
        print(f"Loaded .env file from {env_path}")
        
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            print(f"GEMINI_API_KEY found: {gemini_key[:10]}...")
        
    else:
        print(f".env file not found at {env_path}")
except ImportError:
    # python-dotenv not installed, skip (environment variables must be set manually)
    print("python-dotenv not installed. Install it with: pip install python-dotenv")
except Exception as e:
    print(f"Error loading .env file: {e}")


def get_logo_png_data_uri(size="40"):
    """Load and return ONLY cv-logo.png as a data URI - no other logos."""
    static_dir = Path(__file__).parent / "static"
    # ONLY use cv-logo.png - this is the user's logo
    logo_path = static_dir / "cv-logo.png"
    
    if logo_path.exists():
        try:
            with logo_path.open("rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/png;base64,{data}"
        except Exception as e:
            # Log error but don't show fallback
            return ""
    
    # Return empty if cv-logo.png doesn't exist - no fallbacks
    return ""


def get_logo_html(size="40", variant="default"):
    """Get logo HTML - ALWAYS uses PNG cv-logo.png, no fallbacks."""
    # Force use of cv-logo.png - this is the ONLY logo we want
    png_data_uri = get_logo_png_data_uri(size=size)
    if png_data_uri:
        # cv-logo.png dimensions: 896x2048, aspect ratio ~0.4375 (width/height)
        # Calculate height to maintain proper aspect ratio
        # For navigation: use width-based sizing, but cap height for reasonable display
        width_px = int(size)
        # Original ratio: 896/2048 = 0.4375, so height = width / 0.4375 = width * 2.286
        calculated_height = int(width_px * 2048 / 896)
        
        # For navigation bars, cap the height at a reasonable maximum
        # This ensures the logo fits well in the navigation bar
        max_height = 120 if width_px >= 150 else (100 if width_px >= 100 else (90 if width_px >= 60 else (60 if width_px <= 50 else int(width_px * 1.5))))
        height = min(calculated_height, max_height)
        
        return f'<img src="{png_data_uri}" width="{width_px}" height="{height}" style="width: {width_px}px; height: {height}px; max-height: {max_height}px; object-fit: contain; display: block; transform: none !important; -webkit-transform: none !important; -moz-transform: none !important; -ms-transform: none !important; -o-transform: none !important;" alt="Cognivise Logo" />'
    
    # If PNG is not found, return empty string instead of fallback
    # This ensures no unwanted icons appear
    return ""


def get_hero_image_data_uri() -> str:
    """Return the hero image as a data URI so it always renders correctly in Streamlit HTML."""
    img_path = Path(__file__).parent / "static" / "hero-image.png"
    try:
        with img_path.open("rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{data}"  # Fixed: was returning jpeg, should be png
    except FileNotFoundError:
        # Fallback to empty string if the file is missing; the img tag will just not render properly.
        return ""

def get_landing_bg_image_data_uri() -> str:
    """Return the landing page background image as a data URI."""
    # Try to find the background image - check common names
    static_dir = Path(__file__).parent / "static"
    possible_names = ["landing-bg.jpg", "hero-image.png"]
    
    for name in possible_names:
        img_path = static_dir / name
        if img_path.exists():
            try:
                with img_path.open("rb") as f:
                    data = base64.b64encode(f.read()).decode("utf-8")
                ext = name.split('.')[-1].lower()
                mime_type = f"image/{ext}" if ext != "jpg" else "image/jpeg"
                return f"data:{mime_type};base64,{data}"
            except Exception:
                continue
    
    # Fallback: return empty string
    return ""

def get_auth_bg_image_data_uri() -> str:
    """Return the authentication page background image as a data URI."""
    static_dir = Path(__file__).parent / "static"
    possible_names = ["auth-bg.png", "auth-background.png", "signin-bg.png", "signin-background.png"]
    
    for name in possible_names:
        img_path = static_dir / name
        if img_path.exists():
            try:
                with img_path.open("rb") as f:
                    data = base64.b64encode(f.read()).decode("utf-8")
                ext = name.split('.')[-1].lower()
                mime_type = f"image/{ext}" if ext != "jpg" else "image/jpeg"
                return f"data:{mime_type};base64,{data}"
            except Exception:
                continue
    
    # Fallback: return empty string
    return ""


def create_bloom_chart(result):
    """Create an interactive brand-styled bar chart showing Actual vs Ideal percentages for all Bloom categories."""
    levels = bloom_analyzer_complete.BLOOM_LEVELS
    actual = [result['comparison'][l]['actual'] for l in levels]
    ideal = [result['comparison'][l]['ideal'] for l in levels]
    
    # Create figure with brand styling
    fig = go.Figure()
    
    # Ideal bars - purple-black gradient (brand color)
    fig.add_trace(go.Bar(
        name='Ideal',
        x=levels,
        y=ideal,
        marker=dict(
            color='rgba(111, 91, 255, 0.8)',
            line=dict(color='rgba(111, 91, 255, 1)', width=1.5)
        ),
        text=[f"{val:.1f}%" for val in ideal],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Ideal: %{y:.1f}%<extra></extra>',
    ))
    
    # Actual bars - contrasting color but brand-aligned (lighter purple/cyan)
    fig.add_trace(go.Bar(
        name='Actual',
        x=levels,
        y=actual,
        marker=dict(
            color='rgba(194, 111, 255, 0.8)',
            line=dict(color='rgba(194, 111, 255, 1)', width=1.5)
        ),
        text=[f"{val:.1f}%" for val in actual],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Actual: %{y:.1f}%<extra></extra>',
    ))
    
    # Update layout with brand styling
    fig.update_layout(
        title={
            'text': 'Bloom Taxonomy Distribution: Actual vs Ideal',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#ffffff', 'family': "'Graphik', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Helvetica, Arial, sans-serif"}
        },
        xaxis=dict(
            title=dict(
                text='Bloom Category',
                font=dict(size=14, color='#b29cff')
            ),
            tickfont=dict(size=12, color='#dadffc'),
            gridcolor='rgba(255, 255, 255, 0.1)',
            linecolor='rgba(255, 255, 255, 0.2)',
        ),
        yaxis=dict(
            title=dict(
                text='Percentage (%)',
                font=dict(size=14, color='#b29cff')
            ),
            tickfont=dict(size=12, color='#dadffc'),
            gridcolor='rgba(255, 255, 255, 0.1)',
            linecolor='rgba(255, 255, 255, 0.2)',
            range=[0, max(max(actual), max(ideal)) * 1.2] if max(max(actual), max(ideal)) > 0 else [0, 100]
        ),
        barmode='group',
        bargap=0.3,
        bargroupgap=0.1,
        plot_bgcolor='rgba(9, 13, 31, 0.5)',
        paper_bgcolor='rgba(9, 13, 31, 0.3)',
        font=dict(family="'Graphik', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Helvetica, Arial, sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12, color='#dadffc'),
            bgcolor='rgba(0, 0, 0, 0)',
            bordercolor='rgba(255, 255, 255, 0.2)',
            borderwidth=1
        ),
        height=500,
        margin=dict(l=50, r=50, t=80, b=80),
        hovermode='x unified',
    )
    
    return fig


def init_session():
    # Initialize auth system
    init_auth()
    
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "username" not in st.session_state:
        st.session_state["username"] = None
    
    # Handle sign out
    query_params = st.query_params
    if query_params.get("signout") == "true":
        st.session_state["authenticated"] = False
        st.session_state["username"] = None
        # Remove signout param
        try:
            del st.query_params["signout"]
        except:
            pass
    
    # Check query params FIRST - this handles URL-based navigation from HTML buttons
    raw_page = query_params.get("page")
    if isinstance(raw_page, list):
        raw_page = raw_page[0] if raw_page else None
    
    # Handle dashboard_page query parameter
    raw_dashboard_page = query_params.get("dashboard_page")
    if isinstance(raw_dashboard_page, list):
        raw_dashboard_page = raw_dashboard_page[0] if raw_dashboard_page else None
    if raw_dashboard_page:
        st.session_state["dashboard_page"] = raw_dashboard_page
    
    # If query param exists, use it (this handles HTML button clicks)
    if raw_page:
        current_page = st.session_state.get("page", "landing")
        # Always update dashboard_page if provided in query params (even if already on dashboard)
        if raw_dashboard_page:
            st.session_state["dashboard_page"] = raw_dashboard_page
        if current_page != raw_page:
            st.session_state["page"] = raw_page
            # Force rerun to navigate
            st.rerun()
        elif raw_page == "dashboard" and raw_dashboard_page and st.session_state.get("dashboard_page") != raw_dashboard_page:
            # Already on dashboard but dashboard_page changed - rerun to update
            st.rerun()
    else:
        # Initialize page if not set and no query param
        if "page" not in st.session_state:
            st.session_state["page"] = "landing"


def set_page(name: str):
    st.session_state["page"] = name
    # Update query params for URL consistency
    try:
        st.query_params["page"] = name
    except:
        pass  # Ignore if query params not available
    st.rerun()


def top_nav(show_on_all_pages=True):
    """Display top navigation bar. Standard practice is to show on all pages for easy navigation."""
    # Check if user is authenticated
    is_authenticated = st.session_state.get("authenticated", False)
    username = st.session_state.get("username", None)
    
    if is_authenticated and username:
        # Show user menu when authenticated
        logo_html = get_logo_html(size="200", variant="default")
        nav_content = f"""
        <header class="header top-nav-shell w-full z-50 relative">
          <div class="logo" onclick="window.location.href='?page=landing'" style="cursor: pointer;">
            <div class="logo-icon">{logo_html}</div>
            </div>
          <nav class="nav-menu">
            <a href="?page=dashboard&dashboard_page=analyze" onclick="window.location.href=this.href; return false;">Analyze Exam</a>
            <a href="?page=dashboard&dashboard_page=generate_exam" onclick="window.location.href=this.href; return false;">Generate Exam</a>
            <a href="?page=about">About</a>
          </nav>
          <div class="header-actions">
            <span style="color: rgba(255,255,255,0.7); padding: 0 16px; font-size: 15px; font-weight: 400;">Welcome, {username}</span>
            <button class="btn-login" onclick="window.location.href='?page=landing&signout=true'">Sign Out</button>
          </div>
        </header>
        """
    else:
        # Show sign in button when not authenticated
        logo_html = get_logo_html(size="200", variant="default")
        nav_content = f"""
        <header class="header top-nav-shell w-full z-50 relative">
          <div class="logo" onclick="window.location.href='?page=landing'" style="cursor: pointer;">
            <div class="logo-icon">{logo_html}</div>
            </div>
          <nav class="nav-menu">
            <a href="?page=dashboard&dashboard_page=analyze" onclick="window.location.href=this.href; return false;">Analyze Exam</a>
            <a href="?page=dashboard&dashboard_page=generate_exam" onclick="window.location.href=this.href; return false;">Generate Exam</a>
            <a href="?page=about">About</a>
          </nav>
          <div class="header-actions">
            <a href="?page=signin" class="btn-login" id="signin-btn-nav">Sign In</a>
          </div>
          <script>
          document.addEventListener('DOMContentLoaded', function() {{
              var signinBtn = document.getElementById('signin-btn-nav');
              if (signinBtn) {{
                  signinBtn.addEventListener('click', function(e) {{
                      e.preventDefault();
                      window.location.href = '?page=signin';
                  }});
              }}
          }});
          </script>
        </header>
        """
    
    st.markdown(nav_content, unsafe_allow_html=True)


def footer():
    """Display footer with contact info and copyright."""
    st.markdown("""
    <footer style="margin-top: 4rem; padding: 2rem; text-align: center; border-top: 2px solid rgba(114,9,183,0.3);">
        <div style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">
            <p style="margin: 0.5rem 0;">
                <a href="mailto:contact@cognivise.com" style="color: #4CC9F0; text-decoration: none; margin: 0 1rem;">Email</a>
                <span style="color: rgba(255,255,255,0.3);">|</span>
                <a href="https://github.com/zobiahussain/cognivise-blooms-taxonomy" target="_blank" style="color: #4CC9F0; text-decoration: none; margin: 0 1rem;">GitHub</a>
            </p>
            <p style="margin: 0.5rem 0; color: rgba(255,255,255,0.5);">© Cognivise 2025</p>
        </div>
    </footer>
    """, unsafe_allow_html=True)


def page_landing():
    bg_img_src = get_landing_bg_image_data_uri()
    
    # Inject CSS for new background image and left-aligned content
    st.markdown(f"""
    <style>
    /* Ensure hero section takes full viewport */
    .hero-shell {{
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        background: #0A0519 !important;
    }}
    
    /* New background image */
    .hero-bg-new {{
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        bottom: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        background: #0A0519;
        background-image: url('{bg_img_src}');
        background-size: cover !important;
        background-position: center !important;
        background-repeat: no-repeat !important;
        z-index: 1;
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # Check if user is authenticated for landing page navbar
    is_authenticated = st.session_state.get("authenticated", False)
    username = st.session_state.get("username", None)
    
    if is_authenticated and username:
        logo_html = get_logo_html(size="200", variant="default")
        landing_nav = f"""
        <header class="header top-nav-shell w-full z-50 relative">
          <div class="logo" onclick="window.location.href='?page=landing'" style="cursor: pointer;">
            <div class="logo-icon">{logo_html}</div>
          </div>
          <nav class="nav-menu">
            <a href="?page=dashboard&dashboard_page=analyze" onclick="window.location.href=this.href; return false;">Analyze Exam</a>
            <a href="?page=dashboard&dashboard_page=generate_exam" onclick="window.location.href=this.href; return false;">Generate Exam</a>
            <a href="?page=about">About</a>
          </nav>
          <div class="header-actions">
            <span style="color: rgba(255,255,255,0.7); padding: 0 16px; font-size: 15px; font-weight: 400;">Welcome, {username}</span>
            <button class="btn-login" onclick="window.location.href='?page=landing&signout=true'">Sign Out</button>
          </div>
        </header>
        """
    else:
        logo_html = get_logo_html(size="200", variant="default")
        landing_nav = f"""
        <header class="header top-nav-shell w-full z-50 relative">
          <div class="logo" onclick="window.location.href='?page=landing'" style="cursor: pointer;">
            <div class="logo-icon">{logo_html}</div>
          </div>
          <nav class="nav-menu">
            <a href="?page=dashboard&dashboard_page=analyze" onclick="window.location.href=this.href; return false;">Analyze Exam</a>
            <a href="?page=dashboard&dashboard_page=generate_exam" onclick="window.location.href=this.href; return false;">Generate Exam</a>
            <a href="?page=about">About</a>
          </nav>
          <div class="header-actions">
            <a href="?page=signin" class="btn-login" id="signin-btn-landing">Sign In</a>
              </div>
              <script>
          document.addEventListener('DOMContentLoaded', function() {{
                  var signinBtn = document.getElementById('signin-btn-landing');
              if (signinBtn) {{
                  signinBtn.addEventListener('click', function(e) {{
                          e.preventDefault();
                          window.location.href = '?page=signin';
                  }});
              }}
          }});
              </script>
        </header>
        """
    
    st.markdown(landing_nav, unsafe_allow_html=True)
    
    st.markdown(
        f"""
        <section class="hero-shell">
          <div class="hero-bg-new"></div>
          <div class="hero-content-shell">
            <div class="hero-grid">
              <div class="hero-text-card">
                <div class="hero-title">COGNIVISE</div>
                <div class="hero-tagline-main">Bloom's Taxonomy Analyzer & Exam Generator</div>
                <div class="hero-tagline-sub">Exams designed to measure understanding, not memorization.</div>
                <div class="hero-cta-row">
                  <a href="?page=dashboard" class="hero-cta hero-cta-primary" id="lets-bloom-btn">
                    Let's Bloom
                  </a>
                </div>
                <script>
                // Ensure link navigates in same tab
                document.addEventListener('DOMContentLoaded', function() {{
                    var bloomBtn = document.getElementById('lets-bloom-btn');
                    
                    if (bloomBtn) {{
                        bloomBtn.addEventListener('click', function(e) {{
                            e.preventDefault();
                            window.location.href = '?page=dashboard';
                        }});
                    }}
                }});
                </script>
              </div>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    
    # Check for signout
    query_params = st.query_params
    signout = query_params.get("signout")
    if signout == "true" or signout == True:
        st.session_state["authenticated"] = False
        st.session_state["username"] = None
        st.query_params.clear()
        st.rerun()
    
    # Check for navigation from query params (triggered by button clicks)
    raw_page = query_params.get("page")
    if isinstance(raw_page, list):
        raw_page = raw_page[0] if raw_page else None
    
    if raw_page and raw_page != "landing":
        st.session_state["page"] = raw_page
        # If navigating to dashboard, set default dashboard page
        if raw_page == "dashboard":
            st.session_state["dashboard_page"] = "analyze"
        st.rerun()


def page_about():
    top_nav()
    
    # Combine CSS and HTML in a single markdown call for better rendering
    about_page_html = """<style>
        .about-hero {
            background: linear-gradient(135deg, rgba(15, 5, 25, 0.95) 0%, rgba(31, 15, 47, 0.95) 50%, rgba(15, 5, 25, 0.95) 100%);
            padding: 4rem 2rem;
            text-align: center;
            margin-bottom: 3rem;
        }
        .about-hero h1 {
            font-size: 3.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #9B7ED9 0%, #B19CD9 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 1.5rem;
            letter-spacing: -0.02em;
        }
        .about-hero .tagline {
            font-size: 1.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #9B7ED9 0%, #B19CD9 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            max-width: 800px;
            margin: 0 auto 3rem;
            line-height: 1.6;
        }
        .about-content {
            max-width: 1000px;
            margin: 0 auto;
            padding: 0 2rem;
        }
        .about-intro {
            font-size: 1.25rem;
            line-height: 1.8;
            color: rgba(255, 255, 255, 0.85);
            margin-bottom: 3rem;
            text-align: center;
        }
        .about-intro .highlight {
            color: #9B7ED9;
            font-weight: 600;
        }
        .highlight {
            color: #9B7ED9;
            font-weight: 600;
        }
        .about-section {
            margin-bottom: 4rem;
        }
        .about-section h2 {
            font-size: 2rem;
            color: #9B7ED9;
            font-weight: 600;
            margin-bottom: 1.5rem;
            text-align: center;
        }
        .about-section p {
            font-size: 1.1rem;
            line-height: 1.8;
            color: rgba(255, 255, 255, 0.8);
            margin-bottom: 1.5rem;
            text-align: center;
        }
        .about-features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 2rem;
            margin: 3rem 0;
        }
        .about-feature-card {
            background: linear-gradient(135deg, rgba(31, 15, 47, 0.6) 0%, rgba(20, 10, 30, 0.6) 100%);
            border: 2px solid rgba(155, 126, 217, 0.3);
            border-radius: 16px;
            padding: 2rem;
            transition: all 0.3s ease;
        }
        .about-feature-card:hover {
            border-color: rgba(155, 126, 217, 0.6);
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(155, 126, 217, 0.2);
        }
        .about-feature-card h3 {
            font-size: 1.5rem;
            color: #9B7ED9;
            font-weight: 600;
            margin-bottom: 1rem;
        }
        .about-feature-card p {
            font-size: 1rem;
            line-height: 1.7;
            color: rgba(255, 255, 255, 0.75);
            text-align: left;
            margin: 0;
        }
        .about-cta-section {
            background: linear-gradient(135deg, rgba(155, 126, 217, 0.1) 0%, rgba(177, 156, 217, 0.05) 100%);
            border: 2px solid rgba(155, 126, 217, 0.2);
            border-radius: 20px;
            padding: 3rem;
            text-align: center;
            margin: 4rem 0;
        }
        .about-cta-section h3 {
            font-size: 2rem;
            color: #9B7ED9;
            font-weight: 600;
            margin-bottom: 1rem;
        }
        .about-cta-section p {
            font-size: 1.2rem;
            color: rgba(255, 255, 255, 0.9);
            margin-bottom: 2rem;
        }
        .about-cta-button {
            display: inline-block;
            background: linear-gradient(135deg, rgba(155, 126, 217, 0.3) 0%, rgba(177, 156, 217, 0.2) 100%);
            border: 2px solid rgba(155, 126, 217, 0.6);
            color: #FFFFFF;
            padding: 1rem 2.5rem;
            border-radius: 12px;
            text-decoration: none;
            font-weight: 600;
            font-size: 1.1rem;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        .about-cta-button:hover {
            background: linear-gradient(135deg, rgba(155, 126, 217, 0.4) 0%, rgba(177, 156, 217, 0.3) 100%);
            border-color: rgba(155, 126, 217, 0.8);
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(155, 126, 217, 0.3);
        }
        .about-emphasis {
            font-size: 1.3rem;
            font-weight: 600;
            color: #9B7ED9;
            display: block;
            margin: 2rem 0;
        }
        .about-list {
            list-style: none;
            padding: 0;
            margin: 2rem 0;
        }
        .about-list li {
            font-size: 1.1rem;
            line-height: 2;
            color: rgba(255, 255, 255, 0.85);
            padding-left: 2rem;
            position: relative;
            margin-bottom: 1rem;
        }
        .about-list li:before {
            content: "→";
            position: absolute;
            left: 0;
            color: #9B7ED9;
            font-weight: bold;
        }
        @media (max-width: 768px) {
            .about-hero h1 {
                font-size: 2.5rem;
            }
            .about-hero .tagline {
                font-size: 1.2rem;
            }
            .about-features {
                grid-template-columns: 1fr;
            }
        }
        </style>
        
        <div class="about-hero">
            <h1>About Cognivise</h1>
            <p class="tagline">Making assessments smarter, fairer, and more purpose-driven</p>
        </div>
        
        <div class="about-content">
            <div class="about-section">
                <p>
                    Cognivise shows you where the exam is balanced and where it needs improvement. It gives you a quality score you can trust and guidance that feels practical, not overwhelming.
                </p>
                
                <span class="about-emphasis">And once you see the results, you choose what happens next.</span>
                
                <ul class="about-list">
                    <li>You can ask Cognivise to improve your existing exam.</li>
                    <li>You can create a new one from scratch.</li>
                    <li>You can also give it your own content book and let it generate questions from the material you already teach.</li>
                </ul>
          </div>
            
            <div class="about-section">
                <p style="font-size: 1.3rem; font-weight: 500; color: rgba(255, 255, 255, 0.95);">
                    Cognivise is for anyone who wants their exams to reflect real learning. Not guesswork. Not shortcuts. Real understanding.
              </p>
            </div>
            
            <div class="about-features">
                <div class="about-feature-card">
                    <h3>Analyze</h3>
                    <p>Upload your exam and get instant insights into how questions map across Bloom's Taxonomy levels. See exactly where your assessment stands.</p>
            </div>
                <div class="about-feature-card">
                    <h3>Improve</h3>
                    <p>Get AI-powered suggestions to enhance your existing exam. Fill cognitive gaps and create better-balanced assessments.</p>
            </div>
                <div class="about-feature-card">
                    <h3>Generate</h3>
                    <p>Create new exams from scratch or from your own content. Generate Bloom-aligned questions for any topic or subtopic.</p>
        </div>
            </div>
            
            <div class="about-cta-section">
                <h3>You care about making your assessments better.</h3>
                <p style="font-size: 1.3rem; font-weight: 600;">Cognivise is here to make that simpler.</p>
                <a href="?page=dashboard" class="about-cta-button">Get Started</a>
            </div>
        </div>"""
    
    # Render the complete HTML page using components.v1.html for reliable rendering
    import streamlit.components.v1 as components
    components.html(about_page_html, height=2000, scrolling=True)


def page_signin():
    # Check if we're in sign-in or sign-up mode
    if "auth_mode" not in st.session_state:
        st.session_state["auth_mode"] = "signin"
    
    if st.query_params.get("mode") == "signup":
        st.session_state["auth_mode"] = "signup"
    elif st.query_params.get("mode") == "signin":
        st.session_state["auth_mode"] = "signin"
    
    # Apply glass styling via JavaScript - runs after page loads
    st.markdown("""
    <script>
    (function() {
        function applyGlassStyling() {
            const blockContainer = document.querySelector('.main .block-container');
            if (blockContainer) {
                // Add class to body
                document.body.classList.add('signin-page');
                blockContainer.classList.add('glass-container');
                
                // Apply inline styles directly to ensure they work
                blockContainer.style.cssText = `
                    padding: 35px !important;
                    max-width: 350px !important;
                    width: 100% !important;
                    background: rgba(255, 255, 255, 0.12) !important;
                    border-radius: 20px !important;
                    border: 1px solid rgba(255, 255, 255, 0.28) !important;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25) !important;
                    backdrop-filter: blur(25px) saturate(180%) !important;
                    -webkit-backdrop-filter: blur(25px) saturate(180%) !important;
                    position: relative !important;
                    overflow: visible !important;
                    margin: 0 auto !important;
                    text-align: center !important;
                    color: #ffffffee !important;
                    display: block !important;
                    height: auto !important;
                    max-height: none !important;
                `;
                
                // Ensure all child elements are visible
                const allChildren = blockContainer.querySelectorAll('*');
                allChildren.forEach(child => {
                    child.style.display = child.style.display || 'block';
                    child.style.visibility = 'visible';
                    child.style.opacity = '1';
                });
            }
        }
        
        // Run immediately
        applyGlassStyling();
        
        // Run after DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', applyGlassStyling);
        }
        
        // Run after a short delay to catch late-rendered elements
        setTimeout(applyGlassStyling, 100);
        setTimeout(applyGlassStyling, 500);
        setTimeout(applyGlassStyling, 1000);
        
        // Watch for changes
        const observer = new MutationObserver(applyGlassStyling);
        observer.observe(document.body, { childList: true, subtree: true });
    })();
    </script>
    """, unsafe_allow_html=True)
    
    # Get background image
    bg_image_uri = get_auth_bg_image_data_uri()
    
    # Add background image as HTML element
    if bg_image_uri:
        st.markdown(f"""
        <div style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; 
                    background-image: url('{bg_image_uri}'); 
                    background-size: cover; background-position: center; 
                    background-repeat: no-repeat; z-index: 0; pointer-events: none;"></div>
        <div style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; 
                    background: rgba(10, 5, 25, 0.4); z-index: 1; pointer-events: none;"></div>
        <script>
        // Remove ALL bottom-right elements and prevent scrolling
        (function() {{
            // Aggressively prevent scrolling
            function preventScroll() {{
                document.body.style.overflow = 'hidden';
                document.documentElement.style.overflow = 'hidden';
                document.body.style.height = '100vh';
                document.documentElement.style.height = '100vh';
                document.body.style.maxHeight = '100vh';
                document.documentElement.style.maxHeight = '100vh';
                document.body.style.position = 'fixed';
                document.body.style.width = '100vw';
                document.documentElement.style.width = '100vw';
            }}
            
            preventScroll();
            
            // Remove ALL bottom-right fixed elements (including Gemini logo)
            function removeBottomRightElements() {{
                // Remove all footers and decorations
                document.querySelectorAll('footer, [data-testid="stFooter"], [data-testid="stDecoration"], .stDeployButton').forEach(el => {{
                    el.style.display = 'none';
                    el.remove();
                }});
                
                // Remove ALL fixed bottom-right elements
                document.querySelectorAll('*').forEach(el => {{
                    try {{
                        const style = window.getComputedStyle(el);
                        const rect = el.getBoundingClientRect();
                        const isBottomRight = (style.position === 'fixed' || style.position === 'absolute') &&
                                            ((rect.bottom <= 10 && rect.right <= 200) || 
                                             (style.bottom === '0px' || style.bottom === '0') ||
                                             (style.right === '0px' || style.right === '0'));
                        
                        if (isBottomRight) {{
                            el.style.display = 'none';
                            el.style.visibility = 'hidden';
                            el.style.opacity = '0';
                            el.style.height = '0';
                            el.style.width = '0';
                            el.remove();
                        }}
                    }} catch(e) {{
                        // Ignore errors
                    }}
                }});
                
                // Remove iframes in bottom right
                document.querySelectorAll('iframe').forEach(iframe => {{
                    const rect = iframe.getBoundingClientRect();
                    if (rect.bottom <= 100 && rect.right <= 200) {{
                        iframe.style.display = 'none';
                        iframe.remove();
                    }}
                }});
            }}
            
            // Run immediately and repeatedly
            removeBottomRightElements();
            preventScroll();
            
            // Run on intervals
            const intervals = [50, 100, 200, 500, 1000, 2000];
            intervals.forEach(delay => {{
                setTimeout(() => {{
                    removeBottomRightElements();
                    preventScroll();
                }}, delay);
            }});
            
            // Watch for new elements
            const observer = new MutationObserver(() => {{
                removeBottomRightElements();
                preventScroll();
            }});
            observer.observe(document.body, {{ childList: true, subtree: true, attributes: true }});
            observer.observe(document.documentElement, {{ childList: true, subtree: true, attributes: true }});
            
            // Prevent wheel scroll
            document.addEventListener('wheel', (e) => {{
                if (e.target.closest('.main .block-container')) return;
                e.preventDefault();
                e.stopPropagation();
            }}, {{ passive: false, capture: true }});
            
            // Prevent touch scroll
            document.addEventListener('touchmove', (e) => {{
                if (e.target.closest('.main .block-container')) return;
                e.preventDefault();
                e.stopPropagation();
            }}, {{ passive: false, capture: true }});
        }})();
        </script>
        """, unsafe_allow_html=True)
    
    # Sign in page with gradient background image and liquid glass form
    st.markdown("""
    <style>
    /* Override global styles for signin page - NO SCROLLING */
    .stApp {
        background: transparent !important;
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        bottom: 0 !important;
        overflow: hidden !important;
        height: 100vh !important;
        max-height: 100vh !important;
        width: 100vw !important;
    }
    
    /* Prevent scrolling on body and html */
    html, body {
        overflow: hidden !important;
        height: 100vh !important;
        max-height: 100vh !important;
        width: 100vw !important;
        position: fixed !important;
    }
    
    /* Hide Gemini logo and any footer elements */
    footer[data-testid="stFooter"],
    footer,
    .stDeployButton,
    [data-testid="stDecoration"],
    [class*="gemini"],
    [class*="Gemini"],
    [id*="gemini"],
    [id*="Gemini"],
    a[href*="gemini"],
    a[href*="Gemini"],
    img[alt*="gemini"],
    img[alt*="Gemini"],
    img[src*="gemini"],
    img[src*="Gemini"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0 !important;
        width: 0 !important;
        position: absolute !important;
        left: -9999px !important;
    }
    
    /* Hide ANY element in bottom-right corner - AGGRESSIVE */
    /* Hide all fixed/absolute bottom-right elements */
    [style*="position: fixed"][style*="bottom"],
    [style*="position:fixed"][style*="bottom"],
    [style*="position: absolute"][style*="bottom"],
    [style*="position:absolute"][style*="bottom"],
    [style*="bottom: 0"],
    [style*="bottom:0"],
    [style*="right: 0"],
    [style*="right:0"],
    iframe,
    footer,
    [data-testid="stFooter"],
    [data-testid="stDecoration"],
    .stDeployButton,
    div[style*="position: fixed"][style*="bottom"],
    div[style*="position:fixed"][style*="bottom"],
    div[style*="position: absolute"][style*="bottom"],
    div[style*="position:absolute"][style*="bottom"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0 !important;
        width: 0 !important;
        pointer-events: none !important;
    }
    
    /* Hide Streamlit decoration and all footers */
    [data-testid="stDecoration"],
    [data-testid="stDecoration"] *,
    .stDecoration,
    .stDecoration *,
    footer,
    footer * {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        width: 0 !important;
    }
    
    /* Hide any iframe in bottom right area */
    iframe {
        position: absolute !important;
        left: -9999px !important;
        display: none !important;
    }
    
    /* Target bottom-right corner specifically - where Gemini logo appears */
    body > *:last-child,
    html > body > *:last-child,
    [style*="bottom"][style*="right"],
    [style*="bottom:0"][style*="right:0"],
    [style*="bottom: 0"][style*="right: 0"] {
        display: none !important;
    }
    
    /* Hide any element with z-index in bottom right */
    [style*="z-index"][style*="bottom"],
    [style*="z-index"][style*="right"] {
        display: none !important;
    }
    
    /* Center the form container - FIXED POSITION */
    .main {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        height: 100vh !important;
        max-height: 100vh !important;
        padding: 2rem !important;
        margin: 0 !important;
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        bottom: 0 !important;
        width: 100vw !important;
        overflow: hidden !important;
        z-index: 10 !important;
        box-sizing: border-box !important;
    }
    
    /* Make block-container the glass card - PURE GLASS BLOCK - OVERRIDE ALL OTHER STYLES */
    body.signin-page .main .block-container,
    .main .block-container.glass-container {
        padding: 35px !important;
        max-width: 350px !important;
        width: 100% !important;
        background: rgba(255, 255, 255, 0.12) !important;  /* light frost */
        border-radius: 20px !important;
        border: 1px solid rgba(255, 255, 255, 0.28) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25) !important;
        backdrop-filter: blur(25px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(25px) saturate(180%) !important;
        position: relative !important;
        overflow: visible !important;
        margin: 0 auto !important;
        text-align: center !important;
        color: #ffffffee !important;
        display: block !important;
        height: auto !important;
        max-height: none !important;
    }
    
    /* Ensure ALL Streamlit elements are inside the glass block-container */
    body.signin-page .main .block-container .element-container,
    body.signin-page .main .block-container section,
    body.signin-page .main .block-container .stTextInput,
    body.signin-page .main .block-container .stButton,
    body.signin-page .main .block-container .stCheckbox,
    body.signin-page .main .block-container .stMarkdown,
    body.signin-page .main .block-container [data-testid],
    body.signin-page .main .block-container > *,
    .main .block-container.glass-container .element-container,
    .main .block-container.glass-container section,
    .main .block-container.glass-container .stTextInput,
    .main .block-container.glass-container .stButton,
    .main .block-container.glass-container .stCheckbox,
    .main .block-container.glass-container .stMarkdown,
    .main .block-container.glass-container [data-testid],
    .main .block-container.glass-container > * {
        width: 100% !important;
        max-width: 100% !important;
        position: relative !important;
        z-index: 1 !important;
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    
    /* Glass Card Wrapper - for compatibility */
    .glass-card-wrapper {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 100% !important;
        height: 100% !important;
        min-height: 100vh !important;
        padding: 0 !important;
        box-sizing: border-box !important;
    }
    
    /* Glass Card - for compatibility */
    .glass-card {
        width: 100% !important;
        max-width: 100% !important;
    }
    
    /* Form Header - centered */
    .form-header {
        margin-bottom: 0 !important;
        text-align: center !important;
    }
    
    .form-title,
    .main .block-container h2 {
        font-family: 'Graphik', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #ffffff !important;
        font-size: 27px !important;
        font-weight: 600 !important;
        margin: 0 !important;
        text-align: center !important;
        line-height: 1.2 !important;
    }
    
    .form-subtitle {
        font-family: 'Graphik', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #ffffff !important;
        font-size: 14px !important;
        margin: 6px 0 25px 0 !important;
        text-align: center !important;
        line-height: 1.3 !important;
        opacity: 1 !important;
    }
    
    /* Input Wrapper - Glass Effect Container */
    .input-wrapper {
        margin-bottom: 1rem !important;
        position: relative !important;
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(15px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(15px) saturate(180%) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        padding: 2px !important;
    }
    
    /* Input Fields - Pure Glass Style - Target signin page specifically */
    body.signin-page .main .block-container .input-wrapper,
    .main .block-container.glass-container .input-wrapper {
        width: 100% !important;
        margin-bottom: 18px !important;
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(15px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(15px) saturate(180%) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        padding: 2px !important;
    }
    
    body.signin-page .main .block-container .stTextInput,
    .main .block-container.glass-container .stTextInput {
        width: 100% !important;
        margin-bottom: 0 !important;
    }
    
    body.signin-page .main .block-container .stTextInput > div,
    .main .block-container.glass-container .stTextInput > div {
        background: transparent !important;
    }
    
    body.signin-page .main .block-container .stTextInput > div > div,
    .main .block-container.glass-container .stTextInput > div > div {
        background: transparent !important;
    }
    
    body.signin-page .main .block-container .stTextInput > div > div > input,
    .main .block-container.glass-container .stTextInput > div > div > input {
        width: 100% !important;
        padding: 12px 16px !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(10px) saturate(180%) !important;
        color: #ffffff !important;
        font-size: 14px !important;
        outline: none !important;
        transition: all 0.2s ease !important;
    }
    
    body.signin-page .main .block-container .stTextInput > div > div > input:focus,
    .main .block-container.glass-container .stTextInput > div > div > input:focus {
        border-color: rgba(255, 255, 255, 0.5) !important;
        background: rgba(255, 255, 255, 0.15) !important;
        backdrop-filter: blur(15px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(15px) saturate(180%) !important;
        box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.1) !important;
    }
    
    body.signin-page .main .block-container .stTextInput > div > div > input::placeholder,
    .main .block-container.glass-container .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Remove icon placeholders for cleaner look - matching the example */
    
    /* Responsive styles for mobile */
    @media (max-width: 768px) {
        body.signin-page .main,
        .main {
            padding: 1rem !important;
        }
        
        body.signin-page .main .block-container,
        .main .block-container.glass-container {
            max-width: 100% !important;
            padding: 30px 25px !important;
        }
        
        body.signin-page .form-title,
        .form-title {
            font-size: 24px !important;
        }
        
        body.signin-page .form-subtitle,
        .form-subtitle {
            font-size: 13px !important;
        }
    }
    
    @media (max-width: 480px) {
        body.signin-page .main,
        .main {
            padding: 0.5rem !important;
        }
        
        body.signin-page .main .block-container,
        .main .block-container.glass-container {
            padding: 25px 20px !important;
            border-radius: 18px !important;
        }
        
        body.signin-page .form-title,
        .form-title {
            font-size: 22px !important;
        }
        
        body.signin-page .main .block-container .stTextInput > div > div > input,
        .main .block-container.glass-container .stTextInput > div > div > input {
            padding: 10px 14px !important;
            font-size: 13px !important;
        }
    }
    
    /* Checkbox Styling - Remember me */
    body.signin-page .main .block-container .stCheckbox,
    .main .block-container.glass-container .stCheckbox {
        margin: 0 0 20px 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    body.signin-page .main .block-container .stCheckbox > label,
    .main .block-container.glass-container .stCheckbox > label {
        display: flex !important;
        align-items: center !important;
        gap: 6px !important;
        font-size: 13px !important;
        color: #ffffff !important;
        font-weight: 400 !important;
        opacity: 1 !important;
        margin: 0 !important;
    }
    
    body.signin-page .main .block-container .stCheckbox > label > div:first-child,
    .main .block-container.glass-container .stCheckbox > label > div:first-child {
        background: rgba(255, 255, 255, 0.2) !important;
        border-color: rgba(255, 255, 255, 0.35) !important;
    }
    
    /* Button - Transparent Glass Button */
    body.signin-page .main .block-container .stButton,
    .main .block-container.glass-container .stButton {
        width: 100% !important;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    
    body.signin-page .main .block-container .stButton > button,
    .main .block-container.glass-container .stButton > button {
        width: 100% !important;
        padding: 12px !important;
        border-radius: 12px !important;
        background: rgba(255, 255, 255, 0.15) !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        color: #fff !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: 0.2s ease !important;
        box-shadow: none !important;
    }
    
    body.signin-page .main .block-container .stButton > button:hover,
    .main .block-container.glass-container .stButton > button:hover {
        background: rgba(255, 255, 255, 0.25) !important;
        transform: scale(1.03) !important;
    }
    
    /* Signup Link */
    body.signin-page .main .block-container .signup-link,
    .main .block-container.glass-container .signup-link {
        text-align: center !important;
        margin-top: 20px !important;
        color: rgba(255, 255, 255, 1) !important;
        font-size: 14px !important;
        font-weight: 400 !important;
    }
    
    body.signin-page .main .block-container .signup-link a,
    .main .block-container.glass-container .signup-link a {
        color: #fff !important;
        font-weight: 600 !important;
        text-decoration: none !important;
    }
    
    body.signin-page .main .block-container .signup-link a:hover,
    .main .block-container.glass-container .signup-link a:hover {
        text-decoration: underline !important;
    }
    
    /* Hide default Streamlit labels */
    body.signin-page .main .block-container .stTextInput label,
    .main .block-container.glass-container .stTextInput label {
        display: none !important;
    }
    
    /* Make ALL text on signin page white/bright */
    body.signin-page .main .block-container,
    .main .block-container.glass-container {
        color: #ffffff !important;
    }
    
    body.signin-page .main .block-container *,
    .main .block-container.glass-container * {
        color: inherit !important;
    }
    
    /* Override for specific text elements to ensure they're white */
    body.signin-page .main .block-container p,
    body.signin-page .main .block-container span,
    body.signin-page .main .block-container div,
    .main .block-container.glass-container p,
    .main .block-container.glass-container span,
    .main .block-container.glass-container div {
        color: #ffffff !important;
    }
    
    /* Ensure form header and all markdown content is inside block-container */
    body.signin-page .main .block-container .form-header,
    body.signin-page .main .block-container .form-title,
    body.signin-page .main .block-container .form-subtitle,
    body.signin-page .main .block-container .signup-link,
    .main .block-container.glass-container .form-header,
    .main .block-container.glass-container .form-title,
    .main .block-container.glass-container .form-subtitle,
    .main .block-container.glass-container .signup-link {
        position: relative !important;
        z-index: 1 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Form content - all elements will be inside the glass block-container
    if st.session_state["auth_mode"] == "signup":
        st.markdown("""
        <div class="form-header">
            <h2 class="form-title">Create Account</h2>
            <p class="form-subtitle">Join us and start your journey</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Username field
        st.markdown('<div class="input-wrapper">', unsafe_allow_html=True)
        username = st.text_input("", placeholder="User Name", key="register_username", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Password field
        st.markdown('<div class="input-wrapper">', unsafe_allow_html=True)
        password = st.text_input("", placeholder="Password", type="password", key="register_password", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Confirm Password field
        st.markdown('<div class="input-wrapper">', unsafe_allow_html=True)
        confirm_password = st.text_input("", placeholder="Confirm Password", type="password", key="register_confirm", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Sign Up button
        if st.button("Sign Up", key="signup_btn", use_container_width=True):
            if not username or not password:
                st.error("Please enter username and password.")
            elif password != confirm_password:
                st.error("Passwords do not match.")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                success, message = register_user(username, password)
                if success:
                    st.success(message)
                    st.info("You can now sign in.")
                    st.session_state["auth_mode"] = "signin"
                    st.rerun()
                else:
                    st.error(message)
        
        st.markdown(
            '<p class="signup-link">Already have an account? <a href="?page=signin&mode=signin">Sign in</a></p>',
            unsafe_allow_html=True
        )
    else:
        st.markdown("""
        <div class="form-header">
            <h2 class="form-title">Login</h2>
            <p class="form-subtitle">Welcome back, please login to your account</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Username/Email field
        st.markdown('<div class="input-wrapper">', unsafe_allow_html=True)
        email = st.text_input("", placeholder="User Name", key="signin_username", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Password field
        st.markdown('<div class="input-wrapper">', unsafe_allow_html=True)
        password = st.text_input("", placeholder="Password", type="password", key="signin_password", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Remember me checkbox
        remember = st.checkbox("Remember me", key="remember_me")
        
        # Login button
        if st.button("Login", key="login_btn", use_container_width=True):
            if not email or not password:
                st.error("Please enter both email and password.")
            else:
                success, message = verify_user(email, password)
                if success:
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = email
                    st.success(f"Welcome back, {email}!")
                    set_page("dashboard")
                    st.rerun()
                else:
                    st.error(message)
        
        st.markdown(
            '<p class="signup-link">Don\'t have an account? <a href="?page=signin&mode=signup">Sign up</a></p>',
            unsafe_allow_html=True
        )


def show_loading_screen():
    """Display enhanced full-screen loading screen with neon gradient and animated text."""
    st.markdown("""
    <div class="loading-screen-overlay" id="loading-screen">
        <div class="loading-content">
            <div class="loading-text">We are getting things ready for you…</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def page_dashboard():
    # Check for dashboard_page in query params first (handles navigation from navbar)
    # But only if we're not using session state (sidebar buttons set _use_session_dashboard_page)
    try:
        if not st.session_state.get("_use_session_dashboard_page", False):
            query_params = st.query_params
            if "dashboard_page" in query_params:
                dashboard_page_param = query_params["dashboard_page"]
                if isinstance(dashboard_page_param, list):
                    dashboard_page_param = dashboard_page_param[0] if dashboard_page_param else None
                if dashboard_page_param and dashboard_page_param in ["analyze", "generate_exam", "generate_questions"]:
                    st.session_state["dashboard_page"] = dashboard_page_param
    except Exception as e:
        pass
    
    # Initialize dashboard_page to "analyze" if not set (default)
    if "dashboard_page" not in st.session_state:
        st.session_state["dashboard_page"] = "analyze"
    
    # Show loading screen if model is not loaded
    if "model_loaded" not in st.session_state or not st.session_state.get("model_loaded"):
        # Show the interactive loading screen
        show_loading_screen()
        
        # Load the CPU model (this will take time)
        try:
            model, tokenizer = bloom_analyzer_complete.load_model(use_cpu_model=True)
            
            # Store model in session state
            st.session_state["model"] = model
            st.session_state["tokenizer"] = tokenizer
            st.session_state["model_loaded"] = True
            
            # Rerun to show dashboard (loading screen will disappear automatically)
            st.rerun()
        except Exception as e:
            # Hide loading screen on error
            st.markdown("""
            <script>
            var loadingScreen = document.getElementById('loading-screen');
            if (loadingScreen) {{
                loadingScreen.style.display = 'none';
            }}
            </script>
            """, unsafe_allow_html=True)
            st.error(f"Error loading model: {str(e)}")
            import traceback
            with st.expander("See full error details"):
                st.code(traceback.format_exc())
            st.session_state["model_loaded"] = False
            return
    
    # Model is loaded, show dashboard (loading screen will be hidden automatically)
    top_nav()
    
    # Get current sub-page from session state (default to "analyze")
    dashboard_page = st.session_state.get("dashboard_page", "analyze")
    
    # Enhanced sidebar styling with aggressive visibility enforcement
    st.markdown("""
    <style>
    /* Force sidebar to be visible and expanded - AGGRESSIVE - Brand Kit Colors */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 5, 25, 0.95) 0%, rgba(31, 15, 47, 0.95) 100%) !important;
        border-right: 2px solid rgba(155, 126, 217, 0.3) !important;
        min-width: 280px !important;
        width: 280px !important;
        padding: 1.5rem 1rem !important;
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.5) !important;
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        transform: translateX(0) !important;
        position: relative !important;
        z-index: 100 !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Override any collapsed state */
    [data-testid="stSidebar"][aria-expanded="true"],
    [data-testid="stSidebar"][aria-expanded="false"],
    [data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        transform: translateX(0) !important;
        width: 280px !important;
        min-width: 280px !important;
        max-width: 280px !important;
    }
    
    /* Make ALL sidebar content visible */
    [data-testid="stSidebar"] > *,
    [data-testid="stSidebar"] .element-container,
    [data-testid="stSidebar"] section,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stButton {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    
    /* Sidebar navigation buttons - compact */
    [data-testid="stSidebar"] .stButton {
        width: 100% !important;
        margin: 0.4rem 0 !important;
    }
    
    [data-testid="stSidebar"] .stButton > button {
        width: 100% !important;
        padding: 0.75rem 1.25rem !important;
        border-radius: 12px !important;
        border: 2px solid transparent !important;
        background: rgba(31, 15, 47, 0.6) !important;
        color: #FFFFFF !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        text-align: left !important;
        transition: all 0.3s ease !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.75rem !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(155, 126, 217, 0.2) !important;
        border-color: rgba(155, 126, 217, 0.5) !important;
        transform: translateX(5px) !important;
        color: #FFFFFF !important;
    }
    
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, rgba(155, 126, 217, 0.3) 0%, rgba(177, 156, 217, 0.2) 100%) !important;
        border-color: rgba(155, 126, 217, 0.6) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 15px rgba(155, 126, 217, 0.3) !important;
    }
    
    /* Sidebar text visibility - WHITE TEXT for better contrast */
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6 {
        color: #FFFFFF !important;
        font-family: 'Graphik', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-weight: 600 !important;
        visibility: visible !important;
        opacity: 1 !important;
        margin-top: 0 !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Sidebar paragraph and caption - WHITE for better contrast */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown {
        color: #FFFFFF !important;
        font-family: 'Graphik', -apple-system, BlinkMacSystemFont, sans-serif !important;
        visibility: visible !important;
        opacity: 1 !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Sidebar content - minimal spacing, NO SCROLL, just moved a little lower */
    [data-testid="stSidebar"] {
        overflow-y: hidden !important;
        overflow-x: hidden !important;
        max-height: 100vh !important;
        height: 100vh !important;
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }
    
    [data-testid="stSidebar"] > * {
        overflow: hidden !important;
    }
    
    [data-testid="stSidebar"] .element-container {
        padding-top: 1.5rem !important;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        overflow: hidden !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem !important;
        overflow: hidden !important;
    }
    
    [data-testid="stSidebar"] > div {
        padding-top: 1.5rem !important;
        overflow: hidden !important;
    }
    
    /* Sidebar spacing - compact */
    [data-testid="stSidebar"] .stMarkdown {
        margin-bottom: 0.25rem !important;
        margin-top: 0 !important;
        padding: 0 !important;
    }
    
    [data-testid="stSidebar"] h3 {
        margin-top: 0.5rem !important;
        margin-bottom: 0.25rem !important;
        padding: 0 !important;
    }
    
    [data-testid="stSidebar"] .stCaption {
        margin-top: 0 !important;
        margin-bottom: 0.5rem !important;
        padding: 0 !important;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.2) !important;
        margin: 0.5rem 0 !important;
        visibility: visible !important;
    }
    
    /* Button text - ensure white */
    [data-testid="stSidebar"] .stButton > button {
        color: #FFFFFF !important;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        color: #FFFFFF !important;
    }
    
    /* Fix dashboard content being cut under navbar and apply glass UI - Option 2: Lighter */
    .main .block-container {
        padding-top: 6rem !important;
        margin-top: 0 !important;
        background: linear-gradient(180deg, rgba(20, 10, 30, 0.95) 0%, rgba(35, 20, 50, 0.95) 100%) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    [data-testid="stSidebar"] ~ .main .block-container {
        padding-top: 6rem !important;
        background: linear-gradient(180deg, rgba(20, 10, 30, 0.95) 0%, rgba(35, 20, 50, 0.95) 100%) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Dashboard page background - Option 2: Lighter to match sidebar better */
    .main {
        background: linear-gradient(135deg, rgba(20, 10, 30, 0.95) 0%, rgba(35, 20, 50, 0.95) 50%, rgba(20, 10, 30, 0.95) 100%) !important;
    }
    </style>
    <script>
    // Aggressively force sidebar visibility and white text
    (function() {
        function forceSidebar() {
            const sidebar = document.querySelector('[data-testid="stSidebar"]');
            if (sidebar) {
                sidebar.setAttribute('aria-expanded', 'true');
                sidebar.style.cssText = 'display: block !important; visibility: visible !important; opacity: 1 !important; transform: translateX(0) !important; width: 280px !important; min-width: 280px !important; background: linear-gradient(180deg, rgba(15, 5, 25, 0.95) 0%, rgba(31, 15, 47, 0.95) 100%) !important; border-right: 2px solid rgba(155, 126, 217, 0.3) !important; padding: 1.5rem 1rem !important; backdrop-filter: blur(10px) !important;';
                
                // Force white text on all text elements
                sidebar.querySelectorAll('h1, h2, h3, h4, h5, h6, p, .stCaption, .stMarkdown, button').forEach(el => {
                    el.style.color = '#FFFFFF !important';
                    el.style.visibility = 'visible';
                    el.style.opacity = '1';
                });
                
                // Force white text on buttons
                sidebar.querySelectorAll('button').forEach(btn => {
                    btn.style.color = '#FFFFFF !important';
                    if (btn.querySelector('*')) {
                        btn.querySelectorAll('*').forEach(child => {
                            child.style.color = '#FFFFFF !important';
                        });
                    }
                });
            }
        }
        forceSidebar();
        setTimeout(forceSidebar, 100);
        setTimeout(forceSidebar, 500);
        setTimeout(forceSidebar, 1000);
        setInterval(forceSidebar, 2000);
        new MutationObserver(forceSidebar).observe(document.body, {childList: true, subtree: true});
    })();
    </script>
    <script>
    // Remove ALL Material Icons text (keyboard_arrow_down, keyboard_double_a, etc.) - AGGRESSIVE
    function removeAllMaterialIconsText() {
        // Remove from expander headers
        var headers = document.querySelectorAll('.streamlit-expanderHeader, [class*="expander"]');
        headers.forEach(function(header) {
            var text = header.textContent || header.innerText || '';
            // Remove all keyboard icon text variations
            text = text.replace(/keyboard_arrow_down/gi, '')
                      .replace(/keyboard_double_arrow_down/gi, '')
                      .replace(/keyboard_double_a/gi, '')
                      .replace(/keyboard_arrow/gi, '')
                      .replace(/^key/gi, '')
                      .trim();
            if (text !== (header.textContent || header.innerText)) {
                header.textContent = text;
            }
            // Hide Material Icons elements
            var icons = header.querySelectorAll('.material-icons, [class*="keyboard"], [class*="arrow"], [class*="key"]');
            icons.forEach(function(icon) {
                icon.style.display = 'none';
                icon.style.visibility = 'hidden';
                icon.style.opacity = '0';
                icon.style.width = '0';
                icon.style.height = '0';
            });
        });
        
        // Remove from ALL elements (catch any stray text)
        var allElements = document.querySelectorAll('*');
        allElements.forEach(function(el) {
            var text = el.textContent || el.innerText || '';
            if (text.includes('keyboard_arrow_down') || text.includes('keyboard_double_a') || text.trim() === 'key') {
                el.textContent = text.replace(/keyboard_arrow_down/gi, '')
                                    .replace(/keyboard_double_arrow_down/gi, '')
                                    .replace(/keyboard_double_a/gi, '')
                                    .replace(/keyboard_arrow/gi, '')
                                    .replace(/^key/gi, '')
                                    .trim();
            }
        });
    }
    
    // Run multiple times to catch all elements
    removeAllMaterialIconsText();
    setTimeout(removeAllMaterialIconsText, 50);
    setTimeout(removeAllMaterialIconsText, 100);
    setTimeout(removeAllMaterialIconsText, 300);
    setTimeout(removeAllMaterialIconsText, 500);
    setTimeout(removeAllMaterialIconsText, 1000);
    setInterval(removeAllMaterialIconsText, 2000);
    
    // Watch for new elements
    new MutationObserver(removeAllMaterialIconsText).observe(document.body, { childList: true, subtree: true, characterData: true });
    </script>
    """, unsafe_allow_html=True)
    
    # Sidebar for navigation
    with st.sidebar:
        # Sidebar header with personalized title - compact
        st.markdown("### Workspace")
        
        st.markdown("---")
        
        # Navigation buttons without emojis
        nav_options = [
            ("analyze", "Analyze Exam"),
            ("generate_exam", "Generate Exam"),
            ("generate_questions", "Generate Questions")
        ]
        
        for page_key, label in nav_options:
            is_active = dashboard_page == page_key
            button_type = "primary" if is_active else "secondary"
            
            if st.button(
                label,
                use_container_width=True,
                type=button_type,
                key=f"nav_{page_key}"
            ):
                st.session_state["dashboard_page"] = page_key
                # Mark that we're using session state (not query params)
                st.session_state["_use_session_dashboard_page"] = True
                st.rerun()
        
        st.markdown("---")
        
        # Show user history if authenticated
        username = st.session_state.get("username")
        if username:
            history = get_user_history(username)
            if history:
                st.markdown("### Recent Analyses")
                for i, exam in enumerate(reversed(history[-5:]), 1):  # Show last 5
                    exam_name = exam.get('exam_name', 'Unnamed Exam')
                    quality = exam.get('quality_score', 0)
                    timestamp = exam.get('timestamp', '')
                    st.caption(f"{i}. {exam_name} - {quality:.1f}/100 ({timestamp[:10] if timestamp else ''})")
    
    # Show the selected page content
    dashboard_page = st.session_state.get("dashboard_page", "analyze")
    
    if dashboard_page == "analyze":
        # Show analyze page content (without navbar since dashboard already shows it)
        page_analyze(show_nav=False)
    elif dashboard_page == "generate_exam":
        # Show generate exam page content (without navbar since dashboard already shows it)
        page_generate_exam(show_nav=False)
    elif dashboard_page == "generate_questions":
        # Show generate questions page content (using generate page)
        page_generate(show_nav=False)


def _improve_exam_with_content(questions, result, model, tokenizer, content_file, topic, exam_name):
    """Helper function to improve exam using uploaded content with RAG."""
    with st.spinner("Processing content and generating improved exam..."):
        try:
            # Initialize RAG generator with optimized generation model (separate from analysis)
            # Use Gemini if available, otherwise Qwen (backend handles this)
            gemini_key = os.getenv("GEMINI_API_KEY")
            gemini_available = bool(gemini_key)
            
            if gemini_available:
                rag_generator = RAGExamGenerator(
                    llm_api="gemini",
                    use_local_vector_store=True,
                    api_key=gemini_key
                )
            else:
                # Fallback to Qwen
                rag_generator = RAGExamGenerator(
                    llm_api="local",
                    use_local_vector_store=True,
                    use_optimized_generation=True,
                    generation_model_name="qwen2.5-1.5b"
                )
            
            # Save uploaded file temporarily
            file_ext = content_file.name.split('.')[-1].lower()
            if file_ext == "pdf":
                source_type = "pdf"
            elif file_ext == "json":
                source_type = "json"
            else:
                source_type = "text_file"
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as tmp_file:
                tmp_file.write(content_file.read())
                tmp_path = tmp_file.name
            
            try:
                # Add content to RAG
                rag_generator.add_content(tmp_path, source_type=source_type, metadata={"topic": topic})
                
                # Generate improved exam
                improvement = rag_generator.improve_exam_with_rag(
                    questions,
                    model,
                    tokenizer,
                    topic=topic,
                    exam_name=exam_name
                )
                
                # Display improvement results
                _display_improvement_results(improvement, result)
                
                # Store in session state
                st.session_state["last_improvement"] = improvement
                
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        except Exception as e:
            st.error(f"Error improving exam: {str(e)}")
            import traceback
            with st.expander("See error details"):
                st.code(traceback.format_exc())


def _improve_exam_with_web_search(questions, result, model, tokenizer, search_query, exam_name):
    """Helper function to improve exam using web search with RAG."""
    with st.spinner("Searching web and generating improved exam..."):
        try:
            # Search web for content
            search_results = search_web_content(search_query, num_results=5)
            
            if not search_results:
                st.warning(f"⚠️ No web results found for '{search_query}'. This could be due to:\n"
                          f"- Network connectivity issues\n"
                          f"- DuckDuckGo service temporarily unavailable\n"
                          f"- Search query too specific\n\n"
                          f"**Suggestions:**\n"
                          f"- Try a more general search query\n"
                          f"- Check your internet connection\n"
                          f"- Try again in a few moments")
                return
            
            # Use Gemini Free API
            rag_generator = RAGExamGenerator(
                llm_api="gemini",  # Use Gemini Free API
                use_local_vector_store=True,
                api_key=os.getenv("GEMINI_API_KEY")
            )
            
            # Add web search results as content
            combined_text = "\n\n".join([f"{r['title']}\n{r['text']}" for r in search_results])
            rag_generator.add_content(combined_text, source_type="text", 
                                     metadata={"topic": search_query, "source": "web_search"})
            
            # Generate improved exam
            improvement = rag_generator.improve_exam_with_rag(
                questions,
                model,
                tokenizer,
                topic=search_query,
                exam_name=exam_name
            )
            
            # Display improvement results
            _display_improvement_results(improvement, result)
            
            # Store in session state
            st.session_state["last_improvement"] = improvement
            
        except Exception as e:
            st.error(f"Error improving exam: {str(e)}")
            import traceback
            with st.expander("See error details"):
                st.code(traceback.format_exc())


def _infer_topic_from_questions(questions):
    """Infer topic/subject from a list of questions by extracting common keywords."""
    import re
    from collections import Counter
    
    # Common subject keywords
    subject_keywords = {
        'Computer Science': ['algorithm', 'programming', 'code', 'function', 'variable', 'data structure', 'software', 'computer', 'program', 'code', 'python', 'java', 'c++', 'database', 'network'],
        'Mathematics': ['equation', 'solve', 'calculate', 'formula', 'theorem', 'algebra', 'geometry', 'calculus', 'derivative', 'integral', 'matrix'],
        'Physics': ['force', 'energy', 'velocity', 'acceleration', 'momentum', 'wave', 'particle', 'quantum', 'electricity', 'magnetism'],
        'Chemistry': ['molecule', 'atom', 'reaction', 'compound', 'element', 'bond', 'acid', 'base', 'organic'],
        'Biology': ['cell', 'organism', 'dna', 'gene', 'evolution', 'ecosystem', 'photosynthesis', 'protein'],
        'English': ['literature', 'poem', 'essay', 'grammar', 'syntax', 'metaphor', 'narrative'],
    }
    
    # Combine all questions into one text
    combined_text = ' '.join(questions).lower()
    
    # Count matches for each subject
    subject_scores = {}
    for subject, keywords in subject_keywords.items():
        score = sum(1 for keyword in keywords if keyword in combined_text)
        if score > 0:
            subject_scores[subject] = score
    
    # Return the subject with highest score, or default
    if subject_scores:
        inferred_subject = max(subject_scores, key=subject_scores.get)
        return inferred_subject
    
    # If no match, try to extract common nouns/terms
    words = re.findall(r'\b[a-z]{4,}\b', combined_text)
    if words:
        common_words = Counter(words).most_common(3)
        # Return a generic topic based on exam name or first common word
        return "General"
    
    return "General"


def _display_improvement_results(improvement, original_result):
    """Display exam improvement results."""
    st.markdown("---")
    st.markdown("### Exam Improvement Results")
    
    # Calculate distribution from stored bloom_level values instead of re-analysis
    from utils.bloom_analyzer_complete import IDEAL_DISTRIBUTION, BLOOM_LEVELS
    from collections import Counter
    
    # Count questions by stored bloom_level
    improved_questions = improvement.get('improved_questions', [])
    total_improved = len(improved_questions)
    
    # Count by stored bloom_level
    level_counts = Counter()
    for item in improved_questions:
        level = item.get('bloom_level', 'Unknown')
        if level in BLOOM_LEVELS:
            level_counts[level] += 1
    
    # Calculate actual percentages and differences
    improved_comparison = {}
    total_deviation = 0
    for level in BLOOM_LEVELS:
        count = level_counts.get(level, 0)
        actual_pct = (count / total_improved * 100) if total_improved > 0 else 0
        ideal_pct = IDEAL_DISTRIBUTION[level] * 100
        difference = actual_pct - ideal_pct
        deviation = abs(difference)
        total_deviation += deviation
        
        improved_comparison[level] = {
            'count': count,
            'actual': actual_pct,
            'ideal': ideal_pct,
            'difference': difference,
        }
    
    # Calculate quality score from stored values
    improved_quality_score = max(0, 100 - total_deviation)
    
    # Comparison metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Original Quality", f"{improvement['original_analysis']['quality_score']:.1f}%")
    with col2:
        st.metric("Improved Quality", f"{improved_quality_score:.1f}%")
    with col3:
        delta = improved_quality_score - improvement['original_analysis']['quality_score']
        st.metric("Improvement", f"{delta:+.1f}%", delta=f"{delta:.1f}%")
    with col4:
        st.metric("Total Questions", improvement['total_questions'])
    
    # Show improved distribution chart (vertical bars)
    st.markdown("---")
    st.markdown("#### Improved Bloom Distribution")
    
    # Create distribution table first
    improved_rows = []
    for level in BLOOM_LEVELS:
        improved_comp = improved_comparison[level]
        improved_rows.append({
            "Level": level,
            "Count": improved_comp['count'],
            "Actual %": f"{improved_comp['actual']:.1f}%",
            "Ideal %": f"{improved_comp['ideal']:.1f}%",
            "Difference": f"{improved_comp['difference']:+.1f}%",
        })
    st.dataframe(improved_rows, use_container_width=True, hide_index=True)
    
    # Create chart showing actual vs ideal distribution
    import plotly.graph_objects as go
    levels = BLOOM_LEVELS
    improved_actual = [improved_comparison[level]['actual'] for level in levels]
    improved_ideal = [improved_comparison[level]['ideal'] for level in levels]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=levels,
        y=improved_actual,
        name='Actual Distribution',
        marker_color='#9B7ED9',
        text=[f"{val:.1f}%" for val in improved_actual],
        textposition='outside'
    ))
    fig.add_trace(go.Bar(
        x=levels,
        y=improved_ideal,
        name='Ideal Distribution',
        marker_color='#B19CD9',
        opacity=0.6,
        text=[f"{val:.1f}%" for val in improved_ideal],
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Actual vs Ideal Bloom Taxonomy Distribution',
        xaxis_title='Bloom Level',
        yaxis_title='Percentage (%)',
        barmode='group',
        height=400,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff')
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Show new questions separately
    if improvement['new_questions']:
        st.markdown("---")
        st.markdown("#### New Questions Added")
        st.info(f"**{len(improvement['new_questions'])} new questions** were generated to improve the exam distribution.")
        
        # Group new questions by Bloom level
        new_by_level = {}
        for new_q in improvement['new_questions']:
            level = new_q.get('bloom_level', 'Unknown')
            if level not in new_by_level:
                new_by_level[level] = []
            new_by_level[level].append(new_q)
        
        # Display by level
        for level in bloom_analyzer_complete.BLOOM_LEVELS:
            if level in new_by_level:
                with st.expander(f"**{level}** ({len(new_by_level[level])} questions)"):
                    for i, new_q in enumerate(new_by_level[level], 1):
                        st.write(f"**{i}.** {new_q['question']}")
        
        st.markdown("---")
    
    # Show complete improved exam
    st.markdown("#### Complete Improved Exam")
    st.info(f"**Total Questions:** {len(improvement['improved_questions'])} (Original: {len(improvement['original_questions'])}, New: {len(improvement['new_questions'])})")
    
    # Group all questions by Bloom level
    all_by_level = {}
    all_questions_list = []  # Keep track of all questions in order
    
    for item in improvement['improved_questions']:
        level = item.get('bloom_level', 'Unknown')
        if level not in all_by_level:
            all_by_level[level] = []
        all_by_level[level].append(item)
        all_questions_list.append(item)  # Keep in order
    
    # Display complete exam by level with highlighted new questions
    # Show all questions in order, grouped by level
    question_num = 1
    
    # First, show all recognized Bloom levels
    for level in bloom_analyzer_complete.BLOOM_LEVELS:
        if level in all_by_level and len(all_by_level[level]) > 0:
            with st.expander(f"**{level}** ({len(all_by_level[level])} questions)"):
                for item in all_by_level[level]:
                    if item.get('source') == 'generated':
                        # Highlight new questions with color
                        st.markdown(
                            f'<div style="background-color: rgba(155, 126, 217, 0.2); padding: 10px; border-left: 4px solid #9B7ED9; margin: 5px 0;">'
                            f'<strong>{question_num}.</strong> {item["question"]} <span style="color: #9B7ED9; font-weight: bold;">[NEW]</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.write(f"**{question_num}.** {item['question']}")
                    question_num += 1
    
    # Also show any questions classified as "Unknown" or other levels not in BLOOM_LEVELS
    # Filter out invalid levels (like "1", "2", etc.) and map them to "Unknown"
    for level in sorted(all_by_level.keys()):
        if level not in bloom_analyzer_complete.BLOOM_LEVELS and len(all_by_level[level]) > 0:
            # If level is just a number or invalid, rename to "Unknown/Unclassified"
            display_level = level
            if level.isdigit() or level in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
                display_level = "Unknown/Unclassified"
            
            with st.expander(f"**{display_level}** ({len(all_by_level[level])} questions)"):
                for item in all_by_level[level]:
                    if item.get('source') == 'generated':
                        st.markdown(
                            f'<div style="background-color: rgba(155, 126, 217, 0.2); padding: 10px; border-left: 4px solid #9B7ED9; margin: 5px 0;">'
                            f'<strong>{question_num}.</strong> {item["question"]} <span style="color: #9B7ED9; font-weight: bold;">[NEW]</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.write(f"**{question_num}.** {item['question']}")
                    question_num += 1
    
    # Verify we displayed all questions
    total_displayed = sum(len(all_by_level.get(level, [])) for level in all_by_level.keys())
    expected_count = len(improvement['improved_questions'])
    
    if total_displayed != expected_count:
        st.error(f"ERROR: Displayed {total_displayed} questions but expected {expected_count}. Some questions are missing!")
        st.write("**Debug Info:**")
        st.write(f"- Questions by level: {[(level, len(all_by_level[level])) for level in all_by_level.keys()]}")
        st.write(f"- Total in improved_questions: {expected_count}")
        st.write(f"- Total displayed: {total_displayed}")
        
        # Show all questions that should be there
        with st.expander("Show all questions in improved_questions"):
            for i, item in enumerate(improvement['improved_questions'], 1):
                st.write(f"{i}. [{item.get('bloom_level', 'Unknown')}] {item['question']} (source: {item.get('source', 'unknown')})")
    
    # Download improved exam
    st.markdown("---")
    st.markdown("#### Download Improved Exam")
    improved_exam_text = "\n\n".join([
        f"{i+1}. {item['question']} [{item['bloom_level']}]" 
        for i, item in enumerate(improvement['improved_questions'])
    ])
    
    st.download_button(
        label="Download Improved Exam as Text",
        data=improved_exam_text,
        file_name=f"improved_exam.txt",
        mime="text/plain"
    )


def page_analyze(show_nav=True):
    # Add brand purple background to the page and remove top nav
    st.markdown("""
    <style>
    /* Ensure page has brand background */
    .main .block-container {
        background: radial-gradient(circle at 0% 0%, #151824 0%, #05060a 45%, #05060a 100%) !important;
        padding-top: 2rem !important;
    }
    
    /* Style input elements to match brand */
    .stRadio > div {
        background: rgba(10, 14, 32, 0.6) !important;
        padding: 1rem !important;
        border-radius: 12px !important;
    }
    
    .stTextInput > div > div > input {
        background: rgba(7, 10, 24, 0.9) !important;
        border: 1px solid rgba(132, 138, 220, 0.7) !important;
        color: #f4f6ff !important;
    }
    
    .stTextArea > div > div > textarea {
        background: rgba(7, 10, 24, 0.9) !important;
        border: 1px solid rgba(132, 138, 220, 0.7) !important;
        color: #f4f6ff !important;
    }
    
    .stFileUploader > div {
        background: rgba(10, 14, 32, 0.6) !important;
        border: 1px solid rgba(120, 129, 210, 0.4) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Load model if not already loaded
    if "model" not in st.session_state or "tokenizer" not in st.session_state:
        with st.spinner("Loading Bloom Taxonomy model..."):
            try:
                model, tokenizer = bloom_analyzer_complete.load_model(use_cpu_model=True)
                st.session_state["model"] = model
                st.session_state["tokenizer"] = tokenizer
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                st.exception(e)
                return
    
    model = st.session_state["model"]
    tokenizer = st.session_state["tokenizer"]
    
    # Page title with new design
    if show_nav:
        top_nav()
    st.markdown("""
    <div style="padding: 3rem 2rem 2rem 2rem; text-align: center; max-width: 900px; margin: 0 auto;">
        <h1 class="page-header">Analyze Your Exam with Cognivise</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Add container for form elements
    st.markdown("""
    <div style="max-width: 800px; margin: 0 auto; padding: 0 2rem;">
    """, unsafe_allow_html=True)

    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Upload PDF", "Upload Image", "Paste Text"],
        horizontal=True,
        key="input_method"
    )

    exam_name = st.text_input("Exam Name (optional)", value="My Exam", key="exam_name_input")
    
    # Initialize questions list in session state if not exists
    if "extracted_questions" not in st.session_state:
        st.session_state["extracted_questions"] = []
    
    # Initialize questions_text variable
    questions_text = ""

    if input_method == "Upload PDF":
        st.markdown("### Upload PDF File")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload a PDF file containing exam questions",
            key="pdf_uploader"
        )
        
        if uploaded_file is not None:
            # Check if this is a new file (compare file name)
            if "last_pdf_name" not in st.session_state or st.session_state["last_pdf_name"] != uploaded_file.name:
                with st.spinner("Extracting all text from PDF..."):
                    try:
                        # Save uploaded file temporarily
                        import tempfile
                        import os
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            tmp_path = tmp_file.name
                        
                        # Extract ALL text from PDF (no filtering)
                        full_text = extract_all_text_from_pdf(tmp_path)
                        
                        # Store in session state
                        st.session_state["pdf_full_text"] = full_text
                        st.session_state["last_pdf_name"] = uploaded_file.name
                        st.session_state["pdf_selected_questions"] = []  # User-selected questions
                        
                        # Clean up temp file
                        os.unlink(tmp_path)
                        
                        st.success(f"Extracted all text from PDF! ({len(full_text)} characters)")
                    except Exception as e:
                        st.error(f"Error extracting text from PDF: {str(e)}")
                        import traceback
                        with st.expander("See error details"):
                            st.code(traceback.format_exc())
                        st.session_state["pdf_full_text"] = ""
                        st.session_state["pdf_selected_questions"] = []
            
            # Show extracted text for manual selection
            if "pdf_full_text" in st.session_state and st.session_state["pdf_full_text"]:
                st.markdown("#### Extracted Text from PDF")
                st.info("**Instructions:** Edit the text below to keep only questions (one per line). Delete headers, footers, and junk. Then click 'Extract Questions from Text'.")
                
                # Initialize editable text
                if "pdf_editable_text" not in st.session_state:
                    st.session_state["pdf_editable_text"] = st.session_state["pdf_full_text"]
                
                # Editable text area
                edited_text = st.text_area(
                    "Edit extracted text (remove junk, keep only questions):",
                    value=st.session_state["pdf_editable_text"],
                    height=400,
                    key="pdf_text_editor",
                    help="Delete headers, footers, instructions, and other non-question content. Keep only the actual questions, one per line."
                )
                
                # Update session state
                st.session_state["pdf_editable_text"] = edited_text
                
                # Buttons for actions
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Extract Questions from Text", type="primary", use_container_width=True):
                        # Split by lines and filter
                        lines = [line.strip() for line in edited_text.split('\n') if line.strip()]
                        # Filter out very short lines (likely junk) and common headers
                        filtered_lines = []
                        junk_patterns = ['page', 'page number', 'time allowed', 'maximum marks', 'instructions', 
                                       'section', 'paper', 'exam', 'total marks', 'marks:', 'question paper']
                        for line in lines:
                            line_lower = line.lower()
                            # Skip if too short or matches junk patterns
                            if len(line) < 10:
                                continue
                            if any(pattern in line_lower for pattern in junk_patterns) and len(line) < 50:
                                continue
                            filtered_lines.append(line)
                        
                        st.session_state["extracted_questions"] = filtered_lines
                        st.success(f"Extracted {len(filtered_lines)} questions from edited text!")
                        st.rerun()
                
                with col2:
                    if st.button("Use Auto-Detected Questions", use_container_width=True):
                        # Try auto-detection
                        try:
                            import tempfile
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                                uploaded_file.seek(0)
                                tmp_file.write(uploaded_file.read())
                                tmp_path = tmp_file.name
                            
                            auto_questions = extract_questions_from_pdf(tmp_path)
                            os.unlink(tmp_path)
                            
                            # Update editable text with auto-detected questions
                            auto_text = '\n'.join(auto_questions)
                            st.session_state["pdf_editable_text"] = auto_text
                            st.session_state["extracted_questions"] = auto_questions
                            st.info(f"Auto-detected {len(auto_questions)} questions. Review and edit the text above if needed.")
                            st.rerun()
                        except Exception as e:
                            st.warning(f"Auto-detection failed: {e}")
                
                with col3:
                    if st.button("Reset to Original", use_container_width=True):
                        st.session_state["pdf_editable_text"] = st.session_state["pdf_full_text"]
                        st.rerun()
                
                # Show current questions count
                if "extracted_questions" in st.session_state and st.session_state["extracted_questions"]:
                    st.markdown("---")
                    st.markdown(f"#### Current Questions ({len(st.session_state['extracted_questions'])} questions)")
                    with st.expander("View current questions", expanded=False):
                        for i, q in enumerate(st.session_state["extracted_questions"], 1):
                            st.write(f"{i}. {q}")
            else:
                st.session_state["extracted_questions"] = []
        else:
            st.session_state["extracted_questions"] = []
            st.session_state["last_pdf_name"] = None
            st.session_state["pdf_full_text"] = ""
            st.session_state["pdf_selected_questions"] = []
    
    elif input_method == "Upload Image":
        st.markdown("### Upload Image File")
        uploaded_file = st.file_uploader(
            "Choose an image file (PNG, JPG, JPEG)",
            type=["png", "jpg", "jpeg"],
            help="Upload an image file containing exam questions (will use OCR to extract all text)",
            key="image_uploader"
        )
        
        if uploaded_file is not None:
            # Check if this is a new file (compare file name)
            if "last_image_name" not in st.session_state or st.session_state["last_image_name"] != uploaded_file.name:
                with st.spinner("Extracting all text from image using OCR..."):
                    try:
                        # Extract ALL text from image (no filtering)
                        full_text = extract_all_text_from_image(uploaded_file)
                        
                        # Store in session state
                        st.session_state["image_full_text"] = full_text
                        st.session_state["last_image_name"] = uploaded_file.name
                        st.session_state["image_selected_questions"] = []  # User-selected questions
                        
                        st.success(f"Extracted all text from image! ({len(full_text)} characters)")
                    except Exception as e:
                        st.error(f"Error extracting text from image: {str(e)}")
                        import traceback
                        with st.expander("See error details"):
                            st.code(traceback.format_exc())
                        st.session_state["image_full_text"] = ""
                        st.session_state["image_selected_questions"] = []
            
            # Show extracted text for manual selection
            if "image_full_text" in st.session_state and st.session_state["image_full_text"]:
                st.markdown("#### Extracted Text from Image")
                st.info("**Instructions:** Edit the text below to keep only questions (one per line). Delete headers, footers, and junk. Then click 'Extract Questions from Text'.")
                
                # Initialize editable text
                if "image_editable_text" not in st.session_state:
                    st.session_state["image_editable_text"] = st.session_state["image_full_text"]
                
                # Editable text area
                edited_text = st.text_area(
                    "Edit extracted text (remove junk, keep only questions):",
                    value=st.session_state["image_editable_text"],
                    height=400,
                    key="image_text_editor",
                    help="Delete headers, footers, instructions, and other non-question content. Keep only the actual questions, one per line."
                )
                
                # Update session state
                st.session_state["image_editable_text"] = edited_text
                
                # Buttons for actions
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Extract Questions from Text", type="primary", use_container_width=True, key="image_extract_btn"):
                        # Split by lines and filter
                        lines = [line.strip() for line in edited_text.split('\n') if line.strip()]
                        # Filter out very short lines (likely junk) and common headers
                        filtered_lines = []
                        junk_patterns = ['page', 'page number', 'time allowed', 'maximum marks', 'instructions', 
                                       'section', 'paper', 'exam', 'total marks', 'marks:', 'question paper']
                        for line in lines:
                            line_lower = line.lower()
                            # Skip if too short or matches junk patterns
                            if len(line) < 10:
                                continue
                            if any(pattern in line_lower for pattern in junk_patterns) and len(line) < 50:
                                continue
                            filtered_lines.append(line)
                        
                        st.session_state["extracted_questions"] = filtered_lines
                        st.success(f"Extracted {len(filtered_lines)} questions from edited text!")
                        st.rerun()
                
                with col2:
                    if st.button("Use Auto-Detected Questions", use_container_width=True, key="image_auto_btn"):
                        # Try auto-detection
                        try:
                            # Reset file pointer
                            uploaded_file.seek(0)
                            auto_questions = extract_questions_from_image(uploaded_file)
                            
                            # Update editable text with auto-detected questions
                            auto_text = '\n'.join(auto_questions)
                            st.session_state["image_editable_text"] = auto_text
                            st.session_state["extracted_questions"] = auto_questions
                            st.info(f"Auto-detected {len(auto_questions)} questions. Review and edit the text above if needed.")
                            st.rerun()
                        except Exception as e:
                            st.warning(f"Auto-detection failed: {e}")
                
                with col3:
                    if st.button("Reset to Original", use_container_width=True, key="image_reset_btn"):
                        st.session_state["image_editable_text"] = st.session_state["image_full_text"]
                        st.rerun()
                
                # Show current questions count
                if "extracted_questions" in st.session_state and st.session_state["extracted_questions"]:
                    st.markdown("---")
                    st.markdown(f"#### Current Questions ({len(st.session_state['extracted_questions'])} questions)")
                    with st.expander("View current questions", expanded=False):
                        for i, q in enumerate(st.session_state["extracted_questions"], 1):
                            st.write(f"{i}. {q}")
            else:
                st.session_state["extracted_questions"] = []
        else:
            st.session_state["extracted_questions"] = []
            st.session_state["last_image_name"] = None
            st.session_state["image_full_text"] = ""
            st.session_state["image_selected_questions"] = []
    
    else:  # Paste Text
        st.markdown("### Enter Exam Questions")
        st.markdown("Paste your questions below (one per line):")
        questions_text = st.text_area(
            "Questions (one per line)",
            placeholder="1. Define algorithm\n2. Explain how sorting algorithms work\n3. Compare quicksort and mergesort",
            height=200,
            key="questions_text_area"
        )
        
        # Clear PDF/image questions when switching to text mode
        if "last_pdf_name" in st.session_state:
            st.session_state["last_pdf_name"] = None
        if "last_image_name" in st.session_state:
            st.session_state["last_image_name"] = None
            st.session_state["extracted_questions"] = []
    
    # Get questions based on input method
    if input_method == "Upload PDF" or input_method == "Upload Image":
        questions = st.session_state.get("extracted_questions", [])
    else:
        # Get text from the text area widget directly
        questions = [q.strip() for q in questions_text.splitlines() if q.strip()] if questions_text else []
    
    # Style the Analyze Exam button with brand colors
    st.markdown("""
    <style>
    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(135deg, #6f5bff 0%, #c26fff 100%) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 18px 38px rgba(106, 98, 255, 0.65) !important;
        font-weight: 600 !important;
        letter-spacing: 0.05em !important;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        box-shadow: 0 22px 48px rgba(106, 98, 255, 0.85) !important;
        transform: translateY(-1px) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if st.button("Analyze Exam", type="primary", use_container_width=True):
        # Clear any stale improvement data that might cause errors
        if "last_improvement" in st.session_state:
            del st.session_state["last_improvement"]
        
        if not questions:
            st.warning("Please upload a PDF/image or enter at least one question.")
        else:
            # Show progress for analysis
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text(f"Analyzing {len(questions)} questions...")
            import sys
            sys.stdout.flush()  # Ensure output is visible immediately
            
            try:
                # ✅ FIX: Only run analysis, NO generation code should run here
                result = bloom_analyzer_complete.analyze_exam(
                    questions, 
                    model, 
                    tokenizer, 
                    exam_name=exam_name
                )
                
                progress_bar.progress(1.0)
                status_text.empty()
                progress_bar.empty()
                
                # Store result in session state for later use
                st.session_state["analysis_result"] = result
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Error during analysis: {str(e)}")
                import traceback
                with st.expander("See error details"):
                    st.code(traceback.format_exc())
                st.stop()
                return  # Exit early on error
            
            # Header with quality score
            if 'result' not in locals():
                st.error("Analysis failed. Please try again.")
                st.stop()
                return
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Questions", result['total_questions'])
            with col2:
                st.metric("Quality Score", f"{result['quality_score']:.1f}%")
            with col3:
                st.metric("Rating", result['quality_rating'])

            # Distribution table - include all levels and Unknown if present
            st.markdown("### Bloom Level Distribution")
            rows = []
            total_count = 0
            all_levels = list(bloom_analyzer_complete.BLOOM_LEVELS)
            if "Unknown" in result["comparison"]:
                all_levels.append("Unknown")
            
            for level in all_levels:
                comp = result["comparison"][level]
                total_count += comp["count"]
                if level == "Unknown":
                    status = "UNCLASSIFIED"
                else:
                    status = "OK" if comp['actual'] > 0 and abs(comp['difference']) <= 5 else \
                             "LOW" if comp['actual'] < comp['ideal'] * 0.7 else \
                             "HIGH" if comp['actual'] > comp['ideal'] * 1.3 else \
                             "MISSING" if comp['actual'] == 0 else "WARNING"
                rows.append({
                    "Level": level,
                    "Count": comp["count"],
                    "Actual %": f"{comp['actual']:.1f}%",
                    "Ideal %": f"{comp['ideal']:.1f}%" if level != "Unknown" else "N/A",
                    "Difference": f"{comp['difference']:+.1f}%" if level != "Unknown" else "N/A",
                    "Status": status,
                })
            
            # Display table
            import pandas as pd
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Show total count below table
            st.info(f"**Total Questions:** {total_count} (Expected: {result['total_questions']})")
            if total_count != result['total_questions']:
                st.warning(f"Count mismatch: {total_count} questions in distribution vs {result['total_questions']} total questions.")
            
            # Questions grouped by Bloom level
            st.markdown("### Questions by Bloom Level")
            questions_by_level = {}
            questions_list = result.get('questions', [])
            predictions_list = result.get('predictions', [])
            
            # Group questions by their predicted level
            for i, question in enumerate(questions_list):
                level = predictions_list[i] if i < len(predictions_list) else "Unknown"
                if level not in questions_by_level:
                    questions_by_level[level] = []
                questions_by_level[level].append(question)
            
            # Display questions grouped by level
            for level in all_levels:
                if level in questions_by_level and len(questions_by_level[level]) > 0:
                    with st.expander(f"{level} ({len(questions_by_level[level])} questions)", expanded=False):
                        for i, question in enumerate(questions_by_level[level], 1):
                            st.write(f"**{i}.** {question}")

            # Recommendations
            st.markdown("### Recommendations")
            
            if result['missing']:
                st.warning("**Missing Categories (0 questions):**")
                for item in result['missing']:
                    st.write(f"- **{item['level']}**: {item['action']}")
            
            if result['deficient']:
                st.warning("**Deficient Categories (Below 70% of ideal):**")
                for item in result['deficient']:
                    st.write(f"- **{item['level']}**: Current {item['actual']:.1f}% | Ideal {item['ideal']:.0f}% | {item['action']}")
            
            if result['excessive']:
                st.info("**Excessive Categories (Above 130% of ideal):**")
                for item in result['excessive']:
                    st.write(f"- **{item['level']}**: Current {item['actual']:.1f}% | Ideal {item['ideal']:.0f}% | {item['action']}")
            
            if result['balanced']:
                st.success("**Well-Balanced Categories:**")
                for item in result['balanced']:
                    st.write(f"- **{item['level']}**: {item['actual']:.1f}% (ideal: {item['ideal']:.0f}%)")
            
            if not result['missing'] and not result['deficient'] and not result['excessive']:
                st.success("**Excellent!** Your exam distribution matches the ideal Bloom taxonomy. All categories are well-balanced.")

            # Interactive chart - brand-styled
            st.markdown("### Visual Analysis")
            chart_fig = create_bloom_chart(result)
            st.plotly_chart(chart_fig, use_container_width=True)
            
            # Store result in session state for potential exam generation
            st.session_state["last_analysis_result"] = result
            st.session_state["last_analyzed_exam_name"] = exam_name
            st.session_state["last_analyzed_questions"] = questions
    
    # Show exam improvement option if analysis has been done (outside the button click block)
    if "last_analysis_result" in st.session_state and "last_analyzed_questions" in st.session_state:
        st.markdown("---")
        st.markdown("### Generate Improved Exam")
        
        # Get analysis result
        analysis_result = st.session_state["last_analysis_result"]
        
        st.info("This will generate new questions using AI based on the concepts and content from your analyzed exam questions.")
        
        col1, col2 = st.columns(2)
        with col1:
            # Button to generate improved exam immediately
            generate_improved_clicked = st.button("Generate Improved Exam", type="primary", use_container_width=True, key="btn_generate_improved")
            if generate_improved_clicked:
                # Ensure analysis data is preserved
                if "last_analysis_result" not in st.session_state or "last_analyzed_questions" not in st.session_state:
                    st.error("Analysis data not found. Please analyze the exam first.")
                else:
                    # Get analysis data
                    analysis_result = st.session_state["last_analysis_result"]
                    analyzed_questions = st.session_state["last_analyzed_questions"]
                    analyzed_exam_name = st.session_state.get("last_analyzed_exam_name", "Exam")
                    
                    # Infer topic from questions
                    inferred_topic = _infer_topic_from_questions(analyzed_questions)
                    
                    with st.spinner("Generating improved exam..."):
                        try:
                            from utils.question_generation import improve_exam_smart
                            
                            # Ensure model and tokenizer are available
                            if "model" not in st.session_state or "tokenizer" not in st.session_state:
                                st.error("Model not loaded. Please reload the page.")
                            else:
                                model = st.session_state["model"]
                                tokenizer = st.session_state["tokenizer"]
                                
                                # Generate improved exam - strictly following recommendations
                                # ✅ PERFORMANCE: Pass analysis_result to avoid duplicate analysis
                                improvement = improve_exam_smart(
                                    original_questions=analyzed_questions,
                                    model=model,
                                    tokenizer=tokenizer,
                                    topic=inferred_topic,
                                    exam_name=analyzed_exam_name,
                                    analysis_result=analysis_result
                                )
                                
                                # Add total_questions to improvement dict for display
                                improvement['total_questions'] = len(improvement['improved_questions'])
                                
                                # Store in session state first
                                st.session_state["last_improvement"] = improvement
                                
                                st.success("Improved exam generated successfully!")
                                st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error generating improved exam: {str(e)}")
                            import traceback
                            with st.expander("See error details"):
                                st.code(traceback.format_exc())
        
        with col2:
            # Button to generate from scratch
            generate_scratch_clicked = st.button("Generate from Scratch", use_container_width=True, type="secondary", key="btn_generate_scratch")
            if generate_scratch_clicked:
                # Clear improvement flags
                if "improve_from_analysis" in st.session_state:
                    del st.session_state["improve_from_analysis"]
                if "generation_mode" in st.session_state:
                    del st.session_state["generation_mode"]
                # Navigate to generate exam page within dashboard context (keeps sidebar)
                st.session_state["dashboard_page"] = "generate_exam"
                st.rerun()
        
        # Display improvement results if they exist (from previous generation)
        # ✅ FIX: Only display if improvement is complete and valid (no errors)
        if "last_improvement" in st.session_state and "last_analysis_result" in st.session_state:
            try:
                improvement = st.session_state["last_improvement"]
                analysis_result = st.session_state["last_analysis_result"]
                
                # Validate improvement data is complete
                # Note: improved_analysis is no longer required - we calculate from stored bloom_level values
                if improvement and 'improved_questions' in improvement:
                    # Ensure total_questions is set
                    if 'total_questions' not in improvement:
                        improvement['total_questions'] = len(improvement.get('improved_questions', []))
                    _display_improvement_results(improvement, analysis_result)
                else:
                    # Invalid improvement data - clear it
                    del st.session_state["last_improvement"]
            except Exception as e:
                # Clear invalid improvement data
                if "last_improvement" in st.session_state:
                    del st.session_state["last_improvement"]
    
    # Close container div
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer removed


def page_generate(show_nav=True):
    if show_nav:
        top_nav()
    st.markdown(
        """
        <section class="page-section">
          <div class="section-card">
            <h2>Generate Questions</h2>
            <p>Generate Bloom's Taxonomy-aligned questions for any topic or subtopic.</p>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    
    # Load model if needed
    if "model" not in st.session_state or "tokenizer" not in st.session_state:
        with st.spinner("Loading Bloom Taxonomy model..."):
            try:
                model, tokenizer = bloom_analyzer_complete.load_model(use_cpu_model=True)
                st.session_state["model"] = model
                st.session_state["tokenizer"] = tokenizer
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                st.exception(e)
                return
    
    model = st.session_state["model"]
    tokenizer = st.session_state["tokenizer"]
    
    st.info("**Note:** For generating complete exams with ideal Bloom's Taxonomy distribution, use **Generate Exam Paper** instead.")
    
    if st.button("Go to Generate Exam Paper →", use_container_width=True, type="secondary"):
        set_page("generate_exam")
        st.rerun()
    
    st.markdown("---")
    st.markdown("### Generate Question")
    
    # Form inputs
    col1, col2 = st.columns(2)
    with col1:
        levels = ["Remembering", "Understanding", "Applying", "Analyzing", "Evaluating", "Creating"]
        level = st.selectbox("Bloom Level", levels, help="Select the cognitive level for the question")
    with col2:
        num_questions = st.number_input("Number of Questions", min_value=1, max_value=50, value=1, step=1, 
                                       help="For full exams with ideal distribution, use Generate Exam Paper")
    
    topic = st.text_area(
        "Topic or Description", 
        placeholder="e.g., Computer Science, Algorithms, Data Structures, Python Programming",
        height=100,
        help="Enter the topic or subject area for which you want to generate questions"
    )
    
    if st.button("Generate Questions", type="primary", use_container_width=True):
        if not topic or not topic.strip():
            st.warning("Please enter a topic or description.")
        else:
            with st.spinner(f"Generating {num_questions} {level} question(s)..."):
                try:
                    from utils.question_generation import generate_questions_improved
                    
                    # Generate questions using RAG (pass model/tokenizer to avoid reloading)
                    generated_questions = generate_questions_improved(
                        bloom_level=level,
                        topic=topic.strip(),
                        num_questions=num_questions,
                        model=model,
                        tokenizer=tokenizer
                    )
                    
                    if generated_questions:
                        st.success(f"Generated {len(generated_questions)} question(s)!")
                        st.markdown("---")
                        st.markdown("### Generated Questions")
                        
                        for i, question in enumerate(generated_questions, 1):
                            with st.container():
                                st.markdown(f"**Question {i}:**")
                                st.info(question)
                                
                                # VERIFICATION COMMENTED OUT - Trust generation, model verification causes false rejections
                                # # Verify the generated question's Bloom level
                                # with st.spinner("Verifying Bloom level..."):
                                #     predicted_level = bloom_analyzer_complete.predict(question, model, tokenizer)
                                #     if predicted_level == level:
                                #         st.success(f"Verified as {predicted_level}")
                                #     else:
                                #         st.warning(f"Predicted as {predicted_level} (requested: {level})")
                        
                        # Download option
                        st.markdown("---")
                        import io
                        questions_text = "\n\n".join([f"{i}. {q}" for i, q in enumerate(generated_questions, 1)])
                        st.download_button(
                            label="Download Questions as Text",
                            data=questions_text,
                            file_name=f"{level}_{topic.replace(' ', '_')}_questions.txt",
                            mime="text/plain"
                        )
                    else:
                        st.error("Failed to generate questions. Please try again.")
                        
                except Exception as e:
                    st.error(f"Error generating questions: {str(e)}")
                    import traceback
                    with st.expander("See error details"):
                        st.code(traceback.format_exc())


def page_generate_exam(show_nav=True):
    """Page for generating complete exam papers from content or web search."""
    if show_nav:
        top_nav()
    
    st.markdown("""
    <div style="padding: 3rem 2rem 2rem 2rem; text-align: center; max-width: 900px; margin: 0 auto;">
        <h1 class="page-header">Generate Your Exam or Questions</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model if needed
    if "model" not in st.session_state or "tokenizer" not in st.session_state:
        with st.spinner("Loading Bloom Taxonomy model..."):
            try:
                model, tokenizer = bloom_analyzer_complete.load_model(use_cpu_model=True)
                st.session_state["model"] = model
                st.session_state["tokenizer"] = tokenizer
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                st.exception(e)
                return
    
    model = st.session_state["model"]
    tokenizer = st.session_state["tokenizer"]
    
    # Container for form elements
    st.markdown("""
    <div style="max-width: 800px; margin: 0 auto; padding: 0 2rem;">
    """, unsafe_allow_html=True)
    
    # Check if mode is set from previous page
    generation_mode = st.session_state.get("generation_mode", None)
    improve_from_analysis = st.session_state.get("improve_from_analysis", False)
    
    # Check if we have analysis results to improve from
    has_analysis = "last_analysis_result" in st.session_state and "last_analyzed_questions" in st.session_state
    
    # If improving from analysis but data is missing, show error
    if improve_from_analysis and not has_analysis:
        st.error("Analysis data was not preserved. Please go back to the Analyze page and click 'Generate Improved Exam' again.")
        if st.button("Go to Dashboard"):
            st.session_state["page"] = "dashboard"
            st.query_params["page"] = "dashboard"
            st.rerun()
        st.stop()
    
    # Input method selection - add option to infer from existing questions if analysis exists
    if improve_from_analysis and has_analysis:
        input_method_options = ["Infer from Existing Questions (Recommended)", "Upload Content (PDF/Text/JSON)", "Use Web Search", "Enter Text Content"]
        default_index = 0
    else:
        input_method_options = ["Upload Content (PDF/Text/JSON)", "Use Web Search", "Enter Text Content"]
        default_index = 1 if generation_mode == "web_search" else 0
    
    input_method = st.radio(
        "Choose generation method:",
        input_method_options,
        index=default_index,
        horizontal=False,
        key="exam_generation_method"
    )
    
    # If improving from analysis, show insights and options
    if improve_from_analysis and has_analysis:
        # Ensure analysis data is preserved
        if "last_analysis_result" not in st.session_state:
            st.error("Analysis data was lost. Please go back and analyze the exam again.")
            st.stop()
        
        analysis_result = st.session_state["last_analysis_result"]
        analyzed_questions = st.session_state["last_analyzed_questions"]
        analyzed_exam_name = st.session_state.get("last_analyzed_exam_name", "Previous Exam")
        
        st.markdown("---")
        st.markdown("### Analysis Insights")
        st.info(f"**Improving:** {analyzed_exam_name} | **Quality Score:** {analysis_result['quality_score']:.1f}% | **Total Questions:** {len(analyzed_questions)}")
        
        # Show distribution insights
        if input_method == "Infer from Existing Questions (Recommended)":
            st.markdown("#### Current Distribution vs Ideal:")
            insights_cols = st.columns(6)
            for idx, level in enumerate(bloom_analyzer_complete.BLOOM_LEVELS):
                with insights_cols[idx]:
                    comp = analysis_result['comparison'][level]
                    ideal_pct = comp['ideal']
                    actual_pct = comp['actual']
                    diff = comp['difference']
                    
                    if abs(diff) <= 5:
                        status_text = "OK"
                        status_color = "#4CC9F0"
                    elif diff < -5:
                        status_text = "LOW"
                        status_color = "#F72585"
                    else:
                        status_text = "HIGH"
                        status_color = "#F72585"
                    
                    st.markdown(f"""
                    <div style="text-align: center; padding: 0.5rem; background: rgba(26,0,51,0.5); border-radius: 8px; border: 1px solid rgba(61,26,107,0.4);">
                        <div style="font-size: 1.2rem; margin-bottom: 0.3rem;">{status_text}</div>
                        <div style="font-size: 0.8rem; font-weight: 600; color: {status_color};">{level[:4]}</div>
                        <div style="font-size: 0.7rem; color: rgba(255,255,255,0.7);">Actual: {actual_pct:.1f}%</div>
                        <div style="font-size: 0.7rem; color: rgba(255,255,255,0.5);">Ideal: {ideal_pct:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("**The improved exam will be generated based on your existing questions and will target the ideal Bloom's Taxonomy distribution.**")
        
        # Pre-fill exam name (will be used in the generation section)
        if "exam_name_gen_improve" not in st.session_state:
            st.session_state["exam_name_gen_improve"] = f"Improved {analyzed_exam_name}"
    else:
        # Get exam parameters - only show if not improving from analysis
        if not improve_from_analysis:
            exam_name = st.text_input("Exam Name", value="Generated Exam", key="exam_name_gen")
    
    # Exam parameters (shown for all methods)
    col1, col2 = st.columns(2)
    with col1:
        total_questions = st.number_input(
            "Number of Questions", 
            min_value=10, max_value=100, value=20, step=1,
            help="Total number of questions to generate (minimum 10 for full exam distribution)",
            key="total_questions_gen"
        )
        
        if total_questions < 10:
            st.warning("Minimum 10 questions required for Generate Exam Paper to ensure proper Bloom's Taxonomy distribution.")
    with col2:
        complexity = st.selectbox(
            "Complexity Level",
            ["Easy", "Intermediate", "Complex"],
            index=1,
            help="Difficulty level of the questions",
            key="complexity_gen"
        )
    
    # Backend model selection - use Gemini if available, fallback to Qwen
    # No UI selection needed - handled automatically in backend
    gemini_api_available = bool(os.getenv("GEMINI_API_KEY"))
    if gemini_api_available:
        st.session_state["use_cloud_generation"] = True
    else:
        st.session_state["use_cloud_generation"] = False
    
    # Bloom Level Distribution Selection
    st.markdown("#### Bloom's Taxonomy Distribution")
    bloom_distribution_mode = st.radio(
        "Question Distribution:",
        ["Ideal Distribution (Automatic)", "Specific Bloom Level Only"],
        index=0,
        help="Choose how questions should be distributed across Bloom levels",
        key="bloom_distribution_mode"
    )
    
    specific_bloom_level = None
    
    # Show ideal distribution breakdown for the selected number of questions
    if bloom_distribution_mode == "Ideal Distribution (Automatic)":
        from utils.bloom_analyzer_complete import IDEAL_DISTRIBUTION, BLOOM_LEVELS
        
        # Calculate distribution for current number of questions
        distribution_breakdown = {}
        for level in BLOOM_LEVELS:
            count = max(1, int(total_questions * IDEAL_DISTRIBUTION[level]))
            distribution_breakdown[level] = count
        
        # Adjust to match total
        current_total = sum(distribution_breakdown.values())
        if current_total != total_questions:
            diff = total_questions - current_total
            # Add extra questions to higher levels
            for level in ['Applying', 'Analyzing', 'Evaluating', 'Creating']:
                if diff <= 0:
                    break
                distribution_breakdown[level] += 1
                diff -= 1
        
        st.success(f"**Ideal Distribution** will be applied for **{total_questions} questions**:")
        
        # Display breakdown in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Remembering", f"{distribution_breakdown['Remembering']} ({IDEAL_DISTRIBUTION['Remembering']*100:.0f}%)")
            st.metric("Understanding", f"{distribution_breakdown['Understanding']} ({IDEAL_DISTRIBUTION['Understanding']*100:.0f}%)")
        with col2:
            st.metric("Applying", f"{distribution_breakdown['Applying']} ({IDEAL_DISTRIBUTION['Applying']*100:.0f}%)")
            st.metric("Analyzing", f"{distribution_breakdown['Analyzing']} ({IDEAL_DISTRIBUTION['Analyzing']*100:.0f}%)")
        with col3:
            st.metric("Evaluating", f"{distribution_breakdown['Evaluating']} ({IDEAL_DISTRIBUTION['Evaluating']*100:.0f}%)")
            st.metric("Creating", f"{distribution_breakdown['Creating']} ({IDEAL_DISTRIBUTION['Creating']*100:.0f}%)")
        
        st.markdown("""
        **Ideal Distribution Guidelines:**
        - **Remembering**: 15% (basic recall of facts, terms, concepts)
        - **Understanding**: 20% (comprehension and explanation)
        - **Applying**: 25% (practical application in new situations)
        - **Analyzing**: 20% (analysis, comparison, relationships)
        - **Evaluating**: 10% (judgment, critique, justification)
        - **Creating**: 10% (synthesis, design, creation)
        """)
    else:
        specific_bloom_level = st.selectbox(
            "Select Bloom Taxonomy Level",
            ["Remembering", "Understanding", "Applying", "Analyzing", "Evaluating", "Creating"],
            index=2,
            help="All questions will be generated at this Bloom level only",
            key="specific_bloom_select"
        )
        st.warning(f"**All {total_questions} questions will be generated at the {specific_bloom_level} level only.** Ideal distribution will be ignored.")
    
    generated_exam = None
    
    # Handle "Infer from Existing Questions" option
    if input_method == "Infer from Existing Questions (Recommended)":
        if not has_analysis:
            st.error("No analysis data found. Please analyze an exam first.")
        else:
            analysis_result = st.session_state["last_analysis_result"]
            analyzed_questions = st.session_state["last_analyzed_questions"]
            analyzed_exam_name = st.session_state.get("last_analyzed_exam_name", "Previous Exam")
            
            # Get exam name - check both keys
            if "exam_name_gen_improve" in st.session_state:
                exam_name = st.session_state.get("exam_name_gen_improve", f"Improved {analyzed_exam_name}")
            else:
                exam_name = st.text_input("Exam Name", value=f"Improved {analyzed_exam_name}", key="exam_name_gen_improve")
            
            st.markdown("---")
            st.markdown("### 📝 Content & Topic Configuration")
            
            # Show existing questions preview
            with st.expander("📋 View Existing Questions (Will be used as content)", expanded=False):
                st.write(f"**Total Questions:** {len(analyzed_questions)}")
                for i, q in enumerate(analyzed_questions[:5], 1):
                    st.write(f"{i}. {q[:100]}..." if len(q) > 100 else f"{i}. {q}")
                if len(analyzed_questions) > 5:
                    st.write(f"... and {len(analyzed_questions) - 5} more questions")
            
            # Topic specification
            col1, col2 = st.columns(2)
            with col1:
                topic = st.text_input(
                    "Subject/Topic Area", 
                    value="",
                    help="Specify the subject or topic area (e.g., 'Computer Science', 'Mathematics', 'Biology'). Leave blank to infer from questions.",
                    key="topic_infer",
                    placeholder="e.g., Computer Science, Mathematics"
                )
            with col2:
                use_existing_as_content = st.checkbox(
                    "Use Existing Questions as Content",
                    value=True,
                    help="When enabled, existing questions will be analyzed and used as content to frame new questions around similar topics and concepts. When disabled, only the specified topic will be used.",
                    key="use_existing_content"
                )
            
            if use_existing_as_content:
                st.info("**Existing questions will be analyzed and used as content to generate new questions around similar topics and concepts.**")
                if topic and topic.strip():
                    st.info(f"📝 **Topic specified:** {topic} - Will be combined with existing questions for better context.")
            else:
                st.warning("**Only the specified topic will be used. Existing questions won't be used as content.**")
                if not topic or not topic.strip():
                    st.error("**Please specify a topic** since existing questions won't be used as content.")
            
            if st.button("Generate Improved Exam Paper", type="primary", use_container_width=True):
                if total_questions < 10:
                    st.error("Please enter at least 10 questions for Generate Exam Paper.")
                else:
                    with st.spinner("Generating improved exam based on your configuration..."):
                        try:
                            # Initialize RAG generator - use Gemini if available, otherwise Qwen
                            gemini_key = os.getenv("GEMINI_API_KEY")
                            gemini_available = bool(gemini_key)
                            
                            if gemini_available:
                                rag_generator = RAGExamGenerator(
                                    llm_api="gemini",
                                    use_local_vector_store=True,
                                    api_key=gemini_key
                                )
                            else:
                                rag_generator = RAGExamGenerator(
                                    llm_api="local",
                                    use_local_vector_store=True,
                                    use_optimized_generation=True,
                                    generation_model_name="qwen2.5-1.5b"
                                )
                            
                            # Determine topic
                            if not topic or not topic.strip():
                                topic = f"Content from {analyzed_exam_name}"
                            
                            # Add existing questions as content if checkbox is enabled
                            if use_existing_as_content:
                                questions_text = "\n\n".join(analyzed_questions)
                                with st.spinner("Processing existing questions as content and creating vector store..."):
                                    rag_generator.add_content(questions_text, source_type="text",
                                                             metadata={"topic": topic, "exam_name": exam_name, 
                                                                      "source": "existing_questions", 
                                                                      "original_quality": analysis_result['quality_score'],
                                                                      "num_questions": len(analyzed_questions)})
                                st.success(f"Processed {len(analyzed_questions)} existing questions as content for topic: **{topic}**")
                            else:
                                st.info(f"📝 Using specified topic only: **{topic}** (existing questions not used as content)")
                            
                            # Generate exam with ideal distribution
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            def update_progress(progress, status):
                                # Clamp progress to valid [0.0, 1.0] range to avoid Streamlit errors
                                safe_progress = max(0.0, min(1.0, float(progress)))
                                progress_bar.progress(safe_progress)
                                status_text.text(status)
                            
                            generated_questions = rag_generator.generate_exam_from_content(
                                total_questions=total_questions,
                                topic=topic,
                                complexity=complexity.lower(),
                                specific_bloom_level=None,  # Use ideal distribution
                                progress_callback=update_progress
                            )
                            
                            progress_bar.empty()
                            status_text.empty()
                            
                            # Clean questions before analysis
                            question_list = []
                            for q in generated_questions:
                                clean_q = q['question']
                                if not clean_q:
                                    continue
                                import re
                                clean_q = re.sub(r'\[(INTERMEDIATE|EASY|COMPLEX)\]\s*', '', clean_q, flags=re.IGNORECASE)
                                clean_q = re.sub(r'\[.*?INTERMEDIATE.*?\]', '', clean_q, flags=re.IGNORECASE)
                                clean_q = clean_q.strip('"').strip("'").strip()
                                if '?' in clean_q:
                                    parts = clean_q.split('?', 1)
                                    clean_q = parts[0].strip() + '?'
                                if clean_q and len(clean_q) >= 15:
                                    question_list.append(clean_q)
                            
                            # Analyze generated exam
                            analysis = bloom_analyzer_complete.analyze_exam(
                                question_list, model, tokenizer, exam_name=exam_name
                            )
                            
                            # Clean questions for display
                            cleaned_questions = []
                            for q in generated_questions:
                                clean_q = q['question']
                                if not clean_q:
                                    continue
                                import re
                                for _ in range(3):
                                    clean_q = re.sub(r'\[(INTERMEDIATE|EASY|COMPLEX|intermediate|easy|complex)\]\s*', '', clean_q, flags=re.IGNORECASE)
                                    clean_q = re.sub(r'\[.*?INTERMEDIATE.*?\]', '', clean_q, flags=re.IGNORECASE)
                                clean_q = re.sub(r'\s*Explanation:\s*.*$', '', clean_q, flags=re.IGNORECASE)
                                clean_q = clean_q.strip('"').strip("'").strip()
                                if '?' in clean_q:
                                    clean_q = clean_q.split('?')[0].strip() + '?'
                                if len(clean_q) > 150:
                                    clean_q = clean_q[:150].rstrip(' ,.') + '?'
                                clean_q = re.sub(r'\s+', ' ', clean_q).strip()
                                if clean_q and len(clean_q) >= 10:
                                    cleaned_questions.append({
                                        'question': clean_q,
                                        'bloom_level': q['bloom_level'],
                                        'complexity': q['complexity'],
                                        'source': 'improved_from_analysis'
                                    })
                            
                            generated_exam = {
                                'questions': cleaned_questions,
                                'analysis': analysis,
                                'exam_name': exam_name,
                                'original_analysis': analysis_result,
                                'improvement_delta': analysis['quality_score'] - analysis_result['quality_score']
                            }
                            
                        except Exception as e:
                            st.error(f"Error generating improved exam: {str(e)}")
                            import traceback
                            with st.expander("See error details"):
                                st.code(traceback.format_exc())
    
    elif input_method == "Upload Content (PDF/Text/JSON)":
        username = st.session_state.get("username")
        
        # Show saved books if user is authenticated
        saved_books = []
        if username:
            saved_books = get_user_books(username)
            if saved_books:
                st.markdown("### Your Saved Books")
                with st.expander("Load a Previously Uploaded Book", expanded=False):
                    for book in saved_books:
                        book_name = book.get('name', 'Unnamed Book')
                        book_type = book.get('type', 'unknown')
                        uploaded_at = book.get('uploaded_at', '')[:10] if book.get('uploaded_at') else ''
                        if st.button(f"📖 {book_name} ({book_type.upper()}) - {uploaded_at}", key=f"load_book_{book.get('id')}", use_container_width=True):
                            st.session_state["load_saved_book"] = book.get('id')
                            st.session_state["saved_book_path"] = book.get('path')
                            st.info(f"Loading book: {book_name}. Please proceed to generate exam.")
        
        st.markdown("### Upload Content File")
        uploaded_file = st.file_uploader(
            "Choose a PDF, Text, or JSON file",
            type=["pdf", "txt", "json"],
            help="Upload a PDF, text file, or JSON file (structured book format) containing the content/topics for exam generation",
            key="exam_content_upload"
        )
        
        # Show JSON format hint if JSON file
        if uploaded_file and uploaded_file.name.endswith('.json'):
            st.info("**JSON Book Format**: The system supports structured JSON books with chapters/sections, content/text fields, or nested content structures.")
        
        # Content scope selection
        content_scope = st.radio(
            "Content Scope:",
            ["Complete Book/File", "Specific Chapters/Sections"],
            index=0,
            help="Select whether to use the entire file or specify chapters/sections",
            key="content_scope"
        )
        
        chapters_sections = ""
        if content_scope == "Specific Chapters/Sections":
            chapters_sections = st.text_input(
                "Specify Chapters/Sections",
                placeholder="e.g., Chapter 1-3, Introduction and Conclusion, Sections 2.1-2.5",
                help="Specify which parts of the content to focus on",
                key="chapters_input"
            )
        
        # Topic input for context
        topic = st.text_input(
            "Subject/Topic Area (Optional)", 
            value="",
            help="Enter the subject or topic area to help generate relevant questions",
            key="topic_upload"
        )
        
        # Option to save book if authenticated
        save_book = False
        if username and uploaded_file:
            save_book = st.checkbox("💾 Save this book for future use", value=True, 
                                   help="Save uploaded book to your account for quick access later",
                                   key="save_book_checkbox")
        
        if st.button("Generate Exam Paper", type="primary", use_container_width=True):
            if total_questions < 10:
                st.error("Please enter at least 10 questions for Generate Exam Paper.")
            elif not uploaded_file:
                st.warning("Please upload a content file.")
            else:
                with st.spinner("Processing content and generating exam..."):
                    try:
                        # Use Gemini if available, otherwise Qwen (backend handles this)
                        gemini_key = os.getenv("GEMINI_API_KEY")
                        gemini_available = bool(gemini_key)
                        
                        if gemini_available:
                            rag_generator = RAGExamGenerator(
                                llm_api="gemini",
                                use_local_vector_store=True,
                                api_key=gemini_key
                            )
                        else:
                            rag_generator = RAGExamGenerator(
                                llm_api="local",
                                use_local_vector_store=True,
                                use_optimized_generation=True,
                                generation_model_name="qwen2.5-1.5b"
                            )
                        
                        # Save uploaded file temporarily
                        file_ext = uploaded_file.name.split('.')[-1].lower()
                        if file_ext == "pdf":
                            source_type = "pdf"
                        elif file_ext == "json":
                            source_type = "json"
                        else:
                            source_type = "text_file"
                        
                        # Check if loading saved book
                        use_saved_book = False
                        if st.session_state.get("load_saved_book") and st.session_state.get("saved_book_path"):
                            saved_book_path = st.session_state.get("saved_book_path")
                            if os.path.exists(saved_book_path):
                                tmp_path = saved_book_path
                                # Determine source type from saved book
                                book_info = next((b for b in saved_books if b.get('id') == st.session_state.get("load_saved_book")), None)
                                source_type = book_info.get('type', 'json') if book_info else 'json'
                                st.info(f"Using saved book: {book_info.get('name', 'Book')}")
                                use_saved_book = True
                            else:
                                st.error("Saved book file not found. Please upload again.")
                                uploaded_file = None  # Force re-upload
                        
                        if not use_saved_book:
                            # Save uploaded file temporarily
                            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as tmp_file:
                                tmp_file.write(uploaded_file.read())
                                tmp_path = tmp_file.name
                            
                            # Save book to user's directory if requested
                            if username and save_book:
                                books_dir = os.path.join("user_history", username, "books")
                                os.makedirs(books_dir, exist_ok=True)
                                
                                # Copy to user's books directory
                                book_filename = f"{uploaded_file.name.replace(' ', '_')}"
                                saved_book_path = os.path.join(books_dir, book_filename)
                                shutil.copy2(tmp_path, saved_book_path)
                                
                                # Save book info
                                save_user_book(
                                    username,
                                    uploaded_file.name,
                                    saved_book_path,
                                    book_type=file_ext,
                                    metadata={
                                        "topic": topic,
                                        "content_scope": content_scope,
                                        "chapters": chapters_sections
                                    }
                                )
                                st.success(f"💾 Book saved: {uploaded_file.name}")
                        
                        try:
                            # Add content to RAG
                            metadata = {
                                "topic": topic or "the uploaded content",
                                "exam_name": exam_name,
                                "content_scope": content_scope,
                                "chapters": chapters_sections if chapters_sections else "All"
                            }
                            
                            with st.spinner("Processing content and creating vector store..."):
                                rag_generator.add_content(tmp_path, source_type=source_type, metadata=metadata)
                            
                            # Generate exam with complexity and optional specific bloom level
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            def update_progress(progress, status):
                                # Clamp progress to valid [0.0, 1.0] range to avoid Streamlit errors
                                safe_progress = max(0.0, min(1.0, float(progress)))
                                progress_bar.progress(safe_progress)
                                status_text.text(status)
                            
                            generated_questions = rag_generator.generate_exam_from_content(
                                total_questions=total_questions,
                                topic=topic or "the uploaded content",
                                complexity=complexity.lower(),
                                specific_bloom_level=specific_bloom_level,
                                progress_callback=update_progress
                            )
                            
                            progress_bar.empty()
                            status_text.empty()
                            
                            # Analyze generated exam
                            # Clean questions before analysis
                            question_list = []
                            for q in generated_questions:
                                clean_q = q['question']
                                if not clean_q:
                                    continue
                                import re
                                # Remove complexity tags
                                clean_q = re.sub(r'\[(INTERMEDIATE|EASY|COMPLEX)\]\s*', '', clean_q, flags=re.IGNORECASE)
                                clean_q = re.sub(r'\[.*?INTERMEDIATE.*?\]', '', clean_q, flags=re.IGNORECASE)
                                # Remove quotes
                                clean_q = clean_q.strip('"').strip("'").strip()
                                # Remove explanations after question mark
                                if '?' in clean_q:
                                    parts = clean_q.split('?', 1)
                                    clean_q = parts[0].strip() + '?'
                                    if len(parts) > 1 and len(parts[1].strip()) > 10:
                                        after = parts[1].strip().lower()
                                        if any(after.startswith(word) for word in ['this', 'it', 'the question', 'specifically', 'along', 'followed', 'ensuring', 'consider', 'propose', 'enhance', 'reduce', 'increase', 'improve', 'within', 'maximizing', 'delivering']):
                                            pass
                                # Remove explanation patterns
                                explanation_patterns = [
                                    r'\s+This question.*$', r'\s+It requires.*$', r'\s+The question.*$',
                                    r'\s+This.*because.*$', r'\s+It encourages.*$', r'\s+specifically.*$',
                                    r'\s+along with.*$', r'\s+followed by.*$', r'\s+endowing with.*$',
                                    r'\s+adheres to.*$', r'\s+and ends the question.*$', r'\s+propose ways.*$',
                                    r'\s+enhance efficiency.*$', r'\s+reduce costs.*$', r'\s+increase customer.*$',
                                    r'\s+improve overall.*$', r'\s+within their organization.*$',
                                    r'\s+maximizing profit.*$', r'\s+delivering exceptional.*$',
                                    r'\s+consider their current.*$', r'\s+utilize IT solutions.*$',
                                    r'\s+think creatively.*$', r'\s+ensuring that each step.*$'
                                ]
                                for pattern in explanation_patterns:
                                    clean_q = re.sub(pattern, '', clean_q, flags=re.IGNORECASE).strip()
                                clean_q = clean_q.strip()
                                if clean_q and len(clean_q) >= 15:
                                    question_list.append(clean_q)
                                elif clean_q and len(clean_q) >= 10:
                                    question_list.append(clean_q)
                            
                            analysis = bloom_analyzer_complete.analyze_exam(
                                question_list, model, tokenizer, exam_name=exam_name
                            )
                            
                            # Update questions with cleaned versions for display (ULTRA AGGRESSIVE cleaning)
                            cleaned_questions = []
                            for q in generated_questions:
                                clean_q = q['question']
                                if not clean_q:
                                    continue
                                import re
                                # ULTRA AGGRESSIVE tag removal (3 passes)
                                for _ in range(3):
                                    clean_q = re.sub(r'\[(INTERMEDIATE|EASY|COMPLEX|intermediate|easy|complex)\]\s*', '', clean_q, flags=re.IGNORECASE)
                                    clean_q = re.sub(r'\[.*?INTERMEDIATE.*?\]', '', clean_q, flags=re.IGNORECASE)
                                    clean_q = re.sub(r'\[.*?EASY.*?\]', '', clean_q, flags=re.IGNORECASE)
                                    clean_q = re.sub(r'\[.*?COMPLEX.*?\]', '', clean_q, flags=re.IGNORECASE)
                                    clean_q = re.sub(r'\s*\[(INTERMEDIATE|EASY|COMPLEX)\]\s*', ' ', clean_q, flags=re.IGNORECASE)
                                # Remove "Explanation:", "Example:", "Note:" patterns
                                clean_q = re.sub(r'\s*Explanation:\s*.*$', '', clean_q, flags=re.IGNORECASE)
                                clean_q = re.sub(r'\s*Example:\s*.*$', '', clean_q, flags=re.IGNORECASE)
                                clean_q = re.sub(r'\s*Note:\s*.*$', '', clean_q, flags=re.IGNORECASE)
                                clean_q = re.sub(r'\s*\(Note:.*?\)', '', clean_q, flags=re.IGNORECASE)
                                clean_q = re.sub(r'\s*Please provide.*$', '', clean_q, flags=re.IGNORECASE)
                                clean_q = re.sub(r'\s*Provide details.*$', '', clean_q, flags=re.IGNORECASE)
                                clean_q = re.sub(r'\s*Create a question.*$', '', clean_q, flags=re.IGNORECASE)
                                clean_q = re.sub(r'\s*- Create.*$', '', clean_q, flags=re.IGNORECASE)
                                # Replace placeholders
                                clean_q = re.sub(r'\[technology name\]', 'technology', clean_q, flags=re.IGNORECASE)
                                clean_q = re.sub(r'\[.*?name\]', 'technology', clean_q, flags=re.IGNORECASE)
                                clean_q = re.sub(r'\[.*?\]', '', clean_q)
                                # Remove quotes
                                clean_q = clean_q.strip('"').strip("'").strip()
                                # Take ONLY up to first question mark
                                if '?' in clean_q:
                                    clean_q = clean_q.split('?')[0].strip() + '?'
                                # Limit length to 150 chars
                                if len(clean_q) > 150:
                                    truncated = clean_q[:150]
                                    cut_point = max(truncated.rfind('.'), truncated.rfind(','), truncated.rfind(' '))
                                    if cut_point > 100:
                                        clean_q = clean_q[:cut_point].strip() + '?'
                                    else:
                                        clean_q = clean_q[:150].strip() + '?'
                                # Remove explanation patterns
                                explanation_patterns = [
                                    r'\s+This question.*$', r'\s+It requires.*$', r'\s+The question.*$',
                                    r'\s+The task requires.*$', r'\s+Since.*$', r'\s+What advancements.*$',
                                    r'\s+including steps.*$', r'\s+such as.*$'
                                ]
                                for pattern in explanation_patterns:
                                    clean_q = re.sub(pattern, '', clean_q, flags=re.IGNORECASE).strip()
                                clean_q = re.sub(r'\s+', ' ', clean_q).strip()
                                if clean_q and len(clean_q) >= 10:
                                    cleaned_questions.append({
                                        'question': clean_q,
                                        'bloom_level': q['bloom_level'],
                                        'complexity': q['complexity'],
                                        'source': q.get('source', 'rag_generated')
                                    })
                            
                            generated_exam = {
                                'questions': cleaned_questions,
                                'analysis': analysis,
                                'exam_name': exam_name
                            }
                            
                        finally:
                            # Clean up temp file only if it's a temporary file (not saved book)
                            if os.path.exists(tmp_path) and not st.session_state.get("load_saved_book"):
                                # Only delete if it's in temp directory
                                if "tmp" in tmp_path or tempfile.gettempdir() in tmp_path:
                                    try:
                                        os.unlink(tmp_path)
                                    except:
                                        pass
                            
                            # Clear saved book flag
                            if "load_saved_book" in st.session_state:
                                del st.session_state["load_saved_book"]
                            if "saved_book_path" in st.session_state:
                                del st.session_state["saved_book_path"]
                                
                    except Exception as e:
                        st.error(f"Error generating exam: {str(e)}")
                        import traceback
                        with st.expander("See error details"):
                            st.code(traceback.format_exc())
    
    elif input_method == "Use Web Search":
        st.markdown("### Web Search")
        search_topics = st.text_area(
            "Topics/Search Terms",
            placeholder="e.g., Python programming, Data structures, Machine learning algorithms",
            help="Enter topics or search terms (one per line or comma-separated) to find relevant content on the web",
            height=100,
            key="exam_web_search_topics"
        )
        
        if st.button("Generate Exam Paper", type="primary", use_container_width=True):
            if total_questions < 10:
                st.error("Please enter at least 10 questions for Generate Exam Paper.")
            elif not search_topics or not search_topics.strip():
                st.warning("Please enter topics or search terms.")
            else:
                with st.spinner("Searching web and generating exam..."):
                    try:
                        # Search web for content
                        search_results = search_web_content(search_topics, num_results=5)
                        
                        if not search_results:
                            st.warning(f"⚠️ No web results found for '{search_topics}'. This could be due to:\n"
                                      f"- Network connectivity issues\n"
                                      f"- DuckDuckGo service temporarily unavailable\n"
                                      f"- Search query too specific\n\n"
                                      f"**Suggestions:**\n"
                                      f"- Try a more general search query\n"
                                      f"- Check your internet connection\n"
                                      f"- Try again in a few moments")
                        else:
                            # Use Gemini if available, otherwise use analysis model
                            gemini_key = os.getenv("GEMINI_API_KEY")
                            gemini_available = bool(gemini_key)
                            
                            if gemini_available:
                                rag_generator = RAGExamGenerator(
                                    llm_api="gemini",
                                    use_local_vector_store=True,
                                    api_key=gemini_key
                                )
                            else:
                                # Use analysis model directly
                                rag_generator = RAGExamGenerator(
                                    llm_api="local",
                                    use_local_vector_store=True,
                                    local_model=model,
                                    local_tokenizer=tokenizer
                                )
                            
                            # Add web search results as content
                            with st.spinner("Processing web content and creating vector store..."):
                                combined_text = "\n\n".join([f"{r['title']}\n{r['text']}" for r in search_results])
                                rag_generator.add_content(combined_text, source_type="text",
                                                         metadata={"topic": search_topics, "source": "web_search", 
                                                                  "exam_name": exam_name})
                            
                            # Generate exam with complexity and optional specific bloom level
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            def update_progress(progress, status):
                                # Clamp progress to valid [0.0, 1.0] range to avoid Streamlit errors
                                safe_progress = max(0.0, min(1.0, float(progress)))
                                progress_bar.progress(safe_progress)
                                status_text.text(status)
                            
                            generated_questions = rag_generator.generate_exam_from_content(
                                total_questions=total_questions,
                                topic=search_topics,
                                complexity=complexity.lower(),
                                specific_bloom_level=specific_bloom_level,
                                progress_callback=update_progress
                            )
                            
                            progress_bar.empty()
                            status_text.empty()
                            
                            # Analyze generated exam
                            # Clean questions before analysis
                            question_list = []
                            for q in generated_questions:
                                clean_q = q['question']
                                if not clean_q:
                                    continue
                                import re
                                # Remove complexity tags
                                clean_q = re.sub(r'\[(INTERMEDIATE|EASY|COMPLEX)\]\s*', '', clean_q, flags=re.IGNORECASE)
                                clean_q = re.sub(r'\[.*?INTERMEDIATE.*?\]', '', clean_q, flags=re.IGNORECASE)
                                # Remove quotes
                                clean_q = clean_q.strip('"').strip("'").strip()
                                # Remove explanations after question mark
                                if '?' in clean_q:
                                    parts = clean_q.split('?', 1)
                                    clean_q = parts[0].strip() + '?'
                                    if len(parts) > 1 and len(parts[1].strip()) > 10:
                                        after = parts[1].strip().lower()
                                        if any(after.startswith(word) for word in ['this', 'it', 'the question', 'specifically', 'along', 'followed', 'ensuring', 'consider', 'propose', 'enhance', 'reduce', 'increase', 'improve', 'within', 'maximizing', 'delivering']):
                                            pass
                                # Remove explanation patterns
                                explanation_patterns = [
                                    r'\s+This question.*$', r'\s+It requires.*$', r'\s+The question.*$',
                                    r'\s+This.*because.*$', r'\s+It encourages.*$', r'\s+specifically.*$',
                                    r'\s+along with.*$', r'\s+followed by.*$', r'\s+endowing with.*$',
                                    r'\s+adheres to.*$', r'\s+and ends the question.*$', r'\s+propose ways.*$',
                                    r'\s+enhance efficiency.*$', r'\s+reduce costs.*$', r'\s+increase customer.*$',
                                    r'\s+improve overall.*$', r'\s+within their organization.*$',
                                    r'\s+maximizing profit.*$', r'\s+delivering exceptional.*$',
                                    r'\s+consider their current.*$', r'\s+utilize IT solutions.*$',
                                    r'\s+think creatively.*$', r'\s+ensuring that each step.*$'
                                ]
                                for pattern in explanation_patterns:
                                    clean_q = re.sub(pattern, '', clean_q, flags=re.IGNORECASE).strip()
                                clean_q = clean_q.strip()
                                if clean_q and len(clean_q) >= 15:
                                    question_list.append(clean_q)
                                elif clean_q and len(clean_q) >= 10:
                                    question_list.append(clean_q)
                            
                            analysis = bloom_analyzer_complete.analyze_exam(
                                question_list, model, tokenizer, exam_name=exam_name
                            )
                            
                            # Update questions with cleaned versions for display (ULTRA AGGRESSIVE cleaning)
                            cleaned_questions = []
                            for q in generated_questions:
                                clean_q = q['question']
                                if not clean_q:
                                    continue
                                import re
                                # ULTRA AGGRESSIVE tag removal (3 passes)
                                for _ in range(3):
                                    clean_q = re.sub(r'\[(INTERMEDIATE|EASY|COMPLEX|intermediate|easy|complex)\]\s*', '', clean_q, flags=re.IGNORECASE)
                                    clean_q = re.sub(r'\[.*?INTERMEDIATE.*?\]', '', clean_q, flags=re.IGNORECASE)
                                    clean_q = re.sub(r'\[.*?EASY.*?\]', '', clean_q, flags=re.IGNORECASE)
                                    clean_q = re.sub(r'\[.*?COMPLEX.*?\]', '', clean_q, flags=re.IGNORECASE)
                                    clean_q = re.sub(r'\s*\[(INTERMEDIATE|EASY|COMPLEX)\]\s*', ' ', clean_q, flags=re.IGNORECASE)
                                # Remove "Explanation:", "Example:", "Note:" patterns
                                clean_q = re.sub(r'\s*Explanation:\s*.*$', '', clean_q, flags=re.IGNORECASE)
                                clean_q = re.sub(r'\s*Example:\s*.*$', '', clean_q, flags=re.IGNORECASE)
                                clean_q = re.sub(r'\s*Note:\s*.*$', '', clean_q, flags=re.IGNORECASE)
                                clean_q = re.sub(r'\s*\(Note:.*?\)', '', clean_q, flags=re.IGNORECASE)
                                clean_q = re.sub(r'\s*Please provide.*$', '', clean_q, flags=re.IGNORECASE)
                                clean_q = re.sub(r'\s*Provide details.*$', '', clean_q, flags=re.IGNORECASE)
                                clean_q = re.sub(r'\s*Create a question.*$', '', clean_q, flags=re.IGNORECASE)
                                clean_q = re.sub(r'\s*- Create.*$', '', clean_q, flags=re.IGNORECASE)
                                # Replace placeholders
                                clean_q = re.sub(r'\[technology name\]', 'technology', clean_q, flags=re.IGNORECASE)
                                clean_q = re.sub(r'\[.*?name\]', 'technology', clean_q, flags=re.IGNORECASE)
                                clean_q = re.sub(r'\[.*?\]', '', clean_q)
                                # Remove quotes
                                clean_q = clean_q.strip('"').strip("'").strip()
                                # Take ONLY up to first question mark
                                if '?' in clean_q:
                                    clean_q = clean_q.split('?')[0].strip() + '?'
                                # Limit length to 150 chars
                                if len(clean_q) > 150:
                                    truncated = clean_q[:150]
                                    cut_point = max(truncated.rfind('.'), truncated.rfind(','), truncated.rfind(' '))
                                    if cut_point > 100:
                                        clean_q = clean_q[:cut_point].strip() + '?'
                                    else:
                                        clean_q = clean_q[:150].strip() + '?'
                                # Remove explanation patterns
                                explanation_patterns = [
                                    r'\s+This question.*$', r'\s+It requires.*$', r'\s+The question.*$',
                                    r'\s+The task requires.*$', r'\s+Since.*$', r'\s+What advancements.*$',
                                    r'\s+including steps.*$', r'\s+such as.*$'
                                ]
                                for pattern in explanation_patterns:
                                    clean_q = re.sub(pattern, '', clean_q, flags=re.IGNORECASE).strip()
                                clean_q = re.sub(r'\s+', ' ', clean_q).strip()
                                if clean_q and len(clean_q) >= 10:
                                    cleaned_questions.append({
                                        'question': clean_q,
                                        'bloom_level': q['bloom_level'],
                                        'complexity': q['complexity'],
                                        'source': q.get('source', 'rag_generated')
                                    })
                            
                            generated_exam = {
                                'questions': cleaned_questions,
                                'analysis': analysis,
                                'exam_name': exam_name
                            }
                            
                    except Exception as e:
                        st.error(f"Error generating exam: {str(e)}")
                        import traceback
                        with st.expander("See error details"):
                            st.code(traceback.format_exc())
    
    else:  # Enter Text Content
        st.markdown("### Enter Content Text")
        content_text = st.text_area(
            "Paste or enter content",
            height=300,
            placeholder="Paste your content here...",
            help="Enter the content/topics from which to generate exam questions",
            key="exam_text_content"
        )
        
        # Topic input for context
        topic = st.text_input(
            "Subject/Topic Area (Optional)", 
            value="",
            help="Enter the subject or topic area to help generate relevant questions",
            key="topic_text"
        )
        
        if st.button("Generate Exam Paper", type="primary", use_container_width=True):
            if total_questions < 10:
                st.error("Please enter at least 10 questions for Generate Exam Paper.")
            elif not content_text or not content_text.strip():
                st.warning("Please enter content text.")
            else:
                with st.spinner("Processing content and generating exam..."):
                    try:
                        # Use Gemini if available, otherwise Qwen (backend handles this)
                        gemini_key = os.getenv("GEMINI_API_KEY")
                        gemini_available = bool(gemini_key)
                        
                        if gemini_available:
                            rag_generator = RAGExamGenerator(
                                llm_api="gemini",
                                use_local_vector_store=True,
                                api_key=gemini_key
                            )
                        else:
                            rag_generator = RAGExamGenerator(
                                llm_api="local",
                                use_local_vector_store=True,
                                use_optimized_generation=True,
                                generation_model_name="qwen2.5-1.5b"
                            )
                        
                        # Add text content to RAG
                        with st.spinner("Processing content and creating vector store..."):
                            rag_generator.add_content(content_text, source_type="text",
                                                     metadata={"topic": topic or "the entered content", "exam_name": exam_name})
                        
                        # Generate exam with complexity and optional specific bloom level
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        def update_progress(progress, status):
                            # Clamp progress to valid [0.0, 1.0] range to avoid Streamlit errors
                            safe_progress = max(0.0, min(1.0, float(progress)))
                            progress_bar.progress(safe_progress)
                            status_text.text(status)
                        
                        generated_questions = rag_generator.generate_exam_from_content(
                            total_questions=total_questions,
                            topic=topic or "the entered content",
                            complexity=complexity.lower(),
                            specific_bloom_level=specific_bloom_level,
                            progress_callback=update_progress
                        )
                        
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Analyze generated exam
                        # Clean questions before analysis (same aggressive cleaning)
                        question_list = []
                        for q in generated_questions:
                            clean_q = q['question']
                            if not clean_q:
                                continue
                            import re
                            # ULTRA AGGRESSIVE tag removal
                            for _ in range(3):
                                clean_q = re.sub(r'\[(INTERMEDIATE|EASY|COMPLEX|intermediate|easy|complex)\]\s*', '', clean_q, flags=re.IGNORECASE)
                                clean_q = re.sub(r'\[.*?COMPLEX.*?\]', '', clean_q, flags=re.IGNORECASE)
                            clean_q = re.sub(r'\s*Explanation:\s*.*$', '', clean_q, flags=re.IGNORECASE)
                            clean_q = re.sub(r'\s*Example:\s*.*$', '', clean_q, flags=re.IGNORECASE)
                            clean_q = re.sub(r'\s*Note:\s*.*$', '', clean_q, flags=re.IGNORECASE)
                            clean_q = clean_q.strip('"').strip("'").strip()
                            if '?' in clean_q:
                                clean_q = clean_q.split('?')[0].strip() + '?'
                            if len(clean_q) > 150:
                                clean_q = clean_q[:150].rstrip(' ,.') + '?'
                            clean_q = re.sub(r'\s+', ' ', clean_q).strip()
                            if clean_q and len(clean_q) >= 10:
                                question_list.append(clean_q)
                        
                        analysis = bloom_analyzer_complete.analyze_exam(
                            question_list, model, tokenizer, exam_name=exam_name
                        )
                        
                        generated_exam = {
                            'questions': generated_questions,
                            'analysis': analysis,
                            'exam_name': exam_name
                        }
                        
                    except Exception as e:
                        st.error(f"Error generating exam: {str(e)}")
                        import traceback
                        with st.expander("See error details"):
                            st.code(traceback.format_exc())
    
    # Display generated exam if available
    if generated_exam:
        st.markdown("---")
        st.markdown("### Generated Exam Paper")
        
        # Metrics
        analysis = generated_exam['analysis']
        generated_questions = generated_exam['questions']
        total_generated = len(generated_questions)
        
        # Calculate actual distribution from generated questions' bloom_level field
        from utils.bloom_analyzer_complete import IDEAL_DISTRIBUTION
        actual_distribution = {}
        for level in bloom_analyzer_complete.BLOOM_LEVELS:
            count = sum(1 for q in generated_questions if q.get('bloom_level') == level)
            percentage = (count / total_generated * 100) if total_generated > 0 else 0
            ideal_pct = IDEAL_DISTRIBUTION[level] * 100
            actual_distribution[level] = {
                'count': count,
                'actual': percentage,
                'ideal': ideal_pct,
                'difference': percentage - ideal_pct
            }
        
        # Calculate quality score based on actual distribution
        total_deviation = sum(abs(actual_distribution[level]['difference']) for level in bloom_analyzer_complete.BLOOM_LEVELS)
        quality_score = max(0, 100 - (total_deviation / 2))
        if quality_score >= 90:
            quality_rating = "Excellent"
        elif quality_score >= 80:
            quality_rating = "Good"
        elif quality_score >= 70:
            quality_rating = "Fair"
        elif quality_score >= 60:
            quality_rating = "Needs Improvement"
        else:
            quality_rating = "Poor"
        
        # Show improvement comparison if this is an improved exam
        if 'original_analysis' in generated_exam and 'improvement_delta' in generated_exam:
            st.markdown("### 📈 Improvement Results")
            orig_score = generated_exam['original_analysis']['quality_score']
            new_score = quality_score
            delta = generated_exam['improvement_delta']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Original Quality", f"{orig_score:.1f}%")
            with col2:
                st.metric("Improved Quality", f"{new_score:.1f}%")
            with col3:
                st.metric("Improvement", f"{delta:+.1f}", delta=f"{delta:.1f} points")
            with col4:
                st.metric("Total Questions", total_generated)
            st.markdown("---")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Questions", total_generated)
            with col2:
                st.metric("Quality Score", f"{quality_score:.1f}%")
            with col3:
                st.metric("Rating", quality_rating)
        
        # Distribution table - using actual generated distribution
        st.markdown("#### Bloom Level Distribution")
        rows = []
        for level in bloom_analyzer_complete.BLOOM_LEVELS:
            dist = actual_distribution[level]
            rows.append({
                "Level": level,
                "Count": dist["count"],
                "Percentage": f"{dist['actual']:.1f}%",
                "Ideal %": f"{dist['ideal']:.1f}%",
                "Difference": f"{dist['difference']:+.1f}%",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
        
        # Questions grouped by Bloom level
        st.markdown("#### Generated Questions by Bloom Level")
        questions_by_level = {}
        for q in generated_exam['questions']:
            level = q['bloom_level']
            if level not in questions_by_level:
                questions_by_level[level] = []
            questions_by_level[level].append({
                'question': q['question'],
                'complexity': q.get('complexity', 'intermediate').upper()
            })
        
        for level in bloom_analyzer_complete.BLOOM_LEVELS:
            if level in questions_by_level:
                with st.expander(f"{level} ({len(questions_by_level[level])} questions)", expanded=True):
                    for i, item in enumerate(questions_by_level[level], 1):
                        # Clean question one more time to ensure no tags
                        clean_q = item['question']
                        import re
                        # Remove any remaining tags
                        for _ in range(3):
                            clean_q = re.sub(r'\[(INTERMEDIATE|EASY|COMPLEX|intermediate|easy|complex)\]\s*', '', clean_q, flags=re.IGNORECASE)
                        # Remove complexity badge - don't display it
                        st.write(f"**{i}.** {clean_q}")
        
        # Visual chart - create analysis object from actual distribution
        actual_analysis = {
            'comparison': actual_distribution,
            'quality_score': quality_score,
            'quality_rating': quality_rating,
            'total_questions': total_generated
        }
        st.markdown("#### Visual Analysis")
        chart_fig = create_bloom_chart(actual_analysis)
        st.plotly_chart(chart_fig, use_container_width=True)
        
        # Download options
        st.markdown("---")
        st.markdown("####  Download Exam Paper")
        
        # Text format
        exam_text = f"{generated_exam['exam_name']}\n{'='*70}\n\n"
        question_num = 1
        for level in bloom_analyzer_complete.BLOOM_LEVELS:
            if level in questions_by_level:
                exam_text += f"\n{level.upper()}\n{'-'*70}\n\n"
                for question in questions_by_level[level]:
                    exam_text += f"{question_num}. {question}\n\n"
                    question_num += 1
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download as Text (.txt)",
                data=exam_text,
                file_name=f"{generated_exam['exam_name'].replace(' ', '_')}.txt",
                mime="text/plain"
            )
        with col2:
            import pandas as pd
            df = pd.DataFrame(generated_exam['questions'])
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="Download as CSV (.csv)",
                data=csv_data,
                file_name=f"{generated_exam['exam_name'].replace(' ', '_')}.csv",
                mime="text/csv"
            )
        
        # Clear generation mode
        if "generation_mode" in st.session_state:
            del st.session_state["generation_mode"]
    
    # Close container div
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer removed


def main():
    st.set_page_config(
        page_title="Cognivise – Bloom's Taxonomy Analyzer", 
        page_icon=None, 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Fix: Load Material Icons font from CDN before apply_cognivise_theme
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Material+Icons"
          rel="stylesheet">

    <style>
        /* Force override of ANY Material Icons font Streamlit tries to load */
        @font-face {
            font-family: 'Material Icons';
            src: url('https://fonts.gstatic.com/s/materialicons/v140/flUhRq6tzZclQEJ-Vdg-IuiaDsNc.woff2')
                 format('woff2');
            font-weight: normal;
            font-style: normal;
        }

        .material-icons {
            font-family: 'Material Icons' !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    apply_cognivise_theme()
    
    # Initialize session FIRST
    init_session()
    
    # Get current page from session state
    page = st.session_state.get("page", "landing")
    
    # Check for dashboard_page query parameter
    try:
        if "dashboard_page" in st.query_params:
            dashboard_page_param = st.query_params["dashboard_page"]
            if isinstance(dashboard_page_param, list):
                dashboard_page_param = dashboard_page_param[0] if dashboard_page_param else None
            if dashboard_page_param:
                st.session_state["dashboard_page"] = dashboard_page_param
    except:
        pass
    
    # Route to appropriate page based on session state
    if page == "landing":
        page_landing()
    elif page == "about":
        page_about()
    elif page == "signin":
        page_signin()
    elif page == "dashboard":
        page_dashboard()
    elif page == "analyze":
        page_analyze()
    elif page == "generate":
        page_generate()
    elif page == "generate_exam":
        page_generate_exam()
    else:
        # Default to landing if unknown page
        st.session_state["page"] = "landing"
        st.rerun()


if __name__ == "__main__":
    main()

