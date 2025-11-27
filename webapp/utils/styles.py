import streamlit as st
from pathlib import Path
import base64


def get_material_icons_font_data_uri() -> str:
    """Load Material Icons font as data URI."""
    static_dir = Path(__file__).parent.parent / "static"
    font_path = static_dir / "MaterialIcons-Regular.woff2"
    
    if font_path.exists():
        try:
            with font_path.open("rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            return f"data:font/woff2;base64,{data}"
        except Exception:
            pass
    
    return ""


def apply_cognivise_theme() -> None:
    """Inject futuristic dark purple/neon theme for Cognivise."""
    # Load Material Icons font as data URI
    material_icons_font = get_material_icons_font_data_uri()
    
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Space+Grotesk:wght@300;400;500;600;700&family=Inter+Tight:wght@300;400;500;600;700;800&family=Poppins:wght@200;300;400;500;600;700;800&display=swap');
    
    /* Material Icons Font - Load locally via data URI */
    @font-face {
        font-family: 'Material Icons';
        src: MATERIAL_ICONS_FONT_PLACEHOLDER;
        font-weight: normal;
        font-style: normal;
        font-display: swap;
    }
    
    /* Graphik Font - DISABLED (files missing/corrupted) */
    /* Using system fonts as fallback instead */
    /* Note: Graphik is a commercial font. If you have the font files, place them in /static/fonts/ directory.
    @font-face {
        font-family: 'Graphik';
        src: local('Graphik Regular'),
             local('Graphik-Regular'),
             url('/static/fonts/Graphik-Regular.woff2') format('woff2'),
             url('/static/fonts/Graphik-Regular.woff') format('woff');
        font-weight: 400;
        font-style: normal;
        font-display: swap;
    }
    
    @font-face {
        font-family: 'Graphik';
        src: local('Graphik Semibold'),
             local('Graphik-Semibold'),
             url('/static/fonts/Graphik-Semibold.woff2') format('woff2'),
             url('/static/fonts/Graphik-Semibold.woff') format('woff');
        font-weight: 600;
        font-style: normal;
        font-display: swap;
    }
    */

    :root {
        /* Brand Color Palette - UI Color Kit Gradient System
           24-step gradient from light lavender (#E2DDF8) to dark indigo/black
           See color_kit.md for full documentation */
        --brand-white: #FFFFFF;
        --brand-lavender-light: #E2DDF8; /* Lightest lavender from color kit */
        --brand-lavender: #DDA0DD;
        --brand-lavender-medium: #C8A2C8;
        --brand-purple-light: #B19CD9;
        --brand-purple: #9B7ED9;
        --brand-purple-medium: #8B6F9E;
        --brand-purple-deep: #7A5F8F;
        --brand-purple-dark: #6B4E7F;
        --brand-purple-darker: #5D3D6F;
        --brand-purple-darkest: #4A2D5A;
        --brand-dark: #2D1B3D;
        --brand-darker: #1F0F2F;
        --brand-darkest: #0F0519;
        --brand-black: #000000;
        
        /* Semantic color mappings */
        --bg-primary: var(--brand-darkest);
        --bg-secondary: var(--brand-darker);
        --bg-tertiary: var(--brand-dark);
        --text-primary: var(--brand-white);
        --text-secondary: var(--brand-lavender-light);
        --accent-primary: var(--brand-purple);
        --accent-secondary: var(--brand-purple-medium);
        --accent-hover: var(--brand-lavender);
        
        /* Glow and shadow colors - using brand colors only */
        --glow-purple: var(--brand-purple);
        --glow-purple-light: var(--brand-purple-light);
        --glow-purple-medium: var(--brand-purple-medium);
    }

    * {
        box-sizing: border-box;
        font-family: 'Graphik', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
        font-weight: 400;
    }
    
    /* Headings - Semibold */
    h1, h2, h3, h4, h5, h6,
    .hero-title,
    .page-header,
    .form-title,
    .section-card h2,
    .auth-card h2,
    .feature-title,
    .dashboard-title {
        font-family: 'Graphik', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
        font-weight: 600 !important;
    }

    html, body {
        height: 100%;
        min-height: 100vh;
        margin: 0;
        padding: 0;
        overflow-x: hidden;
        overflow-y: hidden;
        width: 100vw;
        background: linear-gradient(135deg, var(--brand-darkest) 0%, var(--brand-darker) 25%, var(--brand-dark) 50%, var(--brand-purple-darker) 75%, var(--brand-darkest) 100%);
        color: var(--text-primary);
    }
    
    /* Prevent scrolling on sign-in page */
    body[data-page="signin"],
    html {
        overflow: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
        height: 100vh !important;
        max-height: 100vh !important;
    }
    
    /* Hide Streamlit header/footer that might cause scrolling */
    #MainMenu,
    header[data-testid="stHeader"],
    footer[data-testid="stFooter"],
    .stDeployButton {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Remove Streamlit default padding and max-width restrictions */
    .stApp {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100vw !important;
        width: 100vw !important;
        height: 100vh !important;
        max-height: 100vh !important;
        overflow: hidden !important;
    }

    .main {
        padding: 0 !important;
        margin: 0 !important;
        height: 100vh !important;
        max-height: 100vh !important;
        overflow: hidden !important;
    }

    /* Default block-container - will be overridden by signin page styles */
    .main .block-container {
        padding-left: 0 !important;
        padding-right: 0 !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        max-width: 100% !important;
        width: 100% !important;
        height: 100vh !important;
        max-height: 100vh !important;
        overflow: hidden !important;
    }
    
    /* Signin page block-container override - must come after default */
    .main .block-container:has(.glass-card-wrapper),
    .main .block-container:has(.form-header),
    .main .block-container:has(.form-title) {
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
    }

    /* Remove Streamlit's default section padding */
    section[data-testid="stMarkdownContainer"] {
        padding: 0 !important;
        margin: 0 !important;
    }

    /* Utility helpers approximating Tailwind classes used in the markup */
    .relative { position: relative; }
    .absolute { position: absolute; }
    .w-full { width: 100%; }
    .min-h-screen { min-height: 100vh; }
    .overflow-hidden { overflow: hidden; }
    .z-20 { z-index: 20; }
    .z-50 { z-index: 50; }
    .z-10 { z-index: 10; }
    .max-w-7xl { max-width: 80rem; }
    .max-w-5xl { max-width: 64rem; }
    .mx-auto { margin-left: auto; margin-right: auto; }
    .px-6 { padding-left: 1.5rem; padding-right: 1.5rem; }
    .px-8 { padding-left: 2rem; padding-right: 2rem; }
    .py-6 { padding-top: 1.5rem; padding-bottom: 1.5rem; }
    .pt-40 { padding-top: 10rem; }
    .pb-32 { padding-bottom: 8rem; }
    .flex { display: flex; }
    .justify-between { justify-content: space-between; }
    .items-center { align-items: center; }
    .inset-0 { top: 0; right: 0; bottom: 0; left: 0; }

    .top-nav-shell {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        width: 100% !important;
        z-index: 100 !important;
        background: transparent; /* Transparent to show hero image background */
        border-bottom: none;
        box-shadow: none;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        margin: 0 !important;
        padding: 0 !important;
        box-sizing: border-box !important; /* Ensure proper width calculation */
    }

    .top-nav-shell:hover {
        background: transparent; /* Keep transparent on hover */
        border-bottom: none;
    }

    /* Header - Classy Minimal Navbar (matching modern-ui landing page) */
    .header {
        position: relative;
        z-index: 10;
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        align-items: center; /* Vertically center all items */
        padding: 16px 48px; /* Reduced top/bottom padding to move navbar up */
        padding-left: 48px !important; /* Ensure consistent left padding */
        padding-right: 48px !important; /* Ensure consistent right padding */
        backdrop-filter: blur(10px);
        max-width: 100%;
        width: 100%;
        min-height: auto; /* Remove fixed min-height to allow tighter spacing */
        margin: 0 !important; /* Ensure no margin shifts */
        box-sizing: border-box !important; /* Include padding in width calculation */
    }

    .logo {
        display: flex;
        align-items: center; /* Vertically center logo */
        gap: 12px;
        font-size: 24px;
        font-weight: 700;
        height: 100%; /* Take full height of header */
        margin: 0 !important; /* Ensure no margin shifts */
        padding: 0 !important; /* Ensure no padding shifts */
        justify-self: start; /* Align logo to start of grid column */
        /* Ensure logo container doesn't flip */
        transform: none !important;
        direction: ltr !important;
    }

    .logo-icon {
        width: 200px; /* Significantly increased for better visibility */
        height: auto; /* Allow height to adjust based on logo aspect ratio */
        min-height: 70px;
        max-height: 120px; /* Increased to allow larger logo display */
        display: flex;
        align-items: center;
        justify-content: flex-start;
        flex-shrink: 0;
        overflow: visible;
        margin: 0 !important; /* Ensure no margin shifts */
        padding: 0 !important; /* Ensure no padding shifts */
        /* Ensure no transforms on container */
        transform: none !important;
        -webkit-transform: none !important;
        -moz-transform: none !important;
        -ms-transform: none !important;
        -o-transform: none !important;
    }
    
    .logo-icon svg {
        width: 100%;
        height: 100%;
        display: block;
        /* Ensure no flipping or rotation - explicitly set all transform properties */
        transform: none !important;
        -webkit-transform: none !important;
        -moz-transform: none !important;
        -ms-transform: none !important;
        -o-transform: none !important;
        visibility: visible !important;
        opacity: 1 !important;
        /* Ensure proper rendering */
        vertical-align: baseline !important;
        direction: ltr !important;
        /* Prevent any scaling or mirroring */
        scale: 1 !important;
        rotate: 0deg !important;
    }
    
    /* PNG logo image styling */
    .logo-icon img {
        width: 100%;
        height: auto; /* Maintain aspect ratio */
        max-height: 120px; /* Increased to allow larger logo display */
        display: block;
        object-fit: contain;
        /* Ensure no flipping or rotation - explicitly set all transform properties */
        transform: none !important;
        -webkit-transform: none !important;
        -moz-transform: none !important;
        -ms-transform: none !important;
        -o-transform: none !important;
        visibility: visible !important;
        opacity: 1 !important;
        /* Ensure proper rendering */
        vertical-align: baseline !important;
        direction: ltr !important;
        /* Prevent any scaling or mirroring */
        scale: 1 !important;
        rotate: 0deg !important;
    }
    
    /* Ensure logo path is not flipped */
    .logo-icon svg path {
        transform: none !important;
        -webkit-transform: none !important;
        -moz-transform: none !important;
        -ms-transform: none !important;
        -o-transform: none !important;
    }
    
    /* Ensure no parent transforms affect logo */
    .logo * {
        transform: none !important;
    }

    .logo-text {
        background: linear-gradient(135deg, 
            var(--brand-lavender) 0%,
            var(--brand-purple-light) 30%,
            var(--brand-purple) 60%,
            var(--brand-purple-medium) 100%
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-family: 'Graphik', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 600;
        letter-spacing: -0.01em;
        filter: drop-shadow(0 1px 3px rgba(0, 0, 0, 0.2));
    }

    .nav-menu {
        display: flex;
        align-items: center; /* Vertically center menu items */
        justify-content: center; /* Center the nav items horizontally */
        gap: 0;
        height: 100%; /* Take full height of header */
        justify-self: center; /* Center the nav menu in its grid column */
    }

    .nav-menu a {
        font-family: 'Graphik', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #ffffff;
        text-decoration: none;
        font-size: 15px;
        font-weight: 400;
        padding: 0 16px;
        transition: color 0.3s;
        position: relative;
    }

    .nav-menu a:not(:last-child)::after {
        content: '';
        position: absolute;
        right: 0;
        top: 50%;
        transform: translateY(-50%);
        width: 1px;
        height: 14px;
        background: rgba(255, 255, 255, 0.3);
    }

    .nav-menu a:hover {
        color: var(--brand-lavender);
    }

    .header-actions {
        display: flex;
        align-items: center; /* Vertically center actions */
        justify-content: flex-end; /* Align actions to the right */
        gap: 0;
        height: 100%; /* Take full height of header */
        justify-self: end; /* Align actions to end of grid column */
    }

    .header-actions a {
        color: #ffffff !important;
        text-decoration: none !important;
    }

    .header-actions a:link,
    .header-actions a:visited,
    .header-actions a:active {
        color: #ffffff !important;
        text-decoration: none !important;
    }

    .header-actions a:hover {
        color: var(--brand-lavender) !important;
        text-decoration: none !important;
    }

    .btn-login,
    .btn-signup {
        font-family: 'Graphik', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #ffffff !important;
        text-decoration: none !important;
        font-size: 15px;
        font-weight: 400;
        padding: 0 16px;
        cursor: pointer;
        transition: color 0.3s;
        background: none;
        border: none;
        position: relative;
    }

    .btn-login:link,
    .btn-login:visited,
    .btn-login:active,
    .btn-signup:link,
    .btn-signup:visited,
    .btn-signup:active {
        color: #ffffff !important;
        text-decoration: none !important;
    }

    .btn-login::after {
        display: none !important;
        content: none !important;
    }

    .btn-login:hover,
    .btn-signup:hover {
        color: var(--brand-lavender) !important;
        text-decoration: none !important;
    }

    /* Legacy nav styles for backward compatibility */
    .top-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        width: 100%;
        padding: 0 2rem;
    }

    .nav-links {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        width: auto;
    }

    .nav-button {
        padding: 0.5rem 1.25rem;
        border-radius: 0.75rem;
        border: 1px solid rgba(155, 126, 217, 0.3);
        background: rgba(45, 27, 61, 0.2);
        color: #ffffff !important;
        font-size: 0.9rem;
        font-weight: 500;
        letter-spacing: 0.02em;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 0 5px rgba(155, 126, 217, 0.08);
        text-decoration: none !important;
        display: inline-block;
    }

    .nav-button:hover {
        background: rgba(155, 126, 217, 0.3);
        border-color: rgba(155, 126, 217, 0.5);
        box-shadow: 0 0 8px rgba(155, 126, 217, 0.2);
        transform: translateY(-1px);
        color: #ffffff !important;
        text-decoration: none !important;
    }

    .nav-button-primary {
        background: linear-gradient(135deg, var(--brand-purple-dark) 0%, var(--brand-purple-darker) 100%) !important;
        border: 1px solid rgba(155, 126, 217, 0.5) !important;
        box-shadow: 0 0 10px rgba(155, 126, 217, 0.2), 0 0 15px rgba(139, 111, 158, 0.12);
        color: #ffffff !important;
        text-decoration: none !important;
    }

    .nav-button-primary:hover {
        background: linear-gradient(135deg, var(--brand-purple-darker) 0%, var(--brand-purple-darkest) 100%) !important;
        box-shadow: 0 0 12px rgba(155, 126, 217, 0.25), 0 0 20px rgba(139, 111, 158, 0.15);
        transform: translateY(-2px);
        color: #ffffff !important;
        text-decoration: none !important;
    }
    
    .nav-button:visited,
    .nav-button:link,
    .nav-button:active {
        color: #ffffff !important;
        text-decoration: none !important;
    }

    .hero-nav {
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        width: 100vw !important;
        z-index: 30 !important;
        pointer-events: none;
    }

    .hero-nav-inner {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        gap: 1.5rem;
        pointer-events: auto;
        margin-top: 0;
        padding-top: 2.5rem;
        padding-bottom: 1.25rem;
        padding-right: 2rem;
        width: 100%;
        box-sizing: border-box;
    }

    /* Hero (acts like: relative min-h-screen w-full overflow-hidden bg-black) */
    .hero-shell {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        bottom: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        min-height: 100vh !important;
        max-width: 100vw !important;
        overflow: hidden;
        background: linear-gradient(135deg, var(--brand-darkest) 0%, var(--brand-darker) 50%, var(--brand-darkest) 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 !important;
        padding: 0 !important;
        z-index: 1;
    }

    /* Floating shapes animation */
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(5deg); }
    }

    .floating-shape {
        position: absolute;
        opacity: 0.1;
        animation: float 6s ease-in-out infinite;
    }

    /* Background image layer - full viewport */
    .hero-bg {
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        bottom: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        background-size: cover !important;
        background-position: center !important;
        background-repeat: no-repeat !important;
        object-fit: cover;
    }

    /* Gradient overlay layer - full viewport */
    .hero-gradient {
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        bottom: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        background: linear-gradient(
            135deg,
            rgba(45, 27, 61, 0.7) 0%,
            rgba(93, 61, 111, 0.3) 35%,
            rgba(15, 5, 25, 0.9) 100%
        ) !important;
    }

        /* Dark purple gradient overlay for landing page - 30% opacity */
    .hero-gradient-overlay {
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        bottom: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        background: linear-gradient(
            135deg,
            rgba(45, 27, 61, 0.3) 0%,
            rgba(93, 61, 111, 0.3) 50%,
            rgba(15, 5, 25, 0.3) 100%
        ) !important;
        z-index: 5 !important;
    }

    /* Content wrapper - responsive and centered */
    .hero-content-shell {
        position: relative !important;
        z-index: 10 !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        text-align: center !important;
        max-width: 60rem;
        margin: 0 auto;
        padding: 8rem 2rem 6rem 2rem;
        width: 100%;
        box-sizing: border-box;
    }
    
    /* Ensure hero content is above background layers */
    .hero-content-shell,
    .hero-content-shell * {
        position: relative;
        z-index: 10;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .hero-content-shell {
            padding: 6rem 1.5rem 4rem 1.5rem;
        }
        
        .hero-title {
            font-size: 2rem !important;
        }
        
        .hero-subtitle {
            font-size: 1rem !important;
        }
        
        .hero-tagline {
            font-size: 0.9rem !important;
        }
        
        .hero-cta {
            padding: 0.75rem 1.5rem !important;
            font-size: 0.9rem !important;
        }
        
        .hero-cta-row {
            flex-direction: column;
            width: 100%;
        }
        
        .hero-cta-row button {
            width: 100%;
        }
        
        .hero-nav-inner {
            padding-right: 1rem;
        }
        
        /* Navbar responsive */
        .header {
            padding: 24px 24px !important;
            grid-template-columns: 1fr auto 1fr; /* Maintain grid layout on tablet */
        }
        
        .logo {
            font-size: 20px;
        }
        
        .logo-icon {
            width: 80px; /* Increased proportionally */
            height: auto;
            min-height: 50px;
            max-height: 120px; /* Proportional to desktop max-height */
            font-size: 18px;
        }
        
        .nav-menu a {
            font-size: 13px;
            padding: 0 12px;
        }
        
        .btn-login,
        .btn-signup {
            font-size: 13px;
            padding: 0 12px;
        }
    }
    
    @media (max-width: 480px) {
        .hero-title {
            font-size: 1.5rem !important;
        }
        
        .hero-subtitle {
            font-size: 0.9rem !important;
        }
        
        .hero-tagline {
            font-size: 0.85rem !important;
        }
        
        .nav-button {
            padding: 0.4rem 1rem !important;
            font-size: 0.8rem !important;
        }
        
        /* Navbar responsive - mobile */
        .header {
            padding: 20px 16px !important;
            grid-template-columns: 1fr; /* Stack items vertically on mobile */
            grid-template-rows: auto auto auto;
            gap: 12px;
        }
        
        .logo {
            font-size: 18px;
        }
        
        .logo-icon {
            width: 60px; /* Increased proportionally */
            height: auto;
            min-height: 40px;
            max-height: 90px; /* Proportional to desktop max-height */
            font-size: 16px;
        }
        
        .nav-menu {
            grid-row: 2;
            width: 100%;
            justify-content: center;
            margin-top: 0;
            padding-top: 12px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .logo {
            grid-row: 1;
            justify-self: center; /* Center logo on mobile */
        }
        
        .header-actions {
            grid-row: 3;
            justify-self: center; /* Center actions on mobile */
        }
        
        .nav-menu a {
            font-size: 12px;
            padding: 0 8px;
        }
        
        .header-actions {
            margin-left: auto;
        }
        
        .btn-login,
        .btn-signup {
            font-size: 12px;
            padding: 0 8px;
        }
    }

    .hero-grid {
        width: 100%;
        display: flex;
        justify-content: flex-start;
        padding-left: 5%;
    }

    .hero-text-card {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: flex-start;
        max-width: 640px;
        text-align: left;
    }

    .hero-title {
        font-family: 'Graphik', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 4rem;
        font-weight: 600;
        letter-spacing: 0.15em;
        margin-bottom: 0.5em;
        /* Gradient fill using brand kit colors */
        background: linear-gradient(135deg, 
            var(--brand-lavender-light) 0%,
            var(--brand-lavender) 15%,
            var(--brand-purple-light) 30%,
            var(--brand-purple) 50%,
            var(--brand-purple-medium) 70%,
            var(--brand-purple-deep) 85%,
            var(--brand-purple-dark) 100%
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3));
    }

    .hero-subtitle {
        font-family: 'Graphik', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 1.4rem;
        font-weight: 600;
        color: var(--brand-purple-light);
        margin-bottom: 1em;
        text-shadow: 0 0 20px rgba(155, 126, 217, 0.6);
    }

    .hero-tagline {
        font-family: 'Graphik', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 1.1rem;
        font-weight: 400;
        color: #ffffff;
        opacity: 0.9;
        line-height: 1.8;
        margin-bottom: 1em;
        max-width: 600px;
    }

    .hero-tagline-main {
        font-family: 'Graphik', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 1.2rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 1.2em;
    }

    .hero-tagline-sub {
        font-family: 'Graphik', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 1rem;
        font-weight: 400;
        color: rgba(255,255,255,0.8);
        margin-bottom: 2.4em;
    }

    .hero-tagline-accent {
        color: var(--brand-lavender);
    }

    .hero-cta-row {
        display: flex;
        gap: 18px;
        align-items: center;
        justify-content: flex-start;
        flex-wrap: wrap;
        position: relative;
        z-index: 20;
        pointer-events: auto;
    }

    .hero-cta {
        border-radius: 999px;
        padding: 0.95rem 2.4rem;
        font-size: 1.03rem;
        font-weight: 600;
        border: none;
        cursor: pointer;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        transition: transform .14s, box-shadow .14s, background .14s, border-color .14s;
        text-decoration: none;
        display: inline-block;
        color: inherit;
        position: relative;
        z-index: 20;
        pointer-events: auto;
    }

    .hero-cta:link,
    .hero-cta:visited,
    .hero-cta:hover,
    .hero-cta:active {
        text-decoration: none;
        color: inherit;
    }

    .hero-cta-primary {
        /* STRICTLY FROM COLOR KIT - All colors from color-kit.png */
        background: linear-gradient(135deg, 
            var(--brand-purple) 0%,
            var(--brand-purple-medium) 25%,
            var(--brand-purple-deep) 50%,
            var(--brand-purple-dark) 75%,
            var(--brand-purple-darker) 100%
        );
        color: var(--brand-white);
        box-shadow: 0 0 15px rgba(155, 126, 217, 0.3), 0 0 25px rgba(139, 111, 158, 0.2), 0 6px 20px rgba(155, 126, 217, 0.25); /* --brand-purple and --brand-purple-medium from color-kit */
        border: 1px solid var(--brand-purple);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    .hero-cta-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 20px rgba(155, 126, 217, 0.4), 0 0 35px rgba(139, 111, 158, 0.3), 0 8px 25px rgba(155, 126, 217, 0.35); /* --brand-purple and --brand-purple-medium from color-kit */
        background: linear-gradient(135deg, 
            var(--brand-purple-medium) 0%,
            var(--brand-purple-deep) 30%,
            var(--brand-purple-dark) 60%,
            var(--brand-purple-darker) 100%
        );
    }

    .hero-cta-secondary {
        background: rgba(45, 27, 61, 0.5);
        color: var(--brand-lavender-light);
        border: 2px solid rgba(155, 126, 217, 0.6);
        box-shadow: 0 0 10px rgba(155, 126, 217, 0.2), 0 4px 15px rgba(0,0,0,0.3);
        backdrop-filter: blur(10px);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    .hero-cta-secondary:hover {
        background: rgba(45, 27, 61, 0.7);
        border-color: rgba(155, 126, 217, 0.9);
        box-shadow: 0 0 15px rgba(155, 126, 217, 0.3), 0 0 25px rgba(155, 126, 217, 0.2), 0 6px 20px rgba(0,0,0,0.4);
        transform: translateY(-2px);
    }

    /* Removed standalone hero graphic; background image now handles the visual */

    /* Responsive layout */
    @media (max-width: 1024px) {
        .hero-grid {
            padding: 0;
        }

        .hero-title {
            font-size: 2.5rem;
        }

        .hero-subtitle {
            font-size: 1.1rem;
        }

        .hero-tagline {
            max-width: 100%;
        }

        .page-section {
            padding: 5rem 0 3rem 0;
        }
    }

    @media (max-width: 768px) {
        .hero-grid {
            gap: 32px;
            padding: 24px 18px 32px 18px;
        }

        .hero-text-card {
            text-align: center;
            align-items: center;
        }

        .hero-title {
            font-size: 2.1rem;
        }

        .hero-subtitle {
            font-size: 1rem;
        }

        .hero-cta-row {
            flex-direction: column;
            justify-content: center;
        }

        .features-grid {
            grid-template-columns: minmax(0, 1fr);
        }

        .dashboard-grid {
            grid-template-columns: minmax(0, 1fr);
        }

        .section-card {
            border-radius: 22px;
            padding: 1.8rem 1.6rem 2rem 1.6rem;
        }

        .generate-form-card {
            margin: 0 1rem 3rem 1rem;
        }
    }

    @media (max-width: 480px) {
        .hero-title {
            font-size: 1.8rem;
        }
    }

    /* Sections / cards */
    .page-section {
        padding: 7rem 0 4rem 0;
        display: flex;
        justify-content: center;
    }

    .page-section.center {
        padding-top: 8rem;
    }

    .section-card {
        max-width: 840px;
        width: 100%;
        background: radial-gradient(circle at 0% 0%, rgba(155, 126, 217, 0.2), transparent 60%),
                    rgba(45, 27, 61, 0.7);
        border-radius: 32px;
        padding: 2.5rem 2.8rem 2.6rem 2.8rem;
        box-shadow: 0 0 40px rgba(155, 126, 217, 0.3), 0 26px 70px rgba(0,0,0,0.8);
        border: 2px solid var(--brand-purple-medium);
        backdrop-filter: blur(15px);
        transition: all 0.3s ease;
    }

    .section-card:hover {
        border-color: var(--brand-purple);
        box-shadow: 0 0 60px rgba(155, 126, 217, 0.5), 0 30px 80px rgba(0,0,0,0.9);
        transform: translateY(-2px);
    }

    .section-card h2 {
        font-family: 'Graphik', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 1.85rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 1.5rem;
        color: #f5f6ff;
    }

    .section-card p {
        font-family: 'Graphik', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 0.98rem;
        font-weight: 400;
        line-height: 1.7;
        color: #d3d7f5;
    }

    .features-grid {
        max-width: 900px;
        width: 100%;
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 26px;
        margin-top: 2.4rem;
    }

    .feature-card {
        background: radial-gradient(circle at 0% 0%, rgba(155, 126, 217, 0.25), transparent 50%),
                    rgba(45, 27, 61, 0.6);
        border-radius: 26px;
        padding: 1.6rem 1.5rem 1.7rem 1.5rem;
        box-shadow: 0 0 30px rgba(155, 126, 217, 0.3), 0 18px 52px rgba(0,0,0,0.8);
        border: 2px solid var(--brand-purple-medium);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }

    .feature-card:hover {
        border-color: var(--brand-purple);
        box-shadow: 0 0 50px rgba(155, 126, 217, 0.5), 0 0 70px rgba(221, 160, 221, 0.2), 0 22px 60px rgba(0,0,0,0.9);
        transform: translateY(-5px);
    }

    .feature-icon {
        font-size: 1.6rem;
        margin-bottom: 0.6rem;
    }

    .feature-title {
        font-family: 'Graphik', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .feature-copy {
        font-family: 'Graphik', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 0.93rem;
        font-weight: 400;
        color: #cbd0f8;
        line-height: 1.6;
    }

    .auth-card {
        display: flex;
        justify-content: center;
        margin-bottom: 0.5rem;
    }

    .auth-card h2 {
        font-family: 'Graphik', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 1.7rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .auth-card-inner {
        max-width: 380px;
        margin: 0 auto 4rem auto;
        padding: 2.2rem 2.4rem 2.1rem 2.4rem;
        border-radius: 28px;
        background: rgba(45, 27, 61, 0.7);
        border: 2px solid rgba(155, 126, 217, 0.5);
        box-shadow: 0 0 40px rgba(155, 126, 217, 0.3), 0 24px 65px rgba(0,0,0,0.85);
        backdrop-filter: blur(15px);
    }

    .auth-card-inner .stTextInput > div > div > input {
        background: rgba(15, 5, 25, 0.9);
        border-radius: 999px;
        border: 1px solid rgba(155, 126, 217, 0.7);
        color: var(--brand-lavender-light);
    }

    .auth-card-inner .stButton > button {
        border-radius: 999px;
        background: linear-gradient(135deg, var(--brand-purple-dark) 0%, var(--brand-purple-darker) 50%, var(--brand-purple-darkest) 100%);
        box-shadow: 0 8px 20px rgba(155, 126, 217, 0.2), 0 4px 12px rgba(139, 111, 158, 0.15);
        color: #fff;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    /* SIGNIN PAGE - Single centered glass block, NO SCROLLING, ONE PAGE */
    .signin-page-container {
        display: flex !important;
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
        z-index: 100 !important;
        align-items: center !important;
        justify-content: center !important;
        background: linear-gradient(135deg, var(--brand-darkest) 0%, var(--brand-darker) 25%, var(--brand-dark) 50%, var(--brand-purple-darker) 75%, var(--brand-darkest) 100%) !important;
    }
    
    /* Glass Card Wrapper - Centers the glass block */
    .glass-card-wrapper {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 100vw !important;
        height: 100vh !important;
        padding: 2rem !important;
        box-sizing: border-box !important;
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        overflow: hidden !important;
    }
    
    /* Glass Card - Complete glass block containing ALL input fields */
    .glass-card {
        width: 100% !important;
        max-width: 450px !important;
        padding: 3rem 2.5rem !important;
        background: rgba(45, 27, 61, 0.4) !important;
        border-radius: 24px !important;
        border: 2px solid rgba(155, 126, 217, 0.5) !important;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.5),
            0 0 60px rgba(155, 126, 217, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1),
            inset 0 -1px 0 rgba(0, 0, 0, 0.2) !important;
        backdrop-filter: blur(20px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
        position: relative !important;
        overflow: visible !important;
        z-index: 10 !important;
    }
    
    /* Glass effect overlay */
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        border-radius: 24px;
        background: linear-gradient(
            135deg,
            rgba(255, 255, 255, 0.1) 0%,
            rgba(255, 255, 255, 0.05) 30%,
            transparent 50%,
            rgba(255, 255, 255, 0.05) 70%,
            rgba(255, 255, 255, 0.08) 100%
        );
        pointer-events: none;
        z-index: -1;
    }
    
    /* Ensure ALL Streamlit elements are inside the glass block */
    .glass-card .element-container,
    .glass-card section,
    .glass-card .stTextInput,
    .glass-card .stButton,
    .glass-card .stCheckbox,
    .glass-card .stColumns,
    .glass-card .form-header,
    .glass-card .input-wrapper,
    .glass-card .signup-link {
        width: 100% !important;
        max-width: 100% !important;
        padding: 0 !important;
        margin-left: 0 !important;
        margin-right: 0 !important;
    }
    
    /* Make sure Streamlit content doesn't break out of glass card */
    .glass-card > * {
        position: relative !important;
        z-index: 1 !important;
    }
    
    /* Form Header inside glass block */
    .form-header {
        margin-bottom: 1.5rem !important;
        text-align: left !important;
    }
    
    .form-title {
        font-family: 'Graphik', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #ffffff !important;
        font-size: 1.75rem !important;
        font-weight: 600 !important;
        margin: 0 0 0.4rem 0 !important;
        text-align: left !important;
        line-height: 1.2 !important;
    }
    
    .form-subtitle {
        font-family: 'Graphik', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: rgba(255, 255, 255, 0.8) !important;
        font-size: 0.85rem !important;
        font-weight: 400 !important;
        margin: 0 !important;
        text-align: left !important;
        line-height: 1.3 !important;
    }
    
    /* Input Wrapper inside glass block */
    .input-wrapper {
        margin-bottom: 1rem !important;
        position: relative !important;
    }
    
    /* Input Fields inside glass block - Glassmorphism style */
    .glass-card .stTextInput {
        width: 100% !important;
        margin-bottom: 0 !important;
        position: relative !important;
    }
    
    .glass-card .stTextInput > div > div > input {
        width: 100% !important;
        background: linear-gradient(
            135deg,
            rgba(155, 126, 217, 0.15) 0%,
            rgba(139, 111, 158, 0.12) 50%,
            rgba(155, 126, 217, 0.1) 100%
        ) !important;
        color: #ffffff !important;
        border: 2px solid rgba(255, 255, 255, 0.7) !important;
        border-radius: 16px !important;
        padding: 1rem 1rem 1rem 3.5rem !important;
        font-size: 1rem !important;
        font-weight: 400 !important;
        transition: all 0.3s ease !important;
        backdrop-filter: blur(30px) saturate(200%) !important;
        -webkit-backdrop-filter: blur(30px) saturate(200%) !important;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.4),
            inset 0 2px 8px rgba(255, 255, 255, 0.3),
            inset 0 -2px 8px rgba(0, 0, 0, 0.2),
            0 0 0 1px rgba(255, 255, 255, 0.15) !important;
        position: relative !important;
    }
    
    .glass-card .stTextInput > div > div > input:focus {
        border-color: rgba(255, 255, 255, 0.95) !important;
        outline: none !important;
        box-shadow: 
            0 0 0 4px rgba(255, 255, 255, 0.3),
            0 12px 40px rgba(155, 126, 217, 0.6),
            inset 0 3px 10px rgba(255, 255, 255, 0.4),
            inset 0 -3px 10px rgba(0, 0, 0, 0.2),
            0 0 25px rgba(155, 126, 217, 0.4) !important;
        background: linear-gradient(
            135deg,
            rgba(155, 126, 217, 0.25) 0%,
            rgba(139, 111, 158, 0.2) 50%,
            rgba(155, 126, 217, 0.18) 100%
        ) !important;
        transform: translateY(-2px) !important;
    }
    
    .glass-card .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.7) !important;
        font-weight: 400 !important;
    }
    
    /* Password input field - same glassmorphism styling */
    .glass-card input[type="password"],
    .glass-card .stTextInput > div > div > input[type="password"] {
        width: 100% !important;
        background: linear-gradient(
            135deg,
            rgba(155, 126, 217, 0.15) 0%,
            rgba(139, 111, 158, 0.12) 50%,
            rgba(155, 126, 217, 0.1) 100%
        ) !important;
        color: #ffffff !important;
        border: 2px solid rgba(255, 255, 255, 0.7) !important;
        border-radius: 16px !important;
        padding: 1rem 1rem 1rem 3.5rem !important;
        font-size: 1rem !important;
        font-weight: 400 !important;
        transition: all 0.3s ease !important;
        backdrop-filter: blur(30px) saturate(200%) !important;
        -webkit-backdrop-filter: blur(30px) saturate(200%) !important;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.4),
            inset 0 2px 8px rgba(255, 255, 255, 0.3),
            inset 0 -2px 8px rgba(0, 0, 0, 0.2),
            0 0 0 1px rgba(255, 255, 255, 0.15) !important;
    }
    
    .glass-card input[type="password"]:focus,
    .glass-card .stTextInput > div > div > input[type="password"]:focus {
        border-color: rgba(255, 255, 255, 0.95) !important;
        outline: none !important;
        box-shadow: 
            0 0 0 4px rgba(255, 255, 255, 0.3),
            0 12px 40px rgba(155, 126, 217, 0.6),
            inset 0 3px 10px rgba(255, 255, 255, 0.4),
            inset 0 -3px 10px rgba(0, 0, 0, 0.2),
            0 0 25px rgba(155, 126, 217, 0.4) !important;
        background: linear-gradient(
            135deg,
            rgba(155, 126, 217, 0.25) 0%,
            rgba(139, 111, 158, 0.2) 50%,
            rgba(155, 126, 217, 0.18) 100%
        ) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Checkbox inside glass block */
    .glass-card .stCheckbox {
        margin: 1rem 0 !important;
    }
    
    .glass-card .stCheckbox > label {
        color: rgba(255, 255, 255, 0.95) !important;
        font-size: 0.9rem !important;
        font-weight: 400 !important;
    }
    
    /* Button inside glass block */
    .glass-card .stButton {
        width: 100% !important;
        margin-top: 1.5rem !important;
    }
    
    .glass-card .stButton > button {
        width: 100% !important;
        background: linear-gradient(135deg, var(--brand-purple-dark) 0%, var(--brand-purple-darker) 50%, var(--brand-purple-darkest) 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.875rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 2px 8px rgba(155, 126, 217, 0.2), 0 4px 12px rgba(139, 111, 158, 0.12) !important;
        transition: all 0.3s ease !important;
    }
    
    .glass-card .stButton > button:hover {
        background: linear-gradient(135deg, var(--brand-purple-darker) 0%, var(--brand-purple-darkest) 50%, var(--brand-dark) 100%) !important;
        box-shadow: 0 4px 12px rgba(155, 126, 217, 0.25), 0 6px 16px rgba(139, 111, 158, 0.15) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Signup Link inside glass block - CENTERED */
    .glass-card .signup-link,
    .signup-link {
        text-align: center !important;
        margin-top: 1.5rem !important;
        margin-left: auto !important;
        margin-right: auto !important;
        color: rgba(255, 255, 255, 0.8) !important;
        font-size: 0.9rem !important;
        display: block !important;
        width: 100% !important;
    }
    
    .glass-card .signup-link a {
        color: rgba(255, 255, 255, 0.95) !important;
        text-decoration: none !important;
        font-weight: 600 !important;
    }
    
    .glass-card .signup-link a:hover {
        color: #ffffff !important;
        text-decoration: underline !important;
    }
    
    /* Hide default Streamlit labels */
    .glass-card .stTextInput label {
        display: none !important;
    }
    
    /* Force Streamlit containers to not interfere - ensure single page */
    .main .block-container:has(.glass-card-wrapper),
    section:has(.glass-card-wrapper),
    .element-container:has(.glass-card-wrapper),
    .main .block-container:has(.signin-page-container),
    section:has(.signin-page-container),
    .element-container:has(.signin-page-container) {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100vw !important;
        width: 100vw !important;
        height: 100vh !important;
        max-height: 100vh !important;
        overflow: hidden !important;
    }
    
    /* Responsive styles for glass block on mobile */
    @media (max-width: 768px) {
        .glass-card-wrapper {
            padding: 1rem !important;
        }
        
        .glass-card {
            max-width: 100% !important;
            padding: 2rem 1.5rem !important;
        }
        
        .form-title {
            font-size: 1.5rem !important;
        }
        
        .form-subtitle {
            font-size: 0.8rem !important;
        }
    }
    
    @media (max-width: 480px) {
        .glass-card-wrapper {
            padding: 0.5rem !important;
        }
        
        .glass-card {
            padding: 1.5rem 1rem !important;
            border-radius: 20px !important;
        }
        
        .form-title {
            font-size: 1.3rem !important;
        }
        
        .glass-card .stTextInput > div > div > input {
            padding: 0.75rem 0.875rem 0.75rem 2.5rem !important;
            font-size: 0.9rem !important;
        }
    }

    .dashboard-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 22px;
        margin-top: 1.5rem;
    }

    .dashboard-card {
        background: rgba(45, 27, 61, 0.96);
        border-radius: 24px;
        padding: 1.4rem 1.5rem 1.6rem 1.5rem;
        border: 1px solid rgba(155, 126, 217, 0.4);
        box-shadow: 0 20px 60px rgba(0,0,0,0.85);
        text-decoration: none;
        color: inherit;
        display: block;
    }

    .dashboard-title {
        font-family: 'Graphik', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 1.02rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }

    .dashboard-copy {
        font-family: 'Graphik', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 0.9rem;
        font-weight: 400;
        color: #c5caf2;
    }

    .analyze-options {
        display: flex;
        flex-wrap: wrap;
        gap: 14px;
        margin-top: 1rem;
    }

    .analyze-pill {
        background: rgba(45, 27, 61, 0.96);
        border-radius: 999px;
        padding: 0.75rem 1.3rem;
        display: flex;
        align-items: center;
        gap: 8px;
        border: 1px solid rgba(155, 126, 217, 0.6);
        box-shadow: 0 12px 40px rgba(0,0,0,0.85);
    }

    .pill-icon {
        font-size: 1i.2rem;
    }

    .pill-label {
        font-size: 0.9rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    .generate-form-card {
        max-width: 520px;
        margin: 0 auto 4rem auto;
        padding: 2rem 2.1rem;
        border-radius: 26px;
        background: rgba(45, 27, 61, 0.96);
        border: 1px solid rgba(155, 126, 217, 0.4);
        box-shadow: 0 24px 65px rgba(0,0,0,0.85);
    }
    
    /* Enhanced Loading Screen Styles */
    .loading-screen-overlay {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        bottom: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        background: linear-gradient(135deg, #0A0014 0%, #1A0033 50%, #0A0014 100%) !important;
        z-index: 9999 !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        overflow: hidden;
    }

    .loading-screen-overlay::before {
        content: '';
        position: absolute;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(155, 126, 217, 0.3) 0%, transparent 70%);
        animation: rotateGradient 10s linear infinite;
    }

    @keyframes rotateGradient {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .loading-content {
        position: relative;
        z-index: 10;
        text-align: center;
        width: 100%;
    }

    .loading-text {
        font-family: 'Graphik', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        letter-spacing: 0.15em;
        margin-bottom: 1rem;
        /* Gradient fill using brand kit colors - NO GLOW, just gradient */
        background: linear-gradient(135deg, 
            var(--brand-lavender-light) 0%,
            var(--brand-lavender) 15%,
            var(--brand-purple-light) 30%,
            var(--brand-purple) 50%,
            var(--brand-purple-medium) 70%,
            var(--brand-purple-deep) 85%,
            var(--brand-purple-dark) 100%
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        /* Subtle drop shadow for visibility on dark background - NOT a glow */
        filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.5));
        /* Remove all text-shadow glow effects and animations */
        text-shadow: none !important;
        animation: none !important;
    }

    .loading-subtext {
        font-family: 'Graphik', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 1.1rem;
        font-weight: 400;
        color: rgba(255,255,255,0.8);
        text-align: center;
        letter-spacing: 0.05em;
    }

    .loading-spinner {
        width: 60px;
        height: 60px;
        border: 4px solid rgba(155, 126, 217, 0.3);
        border-top: 4px solid var(--brand-purple);
        border-right: 4px solid var(--brand-purple-light);
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin: 2rem auto;
        box-shadow: 0 0 12px rgba(155, 126, 217, 0.25), 0 0 20px rgba(155, 126, 217, 0.15);
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    @keyframes loadingPulse {
        0%, 100% {
            opacity: 1;
            transform: scale(1);
            filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.5));
        }
        50% {
            opacity: 0.95;
            transform: scale(1.01);
            filter: drop-shadow(0 3px 6px rgba(0, 0, 0, 0.6));
        }
    }
    
    /* Responsive loading screen */
    @media (max-width: 768px) {
        .loading-text {
            font-size: 2rem;
        }
    }
    
    @media (max-width: 480px) {
        .loading-text {
            font-size: 1.5rem;
        }
    }
    
    /* Remove Material Icons entirely - FIX B: Remove Material Icons completely */
    /* Disable Material Icons fallback by using system fonts */
    .streamlit-expanderHeader,
    .streamlit-expanderHeader * {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }
    
    /* Hide all SVG icons in expander headers */
    .streamlit-expanderHeader svg {
        display: none !important;
    }
    
    /* Hide Material Icons elements */
    .streamlit-expanderHeader [class*="material-icons"],
    .streamlit-expanderHeader .material-icons,
    .streamlit-expanderHeader button svg,
    .streamlit-expanderHeader [data-testid="stExpanderToggleIcon"],
    .streamlit-expanderHeader .streamlit-expanderHeaderIcon {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
        font-size: 0 !important;
        line-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    .streamlit-expanderHeader::after {
        content: "" !important;  /* no arrow */
        display: none !important;
    }
    
    /* Remove Material Icons text from expander headers */
    .streamlit-expanderHeader {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        display: flex !important;
        align-items: center !important;
        position: relative !important;
        padding-left: 0 !important;
    }
    
    /* Remove custom arrow - no arrow at all */
    .streamlit-expanderHeader::before {
        display: none !important;
        content: "" !important;
        visibility: hidden !important;
    }
    
    /* COMPLETELY HIDE keyboard_arrow elements - they show text when Material Icons font doesn't load */
    .keyboard_arrow_down,
    [class*="keyboard_arrow"],
    [class*="keyboard-arrow"],
    [class*="keyboardArrow"],
    [class*="arrow_down"],
    [class*="arrow-down"],
    [class*="arrowDown"],
    *[class*="keyboard"],
    *[class*="arrow"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        font-size: 0 !important;
        line-height: 0 !important;
        position: absolute !important;
        left: -9999px !important;
        overflow: hidden !important;
        text-indent: -9999px !important;
    }
    
    /* Hide dropdown arrows in selectboxes */
    .stSelectbox svg,
    .stSelectbox [class*="arrow"],
    .stSelectbox [class*="Arrow"],
    .stSelectbox .material-icons,
    .stSelectbox [class*="material-icons"],
    div[data-baseweb="select"] svg,
    div[data-baseweb="select"] [class*="arrow"],
    div[data-baseweb="select"] [class*="Arrow"],
    div[data-baseweb="select"] .material-icons,
    div[data-baseweb="select"] [class*="material-icons"],
    button[data-baseweb="select"] svg,
    button[data-baseweb="select"] [class*="arrow"],
    button[data-baseweb="select"] [class*="Arrow"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
    }
    
    /* Hide all arrow icons in Streamlit widgets */
    [class*="stSelectbox"] svg,
    [class*="stSelectbox"] [class*="arrow"],
    [class*="stSelectbox"] [class*="Arrow"],
    [class*="stSelectbox"] .material-icons,
    [data-testid*="stSelectbox"] svg,
    [data-testid*="stSelectbox"] [class*="arrow"],
    [data-testid*="stSelectbox"] [class*="Arrow"],
    [data-testid*="stSelectbox"] .material-icons {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
    }
    
    /* Material Icons - COMPLETELY HIDE (disable Material Icons entirely) */
    /* CRITICAL: Hide Streamlit's Material Icons fallback text using data-testid */
    [data-testid="stIconMaterial"],
    span[data-testid="stIconMaterial"],
    *[data-testid="stIconMaterial"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
        font-size: 0 !important;
        line-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        position: absolute !important;
        left: -9999px !important;
        overflow: hidden !important;
        text-indent: -9999px !important;
        text-overflow: clip !important;
        white-space: nowrap !important;
    }
    
    .material-icons,
    [class*="material-icons"],
    [class*="materialIcons"],
    [class*="MaterialIcons"],
    span.material-icons,
    i.material-icons,
    *[class*="material-icons"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
        font-size: 0 !important;
        line-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        position: absolute !important;
        left: -9999px !important;
        overflow: hidden !important;
        text-indent: -9999px !important;
    }
    
    /* More specific: Hide arrow icons in BaseWeb components */
    [data-baseweb*="select"] svg,
    [data-baseweb*="select"] [class*="arrow"],
    [data-baseweb*="select"] [class*="Arrow"],
    [data-baseweb*="select"] .material-icons,
    [data-baseweb*="select"] [class*="material-icons"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
    }
    
    /* Override Streamlit error/warning colors to purple brand color */
    div[data-testid="stAlert"] > div:first-child {
        background-color: rgba(155, 126, 217, 0.1) !important;
        border-left-color: var(--brand-purple) !important;
    }
    
    /* Error messages - change red to purple */
    .stAlert[data-baseweb="notification"] {
        background-color: rgba(155, 126, 217, 0.15) !important;
        border-left: 4px solid var(--brand-purple) !important;
    }
    
    /* Warning messages - change yellow/orange to purple */
    div[data-testid="stAlert"]:has(> div > div[class*="alert"]) {
        background-color: rgba(155, 126, 217, 0.1) !important;
    }
    
    /* Streamlit error styling */
    .element-container .stAlert[data-baseweb="notification"][data-kind="error"] {
        background-color: rgba(155, 126, 217, 0.15) !important;
        border-left: 4px solid var(--brand-purple) !important;
    }
    
    /* Streamlit warning styling */
    .element-container .stAlert[data-baseweb="notification"][data-kind="warning"] {
        background-color: rgba(155, 126, 217, 0.1) !important;
        border-left: 4px solid var(--brand-purple) !important;
    }
    
    /* Ensure ALL buttons (primary and default) use brand purple gradient - STRICTLY FROM COLOR KIT */
    .stButton > button,
    .stButton > button[kind="primary"],
    .stButton > button[kind="secondary"],
    button[kind="primary"][data-baseweb="button"],
    button[kind="secondary"][data-baseweb="button"],
    button[data-baseweb="button"],
    div[data-baseweb="button"] > button {
        background: linear-gradient(135deg, var(--brand-purple-dark) 0%, var(--brand-purple-darker) 50%, var(--brand-purple-darkest) 100%) !important;
        background-color: var(--brand-purple-darker) !important; /* Fallback - from color-kit */
        border: 1px solid rgba(155, 126, 217, 0.3) !important; /* --brand-purple #9B7ED9 with 0.3 opacity from color-kit */
        color: var(--brand-white) !important;
        box-shadow: 0 0 8px rgba(155, 126, 217, 0.15), 0 2px 10px rgba(139, 111, 158, 0.1) !important; /* --brand-purple #9B7ED9 and --brand-purple-medium #8B6F9E with opacity from color-kit */
        font-weight: 600 !important;
        letter-spacing: 0.05em !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover,
    .stButton > button[kind="primary"]:hover,
    .stButton > button[kind="secondary"]:hover,
    button[kind="primary"][data-baseweb="button"]:hover,
    button[kind="secondary"][data-baseweb="button"]:hover,
    button[data-baseweb="button"]:hover,
    div[data-baseweb="button"] > button:hover {
        background: linear-gradient(135deg, var(--brand-purple-darker) 0%, var(--brand-purple-darkest) 50%, var(--brand-darkest) 100%) !important;
        background-color: var(--brand-purple-darker) !important; /* Fallback - from color-kit */
        box-shadow: 0 0 12px rgba(155, 126, 217, 0.2), 0 0 20px rgba(139, 111, 158, 0.15), 0 4px 15px rgba(155, 126, 217, 0.2) !important; /* --brand-purple #9B7ED9 and --brand-purple-medium #8B6F9E with opacity from color-kit */
        transform: translateY(-2px) !important;
    }

    /* Glassmorphism cards */
    .glass-card {
        background: rgba(45, 27, 61, 0.5);
        border: 2px solid rgba(155, 126, 217, 0.4);
        border-radius: 20px;
        padding: 2rem;
        backdrop-filter: blur(15px);
        box-shadow: 0 0 30px rgba(155, 126, 217, 0.2), 0 20px 60px rgba(0,0,0,0.7);
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        border-color: rgba(155, 126, 217, 0.7);
        box-shadow: 0 0 50px rgba(155, 126, 217, 0.4), 0 0 80px rgba(155, 126, 217, 0.2), 0 25px 70px rgba(0,0,0,0.8);
        transform: translateY(-3px);
    }

    /* Gradient text - no neon effects */
    .neon-text {
        background: linear-gradient(135deg, 
            var(--brand-lavender-light) 0%,
            var(--brand-purple-light) 30%,
            var(--brand-purple) 60%,
            var(--brand-purple-medium) 100%
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Page headers - gradient fill from color kit */
    .page-header {
        font-family: 'Graphik', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 2.5rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, 
            var(--brand-lavender-light) 0%,
            var(--brand-lavender) 20%,
            var(--brand-purple-light) 40%,
            var(--brand-purple) 60%,
            var(--brand-purple-medium) 80%,
            var(--brand-purple-deep) 100%
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3));
    }

    .page-subheader {
        font-family: 'Graphik', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 1.1rem;
        font-weight: 400;
        color: rgba(255,255,255,0.9);
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Footer styles - HIDDEN */
    footer,
    footer[data-testid="stFooter"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    footer a:hover {
        color: var(--brand-purple) !important;
        text-shadow: 0 0 10px rgba(155, 126, 217, 0.6);
        transition: all 0.3s ease;
    }
    
    /* Hide Gemini logo and any bottom-right decorations - AGGRESSIVE */
    [class*="gemini"],
    [class*="Gemini"],
    [id*="gemini"],
    [id*="Gemini"],
    a[href*="gemini"],
    a[href*="Gemini"],
    img[alt*="gemini"],
    img[alt*="Gemini"],
    img[src*="gemini"],
    img[src*="Gemini"],
    [data-testid="stDecoration"],
    iframe[src*="gemini"],
    iframe[src*="Gemini"],
    svg[class*="gemini"],
    svg[id*="gemini"],
    div[class*="gemini"],
    div[id*="gemini"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0 !important;
        width: 0 !important;
        position: absolute !important;
        left: -9999px !important;
        pointer-events: none !important;
        z-index: -1 !important;
    }
    
    /* Hide any fixed/absolute elements in bottom-right corner */
    [style*="position: fixed"][style*="bottom"],
    [style*="position:fixed"][style*="bottom"],
    [style*="position: absolute"][style*="bottom"],
    [style*="position:absolute"][style*="bottom"],
    [style*="bottom: 0"],
    [style*="bottom:0"][style*="right"],
    [style*="right: 0"][style*="bottom"],
    [style*="right:0"][style*="bottom"] {
        display: none !important;
    }
    
    /* Cover bottom-right corner on signin/login page to hide Gemini logo */
    body[data-page="signin"]::after,
    .signin-page-container::after,
    body:has(.signin-page-container)::after {
        content: '' !important;
        position: fixed !important;
        bottom: 0 !important;
        right: 0 !important;
        width: 200px !important;
        height: 200px !important;
        background: linear-gradient(135deg, var(--brand-darkest) 0%, var(--brand-darker) 50%, var(--brand-darkest) 100%) !important;
        z-index: 9998 !important;
        pointer-events: none !important;
    }
    
    /* Additional overlay for signin page background */
    .signin-page-container::before {
        content: '' !important;
        position: fixed !important;
        bottom: 0 !important;
        right: 0 !important;
        width: 250px !important;
        height: 250px !important;
        background: linear-gradient(135deg, var(--brand-darkest) 0%, var(--brand-darker) 100%) !important;
        z-index: 9997 !important;
        pointer-events: none !important;
    }
    
    /* Style Streamlit File Uploader - Remove boxes, backgrounds, glows - icons only */
    .stFileUploader > div > div > div > div,
    .stFileUploader > div > div > div > div > div,
    .stFileUploader > div > div > div > div > div > div {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        border-color: transparent !important;
        box-shadow: none !important;
        animation: none !important;
    }
    
    /* Remove all circles, indicators, glows */
    .stFileUploader [class*="circle"],
    .stFileUploader [class*="indicator"],
    .stFileUploader [class*="upload"],
    .stFileUploader [class*="inner"],
    .stFileUploader [class*="center"] {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Override any red colors in file uploader - text color white */
    .stFileUploader *,
    .stFileUploader p,
    .stFileUploader span,
    .stFileUploader div,
    .stFileUploader label,
    .stFileUploader button,
    .stFileUploader a {
        color: var(--brand-white) !important;
    }
    
    .stFileUploader [style*="red"],
    .stFileUploader [style*="#f00"],
    .stFileUploader [style*="#ff0000"],
    .stFileUploader [style*="rgb(255, 0, 0)"],
    .stFileUploader [style*="rgba(255, 0, 0"] {
        background-color: var(--brand-purple) !important;
        border-color: var(--brand-purple) !important;
        color: var(--brand-white) !important;
    }
    
    /* File uploader hover state - no boxes, just icons */
    .stFileUploader > div > div > div > div:hover {
        background: transparent !important;
        box-shadow: none !important;
    }
    
    /* Remove all backgrounds, borders, glows from upload indicators */
    div[data-testid="stFileUploader"] > div > div > div > div,
    div[data-testid="stFileUploader"] > div > div > div > div > div,
    div[data-testid="stFileUploader"] > div > div > div > div > div > div {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        animation: none !important;
    }
    
    /* Remove red backgrounds - make transparent */
    .stFileUploader [style*="background"][style*="red"],
    .stFileUploader [style*="background-color"][style*="red"],
    .stFileUploader [style*="background"][style*="#f00"],
    .stFileUploader [style*="background"][style*="#ff0000"],
    .stFileUploader [style*="background"][style*="rgb(255, 0, 0)"],
    .stFileUploader [style*="background"][style*="rgba(255, 0, 0"] {
        background: transparent !important;
        box-shadow: none !important;
    }
    
    /* Override any inline styles with red colors - make transparent */
    .stFileUploader div[style*="red"],
    .stFileUploader div[style*="#f00"],
    .stFileUploader div[style*="#ff0000"],
    .stFileUploader div[style*="rgb(255, 0, 0)"],
    .stFileUploader div[style*="rgba(255, 0, 0"] {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Style any SVG or icon elements in uploader - clean, natural appearance, NO boxes/highlights */
    .stFileUploader svg,
    .stFileUploader svg circle,
    .stFileUploader svg path,
    .stFileUploader svg rect,
    .stFileUploader svg polygon,
    .stFileUploader svg g {
        filter: none !important;
        box-shadow: none !important;
        text-shadow: none !important;
        outline: none !important;
        background: none !important;
        border: none !important;
    }
    
    /* Remove any boxes/backgrounds around icons */
    .stFileUploader > div > div > div,
    .stFileUploader > div > div > div > div {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Remove all backgrounds, borders, glows from upload indicator circles */
    .stFileUploader > div > div > div > div[style],
    .stFileUploader > div > div > div > div > div[style],
    .stFileUploader > div > div > div > div > div > div[style] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Dashboard Sidebar Styling - Clean, Neat, Aesthetic - STRICTLY FROM COLOR KIT */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--brand-darkest) 0%, var(--brand-darker) 100%) !important;
        border-right: 2px solid rgba(155, 126, 217, 0.3) !important; /* --brand-purple #9B7ED9 with 0.3 opacity from color-kit */
        min-width: 280px !important;
        width: 280px !important;
        padding: 1.5rem 1.25rem !important;
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.5) !important; /* --brand-black from color-kit */
        backdrop-filter: blur(10px) !important;
        overflow: visible !important;
        overflow-y: visible !important;
        overflow-x: visible !important;
        height: auto !important;
        min-height: 100vh !important;
        max-height: none !important;
        display: block !important;
    }
    
    /* Sidebar inner container - ensure no height restrictions */
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebar"] > div > div {
        overflow: visible !important;
        overflow-y: visible !important;
        overflow-x: visible !important;
        height: auto !important;
        max-height: none !important;
        min-height: auto !important;
    }
    
    /* Ensure sidebar is always visible on dashboard */
    [data-testid="stSidebar"][aria-expanded="true"],
    [data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    
    /* Sidebar toggle button styling */
    button[data-testid="baseButton-header"] {
        display: block !important;
        visibility: visible !important;
        z-index: 100 !important;
    }
    
    /* Sidebar content styling */
    [data-testid="stSidebar"] .element-container {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Sidebar navigation buttons - WHITE TEXT - compact */
    [data-testid="stSidebar"] .stButton {
        width: 100% !important;
        margin: 0.4rem 0 !important;
    }
    
    [data-testid="stSidebar"] .stButton > button {
        width: 100% !important;
        padding: 0.75rem 1.25rem !important;
        border-radius: 12px !important;
        border: 2px solid transparent !important;
        background: rgba(31, 15, 47, 0.6) !important; /* --brand-darker #1F0F2F from color-kit */
        color: var(--brand-white) !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        text-align: left !important;
        transition: all 0.3s ease !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.75rem !important;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, 
            var(--brand-purple-dark) 0%,
            var(--brand-purple-darker) 50%,
            var(--brand-purple-darkest) 100%
        ) !important;
        border-color: var(--brand-purple-dark) !important;
        transform: translateX(5px) !important;
        color: #FFFFFF !important;
        box-shadow: none !important;
    }
    
    /* Active sidebar button - gradient fill, no glow - darker shades to match sidebar */
    [data-testid="stSidebar"] .stButton > button[kind="primary"],
    [data-testid="stSidebar"] .stButton > button:active,
    [data-testid="stSidebar"] .stButton > button[aria-pressed="true"] {
        background: linear-gradient(135deg, 
            var(--brand-purple-dark) 0%,
            var(--brand-purple-darker) 50%,
            var(--brand-purple-darkest) 100%
        ) !important;
        border-color: var(--brand-purple-dark) !important;
        color: #FFFFFF !important;
        box-shadow: none !important;
    }
    
    /* Remove all glows/shadows from sidebar buttons */
    [data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] .stButton > button:hover,
    [data-testid="stSidebar"] .stButton > button:active,
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        box-shadow: none !important;
        text-shadow: none !important;
        filter: none !important;
    }
    
    /* Sidebar button text - force white */
    [data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] .stButton > button * {
        color: #FFFFFF !important;
    }
    
    /* Sidebar markdown text styling - WHITE for better contrast, fully visible */
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6,
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown h4,
    [data-testid="stSidebar"] .stMarkdown h5,
    [data-testid="stSidebar"] .stMarkdown h6,
    [data-testid="stSidebar"] section[data-testid="stMarkdownContainer"] h1,
    [data-testid="stSidebar"] section[data-testid="stMarkdownContainer"] h2,
    [data-testid="stSidebar"] section[data-testid="stMarkdownContainer"] h3,
    [data-testid="stSidebar"] section[data-testid="stMarkdownContainer"] h4,
    [data-testid="stSidebar"] section[data-testid="stMarkdownContainer"] h5,
    [data-testid="stSidebar"] section[data-testid="stMarkdownContainer"] h6 {
        color: #FFFFFF !important;
        font-family: 'Graphik', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-weight: 600 !important;
        margin: 0 0 1rem 0 !important;
        padding: 0 !important;
        overflow: visible !important;
        text-overflow: clip !important;
        white-space: normal !important;
        word-wrap: break-word !important;
        line-height: 1.4 !important;
        height: auto !important;
        min-height: auto !important;
        max-height: none !important;
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] .stMarkdown p {
        color: #FFFFFF !important;
        font-family: 'Graphik', -apple-system, BlinkMacSystemFont, sans-serif !important;
        margin-bottom: 0.5rem !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #FFFFFF !important;
    }
    
    /* Sidebar content - Clean, no overlapping, no scrolls */
    [data-testid="stSidebar"] > div:first-child,
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebar"] > div > div {
        padding: 0 !important;
        margin: 0 !important;
        overflow: visible !important;
        overflow-y: visible !important;
        overflow-x: visible !important;
        height: auto !important;
        max-height: none !important;
        min-height: auto !important;
    }
    
    [data-testid="stSidebar"] .element-container,
    [data-testid="stSidebar"] section,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] section[data-testid="stMarkdownContainer"] {
        padding: 0 !important;
        margin: 0 0 1rem 0 !important;
        overflow: visible !important;
        overflow-y: visible !important;
        overflow-x: visible !important;
        width: 100% !important;
        height: auto !important;
        max-height: none !important;
        min-height: auto !important;
    }
    
    /* Sidebar spacing - clean and neat */
    [data-testid="stSidebar"] .stMarkdown {
        margin: 0 0 1rem 0 !important;
        padding: 0 !important;
        overflow: visible !important;
        line-height: 1.5 !important;
    }
    
    [data-testid="stSidebar"] h3 {
        margin: 0 0 1rem 0 !important;
        padding: 0 !important;
        overflow: visible !important;
        text-overflow: clip !important;
        white-space: normal !important;
        word-wrap: break-word !important;
        line-height: 1.4 !important;
        height: auto !important;
        min-height: auto !important;
        display: block !important;
        visibility: visible !important;
    }
    
    /* Ensure all sidebar content is visible */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6 {
        overflow: visible !important;
        text-overflow: clip !important;
        white-space: normal !important;
        word-wrap: break-word !important;
        height: auto !important;
        min-height: auto !important;
        max-height: none !important;
    }
    
    [data-testid="stSidebar"] .stCaption {
        margin: 0 0 1rem 0 !important;
        padding: 0 !important;
        overflow: visible !important;
        line-height: 1.5 !important;
    }
    
    /* Prevent any scrolling in sidebar - make everything visible */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > *,
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebar"] > div > div,
    [data-testid="stSidebar"] .element-container,
    [data-testid="stSidebar"] section,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] section[data-testid="stMarkdownContainer"] {
        overflow: visible !important;
        overflow-y: visible !important;
        overflow-x: visible !important;
        height: auto !important;
        max-height: none !important;
        min-height: auto !important;
    }
    
    /* Clean sidebar layout - no overlapping, proper spacing */
    [data-testid="stSidebar"] > div {
        display: block !important;
        gap: 0 !important;
        height: auto !important;
        max-height: none !important;
        min-height: auto !important;
    }
    
    /* Sidebar divider */
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.2) !important;
        margin: 0.5rem 0 !important;
    }
    
    /* Sidebar button text - ensure white */
    [data-testid="stSidebar"] .stButton > button {
        color: #FFFFFF !important;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        color: #FFFFFF !important;
    }
    
    /* Dashboard main content area - Glass UI with lighter background - STRICTLY FROM COLOR KIT */
    .main .block-container {
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        padding-top: 6rem !important;
        padding-bottom: 2rem !important;
        max-width: calc(100vw - 300px) !important;
        margin-top: 0 !important;
        background: linear-gradient(180deg, rgba(15, 5, 25, 0.95) 0%, rgba(31, 15, 47, 0.95) 100%) !important; /* --brand-darkest #0F0519 and --brand-darker #1F0F2F from color-kit */
        backdrop-filter: blur(10px) !important;
    }
    
    /* When sidebar is visible, adjust main content - STRICTLY FROM COLOR KIT */
    [data-testid="stSidebar"][aria-expanded="true"] ~ .main .block-container {
        margin-left: 280px !important;
        padding-top: 6rem !important;
        background: linear-gradient(180deg, rgba(15, 5, 25, 0.95) 0%, rgba(31, 15, 47, 0.95) 100%) !important; /* --brand-darkest #0F0519 and --brand-darker #1F0F2F from color-kit */
        backdrop-filter: blur(10px) !important;
    }
    
    /* Ensure dashboard content starts below navbar - STRICTLY FROM COLOR KIT */
    [data-testid="stSidebar"] ~ .main {
        padding-top: 80px !important;
        background: linear-gradient(135deg, rgba(15, 5, 25, 0.95) 0%, rgba(31, 15, 47, 0.95) 50%, rgba(15, 5, 25, 0.95) 100%) !important; /* --brand-darkest #0F0519 and --brand-darker #1F0F2F from color-kit */
    }
    
    /* Dashboard page background - STRICTLY FROM COLOR KIT */
    .main {
        background: linear-gradient(135deg, rgba(15, 5, 25, 0.98) 0%, rgba(31, 15, 47, 0.98) 50%, rgba(15, 5, 25, 0.98) 100%) !important; /* --brand-darkest #0F0519 and --brand-darker #1F0F2F from color-kit */
    }
    
    /* All UI icons - clean, natural appearance (no forced styling) */
    
    /* Radio Buttons - ULTRA AGGRESSIVE: Force purple, override EVERYTHING */
    /* Target ALL possible radio button selectors */
    [data-baseweb="radio"] svg circle,
    [data-baseweb="radio"] svg circle[fill],
    [data-baseweb="radio"] svg circle[fill*="red"],
    [data-baseweb="radio"] svg circle[fill*="#f00"],
    [data-baseweb="radio"] svg circle[fill*="#ff0000"],
    [data-baseweb="radio"] svg circle[fill*="rgb(255"],
    .stRadio svg circle,
    .stRadio svg circle[fill],
    .stRadio svg circle[fill*="red"],
    [data-testid*="stRadio"] svg circle,
    [data-testid*="stRadio"] svg circle[fill],
    [data-testid*="stRadio"] svg circle[fill*="red"],
    svg circle[fill*="red"],
    svg circle[fill*="#f00"],
    svg circle[fill*="#ff0000"],
    svg circle[fill*="rgb(255, 0"],
    svg circle[fill*="rgba(255, 0"] {
        stroke: #9B7ED9 !important;
        stroke-width: 2px !important;
    }
    
    /* Selected radio button - FORCE purple fill, NO RED ALLOWED */
    [data-baseweb="radio"][aria-checked="true"] svg circle,
    [data-baseweb="radio"][aria-checked="true"] svg circle[fill],
    [data-baseweb="radio"][aria-checked="true"] svg circle[fill*="red"],
    [data-baseweb="radio"][aria-checked="true"] svg circle[fill*="#f00"],
    .stRadio svg circle[fill]:not([fill="none"]):not([fill="transparent"]),
    .stRadio svg circle[fill*="red"],
    [data-testid*="stRadio"] svg circle[fill]:not([fill="none"]):not([fill="transparent"]),
    [data-testid*="stRadio"] svg circle[fill*="red"],
    svg circle[fill*="red"][fill]:not([fill="none"]):not([fill="transparent"]) {
        fill: #9B7ED9 !important;
        stroke: #9B7ED9 !important;
        stroke-width: 2px !important;
    }
    
    /* Unselected radio button - transparent fill, purple border */
    [data-baseweb="radio"]:not([aria-checked="true"]) svg circle,
    [data-baseweb="radio"]:not([aria-checked="true"]) svg circle[fill],
    .stRadio svg circle:not([fill]):not([fill="none"]):not([fill="transparent"]),
    .stRadio svg circle[fill="none"],
    .stRadio svg circle[fill="transparent"] {
        fill: transparent !important;
        stroke: #9B7ED9 !important;
        stroke-width: 2px !important;
    }
    
    /* Radio button labels - white text */
    .stRadio label,
    [data-testid*="stRadio"] label,
    [data-baseweb="radio"] label {
        color: var(--brand-white) !important;
    }
    
    /* NUCLEAR OPTION: Override ANY red color in SVG circles */
    svg circle[style*="red"],
    svg circle[style*="#f00"],
    svg circle[style*="#ff0000"],
    svg circle[style*="rgb(255, 0"],
    svg circle[style*="rgba(255, 0"] {
        fill: #9B7ED9 !important;
        stroke: #9B7ED9 !important;
    }
    
    </style>
    <script>
    // Remove ALL Material Icons TEXT (but keep icons working) - ULTRA AGGRESSIVE
    // Based on Streamlit's data-testid="stIconMaterial" pattern
    function removeAllMaterialIconsText() {
        // CRITICAL: Remove all [data-testid="stIconMaterial"] elements directly
        // This is the exact attribute Streamlit uses for Material Icons
        var iconElements = document.querySelectorAll('[data-testid="stIconMaterial"]');
        iconElements.forEach(function(element) {
            // Remove the element completely
            if (element.parentNode) {
                element.parentNode.removeChild(element);
            } else {
                element.style.display = 'none';
                element.style.visibility = 'hidden';
                element.style.opacity = '0';
                element.style.width = '0';
                element.style.height = '0';
                element.style.fontSize = '0';
                element.style.lineHeight = '0';
                element.style.margin = '0';
                element.style.padding = '0';
                element.style.position = 'absolute';
                element.style.left = '-9999px';
                element.style.overflow = 'hidden';
                element.style.textIndent = '-9999px';
                element.textContent = '';
            }
        });
        
        // Remove from ALL text nodes in document - AGGRESSIVE
        var walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null, false);
        var textNodes = [];
            var node;
            while (node = walker.nextNode()) {
            if (node.textContent && (
                node.textContent.includes('keyboard_arrow_down') || 
                node.textContent.includes('keyboard_double_arrow_down') || 
                node.textContent.includes('keyboard_double_a') || 
                node.textContent.includes('keyboard_arrow') ||
                node.textContent.includes('arrow_upward') ||
                node.textContent.includes('arrow_downward') ||
                node.textContent.includes('keep') ||
                node.textContent.trim() === 'key' ||
                /^key/i.test(node.textContent.trim())
            )) {
                // Check if parent has data-testid="stIconMaterial" - if so, remove parent
                var parent = node.parentNode;
                if (parent && parent.getAttribute && parent.getAttribute('data-testid') === 'stIconMaterial') {
                    if (parent.parentNode) {
                        parent.parentNode.removeChild(parent);
                    }
                    continue;
                }
                textNodes.push(node);
            }
        }
        
        // Remove text from all matching nodes
        textNodes.forEach(function(textNode) {
            var parent = textNode.parentNode;
            if (parent && parent.classList && parent.classList.contains('material-icons')) {
                // Keep Material Icons elements but ensure font loads
                return;
            }
            // Remove the text completely
            var newText = textNode.textContent
                .replace(/keyboard_arrow_down/gi, '')
                .replace(/keyboard_double_arrow_down/gi, '')
                .replace(/keyboard_double_a/gi, '')
                .replace(/keyboard_arrow/gi, '')
                .replace(/arrow_upward/gi, '')
                .replace(/arrow_downward/gi, '')
                .replace(/^key/gi, '')
                .trim();
            if (newText === '') {
                textNode.textContent = '';
            } else {
                textNode.textContent = newText;
            }
        });
        
        // Remove from expander headers specifically
        var headers = document.querySelectorAll('.streamlit-expanderHeader, [class*="expander"]');
        headers.forEach(function(header) {
            // Remove [data-testid="stIconMaterial"] elements in headers
            var iconElements = header.querySelectorAll('[data-testid="stIconMaterial"]');
            iconElements.forEach(function(element) {
                if (element.parentNode) {
                    element.parentNode.removeChild(element);
                }
            });
            
            // Remove text nodes with keyboard_arrow_down
            var walker = document.createTreeWalker(header, NodeFilter.SHOW_TEXT, null, false);
            var textNode;
            while (textNode = walker.nextNode()) {
                if (textNode.textContent && (
                    textNode.textContent.includes('keyboard_arrow_down') ||
                    textNode.textContent.includes('keyboard_arrow') ||
                    /^key/i.test(textNode.textContent.trim())
                )) {
                    textNode.textContent = '';
                }
            }
        });
    }
    
    // Run immediately and on DOM changes - MORE FREQUENT
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', removeAllMaterialIconsText);
    } else {
        removeAllMaterialIconsText();
    }
    
    // Run on every Streamlit rerun - MORE FREQUENT
    setTimeout(removeAllMaterialIconsText, 50);
    setTimeout(removeAllMaterialIconsText, 100);
    setTimeout(removeAllMaterialIconsText, 200);
    setTimeout(removeAllMaterialIconsText, 500);
    setTimeout(removeAllMaterialIconsText, 1000);
    setTimeout(removeAllMaterialIconsText, 2000);
    
    // Watch for new expanders - MORE AGGRESSIVE
    var observer = new MutationObserver(function(mutations) {
        removeAllMaterialIconsText();
    });
    observer.observe(document.body, { 
        childList: true, 
        subtree: true, 
        characterData: true,
        attributes: false
    });
    
    // ULTRA AGGRESSIVE: Remove ALL red colors and force purple - NO EXCEPTIONS
    function replaceRedWithPurple() {
        console.log('🔴 FORCING PURPLE ON RADIO BUTTONS...');
        var brandPurple = '#9B7ED9';
        
        // Find ALL possible radio button circles - multiple selectors
        var selectors = [
            '[data-baseweb="radio"] svg circle',
            '[data-baseweb="radio"] svg path',
            '[data-testid*="stRadio"] svg circle',
            '[data-testid*="stRadio"] svg path',
            '.stRadio svg circle',
            '.stRadio svg path',
            'svg circle[fill*="red"]',
            'svg circle[fill*="#f00"]',
            'svg circle[fill*="#ff0000"]',
            'svg circle[fill*="rgb(255"]',
            'svg path[fill*="red"]',
            'svg path[fill*="#f00"]'
        ];
        
        var allCircles = [];
        selectors.forEach(function(selector) {
            try {
                var circles = document.querySelectorAll(selector);
                circles.forEach(function(circle) {
                    if (allCircles.indexOf(circle) === -1) {
                        allCircles.push(circle);
                    }
                });
            } catch(e) {}
        });
        
        // Also find by walking the DOM
        var allSVGs = document.querySelectorAll('svg');
        allSVGs.forEach(function(svg) {
            var circles = svg.querySelectorAll('circle, path');
            circles.forEach(function(circle) {
                if (allCircles.indexOf(circle) === -1) {
                    allCircles.push(circle);
                }
            });
        });
        
        console.log('Found', allCircles.length, 'radio button elements');
        
        allCircles.forEach(function(circle) {
            var fill = circle.getAttribute('fill') || '';
            var stroke = circle.getAttribute('stroke') || '';
            var computedFill = window.getComputedStyle(circle).fill || '';
            var computedStroke = window.getComputedStyle(circle).stroke || '';
            
            // Check if it's a radio button circle (has a parent with radio attributes)
            var parent = circle.closest('[data-baseweb="radio"], [data-testid*="stRadio"], .stRadio');
            if (!parent) {
                // Check if it's near radio button text
                var nearbyText = circle.closest('label, div, span');
                if (nearbyText && (nearbyText.textContent.includes('Upload PDF') || nearbyText.textContent.includes('Upload Image') || nearbyText.textContent.includes('Paste Text'))) {
                    parent = nearbyText;
                }
            }
            
            // If it's a radio button OR if it has red color, force purple
            var isRed = fill.includes('red') || fill.includes('#f00') || fill.includes('#ff0000') || 
                       fill.includes('rgb(255, 0') || fill.includes('rgba(255, 0') ||
                       computedFill.includes('rgb(255, 0') || computedFill.includes('rgba(255, 0') ||
                       computedFill.includes('red');
            
            if (parent || isRed) {
                // FORCE PURPLE - ULTRA AGGRESSIVE
                // Check if it's selected (filled) or unselected (outline only)
                var isSelected = fill && fill !== 'none' && fill !== 'transparent' && 
                                fill !== '' && !fill.includes('transparent');
                
                if (isSelected || isRed) {
                    // Selected state - purple fill
                    circle.setAttribute('fill', brandPurple);
                    circle.style.setProperty('fill', brandPurple, 'important');
                    circle.style.fill = brandPurple;
                } else {
                    // Unselected state - transparent fill, purple stroke
                    circle.setAttribute('fill', 'transparent');
                    circle.style.setProperty('fill', 'transparent', 'important');
                }
                
                // Always set stroke to purple
                circle.setAttribute('stroke', brandPurple);
                circle.style.setProperty('stroke', brandPurple, 'important');
                circle.style.stroke = brandPurple;
                
                // Remove any inline style that might override
                var currentStyle = circle.getAttribute('style') || '';
                currentStyle = currentStyle.replace(/fill\s*:\s*[^;]+/gi, '');
                currentStyle = currentStyle.replace(/stroke\s*:\s*[^;]+/gi, '');
                currentStyle += '; fill: ' + (isSelected ? brandPurple : 'transparent') + ' !important; stroke: ' + brandPurple + ' !important;';
                circle.setAttribute('style', currentStyle);
            }
        });
        
        console.log('Purple applied to radio buttons');
    }
    
    // Run IMMEDIATELY - multiple times
    console.log('INITIALIZING PURPLE RADIO BUTTONS...');
    replaceRedWithPurple();
    
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', replaceRedWithPurple);
    }
    
    window.addEventListener('load', replaceRedWithPurple);
    
    // Run VERY FREQUENTLY - catch Streamlit's dynamic updates
    var intervals = [10, 25, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000];
    intervals.forEach(function(delay) {
        setTimeout(replaceRedWithPurple, delay);
    });
    
    // Watch for ANY changes - ULTRA AGGRESSIVE
    var redObserver = new MutationObserver(function(mutations) {
        var shouldUpdate = false;
        mutations.forEach(function(mutation) {
            if (mutation.addedNodes.length > 0 || 
                mutation.type === 'attributes' || 
                (mutation.type === 'childList' && mutation.addedNodes.length > 0)) {
                shouldUpdate = true;
            }
        });
        if (shouldUpdate) {
            setTimeout(replaceRedWithPurple, 5);
            setTimeout(replaceRedWithPurple, 10);
            setTimeout(replaceRedWithPurple, 25);
        }
    });
    redObserver.observe(document.body, { 
        childList: true, 
        subtree: true, 
        attributes: true, 
        attributeFilter: ['style', 'fill', 'stroke', 'class', 'data-baseweb', 'aria-checked'] 
    });
    
    // Also watch for style attribute changes specifically
    var styleObserver = new MutationObserver(function() {
        setTimeout(replaceRedWithPurple, 5);
    });
    var allElements = document.querySelectorAll('*');
    allElements.forEach(function(el) {
        styleObserver.observe(el, { attributes: true, attributeFilter: ['style', 'fill', 'stroke'] });
    });
    
    // Hide Gemini logo on login/signin page - aggressive removal
    function hideGeminiLogo() {
        // Remove any Gemini-related elements
        var geminiElements = document.querySelectorAll('[class*="gemini"], [id*="gemini"], [data-testid="stDecoration"], img[src*="gemini"], iframe[src*="gemini"], svg[class*="gemini"]');
        geminiElements.forEach(function(el) {
            el.style.display = 'none';
            el.style.visibility = 'hidden';
            el.style.opacity = '0';
            el.style.height = '0';
            el.style.width = '0';
            el.style.position = 'absolute';
            el.style.left = '-9999px';
            el.style.pointerEvents = 'none';
            el.style.zIndex = '-1';
        });
        
        // Cover bottom-right corner on signin page
        if (document.body.classList.contains('signin-page') || document.querySelector('.signin-page-container')) {
            var overlay = document.createElement('div');
            overlay.id = 'gemini-cover-overlay';
            overlay.style.cssText = 'position: fixed !important; bottom: 0 !important; right: 0 !important; width: 300px !important; height: 300px !important; background: linear-gradient(135deg, #0F0519 0%, #1F0F2F 100%) !important; z-index: 9999 !important; pointer-events: none !important;';
            document.body.appendChild(overlay);
        }
    }
    
    // Run immediately and on DOM changes
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', hideGeminiLogo);
    } else {
        hideGeminiLogo();
    }
    
    // Run on every Streamlit rerun
    setTimeout(hideGeminiLogo, 100);
    setTimeout(hideGeminiLogo, 500);
    setTimeout(hideGeminiLogo, 1000);
    
    // Watch for new elements
    var geminiObserver = new MutationObserver(hideGeminiLogo);
    geminiObserver.observe(document.body, { childList: true, subtree: true, attributes: true });
    
    // DIAGNOSTICS: Log expander header information
    function logExpanderDiagnostics() {
        console.log("=== EXPANDER DIAGNOSTICS ===");
        var expanders = document.querySelectorAll('div[data-testid="stExpander"] summary, .streamlit-expanderHeader');
        console.log("Expander headers found:", expanders.length);
        
        expanders.forEach(function(el, idx) {
            console.log('--- Expander', idx, '---');
            console.log('HTML:', el.innerHTML);
            console.log('Text content:', el.textContent);
            console.log('Classes:', el.className);
            console.log('Has SVG:', el.querySelector('svg') !== null);
            console.log('Has Material Icons:', el.querySelector('.material-icons') !== null);
            console.log('Computed display:', window.getComputedStyle(el).display);
            
            // Check for keyboard_arrow_down text
            var walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null, false);
            var textNode;
            while (textNode = walker.nextNode()) {
                if (textNode.textContent && textNode.textContent.includes('keyboard')) {
                    console.log('FOUND keyboard text:', textNode.textContent);
                }
            }
        });
        console.log("=== END DIAGNOSTICS ===");
    }
    
    // Run diagnostics after a delay to catch dynamically loaded expanders
    setTimeout(logExpanderDiagnostics, 1000);
    setTimeout(logExpanderDiagnostics, 3000);
    
    // Watch for new expanders and log them
    var expanderObserver = new MutationObserver(function(mutations) {
        var hasNewExpander = false;
        mutations.forEach(function(mutation) {
            if (mutation.addedNodes.length > 0) {
                mutation.addedNodes.forEach(function(node) {
                    if (node.nodeType === 1 && (node.querySelector && (node.querySelector('div[data-testid="stExpander"]') || node.querySelector('.streamlit-expanderHeader')))) {
                        hasNewExpander = true;
                    }
                });
            }
        });
        if (hasNewExpander) {
            setTimeout(logExpanderDiagnostics, 500);
        }
    });
    expanderObserver.observe(document.body, { childList: true, subtree: true });
    
    // FORCE BUTTON GRADIENTS - Apply color-kit gradients directly via JavaScript
    // ULTRA AGGRESSIVE - Override everything
    function forceButtonGradients() {
        console.log('FORCING BUTTON GRADIENTS...');
        // Color-kit purple gradient colors
        var purpleDark = '#6B4E7F';
        var purpleDarker = '#5D3D6F';
        var purpleDarkest = '#4A2D5A';
        var purpleBorder = 'rgba(155, 126, 217, 0.3)';
        var gradient = 'linear-gradient(135deg, ' + purpleDark + ' 0%, ' + purpleDarker + ' 50%, ' + purpleDarkest + ' 100%)';
        
        // Find ALL possible button selectors
        var selectors = [
            '.stButton > button',
            'button[data-baseweb="button"]',
            'div[data-baseweb="button"] > button',
            'button[kind="primary"]',
            'button[kind="secondary"]',
            '[data-testid*="button"] button',
            'button.stButton',
            'button'
        ];
        
        var allButtons = [];
        selectors.forEach(function(selector) {
            try {
                var buttons = document.querySelectorAll(selector);
                buttons.forEach(function(btn) {
                    if (allButtons.indexOf(btn) === -1) {
                        allButtons.push(btn);
                    }
                });
            } catch(e) {
                console.error('Selector error:', selector, e);
            }
        });
        
        console.log('Found', allButtons.length, 'buttons');
        
        allButtons.forEach(function(button) {
            // Skip sidebar buttons that should have different styling
            var isSidebarButton = button.closest('[data-testid="stSidebar"]');
            
            if (!isSidebarButton && button.offsetParent !== null) {
                // Apply gradient background for main buttons - ULTRA AGGRESSIVE
                button.style.cssText += 'background: ' + gradient + ' !important;';
                button.style.cssText += 'background-image: ' + gradient + ' !important;';
                button.style.cssText += 'background-color: ' + purpleDarker + ' !important;';
                button.style.cssText += 'border-color: ' + purpleBorder + ' !important;';
                button.setAttribute('style', button.getAttribute('style') + ' background: ' + gradient + ' !important; background-image: ' + gradient + ' !important;');
                button.style.setProperty('background', gradient, 'important');
                button.style.setProperty('background-image', gradient, 'important');
                button.style.setProperty('background-color', purpleDarker, 'important');
                button.style.setProperty('border-color', purpleBorder, 'important');
            }
        });
        
        console.log('Button gradients applied');
    }
    
    // Force sidebar background gradient
    function forceSidebarBackground() {
        console.log('FORCING SIDEBAR BACKGROUND...');
        var sidebar = document.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {
            var gradient = 'linear-gradient(180deg, #0F0519 0%, #1F0F2F 100%)';
            sidebar.style.cssText += 'background: ' + gradient + ' !important;';
            sidebar.style.setProperty('background', gradient, 'important');
            sidebar.setAttribute('style', sidebar.getAttribute('style') + ' background: ' + gradient + ' !important;');
            console.log('Sidebar background applied');
        } else {
            console.log('Sidebar not found');
        }
    }
    
    // Run immediately
    console.log('INITIALIZING FORCE STYLES...');
    forceButtonGradients();
    forceSidebarBackground();
    
    // Run on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            console.log('DOM Content Loaded');
            forceButtonGradients();
            forceSidebarBackground();
        });
    }
    
    // Run on window load
    window.addEventListener('load', function() {
        console.log('Window Loaded');
        forceButtonGradients();
        forceSidebarBackground();
    });
    
    // Run on every Streamlit rerun - MORE FREQUENT
    var intervals = [10, 50, 100, 200, 500, 1000, 2000, 3000];
    intervals.forEach(function(delay) {
        setTimeout(function() {
            forceButtonGradients();
            forceSidebarBackground();
        }, delay);
    });
    
    // Watch for new buttons - AGGRESSIVE
    var buttonObserver = new MutationObserver(function(mutations) {
        var shouldUpdate = false;
        mutations.forEach(function(mutation) {
            if (mutation.addedNodes.length > 0 || mutation.type === 'attributes') {
                shouldUpdate = true;
            }
        });
        if (shouldUpdate) {
            setTimeout(forceButtonGradients, 10);
            setTimeout(forceSidebarBackground, 10);
        }
    });
    buttonObserver.observe(document.body, { 
        childList: true, 
        subtree: true, 
        attributes: true, 
        attributeFilter: ['style', 'class', 'data-baseweb'] 
    });
    
    // Streamlit specific: Hook into Streamlit's rerun
    if (window.parent && window.parent.postMessage) {
        window.addEventListener('message', function(event) {
            if (event.data && (event.data.type === 'streamlit:rerun' || event.data.type === 'streamlit:render')) {
                setTimeout(forceButtonGradients, 100);
                setTimeout(forceSidebarBackground, 100);
            }
        });
    }
    </script>
    """
    
    # Replace Material Icons font placeholder with actual font data URI
    font_src = material_icons_font if material_icons_font else "url('https://fonts.gstatic.com/s/materialicons/v139/flUhRq6tzZclQEJ-Vdg-IuiaDsNcIhQ8tQ.woff2') format('woff2')"
    css = css.replace('MATERIAL_ICONS_FONT_PLACEHOLDER', font_src)
    
    st.markdown(css, unsafe_allow_html=True)


                            