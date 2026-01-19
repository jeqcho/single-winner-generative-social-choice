"""
Shared configuration for plotting scripts.
"""
from pathlib import Path

# Directories
OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "outputs" / "full_sampling_experiment"
DATA_DIR = OUTPUT_DIR / "data"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Method groups
TRADITIONAL_METHODS = ['schulze', 'borda', 'irv', 'plurality', 'veto_by_consumption']
CHATGPT_METHODS = ['chatgpt', 'chatgpt_personas', 'chatgpt_rankings']
CHATGPT_STAR_METHODS = ['chatgpt_star', 'chatgpt_star_personas', 'chatgpt_star_rankings']
CHATGPT_DOUBLE_STAR_METHODS = ['chatgpt_double_star', 'chatgpt_double_star_personas', 'chatgpt_double_star_rankings']

METHODS_ORDER = [
    'schulze', 'borda', 'irv', 'plurality', 'veto_by_consumption',
    'chatgpt', 'chatgpt_personas', 'chatgpt_rankings',
    'chatgpt_star', 'chatgpt_star_personas', 'chatgpt_star_rankings',
    'chatgpt_double_star', 'chatgpt_double_star_personas', 'chatgpt_double_star_rankings',
    'chatgpt_triple_star'
]

METHODS_ORDER_SHORT = [
    'schulze', 'borda', 'irv', 'plurality', 'veto_by_consumption',
    'chatgpt', 'chatgpt_star', 'chatgpt_double_star', 'chatgpt_triple_star'
]

# Display names
METHOD_DISPLAY = {
    'schulze': 'Schulze',
    'borda': 'Borda',
    'irv': 'IRV',
    'plurality': 'Plurality',
    'veto_by_consumption': 'VBC',
    'chatgpt': 'GPT',
    'chatgpt_rankings': 'GPT+Rank',
    'chatgpt_personas': 'GPT+Pers',
    'chatgpt_star': 'GPT*',
    'chatgpt_star_rankings': 'GPT*+Rank',
    'chatgpt_star_personas': 'GPT*+Pers',
    'chatgpt_double_star': 'GPT**',
    'chatgpt_double_star_rankings': 'GPT**+Rank',
    'chatgpt_double_star_personas': 'GPT**+Pers',
    'chatgpt_triple_star': 'GPT***',
}

METHOD_DISPLAY_LONG = {
    'schulze': 'Schulze',
    'borda': 'Borda',
    'irv': 'IRV',
    'plurality': 'Plurality',
    'veto_by_consumption': 'Veto by Consumption',
    'chatgpt': 'ChatGPT',
    'chatgpt_rankings': 'ChatGPT+Rankings',
    'chatgpt_personas': 'ChatGPT+Personas',
    'chatgpt_star': 'ChatGPT*',
    'chatgpt_star_rankings': 'ChatGPT*+Rankings',
    'chatgpt_star_personas': 'ChatGPT*+Personas',
    'chatgpt_double_star': 'ChatGPT**',
    'chatgpt_double_star_rankings': 'ChatGPT**+Rankings',
    'chatgpt_double_star_personas': 'ChatGPT**+Personas',
    'chatgpt_triple_star': 'ChatGPT***',
}

# Topic short names
TOPIC_SHORT_NAMES = {
    'how-should-we-increase-the-general-publics-trust-i': 'Public Trust',
    'what-are-the-best-policies-to-prevent-littering-in': 'Littering',
    'what-are-your-thoughts-on-the-way-university-campu': 'Campus Speech',
    'what-balance-should-be-struck-between-environmenta': 'Environment',
    'what-balance-should-exist-between-gun-safety-laws-': 'Gun Safety',
    'what-limits-if-any-should-exist-on-free-speech-reg': 'Free Speech',
    'what-principles-should-guide-immigration-policy-an': 'Immigration',
    'what-reforms-if-any-should-replace-or-modify-the-e': 'Electoral College',
    'what-responsibilities-should-tech-companies-have-w': 'Tech Responsibility',
    'what-role-should-artificial-intelligence-play-in-s': 'AI in Society',
    'what-role-should-the-government-play-in-ensuring-u': 'Healthcare Access',
    'what-should-guide-laws-concerning-abortion': 'Abortion',
    'what-strategies-should-guide-policing-to-address-b': 'Policing',
}

# Colors - grouped by method family
COLORS_GROUPED = {
    # Traditional - distinct colors
    'schulze': '#1f77b4',
    'borda': '#2ca02c',
    'irv': '#17becf',
    'plurality': '#8c564b',
    'veto_by_consumption': '#9467bd',
    # ChatGPT group - orange
    'chatgpt': '#ff7f0e',
    'chatgpt_rankings': '#ff7f0e',
    'chatgpt_personas': '#ff7f0e',
    # ChatGPT* group - red
    'chatgpt_star': '#d62728',
    'chatgpt_star_rankings': '#d62728',
    'chatgpt_star_personas': '#d62728',
    # ChatGPT** group - purple
    'chatgpt_double_star': '#7b4bbd',
    'chatgpt_double_star_rankings': '#7b4bbd',
    'chatgpt_double_star_personas': '#7b4bbd',
    # ChatGPT*** - green
    'chatgpt_triple_star': '#2ca02c',
}

# Line styles for variants
LINESTYLES = {
    'schulze': '-',
    'borda': '-',
    'irv': '-',
    'plurality': '-',
    'veto_by_consumption': '-',
    'chatgpt': '-',
    'chatgpt_rankings': '--',
    'chatgpt_personas': ':',
    'chatgpt_star': '-',
    'chatgpt_star_rankings': '--',
    'chatgpt_star_personas': ':',
    'chatgpt_double_star': '-',
    'chatgpt_double_star_rankings': '--',
    'chatgpt_double_star_personas': ':',
    'chatgpt_triple_star': '-',
}

# Contrasting colors for subplots
CONTRAST_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
