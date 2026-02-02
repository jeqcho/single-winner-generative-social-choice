"""
Render Mermaid diagrams to PNG/PDF using kroki.io service.
"""

import base64
import zlib
import urllib.request
import urllib.parse
from pathlib import Path


def encode_mermaid(mermaid_code: str) -> str:
    """Encode Mermaid code for kroki URL."""
    compressed = zlib.compress(mermaid_code.encode('utf-8'), 9)
    encoded = base64.urlsafe_b64encode(compressed).decode('ascii')
    return encoded


def render_mermaid_kroki(mermaid_code: str, output_path: Path, format: str = 'png'):
    """
    Render Mermaid diagram using kroki.io service.
    
    Args:
        mermaid_code: Mermaid diagram source
        output_path: Path to save output
        format: Output format ('png', 'svg', 'pdf')
    """
    encoded = encode_mermaid(mermaid_code)
    url = f"https://kroki.io/mermaid/{format}/{encoded}"
    
    print(f"Fetching from kroki.io...")
    
    req = urllib.request.Request(url)
    req.add_header('User-Agent', 'Python/experiment-flow-renderer')
    
    with urllib.request.urlopen(req, timeout=30) as response:
        data = response.read()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(data)
    
    print(f"Saved: {output_path}")
    return output_path


def main():
    """Render the experiment flow Mermaid diagram."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Render Mermaid diagram")
    parser.add_argument("--input", "-i", type=str, help="Input .mmd file")
    parser.add_argument("--output-dir", "-o", type=str, help="Output directory")
    args = parser.parse_args()
    
    # Default paths
    base_dir = Path(__file__).parent.parent.parent
    input_path = Path(args.input) if args.input else base_dir / "outputs" / "paper" / "experiment_flow.mmd"
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "outputs" / "paper"
    
    # Read Mermaid code
    mermaid_code = input_path.read_text()
    
    # Render to PNG and SVG (SVG is better for LaTeX)
    render_mermaid_kroki(mermaid_code, output_dir / "experiment_flow_mermaid.png", 'png')
    render_mermaid_kroki(mermaid_code, output_dir / "experiment_flow_mermaid.svg", 'svg')
    
    print("\nFor LaTeX, use the SVG with inkscape or convert to PDF:")
    print("  inkscape experiment_flow_mermaid.svg --export-pdf=experiment_flow_mermaid.pdf")
    print("\nOr include SVG directly with svg package:")
    print("  \\usepackage{svg}")
    print("  \\includesvg[width=\\textwidth]{experiment_flow_mermaid}")


if __name__ == "__main__":
    main()
