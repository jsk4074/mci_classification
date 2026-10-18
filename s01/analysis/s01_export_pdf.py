#!/usr/bin/env python3
"""Export the s01 XLS-R report as an A4 PDF without changing its HTML."""

import argparse
from pathlib import Path
import tempfile

def export_pdf(html_path: Path, output_path: Path, stylesheet: Path | None = None) -> Path:
    """Publish a standalone PDF; import optional rendering dependencies lazily."""

    source = Path(html_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if destination == source:
        raise ValueError("PDF output must differ from the HTML source.")
    text = source.read_text(encoding="utf-8")
    stylesheet = Path(stylesheet) if stylesheet is not None else Path(__file__).with_name("report_a4.css")
    style = stylesheet.read_text(encoding="utf-8")
    if "</head>" not in text:
        raise ValueError("HTML source must contain a closing head element.")
    from weasyprint import HTML

    # Add the print stylesheet last in the author cascade for this export only.
    text = text.replace("</head>", f"<style>{style}</style></head>", 1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".report-", suffix=".pdf", delete=False) as handle:
            temporary = Path(handle.name)
        HTML(string=text, base_url=source.parent.as_uri() + "/").write_pdf(temporary)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stylesheet", type=Path)
    args = parser.parse_args(argv)
    try:
        destination = export_pdf(args.html, args.output, args.stylesheet)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(f"Saved A4 PDF: {destination}")
    return 0


if __name__ == "__main__":
    main()
