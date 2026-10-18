#!/usr/bin/env python3
"""Export an existing HTML report as an A4 PDF without changing the HTML."""

import argparse
from pathlib import Path
import tempfile

from weasyprint import HTML


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stylesheet", type=Path, default=Path(__file__).with_name("report_a4.css"))
    args = parser.parse_args()
    source = args.html.resolve()
    destination = args.output.resolve()
    if destination == source:
        parser.error("PDF output must differ from the HTML source.")
    text = source.read_text(encoding="utf-8")
    style = args.stylesheet.read_text(encoding="utf-8")
    if "</head>" not in text:
        parser.error("HTML source must contain a closing head element.")
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
    print(f"Saved A4 PDF: {destination}")


if __name__ == "__main__":
    main()
