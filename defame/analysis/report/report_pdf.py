"""
PDF rendering utilities for DEFAME analysis reports.

This module uses the same markdown_pdf library that DEFAME uses for
generating PDF reports during fact-checking runs.
"""

from pathlib import Path

from markdown_pdf import MarkdownPdf, Section


def markdown_to_pdf(markdown_content: str, output_path: str | Path) -> str | None:
    """
    Convert markdown content to PDF using markdown_pdf library.

    This uses the same approach as DEFAME's Report.save_to() method.

    Args:
        markdown_content: Markdown-formatted text
        output_path: Path to save the PDF file

    Returns:
        Success message if PDF was generated, error message if failed
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Create PDF using markdown_pdf (same as DEFAME's Report class)
        pdf = MarkdownPdf(toc_level=0)
        pdf.add_section(Section(markdown_content, toc=False))
        pdf.meta["title"] = "DEFAME Analysis Report"
        pdf.save(output_path)

        return f"PDF generated: {output_path}"

    except Exception as e:
        return f"PDF generation failed: {e}"
