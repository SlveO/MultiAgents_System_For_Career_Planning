"""Generate the six anonymous, self-created pilot media fixtures.

This is a maintainer utility. The generated PNG/PDF files are versioned; model
weights and experiment outputs remain ignored. Pillow is required only when
regenerating these fixtures.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = ROOT / "dataset" / "research_cases" / "assets"
WIDTH = 1400
HEIGHT = 900

ASSETS = {
    "image-001.png": (
        "DATA ANALYST INTERNSHIP",
        [
            "REQUIRED: SQL QUERYING",
            "DELIVERABLE: ONE DASHBOARD PORTFOLIO",
            "TASK: WEEKLY METRIC REPORTING",
            "ELIGIBILITY: CURRENT UNDERGRADUATE",
        ],
    ),
    "image-002.png": (
        "EMBEDDED SOFTWARE ENGINEER",
        [
            "REQUIRED: C OR C++",
            "REQUIRED: FREERTOS",
            "INTERFACES: UART / I2C / SPI",
            "DELIVERABLE: PUBLIC GIT REPOSITORY",
        ],
    ),
    "image-003.png": (
        "AI PRODUCT MANAGER",
        [
            "DELIVERABLE: PRODUCT REQUIREMENTS DOCUMENT",
            "REQUIRED: PRODUCT METRICS",
            "TASK: USER INTERVIEWS",
            "REQUIRED: LLM EVALUATION DESIGN",
        ],
    ),
    "pdf-001.pdf": (
        "HARDWARE ENGINEER - ROLE BRIEF",
        [
            "REQUIRED: PCB LAYOUT",
            "REQUIRED: OSCILLOSCOPE OPERATION",
            "TASK: EMC TESTING",
            "CONSTRAINT: TRAVEL UP TO 20 PERCENT",
        ],
    ),
    "pdf-002.pdf": (
        "POLYMER MATERIALS ENGINEER - ROLE BRIEF",
        [
            "TASK: POLYMER SYNTHESIS",
            "REQUIRED: DSC AND GPC CHARACTERIZATION",
            "EDUCATION: MASTER DEGREE PREFERRED",
            "REQUIRED: CHEMICAL SAFETY TRAINING",
        ],
    ),
    "pdf-003.pdf": (
        "OBSTETRICS PHYSICIAN - ROLE BRIEF",
        [
            "EDUCATION: GRADUATE DEGREE",
            "REQUIRED: PHYSICIAN LICENSE",
            "REQUIRED: RESIDENCY CERTIFICATE",
            "CONSTRAINT: NIGHT SHIFT ROTATION",
        ],
    ),
}


def _draw_fixture(title: str, evidence_lines: list[str]) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=34)
    small = ImageFont.load_default(size=26)

    draw.rectangle((0, 0, WIDTH, 130), fill="#17324d")
    draw.text((70, 45), title, fill="white", font=font)
    draw.text((70, 165), "SYNTHETIC ANONYMOUS RESEARCH FIXTURE", fill="#4b5563", font=small)

    y = 245
    for index, line in enumerate(evidence_lines, start=1):
        draw.rounded_rectangle(
            (70, y, WIDTH - 70, y + 115),
            radius=10,
            fill="#f4f7fa" if index % 2 else "#eef6f2",
            outline="#9aa8b5",
            width=2,
        )
        draw.rectangle((90, y + 28, 145, y + 83), fill="#c43b32")
        draw.text((107, y + 36), str(index), fill="white", font=small)
        draw.text((180, y + 37), line, fill="#15202b", font=small)
        y += 135

    draw.text(
        (70, HEIGHT - 55),
        "Fixture ID is encoded by the filename. No person or employer is represented.",
        fill="#4b5563",
        font=small,
    )
    return image


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _write_pdf(path: Path, title: str, evidence_lines: list[str]) -> None:
    """Write a deterministic one-page vector PDF using built-in Helvetica."""
    commands = [
        "0.09 0.20 0.30 rg 0 770 1400 130 re f",
        f"BT /F1 34 Tf 1 1 1 rg 70 830 Td ({_pdf_escape(title)}) Tj ET",
        "BT /F1 26 Tf 0.29 0.33 0.39 rg 70 705 Td "
        "(SYNTHETIC ANONYMOUS RESEARCH FIXTURE) Tj ET",
    ]
    y = 540
    for index, line in enumerate(evidence_lines, start=1):
        fill = "0.96 0.97 0.98" if index % 2 else "0.93 0.97 0.95"
        commands.extend(
            [
                f"{fill} rg 70 {y} 1260 115 re f",
                f"0.60 0.66 0.71 RG 2 w 70 {y} 1260 115 re S",
                f"0.77 0.23 0.20 rg 90 {y + 28} 55 55 re f",
                f"BT /F1 26 Tf 1 1 1 rg 107 {y + 45} Td ({index}) Tj ET",
                f"BT /F1 26 Tf 0.08 0.13 0.18 rg 180 {y + 47} Td "
                f"({_pdf_escape(line)}) Tj ET",
            ]
        )
        y -= 135
    commands.append(
        "BT /F1 22 Tf 0.29 0.33 0.39 rg 70 30 Td "
        "(Fixture ID is encoded by the filename. No person or employer is represented.) Tj ET"
    )
    stream = ("\n".join(commands) + "\n").encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 1400 900] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n"
        + stream
        + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    content = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(content))
        content.extend(f"{number} 0 obj\n".encode("ascii"))
        content.extend(obj)
        content.extend(b"\nendobj\n")
    xref_offset = len(content)
    content.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    content.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        content.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    content.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(content)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for filename, (title, evidence_lines) in ASSETS.items():
        image = _draw_fixture(title, evidence_lines)
        output = ASSET_DIR / filename
        if output.suffix == ".pdf":
            _write_pdf(output, title, evidence_lines)
        else:
            image.save(output, "PNG", optimize=False)
        print(output.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
