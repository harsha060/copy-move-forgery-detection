"""Utility to generate sample authentic and forged images for testing."""

from pathlib import Path
import cv2
import numpy as np


def generate_sample_images(output_dir: Path):
    """Generate authentic.png and forged.png for testing and demonstration."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    h, w = 500, 500
    # Create smooth natural-looking synthetic background
    x = np.linspace(0, 4 * np.pi, w)
    y = np.linspace(0, 4 * np.pi, h)
    xx, yy = np.meshgrid(x, y)
    background = (np.sin(xx) * np.cos(yy) * 60 + 160).astype(np.uint8)
    authentic = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)

    # Add rich distinctive textured shapes to the background
    cv2.rectangle(authentic, (300, 80), (450, 180), (180, 50, 180), -1)
    cv2.line(authentic, (300, 80), (450, 180), (255, 255, 255), 3)
    cv2.ellipse(authentic, (250, 380), (80, 40), 30, 0, 360, (50, 150, 220), -1)

    # Create a feature-rich textured stamp object at (100, 100)
    for r in range(40, 8, -6):
        color = ((r * 37) % 255, (r * 67) % 255, (r * 97) % 255)
        cv2.circle(authentic, (100, 100), r, color, 2)
    cv2.rectangle(authentic, (75, 75), (125, 125), (40, 220, 220), 2)
    cv2.line(authentic, (70, 70), (130, 130), (255, 0, 255), 2)
    cv2.line(authentic, (70, 130), (130, 70), (0, 255, 255), 2)
    cv2.putText(authentic, "CMFD", (80, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Save authentic image
    auth_path = output_dir / "authentic.png"
    cv2.imwrite(str(auth_path), authentic)
    print(f"Generated authentic image: {auth_path}")

    # Create forged image by copying the entire stamp patch [50:150, 50:150]
    forged = authentic.copy()
    cloned_patch = authentic[50:150, 50:150].copy()

    # Paste cloned patch at target location [300:400, 280:380] (spatial distance ~260px)
    forged[300:400, 280:380] = cloned_patch

    forged_path = output_dir / "forged.png"
    cv2.imwrite(str(forged_path), forged)
    print(f"Generated forged image: {forged_path}")


if __name__ == "__main__":
    generate_sample_images(Path(__file__).parent)
