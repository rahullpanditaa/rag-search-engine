import argparse
from lib.describe_image import describe_image

def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Query rewriting - describe image CLI")
    parser.add_argument("--image", type=str, help="Path to an image file", required=True)
    parser.add_argument("--query", type=str, help="Query to rewrite based on the image", required=True)

    args = parser.parse_args()
    describe_image(path=args.image, query=args.query)

