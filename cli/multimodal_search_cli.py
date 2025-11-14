import argparse
from lib.multimodal_search import verify_image_embedding

def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_img_embedding_parser = subparsers.add_parser("verify_image_embedding", help="Genrate a vector embedding for given image")
    verify_img_embedding_parser.add_argument("--image", type=str, help="Image path", required=True)
    

    args = parser.parse_args()
    
    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(img_path=args.image)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()