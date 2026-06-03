import os
import csv
import argparse
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import openvino as ov

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff')


def download_model_files(repo_id):
    """
    Downloads the ONNX model and the tags CSV from a specified
    Hugging Face repository.
    """
    print(
        f"Fetching model artifacts from Hugging Face repository: {repo_id}...")
    model_path = hf_hub_download(repo_id=repo_id, filename="model.onnx")
    tags_path = hf_hub_download(repo_id=repo_id, filename="selected_tags.csv")
    return model_path, tags_path


def load_tags(tags_csv_path):
    """Parses the Danbooru tags file mapping index to tag names."""
    with open(tags_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        tags = [row['name'] for row in reader]
    return tags


def preprocess_image(image_path, target_size):
    """Resizes, pads, and normalizes the image to match model specifications."""
    img = Image.open(image_path).convert('RGB')

    # Calculate padding to preserve aspect ratio
    old_size = img.size
    ratio = float(target_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = img.resize(new_size, Image.Resampling.LANCZOS)

    # Create canvas and paste image into center (using a white background)
    new_img = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    new_img.paste(img, ((target_size - new_size[0]) // 2,
                        (target_size - new_size[1]) // 2))

    # Convert to BGR array as expected by the SmilingWolf vision pipeline
    img_array = np.array(new_img, dtype=np.float32)
    img_array = img_array[:, :, ::-1]  # RGB to BGR

    # Expand dims to represent batch size [1, H, W, 3]
    return np.expand_dims(img_array, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Auto-tag images using tagger models via OpenVINO.")
    parser.add_argument(
        "directory", type=str, help="Directory containing images to tag.")
    parser.add_argument(
        "-m", "--model", type=str,
        default="SmilingWolf/wd-v1-4-moat-tagger-v2",
        help=(
            "Hugging Face repository ID for the target model "
            "(default: SmilingWolf/wd-v1-4-moat-tagger-v2)."
        )
    )
    parser.add_argument(
        "-d", "--dry-run", action="store_true",
        help="Print tags to stdout instead of saving to file.")
    parser.add_argument(
        "-c", "--confidence", type=float, default=0.37,
        help="Tag confidence threshold (default: 0.37).")
    parser.add_argument(
        "-g", "--gpu", type=str, default="GPU",
        help="Target hardware accelerator device index (default: GPU).")
    parser.add_argument(
        "-u", "--keep-underscores", action="store_true",
        help="Keep underscores in tags instead of replacing them with spaces.")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: Target directory '{args.directory}' does not exist "
              "or is not a directory.")
        return

    # 1. Prepare assets from the designated repository
    try:
        model_path, tags_path = download_model_files(args.model)
    except Exception as e:
        print(f"Error downloading from repository '{args.model}': {e}")
        return

    tags_list = load_tags(tags_path)

    # 2. Initialize Core OpenVINO Runtime
    core = ov.Core()
    print("Loading model graph into OpenVINO...")
    model = core.read_model(model_path)

    # Configure FP16 execution hint for the hardware compiler
    core.set_property(args.gpu, {"INFERENCE_PRECISION_HINT": "f16"})

    print(f"Compiling model for target hardware device: {args.gpu}")
    compiled_model = core.compile_model(model, device_name=args.gpu)
    infer_request = compiled_model.create_infer_request()

    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    # Dynamically extract image resolution from the model's structural inputs
    shape = input_layer.shape
    if shape[1] == 3:  # NCHW format
        image_size = shape[2]
    else:              # NHWC format (Default for MOAT/ConvNeXt ONNX models)
        image_size = shape[1]

    print(
        "Detected model input resolution requirement: "
        f"{image_size}x{image_size}")

    # Scan target directory for images
    files = [
        os.path.join(args.directory, f)
        for f in os.listdir(args.directory)]
    image_paths = [
        f for f in files if os.path.isfile(f)
        and f.lower().endswith(VALID_EXTENSIONS)]

    if not image_paths:
        print(f"No matching images found in directory: {args.directory}")
        return

    print(f"Found {len(image_paths)} images to process.\n")

    # 3. Process Batch Loop
    for image_path in image_paths:
        try:
            processed_image = preprocess_image(image_path, image_size)

            results_dict = infer_request.infer({input_layer: processed_image})
            probs = results_dict[output_layer][0]

            # Extract tags that clear the required threshold
            matched_tags = []
            for i, prob in enumerate(probs):
                if prob >= args.confidence:
                    raw_tag = tags_list[i]
                    if not args.keep_underscores:
                        raw_tag = raw_tag.replace('_', ' ')
                    matched_tags.append(raw_tag)

            tag_string = ", ".join(matched_tags)

            if args.dry_run:
                print(
                    "--- Dry Run Results for: "
                    f"{os.path.basename(image_path)} ---")
                print(
                    tag_string if tag_string
                    else "[No tags matched threshold]")
                print("-" * 40 + "\n")
            else:
                # Replace image extension with .txt
                txt_path = os.path.splitext(image_path)[0] + ".txt"
                with open(txt_path, "w", encoding="utf-8") as txt_file:
                    txt_file.write(tag_string)
                print(
                    f"Successfully processed: {os.path.basename(image_path)} "
                    f"-> {os.path.basename(txt_path)}")

        except Exception as e:
            print(
                f"Failed to process {os.path.basename(image_path)}. "
                f"Error: {e}")


if __name__ == "__main__":
    main()
