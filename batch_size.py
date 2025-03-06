import torch
import sys

MODEL_PATH = './model_5_version_2_binary.pt'
MODEL_FEATURE_IDS = [3,6,10,11,36,71,93,94,95,96,97,98,99,100,101,102,103,104,107,154,155,181,182,192,193,194]


def infer_batch_size(model_path=MODEL_PATH):
    """Infer input batch size from output and verify it by running a forward pass."""
    model = torch.jit.load(model_path, map_location=torch.device("cpu"))
    model.eval()

    print(f"Model loaded: {model_path}. Set to evaluation mode")
    print("Passing an input with batch size = 1")
    
    dummy_batch_size = 1
    example_input = {}
    for id in MODEL_FEATURE_IDS:
        array_size = 1
        if id == 71:
            array_size = 200
        elif id == 154:
            array_size = 40
        example_input[id] = torch.randn(dummy_batch_size, array_size)

    with torch.no_grad():
        output = model(example_input)
    print("Output from model with single input:")
    print(output)

    if isinstance(output, torch.Tensor):
        batch_size = output.shape[0]
    elif isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor):
        batch_size = output[0].shape[0]
    else:
        raise ValueError("Unexpected output format from the model.")
    print(f"Inferred input batch size: {batch_size}")

    # Verify by running inference with new batch input
    batch_input = {}
    for id in MODEL_FEATURE_IDS:
        array_size = 1
        if id == 71:
            array_size = 200
        elif id == 154:
            array_size = 40
        batch_input[id] = torch.randn(batch_size, array_size)
    
    with torch.no_grad():
        new_output = model(batch_input)
    print("Output from model with batch input:")
    print(new_output)

    # Check if the new output matches the expected batch size
    if isinstance(new_output, torch.Tensor):
        output_batch_size = new_output.shape[0]
    elif isinstance(new_output, (tuple, list)) and isinstance(new_output[0], torch.Tensor):
        output_batch_size = new_output[0].shape[0]
    else:
        raise ValueError("Unexpected output format from verification step.")

    if output_batch_size == batch_size:
        print(f"Verification successful: Model correctly handles batch size {batch_size}")
        return True
    else:
        print(f"Verification failed: Expected batch size {batch_size}, but got {output_batch_size}")
        return False

def main():
    batch_size = infer_batch_size()
    
if __name__ == '__main__':
    main()
