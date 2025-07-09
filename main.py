import torch


def main():
    assert torch.cuda.is_available()
    print("Hello, World! You have a GPU available!")
    print("CUDA version: " + torch.version.cuda)


if __name__ == "__main__":
    main()
