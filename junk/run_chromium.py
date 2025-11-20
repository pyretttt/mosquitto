import subprocess

pages = [
    ("Demystifying_Tensor_Parallelism", "https://robotchinwag.com/posts/demystifying-tensor-parallelism/"),
    ("The_Tensor_Calculus_You_Need_for_Deep_Learning", "https://robotchinwag.com/posts/the-tensor-calculus-you-need-for-deep-learning/"),
    ("Einsum", "https://robotchinwag.com/posts/einsum-gradient/"),
    ("Matrix_Inverse", "https://robotchinwag.com/posts/maxtrix-inverse-gradient/"),
    ("Cross-Entropy", "https://robotchinwag.com/posts/crossentropy-loss-gradient/"),
    ("Linear_Layer", "https://robotchinwag.com/posts/linear-layer-deriving-the-gradient-for-the-backward-pass/"),
    ("Layer_Normalization", "https://robotchinwag.com/posts/layer-normalization-deriving-the-gradient-for-the-backward-pass/"),
    ("Backpropagation_and_Multivariable_Calculus", "https://robotchinwag.com/posts/backpropagation-and-multivariable-calculus/")
]


if __name__ == "__main__":
    for name, link in pages:
        subprocess.run([
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "--headless",
            "--disable-gpu",
            f"--print-to-pdf={name}.pdf",
            link
        ])