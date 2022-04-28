from typing import List

import cv2
import io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.data import camera
from tqdm import tqdm


def apply_circular_filter(image: np.ndarray) -> None:
    """
    Set image contents outside an inscribed circular region of radius side//2 to 0.
    In place operation.

    Args:
        image: Image to filter.
    Returns:
        Filtered image.
    """

    img_shape = image.shape[0]
    radius = img_shape // 2

    # calculate grid of distances from center pixel
    x_coords, y_coords = np.meshgrid(
        np.arange(-radius, radius), np.arange(-radius, radius)
    )
    distance_from_center = np.sqrt(x_coords ** 2 + y_coords ** 2)

    image[distance_from_center > radius] = 0

    return image


def rotate(image: np.ndarray, theta: float) -> np.ndarray:
    """
    Apply arbitrary angular rotation to image.

    Args:
        image: Numpy array to rotate.
        theta: Angle to rotate image by.
    Returns:
        Numpy array of rotated image.
    """

    pil_image = Image.fromarray(image)
    pil_image = pil_image.rotate(angle=theta)

    rotated = np.array(pil_image)

    return rotated


def sample_image() -> np.ndarray:
    """
    Example image to demonstrate Radon.

    Returns:
        Sample image to perform Radon analysis.
    """

    image = camera().astype(np.float64)
    apply_circular_filter(image)

    return image


def ramp_filter(oned_fft: np.ndarray) -> np.ndarray:
    """
    Ramp filter duplicated from scikit-image.
    https://github.com/scikit-image/scikit-image/blob/main/skimage/transform/radon_transform.py

    Args:
        oned_fft: Unfiltered 1D FFT.
    Returns:
        1D FFT with mitigated low-frequency artifacts. 
    """    

    n = np.concatenate(
        (
            np.arange(1, len(oned_fft) / 2 + 1, 2, dtype=int),
            np.arange(len(oned_fft) / 2 - 1, 0, -2, dtype=int),
        )
    )
    f = np.zeros(len(oned_fft))
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2
    fourier_filter = 2 * np.real(np.fft.fft(f))

    filtered_oned_fft = oned_fft * fourier_filter

    return filtered_oned_fft


def calculate_oned_fft_of_sums(image: np.ndarray, theta: float) -> np.ndarray:
    """
    Rotate the image by theta degrees, sum the columns, and calculate the 1D FFT of the sums.

    Args:
        image: Image as np.ndarray.
        theta: Azimuth angle theta.
    Returns:
        1D FFT at azimuth angle theta.
    """

    rotated = rotate(image, theta)
    summed = rotated.sum(axis=0)
    oned_fft = np.fft.fft(np.fft.ifftshift(summed))

    oned_fft = ramp_filter(oned_fft)
    oned_fft = np.fft.fftshift(oned_fft)

    return rotated, summed, oned_fft


def calculate_updated_reconstructed_twod_fft(oned_fft: np.ndarray, angle: float, reconstructed_twod_fft: np.ndarray) -> None:
    """
    """

    reconstructed_temp = np.zeros(reconstructed_twod_fft.shape, dtype=np.complex64)
    reconstructed_temp[reconstructed_twod_fft.shape[0]//2, :] = oned_fft

    reconstructed_twod_fft += rotate(reconstructed_temp.real, -angle) + (
        1j * rotate(reconstructed_temp.imag, -angle)
    )

    reconstructed_image = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(reconstructed_twod_fft))
    )

    apply_circular_filter(reconstructed_image)


def update_subplot_1(image: np.ndarray) -> None:
    """
    Plot the original 2D FFT of the original image.

    Args:
        iamge: Original sample image.
    """

    plt.subplot(2, 4, 1)
    plt.title("Original 2D FFT")
    plt.imshow(np.log10(np.abs(np.fft.fftshift(np.fft.fft2(image)))))
    plt.axis("off")


def update_subplot_2(rotated: np.ndarray) -> None:
    """
    Plot the image with a circular mask applied, randomly rotated.

    Args:
        rotated: Rotated image.
    """

    plt.subplot(2, 4, 2)
    plt.title("Rotated image")
    plt.imshow(rotated)
    plt.axis("off")


def update_subplot_3(reconstructed_twod_fft: np.ndarray) -> None:
    """
    Plot

    Args:
        reconstructed_twod_fft:
    """

    plt.subplot(2, 4, 3)
    plt.title("Radon 2D FFT")
    plt.imshow(np.abs(np.log10(reconstructed_twod_fft + 0.00000001)))
    plt.axis("off")


def update_subplot_4(reconstructed_twod_fft: np.ndarray) -> None:
    """
    Args:
        reconstructed_image:
    """

    reconstructed_image = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(reconstructed_twod_fft))
    )

    apply_circular_filter(reconstructed_image)

    plt.subplot(2, 4, 4)
    plt.title("Radon 2D IFFT Image")
    plt.imshow(np.abs(reconstructed_image))
    plt.axis("off")


def update_subplot_5(column_sums: np.ndarray) -> None:
    """
    Args:
        column_sums: 
    """

    plt.subplot(2, 4, 6)
    plt.title("Column sums")
    plt.plot(column_sums)
    plt.xticks([])
    plt.yticks([])


def update_subplot_6(oned_fft: np.ndarray) -> None:
    """
    Args:
        oned_fft:
    """

    plt.subplot(2, 4, 7)
    plt.title("1D FFT of column sums")
    plt.plot(np.abs(oned_fft))
    plt.xticks([0, len(oned_fft)//2, len(oned_fft)], [-len(oned_fft)//2, 0, len(oned_fft)//2])
    plt.yticks([])


def get_arr_from_fig(dpi: int = 100) -> Image:
    """
    Get np.ndarray from a figure.

    Args:
        fig: Matplotlib Figure.
        dpi: Resolution.
    Returns:
        Numpy array of image contents.
    """

    buf = io.BytesIO()
    plt.gcf().savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return Image.fromarray(img)


def write_gif(
    plots: List[Image.Image],
    path_str: str = "radon_transform.gif",
    last_frame_repeats: int = 10,
    duration: int = 120,
) -> None:
    """
    Write animated gif.

    Args:
        plots: Output plots to write.
        path_str: Path to output gif.
        last_frame_repeats: Number of repeats of last still.
        duration: Duration in seconds.
    """

    for _ in range(last_frame_repeats):
        plots.append(plots[-1])

    plots[0].save(
        path_str, save_all=True, append_images=plots[1:], duration=duration, loop=0,
    )


def radon_transform_example():
    """
    
    """

    plt.figure(figsize=(10.5, 5))

    plots = []

    image = sample_image()

    reconstructed_twod_fft = np.zeros(image.shape, dtype=np.complex64)

    for angle in tqdm(np.arange(0, 180, 5)):

        rotated, column_sums, oned_fft = calculate_oned_fft_of_sums(image, theta=angle)

        plt.clf()

        plt.suptitle(f"Rotation angle is {angle}", fontweight="bold")

        update_subplot_1(image)
        update_subplot_2(rotated)

        calculate_updated_reconstructed_twod_fft(oned_fft, angle, reconstructed_twod_fft)

        update_subplot_3(reconstructed_twod_fft)
        update_subplot_4(reconstructed_twod_fft)
        update_subplot_5(column_sums)
        update_subplot_6(oned_fft)

        plots.append(get_arr_from_fig())

    write_gif(plots)

if __name__ == "__main__":
    radon_transform_example()
