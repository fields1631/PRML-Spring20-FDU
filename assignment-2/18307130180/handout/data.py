
import tensorflow as tf
import numpy as np

def generate_batch(batch_size, max_len):
    """
    Generate a batch of data.

    Parameters
    ----------
    batch_size: int, positive
        Batch size.
    max_len: int, positive
        Max length of data.

    Returns
    -------
    nums1:
        The first operand.
    nums2:
        The second operand.
    results:
        nums1 + nums2.
    """
    nums1 = np.random.randint(0, 10, size=(batch_size, max_len))
    nums2 = np.random.randint(0, 10, size=(batch_size, max_len))
    results = np.zeros(shape=(batch_size, max_len))

    for i in range(batch_size):
        carry = 0
        for j in range(max_len):
            num1, num2 = nums1[i, j], nums2[i, j]
            results[i, j] = (carry + num1 + num2) % 10
            carry = (carry + num1 + num2) // 10

    return nums1, nums2, results

def row_to_string(row):
    """
    Convert a Tensor of digits to string.

    Parameters
    ----------
    row: Tensor
        Digits of long integer.

    Returns
    -------
    line:
        string of digits.
    """
    row = tf.keras.backend.get_value(row).astype(int)
    line = ''.join([str(_) for _ in row])
    line = line[::-1]

    return line
