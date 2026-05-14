import numpy as np
import constriction


# =========================================================
# Basic utils
# =========================================================
def judege_type(min_value, max_value):
    """Determine a compact numpy dtype given value range."""
    if min_value >= 0:
        if max_value <= 255:
            return np.uint8
        elif max_value <= 65535:
            return np.uint16
        else:
            return np.uint32
    else:
        if max_value <= 127 and min_value >= -128:
            return np.int8
        elif max_value <= 32767 and min_value >= -32768:
            return np.int16
        else:
            return np.int32


def get_np_size(x):
    """Compute total byte size of numpy-style encoded outputs."""
    if isinstance(x, np.ndarray):
        return x.size * x.itemsize

    if isinstance(x, np.generic):
        return x.itemsize

    if isinstance(x, (list, tuple)):
        return sum(get_np_size(item) for item in x)

    array_x = np.asarray(x)
    return array_x.size * array_x.itemsize


def entropy_from_counts(counts):
    counts = np.asarray(counts, dtype=np.float64)
    probs = counts / counts.sum()
    nz = probs > 0
    return float(-np.sum(probs[nz] * np.log2(probs[nz])))


def _varint_num_bytes_scalar(v: int) -> int:
    v = int(v)
    n = 1
    while v >= 0x80:
        v >>= 7
        n += 1
    return n


def _varint_bits(arr):
    arr = np.asarray(arr).reshape(-1)
    return sum(_varint_num_bytes_scalar(int(v)) for v in arr) * 8


def _zigzag_encode(x):
    x = np.asarray(x, dtype=np.int64)
    return np.where(x >= 0, 2 * x, -2 * x - 1).astype(np.uint64)


def _delta_encode(x):
    x = np.asarray(x, dtype=np.int64)
    if x.size == 0:
        return x
    out = np.empty_like(x, dtype=np.int64)
    out[0] = x[0]
    out[1:] = x[1:] - x[:-1]
    return out


def _minimal_unsigned_dtype(max_value):
    if max_value <= np.iinfo(np.uint8).max:
        return np.uint8
    elif max_value <= np.iinfo(np.uint16).max:
        return np.uint16
    elif max_value <= np.iinfo(np.uint32).max:
        return np.uint32
    else:
        return np.uint64


# =========================================================
# Internal helpers
# =========================================================
def _encode_one_vector(vec):
    """
    Encode one 1D vector with categorical ANS.
    Returns stream package.
    """
    unique, unique_inverse, unique_counts = np.unique(
        vec, return_inverse=True, return_counts=True
    )

    min_value = unique.min()
    max_value = unique.max()
    unique = unique.astype(judege_type(min_value, max_value))

    message = unique_inverse.astype(np.int32)
    probabilities = unique_counts.astype(np.float64)
    probabilities /= probabilities.sum()

    entropy_model = constriction.stream.model.Categorical(probabilities, perfect=False)
    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(message, entropy_model)
    compressed = encoder.get_compressed()

    payload_bits = get_np_size(compressed) * 8

    # ---- estimate side info bits ----
    # unique values
    unique_bits = get_np_size(unique) * 8

    # counts: use a lighter estimate than raw int64
    counts_delta = _delta_encode(unique_counts)
    counts_zigzag = _zigzag_encode(counts_delta)
    counts_bits = _varint_bits(counts_zigzag)

    # length
    length_bits = _varint_bits(np.array([len(message)], dtype=np.uint64))

    side_bits = unique_bits + counts_bits + length_bits
    total_bits = payload_bits + side_bits

    return {
        "compressed": compressed,
        "unique_counts": unique_counts,
        "unique": unique,
        "length": int(len(message)),
        "payload_bits": int(payload_bits),
        "side_bits": int(side_bits),
        "total_bits": int(total_bits),
        "num_unique": int(len(unique)),
        "entropy_bits_per_symbol": entropy_from_counts(unique_counts),
        "bits_per_symbol_payload": float(payload_bits / max(len(message), 1)),
        "bits_per_symbol_total": float(total_bits / max(len(message), 1)),
    }


def _decode_one_vector(compressed, unique_counts, quant_symbol, symbol_length):
    probabilities = np.asarray(unique_counts, dtype=np.float64)
    probabilities /= probabilities.sum()

    entropy_model = constriction.stream.model.Categorical(probabilities, perfect=False)
    decoder = constriction.stream.stack.AnsCoder(compressed)
    decoded = decoder.decode(entropy_model, symbol_length)

    decoded = np.asarray(quant_symbol)[decoded]
    return decoded


# =========================================================
# Main compress API
# =========================================================
def compress_matrix_flatten_categorical(matrix, mode="split_channel", return_table=False):
    """
    mode:
        - "joint"          : flatten all symbols together
        - "split_channel"  : 2D per-channel ANS, 1D behaves as single stream
        - "delta_split"    : 2D each channel delta+zigzag then ANS

    Main return:
        summary dict with:
            original_bits
            compressed_bits
            compression_ratio
            shape

    If return_table=True:
        also returns an auxiliary table dict.
    """
    matrix = np.array(matrix)
    original_bits = int(matrix.size * matrix.dtype.itemsize * 8)
    shape = tuple(matrix.shape)

    if matrix.ndim == 1:
        pkg = _encode_one_vector(matrix)
        shape_bits = _varint_bits(np.array(shape, dtype=np.uint64))
        compressed_bits = int(pkg["total_bits"] + shape_bits)

        summary = {
            "original_bits": original_bits,
            "compressed_bits": compressed_bits,
            "compression_ratio": float(original_bits / max(compressed_bits, 1)),
            "shape": shape,
        }

        table = {
            "mode": "1d",
            "streams": [
                {
                    "stream_id": 0,
                    "length": pkg["length"],
                    "num_unique": pkg["num_unique"],
                    "payload_bits": pkg["payload_bits"],
                    "side_bits": pkg["side_bits"],
                    "total_bits": pkg["total_bits"],
                    "entropy_bits_per_symbol": pkg["entropy_bits_per_symbol"],
                    "bits_per_symbol_payload": pkg["bits_per_symbol_payload"],
                    "bits_per_symbol_total": pkg["bits_per_symbol_total"],
                }
            ],
            "shape_bits": int(shape_bits),
            "compressed_obj": pkg["compressed"],
            "unique_counts": pkg["unique_counts"],
            "unique": pkg["unique"],
        }

        if return_table:
            return summary, table
        return summary

    elif matrix.ndim == 2:
        N, C = matrix.shape

        if mode == "joint":
            vec = matrix.reshape(-1)
            pkg = _encode_one_vector(vec)
            shape_bits = _varint_bits(np.array(shape, dtype=np.uint64))
            compressed_bits = int(pkg["total_bits"] + shape_bits)

            summary = {
                "original_bits": original_bits,
                "compressed_bits": compressed_bits,
                "compression_ratio": float(original_bits / max(compressed_bits, 1)),
                "shape": shape,
            }

            table = {
                "mode": "joint",
                "streams": [
                    {
                        "stream_id": 0,
                        "length": pkg["length"],
                        "num_unique": pkg["num_unique"],
                        "payload_bits": pkg["payload_bits"],
                        "side_bits": pkg["side_bits"],
                        "total_bits": pkg["total_bits"],
                        "entropy_bits_per_symbol": pkg["entropy_bits_per_symbol"],
                        "bits_per_symbol_payload": pkg["bits_per_symbol_payload"],
                        "bits_per_symbol_total": pkg["bits_per_symbol_total"],
                    }
                ],
                "shape_bits": int(shape_bits),
                "compressed_obj": [pkg["compressed"]],
                "unique_counts": [pkg["unique_counts"]],
                "unique": [pkg["unique"]],
                "symbol_length": int(matrix.size),
                "symbol_shape": shape,
            }

            if return_table:
                return summary, table
            return summary

        elif mode in ["split_channel", "delta_split"]:
            streams = []
            payload_bits_sum = 0
            side_bits_sum = 0
            stream_table = []

            compressed_obj = []
            unique_counts_all = []
            unique_all = []

            for c in range(C):
                vec = matrix[:, c]

                if mode == "delta_split":
                    d = _delta_encode(vec.astype(np.int64))
                    z = _zigzag_encode(d)
                    vec_to_encode = z.astype(_minimal_unsigned_dtype(int(z.max())))
                else:
                    vec_to_encode = vec

                pkg = _encode_one_vector(vec_to_encode)
                streams.append(pkg)

                payload_bits_sum += pkg["payload_bits"]
                side_bits_sum += pkg["side_bits"]

                compressed_obj.append(pkg["compressed"])
                unique_counts_all.append(pkg["unique_counts"])
                unique_all.append(pkg["unique"])

                stream_table.append({
                    "stream_id": c,
                    "length": pkg["length"],
                    "num_unique": pkg["num_unique"],
                    "payload_bits": pkg["payload_bits"],
                    "side_bits": pkg["side_bits"],
                    "total_bits": pkg["total_bits"],
                    "entropy_bits_per_symbol": pkg["entropy_bits_per_symbol"],
                    "bits_per_symbol_payload": pkg["bits_per_symbol_payload"],
                    "bits_per_symbol_total": pkg["bits_per_symbol_total"],
                })

            shape_bits = _varint_bits(np.array(shape, dtype=np.uint64))
            compressed_bits = int(payload_bits_sum + side_bits_sum + shape_bits)

            summary = {
                "original_bits": original_bits,
                "compressed_bits": compressed_bits,
                "compression_ratio": float(original_bits / max(compressed_bits, 1)),
                "shape": shape,
            }

            table = {
                "mode": mode,
                "streams": stream_table,
                "shape_bits": int(shape_bits),
                "compressed_obj": compressed_obj,
                "unique_counts": unique_counts_all,
                "unique": unique_all,
                "symbol_length": int(matrix.size),
                "symbol_shape": shape,
            }

            if return_table:
                return summary, table
            return summary

        else:
            raise ValueError("mode must be one of ['joint', 'split_channel', 'delta_split']")

    else:
        raise ValueError("matrix must be 1D or 2D")


# =========================================================
# Main decompress API
# =========================================================
def decompress_matrix_flatten_categorical(table):
    """
    Decompress from auxiliary table returned by compress(..., return_table=True).

    Supports:
        - 1d
        - joint
        - split_channel
        - delta_split
    """
    mode = table["mode"]

    # 1D
    if mode == "1d":
        stream = table["streams"][0]
        decoded = _decode_one_vector(
            compressed=table["compressed_obj"],
            unique_counts=table["unique_counts"],
            quant_symbol=table["unique"],
            symbol_length=stream["length"],
        )
        return decoded

    # joint
    elif mode == "joint":
        stream = table["streams"][0]
        decoded = _decode_one_vector(
            compressed=table["compressed_obj"][0],
            unique_counts=table["unique_counts"][0],
            quant_symbol=table["unique"][0],
            symbol_length=stream["length"],
        )
        return decoded.reshape(table["symbol_shape"])

    # split_channel
    elif mode == "split_channel":
        shape = table["symbol_shape"]
        N, C = shape
        decoded_matrix = np.zeros((N, C), dtype=np.int64)

        for c in range(C):
            length = table["streams"][c]["length"]
            decoded = _decode_one_vector(
                compressed=table["compressed_obj"][c],
                unique_counts=table["unique_counts"][c],
                quant_symbol=table["unique"][c],
                symbol_length=length,
            )
            decoded_matrix[:, c] = decoded

        return decoded_matrix.reshape(shape)

    # delta_split
    elif mode == "delta_split":
        shape = table["symbol_shape"]
        N, C = shape
        decoded_matrix = np.zeros((N, C), dtype=np.int64)

        for c in range(C):
            length = table["streams"][c]["length"]
            z = _decode_one_vector(
                compressed=table["compressed_obj"][c],
                unique_counts=table["unique_counts"][c],
                quant_symbol=table["unique"][c],
                symbol_length=length,
            ).astype(np.int64)

            # inverse zigzag
            d = np.where((z & 1) == 0, z // 2, -(z // 2) - 1)

            # inverse delta
            x = np.cumsum(d, axis=0)
            decoded_matrix[:, c] = x

        return decoded_matrix.reshape(shape)

    else:
        raise ValueError(f"Unsupported mode: {mode}")