from functools import lru_cache
import math
from os import PathLike
import librosa.util
from transformers import EncodecModel, AutoProcessor
from encodec.utils import convert_audio
import torchaudio
import torch
from torch import Tensor
from typing import BinaryIO, List, Optional, Tuple, Union

from stem_continuation_dataset_generator.utils.device import Device

ENCODER_BATCH_SIZE = 1
ENCODED_TOKENS_PER_CHUNK = 512  # large values (over 1024) require a large amount of memory and can produce OOM errors


@lru_cache(maxsize=1)
def get_codec(device: Device):
    print(f'Encoding using device {device}')
    model = EncodecModel.from_pretrained("facebook/encodec_32khz", normalize=False, device_map=device)
    # print(model.config)
    return model.to(device).eval()


@lru_cache(maxsize=1)
def get_processor(device: Device):
    return AutoProcessor.from_pretrained("facebook/encodec_32khz", device_map=device)


def encode_file(audio_path: Union[BinaryIO, str, PathLike], device: Device, format: Optional[str] = None, batch_size: int = ENCODER_BATCH_SIZE) -> Tuple[Tensor, float]:
    # Load and pre-process the audio waveform
    wav, sr = torchaudio.load(audio_path, format=format, normalize=False)  # Normalization is later performed using librosa as it seems to work better
    return encode(wav, sr, device, batch_size=batch_size)


def get_total_chunks(samples_per_chunk: int, num_samples: int) -> int:

    return math.ceil(num_samples / samples_per_chunk)


def concat_chunks(chunks: List[Tensor], device: Device) -> Tensor:

    sequence = torch.cat(chunks, dim=-1)

    return sequence


def normalize_audio(audio: Tensor) -> Tensor:

    return torch.tensor(librosa.util.normalize(audio.numpy(), axis=1))


def chunk_list(lst, n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def encode(audio: Tensor, sr: int, device: Device, batch_size: int = ENCODER_BATCH_SIZE) -> Tuple[Tensor, float]:

    device = device if not device.startswith('mps') else 'cpu'  # Encoding is not supported on MPS
    processor = get_processor(device)
    codec = get_codec(device)

    wav = convert_audio(audio, sr, processor.sampling_rate, codec.config.audio_channels)
    wav = normalize_audio(wav)
    length_in_seconds = wav.shape[1] / processor.sampling_rate
    frames_no = math.ceil(length_in_seconds * codec.config.frame_rate)

    # split wav in chunks
    num_samples = wav.shape[1]
    samples_per_chunk = math.ceil((ENCODED_TOKENS_PER_CHUNK / codec.config.frame_rate) * processor.sampling_rate)
    total_chunks = get_total_chunks(samples_per_chunk, num_samples)
    chunks = []
    start_index = 0

    # Split into chunks
    for _ in range(total_chunks):
        end_index = start_index + samples_per_chunk
        chunk = wav[:, start_index:end_index]
        chunks.append(chunk.squeeze(0).numpy())  # Remove the first empty dimension
        start_index = end_index
    
    encoded_chunks = []

    # create audio chunks
    batches: List[List[Tensor]] = list(chunk_list(chunks, batch_size))

    for batch in batches:
        inputs = processor(raw_audio=batch, sampling_rate=processor.sampling_rate, return_tensors="pt")
        bandwidth = 2.2
        result = codec.encode(inputs["input_values"].to(device), inputs["padding_mask"].to(device), bandwidth=bandwidth)
        assert result.audio_codes.shape[0] == 1, 'Multiple elements returned by codec encoding, expected one'        
        sequence = result.audio_codes[0]
        
        # Concatenate the batch items into a single sequence
        batch_size, codebooks, seq_len = sequence.shape
        result = sequence.permute(1, 0, 2).reshape(codebooks, seq_len * batch_size)
        encoded_chunks.append(result)

    encoded_audio = concat_chunks(encoded_chunks, device=device)

    # Remove padding from the encoded audio
    encoded_audio = encoded_audio[:, :frames_no]
    
    return encoded_audio, codec.config.frame_rate


def decode(codes: Tensor, device: Device) -> Tuple[Tensor, int]:
    device = device if not device.startswith('mps') else 'cpu'  # Decoding is not supported on MPS
    codec = get_codec(device)
    decoded_wav = codec.decode(codes.unsqueeze(0).to(device), [None])
    output_tensor = decoded_wav['audio_values'].detach().squeeze(0)
    return output_tensor, codec.config.sampling_rate
