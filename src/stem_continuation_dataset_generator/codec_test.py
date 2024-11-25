import math

import torchaudio
from stem_continuation_dataset_generator.codec import encode_file, get_codec, get_processor
from stem_continuation_dataset_generator.utils.device import get_device

FILE_PATH = 'resources/audio.ogg'

device = get_device()
codec = get_codec(device)
processor = get_processor(device)


def test_encode_file():
    file, frame_rate = encode_file(FILE_PATH, device)

    wav, sr = torchaudio.load(FILE_PATH, normalize=False)

    assert frame_rate == codec.config.frame_rate
    assert file.shape[0] == 4  # Number of codebooks

    # Enable these lines to produce an audio file for the encoded audio
    # 
    # decoded_audio, decoded_sr = decode(file.unsqueeze(0), device)
    # torchaudio.save('output/prova.wav', decoded_audio, sample_rate=decoded_sr)
    
    length_in_seconds = wav.shape[-1] / sr
    assert file.shape[1] == math.ceil(length_in_seconds * frame_rate)
