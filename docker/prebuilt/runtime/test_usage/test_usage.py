import soundfile, time
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text
d = ModelDownloader()
speech2text = Speech2Text("exp/asr_train_asr_transformer_e18_raw_bpe_sp/config.yaml", "exp/asr_train_asr_transformer_e18_raw_bpe_sp/54epoch.pth", maxlenratio=0.0, minlenratio=0.0, beam_size=50,ctc_weight=0.3,lm_weight=0.5,penalty=0.0,nbest=1)
#speech2text = Speech2Text(**d.download_and_unpack("kamo-naoyuki/mini_an4_asr_train_raw_bpe_valid.acc.best"), maxlenratio=0.0, minlenratio=0.0, beam_size=20,ctc_weight=0.3,lm_weight=0.5,penalty=0.0,nbest=1)
# Confirm the sampling rate is equal to that of the training corpus.
# If not, you need to resample the audio data before inputting to speech2text
start = time.time()
#speech, rate = soundfile.read("output_5220e1b43f5d488caa6be43b735e3365.wav")
speech, rate = soundfile.read("why_r_useful.wav")
nbests = speech2text(speech)

text, *_ = nbests[0]
end = time.time()
print(text)
print(end - start)