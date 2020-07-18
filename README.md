# End-to-end Automatic Speech Recognition Systems - PyTorch Implementation
For complete introdution and usage, please see the original repository [Alexander-H-Liu/End-to-end-ASR-Pytorch](https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch).
## New features
1. Added layer-wise transfer learning
2. Supports multiple development sets
3. Fixed some bugs

## Instructions
### Training
Modify `script/train.sh`, `script/train_lm.sh`, `config/librispeech_asr.yaml`, and `config/librispeech_lm.yaml` first. GPU is required.
```
bash script/train.sh <asr name> <cuda id>
bash script/train_lm.sh <lm name> <cuda id>
```
### Testing
Modify `script/test.sh` and `config/librispeech_test.sh` first. Increase the number of `--njobs` can speed up decoding process, but might cause OOM.
```
bash script/test.sh <asr name> <cuda id>
```

## Baseline model
Both joint CTC-attention ASR model and RNNLM are trained on the LibriSpeech `train-clean-100` set. The perplexity of the LM on `dev-clean` set is 3.66. 

| Decoding | DEV WER(%) | TEST WER(%) |
| -------- | ---------- | ----------- |
| Greedy   |            |             |
| Beam=2   |            |             |
| Beam=4   |            |             |
| Beam=8   |            |             |

## TODO
1. CTC beam decoding (testing)
2. SpecAugment (will be released)
3. Multiple corpora training (will be released)
4. Evaluation of the results of beam decoding (will be released)
5. Combination of CTC and RNNLM: RNN transducer (under construction)
6. Combining voice conversion model with ASR (under construction)

## Citation

```
@inproceedings{liu2019adversarial,
  title={Adversarial Training of End-to-end Speech Recognition Using a Criticizing Language Model},
  author={Liu, Alexander and Lee, Hung-yi and Lee, Lin-shan},
  booktitle={International Conference on Speech RecognitionAcoustics, Speech and Signal Processing (ICASSP)},
  year={2019},
  organization={IEEE}
}

@inproceedings{alex2019sequencetosequence,
    title={Sequence-to-sequence Automatic Speech Recognition with Word Embedding Regularization and Fused Decoding},
    author={Alexander H. Liu and Tzu-Wei Sung and Shun-Po Chuang and Hung-yi Lee and Lin-shan Lee},
    booktitle={International Conference on Speech RecognitionAcoustics, Speech and Signal Processing (ICASSP)},
    year={2020},
    organization={IEEE}
}

@misc{chang2020endtoend,
    title={End-to-end Whispered Speech Recognition with Frequency-weighted Approaches and Layer-wise Transfer Learning},
    author={Heng-Jui Chang and Alexander H. Liu and Hung-yi Lee and Lin-shan Lee},
    year={2020},
    eprint={2005.01972},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
