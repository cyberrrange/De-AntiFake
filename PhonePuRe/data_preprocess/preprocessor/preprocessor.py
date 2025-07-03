import os
import random
import math

from praatio import tgio
import random


class Preprocessor:
    def __init__(self, config):
        self.config = config
        # Path
        self.clean_flist = config["path"]["clean_flist"]
        self.tgt_dir = config["path"]["textgrid_path"]
        self.out_file = config["path"]["dataset_file_path"]
        # audio preprocessing parameters
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.filter_length = config["preprocessing"]["stft"]["filter_length"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]
        self.win_length = config["preprocessing"]["stft"]["win_length"]

    def build_from_path(self):
        print("Processing Data ...")
        out = list()
        n_frames = 0

        with open(self.clean_flist) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                audiopath = line.strip()

                basename = os.path.basename(audiopath).split(".")[0]
                speaker = audiopath.split('/')[-2]
                tg_path = os.path.join(
                    self.tgt_dir, speaker, "{}.TextGrid".format(basename)
                )
                if self.config['dataset'] == 'LIBRI_NO_TEXT':
                    speaker = basename.split('-')[0]
                    book = basename.split('-')[1]
                    tg_path = os.path.join(
                        self.tgt_dir, speaker, book, "{}.TextGrid".format(basename)
                    )
                if self.config['dataset'] == 'LIBRI_PROCESSED':
                    speaker = basename.split('-')[0]
                    book = basename.split('-')[1]
                    tg_path = os.path.join(
                        self.tgt_dir, speaker, "{}.TextGrid".format(basename)
                    )
                if self.config['dataset'] == 'RULIBRI':
                    speaker = audiopath.split('/')[-3]
                    book = audiopath.split('/')[-2]
                    tg_path = os.path.join(
                        self.tgt_dir, speaker, book, "{}.TextGrid".format(basename)
                    )
                print(tg_path)
                if os.path.exists(tg_path):
                    ret = self.process_utterance(speaker, tg_path, audiopath)
                    if ret is None:
                        continue
                    else:
                        info, n = ret
                    out.append(info)

                n_frames += n

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        random.shuffle(out)
        out = [r for r in out if r is not None]

        # Write metadata
        with open(self.out_file, "w", encoding="utf-8") as f:
            for m in out:
                f.write(m + "\n")

        return out

    def process_utterance(self, speaker, tgpath, audiopath):
        # Get alignments
        print("open textgrid: ", tgpath)
        textgrid = tgio.openTextgrid(tgpath)
        print("finish open textgrid")
        phone, durations, stamps = self.get_alignment(
            textgrid.tierDict["phones"].entryList
        )
        #total_seconds = sum(durations) * self.hop_length / float(self.sampling_rate)
        #if total_seconds < 1.5:
        #    return None
        text = " ".join(phone)

        stamps_str = " ".join([str(stamp) for stamp in stamps])
        durations_str = " ".join([str(duration) for duration in durations])

        return (
            "|".join([stamps_str, durations_str,
                     text, audiopath, speaker]),
            sum(durations),
        )

    def get_alignment(self, tier):
        def time2frame(t):
            return math.floor((t+1e-9) * self.sampling_rate / self.hop_length)

        sil_phones = ["", "sp", "spn", '', "''", '""']

        phones = []
        stamps = []
        durations = []
        #print(tier)
        previous_end = 0.0  # 初始化前一个结束时间
        for t in tier:
            s, e, p = t.start, t.end, t.label

            # 检查当前间隔是否存在间隙（静音）
            if s > previous_end:  # 有间隙，添加静音
                phones.append('sil')
                start_frame = time2frame(previous_end)
                end_frame = time2frame(s)
                stamps.append(start_frame)
                durations.append(end_frame - start_frame)

            # 添加当前间隔
            if p not in sil_phones:
                phones.append(p)  # 普通音素
            else:
                phones.append('sil')  # 静音音素

            start_frame = time2frame(s)
            end_frame = max(time2frame(e), start_frame + 1)
            stamps.append(start_frame)
            durations.append(end_frame - start_frame)

            previous_end = e  # 更新上一个结束时间

        # 如果最后一个间隔未结束于总时长，补充静音
        if previous_end < tier[-1].end:
            phones.append('sil')
            start_frame = time2frame(previous_end)
            end_frame = time2frame(tier[-1].end)
            stamps.append(start_frame)
            durations.append(end_frame - start_frame)

        return phones, durations, stamps
